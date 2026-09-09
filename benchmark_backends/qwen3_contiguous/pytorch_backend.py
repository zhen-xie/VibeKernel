"""PyTorch eager / cuBLAS-oriented contiguous-KV Qwen3 baseline."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from common import ContiguousKVCache, Qwen3BenchmarkConfig, Qwen3Weights, build_rope_table, rms_norm


class PyTorchBackend:
    name = "pytorch-eager"

    def __init__(self, cfg: Qwen3BenchmarkConfig, weights: Qwen3Weights, batch: int) -> None:
        self.cfg, self.weights, self.batch = cfg, weights, batch
        self.cache = ContiguousKVCache(cfg, batch, weights.embedding.device, weights.embedding.dtype)
        self.cos, self.sin = build_rope_table(cfg, weights.embedding.device, weights.embedding.dtype)

    def reset_cache(self) -> None:
        self.cache.reset()

    def _rope(self, q: torch.Tensor, k: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        size, half = q.shape[2], self.cfg.head_dim // 2
        cos = self.cos[start:start + size][None, None]
        sin = self.sin[start:start + size][None, None]
        rotate = lambda x: torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past: int, causal: bool) -> torch.Tensor:
        # Keep GQA grouped: [B,KVH,R,S,D], never materialize repeated K/V.
        b, _, sq, d = q.shape
        qg = q.view(b, self.cfg.num_key_value_heads, self.cfg.q_per_kv, sq, d)
        scores = torch.einsum("bhrsd,bhtd->bhrst", qg.float(), k.float()) * (d ** -0.5)
        if causal:
            qpos = torch.arange(past, past + sq, device=q.device)[:, None]
            kpos = torch.arange(k.shape[2], device=q.device)[None, :]
            scores.masked_fill_(kpos > qpos, float("-inf"))
        probs = scores.softmax(dim=-1).to(q.dtype)
        out = torch.einsum("bhrst,bhtd->bhrsd", probs, v)
        return out.reshape(b, self.cfg.num_attention_heads, sq, d)

    def _layer(self, x: torch.Tensor, layer_id: int, causal: bool) -> torch.Tensor:
        c, w = self.cfg, self.weights.layers[layer_id]
        residual = x
        h = rms_norm(x, w.input_norm, c.rms_norm_eps)
        qkv = F.linear(h, w.qkv)
        q, k, v = qkv.split((c.hidden_size, c.kv_size, c.kv_size), dim=-1)
        b, s, _ = q.shape
        q = rms_norm(q.view(b, s, c.num_attention_heads, c.head_dim), w.q_norm, c.rms_norm_eps).transpose(1, 2)
        k = rms_norm(k.view(b, s, c.num_key_value_heads, c.head_dim), w.k_norm, c.rms_norm_eps).transpose(1, 2)
        v = v.view(b, s, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        past = self.cache.length
        q, k = self._rope(q, k, past)
        k_all, v_all, _ = self.cache.append(layer_id, k, v)
        attn = self._attention(q, k_all, v_all, past, causal).transpose(1, 2).reshape(b, s, c.hidden_size)
        x = residual + F.linear(attn, w.o_proj)
        residual = x
        h = rms_norm(x, w.post_norm, c.rms_norm_eps)
        gate, up = F.linear(h, w.gate_up).chunk(2, dim=-1)
        return residual + F.linear(F.silu(gate) * up, w.down)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.reset_cache()
        x = F.embedding(input_ids, self.weights.embedding)
        for layer_id in range(self.cfg.num_layers):
            x = self._layer(x, layer_id, causal=True)
        self.cache.finish_layer_stack(input_ids.shape[1])
        x = rms_norm(x, self.weights.final_norm, self.cfg.rms_norm_eps)
        return F.linear(x[:, -1], self.weights.lm_head)

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        x = F.embedding(token_ids, self.weights.embedding)
        for layer_id in range(self.cfg.num_layers):
            x = self._layer(x, layer_id, causal=False)
        self.cache.finish_layer_stack(1)
        x = rms_norm(x, self.weights.final_norm, self.cfg.rms_norm_eps)
        return F.linear(x[:, 0], self.weights.lm_head)
