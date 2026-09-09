"""Shared shapes, weights, and contiguous KV-cache state for all backends.

This module deliberately contains no Mirage- or Triton-specific code.  It is
the contract that keeps the three benchmark backends semantically identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class Qwen3BenchmarkConfig:
    vocab_size: int = 4096
    hidden_size: int = 256
    intermediate_size: int = 768
    num_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 32
    max_seq_len: int = 160
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    def __post_init__(self) -> None:
        assert self.hidden_size == self.num_attention_heads * self.head_dim
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_size(self) -> int:
        return self.hidden_size + 2 * self.kv_size

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def qwen3_8b_core(cls, *, num_layers: int = 1, vocab_size: int = 4096, max_seq_len: int = 160) -> "Qwen3BenchmarkConfig":
        """Real Qwen3-8B Transformer dimensions with a configurable LM vocab.

        Keeping the default vocabulary at 4096 isolates Transformer latency.
        Pass ``--vocab-size 151936`` when the full LM-head cost is desired.
        """
        return cls(
            vocab_size=vocab_size,
            hidden_size=4096,
            intermediate_size=12288,
            num_layers=num_layers,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            max_seq_len=max_seq_len,
        )


class LayerWeights(NamedTuple):
    input_norm: torch.Tensor
    qkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: torch.Tensor
    post_norm: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor


class Qwen3Weights:
    """One deterministic BF16 weight set shared by all benchmark backends."""

    def __init__(self, cfg: Qwen3BenchmarkConfig, device: torch.device, dtype: torch.dtype, seed: int) -> None:
        self.cfg = cfg
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        rand = lambda *shape: torch.randn(*shape, device=device, dtype=dtype, generator=gen) * 0.02
        self.embedding = rand(cfg.vocab_size, cfg.hidden_size)
        self.layers = tuple(
            LayerWeights(
                torch.ones(cfg.hidden_size, device=device, dtype=dtype),
                rand(cfg.qkv_size, cfg.hidden_size),
                torch.ones(cfg.head_dim, device=device, dtype=dtype),
                torch.ones(cfg.head_dim, device=device, dtype=dtype),
                rand(cfg.hidden_size, cfg.hidden_size),
                torch.ones(cfg.hidden_size, device=device, dtype=dtype),
                rand(2 * cfg.intermediate_size, cfg.hidden_size),
                rand(cfg.hidden_size, cfg.intermediate_size),
            )
            for _ in range(cfg.num_layers)
        )
        self.final_norm = torch.ones(cfg.hidden_size, device=device, dtype=dtype)
        self.lm_head = rand(cfg.vocab_size, cfg.hidden_size)


class ContiguousKVCache:
    """Preallocated [layer, K/V, batch, KVH, position, D] cache.

    Writes are in-place; decode never reallocates or copies its history.
    """

    def __init__(self, cfg: Qwen3BenchmarkConfig, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg, self.batch = cfg, batch
        self.storage = torch.empty(
            cfg.num_layers, 2, batch, cfg.num_key_value_heads, cfg.max_seq_len, cfg.head_dim,
            device=device, dtype=dtype,
        )
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        # k/v are [B, KVH, S, D].  The returned tensors include all history.
        count = k.shape[2]
        start, end = self.length, self.length + count
        if end > self.cfg.max_seq_len:
            raise ValueError(f"KV cache overflow: need {end}, max is {self.cfg.max_seq_len}")
        self.storage[layer, 0, :, :, start:end, :].copy_(k)
        self.storage[layer, 1, :, :, start:end, :].copy_(v)
        return self.storage[layer, 0, :, :, :end, :], self.storage[layer, 1, :, :, :end, :], start

    def finish_layer_stack(self, appended_tokens: int) -> None:
        self.length += appended_tokens


def build_rope_table(cfg: Qwen3BenchmarkConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(cfg.max_seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim))
    phase = torch.outer(positions, inv_freq)
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos().to(dtype), phase.sin().to(dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * weight
