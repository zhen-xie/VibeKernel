"""B=1 paged-KV variant of the validated Triton Qwen3 backend."""
from pathlib import Path
import sys
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen3_paged"))
from paged_cache import PagedKVCache
from triton_backend import TritonBackend, triton_gqa_attention, triton_rope
_local_spec = importlib.util.spec_from_file_location("_triton_paged_cache", Path(__file__).with_name("paged_cache.py"))
_local = importlib.util.module_from_spec(_local_spec)
_local_spec.loader.exec_module(_local)
paged_gather, paged_write = _local.paged_gather, _local.paged_write


class PagedTritonBackend(TritonBackend):
    name = "triton-paged"

    def __init__(self, cfg, weights, batch, profiler=None, page_size=64):
        if batch != 1:
            raise ValueError("initial paged Triton backend supports one request")
        super().__init__(cfg, weights, batch, profiler)
        pages = (cfg.max_seq_len + page_size - 1) // page_size
        self.cache = PagedKVCache(cfg.num_layers, pages, page_size, cfg.num_key_value_heads,
                                  cfg.head_dim, weights.embedding.device, weights.embedding.dtype)

    def _layer(self, x, layer_id, sequence, causal):
        # Reuse every validated Triton operation up through RoPE; replace only
        # contiguous cache traffic and pass gathered logical history to GQA.
        c, w = self.cfg, self.weights.layers[layer_id]; residual = x
        from triton_backend import triton_rmsnorm, triton_linear, triton_split_qkv, triton_heads_to_rows, triton_add, triton_split_halves, triton_silu_mul
        h = self._m("input_rmsnorm", lambda: triton_rmsnorm(x, w.input_norm, c.rms_norm_eps))
        qkv = self._m("qkv_gemm", lambda: triton_linear(h, w.qkv)); q, k, v = triton_split_qkv(qkv, 1, sequence, c)
        q = self._m("qk_rmsnorm", lambda: triton_rmsnorm(q, w.q_norm, c.rms_norm_eps)); k = triton_rmsnorm(k, w.k_norm, c.rms_norm_eps)
        past = self.cache.length; q = self._m("rope", lambda: triton_rope(q, self.cos, self.sin, past)); k = triton_rope(k, self.cos, self.sin, past)
        kc, vc = self.cache.storage[layer_id, 0], self.cache.storage[layer_id, 1]
        self._m("kv_cache_write", lambda: (paged_write(k, kc, self.cache.metadata.indices, past), paged_write(v, vc, self.cache.metadata.indices, past)))
        kh, vh = paged_gather(kc, self.cache.metadata.indices, past + sequence), paged_gather(vc, self.cache.metadata.indices, past + sequence)
        attn = self._m("attention", lambda: triton_gqa_attention(q, kh, vh, past=past, causal=causal)); attn = triton_heads_to_rows(attn)
        x = self._m("o_gemm_residual", lambda: triton_add(residual, triton_linear(attn, w.o_proj))); residual = x
        h = self._m("post_rmsnorm", lambda: triton_rmsnorm(x, w.post_norm, c.rms_norm_eps)); gate, up = triton_split_halves(self._m("gateup_gemm", lambda: triton_linear(h, w.gate_up)))
        return self._m("down_gemm_residual", lambda: triton_add(residual, triton_linear(self._m("silu_mul", lambda: triton_silu_mul(gate, up)), w.down)))
