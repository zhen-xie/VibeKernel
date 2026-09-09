"""Pure-Triton kernels and backend for the Qwen3 benchmark.

The shared tensor shapes, random weights, and contiguous-KV contract live in
the sibling ``qwen3_contiguous`` reference backend.  Keeping the compute code
here prevents PyTorch and Triton implementations from being conflated.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Optional, TypeVar

import torch

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "qwen3_contiguous"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))
from common import ContiguousKVCache, Qwen3BenchmarkConfig, Qwen3Weights, build_rope_table  # noqa: E402

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # clearer error on an incomplete benchmark env
    raise ImportError("The Triton backend needs `pip install triton` in this CUDA environment.") from exc


T = TypeVar("T")


@triton.jit
def _rmsnorm_kernel(x, weight, out, cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    value = tl.load(x + row * cols + col, mask=col < cols, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(value * value, axis=0) / cols + eps)
    w = tl.load(weight + col, mask=col < cols, other=0.0)
    tl.store(out + row * cols + col, value * scale * w, mask=col < cols)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference Triton RMSNorm for a contiguous ``[..., hidden]`` tensor."""
    cols = x.shape[-1]
    if cols > 65536:
        raise ValueError(f"Triton reference RMSNorm supports <= 65536 columns, got {cols}")
    rows = x.numel() // cols
    out = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](
        x, weight, out, cols, eps=eps, block=triton.next_power_of_2(cols), num_warps=8
    )
    return out


@triton.jit
def _linear_kernel(
    x, weight, out,
    M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute ``out[M, N] = x[M, K] @ weight[N, K].T`` in FP32 accumulate."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        lhs = tl.load(
            x + rows[:, None] * K + kk[None, :],
            mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0,
        )
        # ``weight`` follows torch.nn.functional.linear's [out_features, in_features] layout.
        rhs = tl.load(
            weight + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < N) & (kk[:, None] < K), other=0.0,
        )
        acc += tl.dot(lhs, rhs)
    tl.store(out + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def triton_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Triton equivalent of ``F.linear(x, weight)`` for contiguous 2D BF16 tensors.

    A single explicit implementation is important for the benchmark: unlike
    calling ``torch.matmul``, it cannot silently dispatch to cuBLAS/cuBLASLt.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("triton_linear expects x[M, K] and weight[N, K]")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("triton_linear requires contiguous inputs")
    m, k, n = x.shape[0], x.shape[1], weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _linear_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x, weight, out, m, n, k,
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4,
    )
    return out


@triton.jit
def _rope_kernel(x, cos, sin, out, S, D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    """Rotate one [B, head, sequence] row with Qwen's half-split RoPE."""
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    position = row % S
    first = tl.load(x + row * D + d, mask=d < HALF, other=0.0)
    second = tl.load(x + row * D + HALF + d, mask=d < HALF, other=0.0)
    c = tl.load(cos + position * D + d, mask=d < HALF, other=1.0)
    s = tl.load(sin + position * D + d, mask=d < HALF, other=0.0)
    tl.store(out + row * D + d, first * c - second * s, mask=d < HALF)
    tl.store(out + row * D + HALF + d, second * c + first * s, mask=d < HALF)


def triton_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int) -> torch.Tensor:
    """Apply precomputed RoPE to contiguous ``x[B, heads, S, head_dim]``."""
    if x.ndim != 4 or not x.is_contiguous():
        raise ValueError("triton_rope expects contiguous x[B, heads, S, head_dim]")
    _, _, s, d = x.shape
    if d % 2 or d > 512:
        raise ValueError("reference Triton RoPE expects an even head_dim <= 512")
    out = torch.empty_like(x)
    _rope_kernel[(x.numel() // d,)](
        x, cos[start:], sin[start:], out, s, D=d, HALF=d // 2,
        BLOCK=triton.next_power_of_2(d // 2), num_warps=4,
    )
    return out


@triton.jit
def _cache_write_kernel(src, cache, start, elements, S: tl.constexpr, D: tl.constexpr,
                        MAX_T: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # p indexes src[B, KVH, S, D].  Cache is [B, KVH, MAX_T, D].
    bh = p // (S * D)
    within = p % (S * D)
    position = within // D
    column = within % D
    cache_offset = bh * (MAX_T * D) + (start + position) * D + column
    value = tl.load(src + p, mask=p < elements)
    tl.store(cache + cache_offset, value, mask=p < elements)


def triton_cache_write(src: torch.Tensor, cache: torch.Tensor, start: int) -> None:
    """In-place append ``src[B, KVH, S, D]`` into contiguous cache history."""
    if src.ndim != 4 or cache.ndim != 4 or not src.is_contiguous() or not cache.is_contiguous():
        raise ValueError("triton_cache_write requires contiguous [B, heads, sequence, D] tensors")
    b, h, s, d = src.shape
    if cache.shape[0] != b or cache.shape[1] != h or cache.shape[3] != d:
        raise ValueError("KV cache shape is incompatible with source")
    if start < 0 or start + s > cache.shape[2]:
        raise ValueError("KV cache write would exceed allocated sequence length")
    n = src.numel()
    _cache_write_kernel[(triton.cdiv(n, 256),)](
        src, cache, start, n, S=s, D=d, MAX_T=cache.shape[2], BLOCK=256,
    )


@triton.jit
def _gqa_attention_kernel(
    q, k, v, out,
    HQ: tl.constexpr, HKV: tl.constexpr, S: tl.constexpr, T: tl.constexpr, CACHE_T: tl.constexpr,
    D: tl.constexpr, GROUP_SIZE: tl.constexpr, PAST: tl.constexpr,
    CAUSAL: tl.constexpr, SM_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program computes one query row using online softmax over K/V."""
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    sequence = pid % S
    tmp = pid // S
    q_head = tmp % HQ
    batch = tmp // HQ
    kv_head = q_head // GROUP_SIZE

    q_offset = ((batch * HQ + q_head) * S + sequence) * D
    q_value = tl.load(q + q_offset + d, mask=d < D, other=0.0).to(tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    # In decode, S=1 and PAST is the already-filled cache length.  In
    # prefill, each row may attend only through PAST + its own position.
    last_visible = PAST + sequence if CAUSAL else T - 1
    for start_n in range(0, T, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        valid = n < T
        if CAUSAL:
            valid = valid & (n <= last_visible)
        kv_offset = ((batch * HKV + kv_head) * CACHE_T + n[:, None]) * D + d[None, :]
        key = tl.load(k + kv_offset, mask=(n[:, None] < T) & (d[None, :] < D), other=0.0).to(tl.float32)
        score = tl.sum(key * q_value[None, :], axis=1) * SM_SCALE
        score = tl.where(valid, score, -float("inf"))
        tile_max = tl.max(score, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        probability = tl.exp(score - new_max)
        alpha = tl.exp(running_max - new_max)
        value = tl.load(v + kv_offset, mask=(n[:, None] < T) & (d[None, :] < D), other=0.0).to(tl.float32)
        accumulator = accumulator * alpha + tl.sum(probability[:, None] * value, axis=0)
        running_sum = running_sum * alpha + tl.sum(probability, axis=0)
        running_max = new_max
    out_offset = ((batch * HQ + q_head) * S + sequence) * D
    tl.store(out + out_offset + d, accumulator / running_sum, mask=d < D)


def triton_gqa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, past: int, causal: bool, length: int | None = None
) -> torch.Tensor:
    """Causal/non-causal grouped-query attention over a contiguous KV cache.

    ``q`` is ``[B, QH, S, D]`` and K/V are ``[B, KVH, T, D]``.  K/V heads
    are grouped logically inside the kernel; no repeat-interleave materializes
    the GQA expansion.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("attention expects q[B,QH,S,D], k/v[B,KVH,T,D]")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Triton attention requires contiguous Q/K/V")
    b, qh, s, d = q.shape
    bk, kvh, cache_t, kd = k.shape
    t = cache_t if length is None else length
    if bk != b or kd != d or qh % kvh:
        raise ValueError("incompatible GQA dimensions")
    if d > 256 or t > cache_t or t > 4096:
        raise ValueError("reference Triton attention supports D <= 256 and T <= 4096")
    if past < 0 or (causal and past + s > t):
        raise ValueError("invalid cache length for causal attention")
    out = torch.empty_like(q)
    _gqa_attention_kernel[(b * qh * s,)](
        q, k, v, out,
        HQ=qh, HKV=kvh, S=s, T=t, CACHE_T=cache_t, D=d, GROUP_SIZE=qh // kvh,
        PAST=past, CAUSAL=causal, SM_SCALE=d ** -0.5,
        BLOCK_N=64, BLOCK_D=triton.next_power_of_2(d), num_warps=4,
    )
    return out


@triton.jit
def _embedding_kernel(ids, table, out, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(ids + row)
    value = tl.load(table + token * H + col, mask=col < H, other=0.0)
    tl.store(out + row * H + col, value, mask=col < H)


def triton_embedding(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    ids = ids.reshape(-1)
    h = table.shape[1]
    out = torch.empty((ids.numel(), h), device=table.device, dtype=table.dtype)
    _embedding_kernel[(ids.numel(),)](ids, table, out, H=h, BLOCK=triton.next_power_of_2(h), num_warps=8)
    return out


@triton.jit
def _binary_kernel(a, b, out, n, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + p, tl.load(a + p, mask=p < n) + tl.load(b + p, mask=p < n), mask=p < n)


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a); n = a.numel()
    _binary_kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK=256)
    return out


@triton.jit
def _silu_mul_kernel(gate, up, out, n, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(gate + p, mask=p < n, other=0.0).to(tl.float32)
    u = tl.load(up + p, mask=p < n, other=0.0)
    tl.store(out + p, (g / (1.0 + tl.exp(-g))) * u, mask=p < n)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate); n = gate.numel()
    _silu_mul_kernel[(triton.cdiv(n, 256),)](gate, up, out, n, BLOCK=256)
    return out


@triton.jit
def _copy_projection_kernel(src, dst, base, row_stride, rows, heads: tl.constexpr, S: tl.constexpr,
                            D: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = p % D
    t = p // D
    head = t % heads
    row = t // heads
    batch = row // S
    seq = row % S
    source = row * row_stride + base + head * D + d
    target = ((batch * heads + head) * S + seq) * D + d
    tl.store(dst + target, tl.load(src + source, mask=p < rows * heads * D), mask=p < rows * heads * D)


def triton_split_qkv(qkv: torch.Tensor, batch: int, seq: int, cfg: Qwen3BenchmarkConfig):
    """Materialize Q/K/V in the head-major layout required by attention."""
    rows, width = qkv.shape
    assert rows == batch * seq and width == cfg.qkv_size
    q = torch.empty((batch, cfg.num_attention_heads, seq, cfg.head_dim), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((batch, cfg.num_key_value_heads, seq, cfg.head_dim), device=qkv.device, dtype=qkv.dtype)
    v = torch.empty_like(k)
    # Source row stride is qkv_size; pass it as offset to avoid any torch layout copy.
    for dst, base in ((q, 0), (k, cfg.hidden_size), (v, cfg.hidden_size + cfg.kv_size)):
        heads = dst.shape[1]; n = rows * heads * cfg.head_dim
        _copy_projection_kernel[(triton.cdiv(n, 256),)](
            qkv, dst, base, cfg.qkv_size, rows, heads=heads, S=seq, D=cfg.head_dim, BLOCK=256
        )
    return q, k, v


@triton.jit
def _heads_to_rows_kernel(src, dst, n, HQ: tl.constexpr, S: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = p % D; t = p // D; h = t % HQ; row = t // HQ; b = row // S; s = row % S
    source = ((b * HQ + h) * S + s) * D + d
    target = (row * HQ + h) * D + d
    tl.store(dst + target, tl.load(src + source, mask=p < n), mask=p < n)


def triton_heads_to_rows(src: torch.Tensor) -> torch.Tensor:
    b, h, s, d = src.shape; n = src.numel()
    out = torch.empty((b * s, h * d), device=src.device, dtype=src.dtype)
    _heads_to_rows_kernel[(triton.cdiv(n, 256),)](src, out, n, HQ=h, S=s, D=d, BLOCK=256)
    return out


@triton.jit
def _split_halves_kernel(src, left, right, rows, width: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = p // width; col = p % width
    tl.store(left + p, tl.load(src + row * (2 * width) + col, mask=p < rows * width), mask=p < rows * width)
    tl.store(right + p, tl.load(src + row * (2 * width) + width + col, mask=p < rows * width), mask=p < rows * width)


def triton_split_halves(src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, doubled = src.shape; width = doubled // 2
    left, right = torch.empty((rows, width), device=src.device, dtype=src.dtype), torch.empty((rows, width), device=src.device, dtype=src.dtype)
    _split_halves_kernel[(triton.cdiv(rows * width, 256),)](src, left, right, rows, width=width, BLOCK=256)
    return left, right


@triton.jit
def _last_token_kernel(src, out, S: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); h = tl.arange(0, BLOCK)
    tl.store(out + b * H + h, tl.load(src + (b * S + S - 1) * H + h, mask=h < H), mask=h < H)


def triton_last_token(src: torch.Tensor, batch: int, sequence: int) -> torch.Tensor:
    """Gather final hidden state from contiguous rows ``[B*S, H]``."""
    h = src.shape[1]
    out = torch.empty((batch, h), device=src.device, dtype=src.dtype)
    _last_token_kernel[(batch,)](src, out, S=sequence, H=h, BLOCK=triton.next_power_of_2(h), num_warps=8)
    return out

@triton.jit
def _argmax_kernel(x, out, vocab: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0); col = tl.arange(0, BLOCK)
    vals = tl.load(x + row * vocab + col, mask=col < vocab, other=-float("inf"))
    tl.store(out + row, tl.argmax(vals, axis=0))

def triton_argmax(logits: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    if vocab > 65536:
        return torch.argmax(logits, dim=-1)
    out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    _argmax_kernel[(logits.shape[0],)](logits, out, vocab=vocab, BLOCK=triton.next_power_of_2(vocab), num_warps=8)
    return out


class TritonBackend:
    """Reserved public interface for the pure-Triton backend.

    No model computation is routed through this class until every involved
    kernel, especially causal GQA attention, passes numerical comparison with
    the PyTorch semantic reference.
    """

    name = "triton"

    def __init__(
        self,
        cfg: Qwen3BenchmarkConfig,
        weights: Qwen3Weights,
        batch: int,
        profiler: Optional[Callable[[str, Callable[[], T]], T]] = None,
    ) -> None:
        self.cfg, self.weights, self.batch, self.profiler = cfg, weights, batch, profiler
        self.cache = ContiguousKVCache(cfg, batch, weights.embedding.device, weights.embedding.dtype)
        self.cos, self.sin = build_rope_table(cfg, weights.embedding.device, weights.embedding.dtype)
        self.phase = "run"

    def _m(self, name, fn):
        return fn() if self.profiler is None else self.profiler(f"{self.phase}/{name}", fn)

    def _layer(self, x, layer_id, sequence, causal):
        c, w = self.cfg, self.weights.layers[layer_id]; residual = x
        h = self._m("input_rmsnorm", lambda: triton_rmsnorm(x, w.input_norm, c.rms_norm_eps))
        qkv = self._m("qkv_gemm", lambda: triton_linear(h, w.qkv))
        q, k, v = triton_split_qkv(qkv, self.batch, sequence, c)
        q = self._m("qk_rmsnorm", lambda: triton_rmsnorm(q, w.q_norm, c.rms_norm_eps))
        k = triton_rmsnorm(k, w.k_norm, c.rms_norm_eps)
        past = self.cache.length
        q = self._m("rope", lambda: triton_rope(q, self.cos, self.sin, past))
        k = triton_rope(k, self.cos, self.sin, past)
        kc, vc = self.cache.storage[layer_id, 0], self.cache.storage[layer_id, 1]
        self._m("kv_cache_write", lambda: (triton_cache_write(k, kc, past), triton_cache_write(v, vc, past)))
        attn = self._m("attention", lambda: triton_gqa_attention(q, kc, vc, past=past, causal=causal, length=past + sequence))
        attn = triton_heads_to_rows(attn)
        x = self._m("o_gemm_residual", lambda: triton_add(residual, triton_linear(attn, w.o_proj)))
        residual = x; h = self._m("post_rmsnorm", lambda: triton_rmsnorm(x, w.post_norm, c.rms_norm_eps))
        gu = self._m("gateup_gemm", lambda: triton_linear(h, w.gate_up)); gate, up = triton_split_halves(gu)
        mlp = self._m("silu_mul", lambda: triton_silu_mul(gate, up))
        return self._m("down_gemm_residual", lambda: triton_add(residual, triton_linear(mlp, w.down)))

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.phase = "prefill"; self.cache.reset(); s = input_ids.shape[1]
        x = self._m("embedding", lambda: triton_embedding(input_ids, self.weights.embedding))
        for i in range(self.cfg.num_layers): x = self._layer(x, i, s, True)
        self.cache.finish_layer_stack(s); x = self._m("final_rmsnorm", lambda: triton_rmsnorm(x, self.weights.final_norm, self.cfg.rms_norm_eps))
        return self._m("lm_head", lambda: triton_linear(triton_last_token(x, self.batch, s), self.weights.lm_head))

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.phase = "decode"; x = self._m("embedding", lambda: triton_embedding(token_ids, self.weights.embedding))
        for i in range(self.cfg.num_layers): x = self._layer(x, i, 1, False)
        self.cache.finish_layer_stack(1); x = self._m("final_rmsnorm", lambda: triton_rmsnorm(x, self.weights.final_norm, self.cfg.rms_norm_eps))
        return self._m("lm_head", lambda: triton_linear(x, self.weights.lm_head))

    def argmax(self, logits: torch.Tensor) -> torch.Tensor:
        return self._m("argmax", lambda: triton_argmax(logits))
