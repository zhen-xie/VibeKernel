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
from common import Qwen3BenchmarkConfig, Qwen3Weights  # noqa: E402

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

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Triton prefill is enabled after the operator-validation milestone.")

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Triton decode is enabled after the operator-validation milestone.")
