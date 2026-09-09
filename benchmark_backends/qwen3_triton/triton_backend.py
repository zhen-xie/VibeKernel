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
