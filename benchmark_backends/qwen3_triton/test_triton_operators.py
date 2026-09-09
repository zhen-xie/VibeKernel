"""Numerical gate for Triton kernels before using them in timing results."""
from pathlib import Path
import sys

import torch

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "qwen3_contiguous"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))
from common import rms_norm  # noqa: E402
from triton_backend import triton_rmsnorm


def test_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn((17, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
    actual = triton_rmsnorm(x, w, 1e-6)
    expected = rms_norm(x, w, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print("PASSED: Triton RMSNorm matches PyTorch reference")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")
    test_rmsnorm()
