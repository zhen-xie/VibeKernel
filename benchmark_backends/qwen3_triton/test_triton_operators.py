"""Numerical gate for Triton kernels before using them in timing results."""
from pathlib import Path
import sys

import torch

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "qwen3_contiguous"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))
from common import Qwen3BenchmarkConfig, build_rope_table, rms_norm  # noqa: E402
from triton_backend import triton_cache_write, triton_linear, triton_rmsnorm, triton_rope


def test_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn((17, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
    actual = triton_rmsnorm(x, w, 1e-6)
    expected = rms_norm(x, w, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print("PASSED: Triton RMSNorm matches PyTorch reference")


def test_linear() -> None:
    # QKV uses K=4096 and N=5120 in the Qwen3-8B-core configuration.  This
    # smaller M still exercises that exact projection layout without making a
    # correctness test needlessly long.
    torch.manual_seed(1)
    x = torch.randn((16, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((5120, 4096), device="cuda", dtype=torch.bfloat16)
    actual = triton_linear(x, weight)
    expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print("PASSED: Triton linear matches torch.nn.functional.linear")


def test_rope() -> None:
    torch.manual_seed(2)
    cfg = Qwen3BenchmarkConfig(max_seq_len=32, head_dim=128, hidden_size=256,
                               num_attention_heads=2, num_key_value_heads=1)
    cos, sin = build_rope_table(cfg, torch.device("cuda"), torch.bfloat16)
    x = torch.randn((2, 2, 7, 128), device="cuda", dtype=torch.bfloat16)
    actual = triton_rope(x, cos, sin, start=5)
    half = x.shape[-1] // 2
    expected = x * cos[5:12][None, None] + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin[5:12][None, None]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print("PASSED: Triton RoPE matches PyTorch reference")


def test_cache_write() -> None:
    torch.manual_seed(3)
    src = torch.randn((2, 8, 3, 128), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((2, 8, 16, 128), device="cuda", dtype=torch.bfloat16)
    expected = cache.clone()
    expected[:, :, 5:8, :].copy_(src)
    triton_cache_write(src, cache, start=5)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    print("PASSED: Triton KV cache write matches PyTorch reference")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")
    test_rmsnorm()
    test_linear()
    test_rope()
    test_cache_write()
