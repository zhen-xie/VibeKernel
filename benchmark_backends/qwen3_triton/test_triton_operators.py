"""Numerical gate for Triton kernels before using them in timing results."""
from pathlib import Path
import sys

import torch

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "qwen3_contiguous"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))
from common import Qwen3BenchmarkConfig, build_rope_table, rms_norm  # noqa: E402
from triton_backend import (
    triton_add, triton_cache_write, triton_embedding, triton_gqa_attention,
    triton_linear, triton_rmsnorm, triton_rope, triton_silu_mul, triton_split_qkv,
)


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


def _attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past: int, causal: bool) -> torch.Tensor:
    b, qh, s, d = q.shape
    kvh = k.shape[1]
    grouped_q = q.view(b, kvh, qh // kvh, s, d)
    scores = torch.einsum("bhrsd,bhtd->bhrst", grouped_q.float(), k.float()) * (d ** -0.5)
    if causal:
        qpos = torch.arange(past, past + s, device=q.device)[:, None]
        kpos = torch.arange(k.shape[2], device=q.device)[None, :]
        scores.masked_fill_(kpos > qpos, float("-inf"))
    probs = scores.softmax(dim=-1).to(q.dtype)
    return torch.einsum("bhrst,bhtd->bhrsd", probs, v).reshape_as(q)


def test_gqa_attention() -> None:
    torch.manual_seed(4)
    # Prefill: causal rows grow from one visible key to the full sequence.
    q = torch.randn((2, 8, 11, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 2, 11, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = triton_gqa_attention(q, k, v, past=0, causal=True)
    expected = _attention_reference(q, k, v, past=0, causal=True)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    # Decode: one query sees a pre-existing contiguous history.
    q = torch.randn((2, 8, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 2, 16, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = triton_gqa_attention(q, k, v, past=15, causal=False)
    expected = _attention_reference(q, k, v, past=15, causal=False)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    print("PASSED: Triton GQA attention matches PyTorch reference")


def test_elementwise_and_embedding() -> None:
    torch.manual_seed(5)
    table = torch.randn((101, 256), device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[1, 99], [7, 42]], device="cuda", dtype=torch.int64)
    torch.testing.assert_close(triton_embedding(ids, table), table[ids].reshape(-1, 256), rtol=0, atol=0)
    gate = torch.randn((13, 768), device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    torch.testing.assert_close(triton_silu_mul(gate, up), torch.nn.functional.silu(gate) * up, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(triton_add(gate, up), gate + up, rtol=0, atol=0)
    print("PASSED: Triton embedding and elementwise kernels match PyTorch reference")


def test_qkv_layout() -> None:
    torch.manual_seed(6)
    cfg = Qwen3BenchmarkConfig(hidden_size=256, intermediate_size=768, num_layers=1,
                               num_attention_heads=8, num_key_value_heads=2, head_dim=32)
    batch, seq = 2, 3
    qkv = torch.randn((batch * seq, cfg.qkv_size), device="cuda", dtype=torch.bfloat16)
    q, k, v = triton_split_qkv(qkv, batch, seq, cfg)
    raw_q, raw_k, raw_v = qkv.split((cfg.hidden_size, cfg.kv_size, cfg.kv_size), dim=-1)
    expected_q = raw_q.view(batch, seq, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2).contiguous()
    expected_k = raw_k.view(batch, seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2).contiguous()
    expected_v = raw_v.view(batch, seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2).contiguous()
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v, expected_v, rtol=0, atol=0)
    print("PASSED: Triton QKV split/layout matches PyTorch reference")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")
    test_rmsnorm()
    test_linear()
    test_rope()
    test_cache_write()
    test_gqa_attention()
    test_elementwise_and_embedding()
    test_qkv_layout()
