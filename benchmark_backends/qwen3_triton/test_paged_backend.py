"""End-to-end correctness gate: paged Triton versus paged PyTorch, B=1."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen3_paged"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Qwen3BenchmarkConfig, Qwen3Weights
from pytorch_backend import PyTorchBackend
from paged_backend import PagedTritonBackend


def main():
    cfg = Qwen3BenchmarkConfig(vocab_size=257, hidden_size=256, intermediate_size=768, num_layers=1,
                               num_attention_heads=8, num_key_value_heads=2, head_dim=32, max_seq_len=80)
    weights = Qwen3Weights(cfg, torch.device("cuda"), torch.bfloat16, seed=10)
    prompt = torch.arange(70, device="cuda").reshape(1, 70) % cfg.vocab_size
    pytorch, triton = PyTorchBackend(cfg, weights, 1), PagedTritonBackend(cfg, weights, 1)
    ref, actual = pytorch.prefill(prompt), triton.prefill(prompt)
    torch.testing.assert_close(actual, ref, rtol=6e-2, atol=6e-2)
    token = torch.argmax(ref, dim=-1)
    torch.testing.assert_close(triton.decode(token), pytorch.decode(token), rtol=6e-2, atol=6e-2)
    print("PASSED: paged Triton prefill/decode matches paged PyTorch across a page boundary")


if __name__ == "__main__":
    main()
