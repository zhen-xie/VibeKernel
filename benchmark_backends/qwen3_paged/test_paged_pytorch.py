"""Verify paged cache preserves the contiguous Qwen3 computation semantics."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))
sys.path.insert(1, str(ROOT / "qwen3_contiguous"))
from common import Qwen3BenchmarkConfig, Qwen3Weights
from pytorch_backend import PyTorchBackend
from pytorch_backend import PagedPyTorchBackend


def main():
    cfg = Qwen3BenchmarkConfig(vocab_size=257, hidden_size=256, intermediate_size=768, num_layers=1,
                               num_attention_heads=8, num_key_value_heads=2, head_dim=32, max_seq_len=80)
    w = Qwen3Weights(cfg, torch.device("cuda"), torch.bfloat16, seed=9)
    x = torch.arange(70, device="cuda").reshape(1, 70) % cfg.vocab_size
    a, b = PyTorchBackend(cfg, w, 1), PagedPyTorchBackend(cfg, w, 1, page_size=64)
    torch.testing.assert_close(b.prefill(x), a.prefill(x), rtol=3e-2, atol=3e-2)
    token = torch.argmax(a.prefill(x), -1)
    torch.testing.assert_close(b.decode(token), a.decode(token), rtol=3e-2, atol=3e-2)
    print("PASSED: paged PyTorch prefill/decode matches contiguous reference across a page boundary")


if __name__ == "__main__": main()
