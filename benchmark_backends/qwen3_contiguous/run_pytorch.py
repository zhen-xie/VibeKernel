"""Run correctness and CUDA-event timing for the first benchmark backend.

Usage from repository root:
  python benchmark/qwen3_contiguous/run_pytorch.py --batch 2 --prompt-len 128 --decode-steps 32
"""
from __future__ import annotations

import argparse
import statistics
import torch

from common import Qwen3BenchmarkConfig, Qwen3Weights
from pytorch_backend import PyTorchBackend


def time_ms(fn, repeats: int) -> list[float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(repeats):
        start.record(); fn(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    cfg = Qwen3BenchmarkConfig(max_seq_len=args.prompt_len + args.decode_steps)
    device = torch.device("cuda")
    weights = Qwen3Weights(cfg, device, torch.bfloat16, seed=1234)
    backend = PyTorchBackend(cfg, weights, args.batch)
    prompt = torch.randint(cfg.vocab_size, (args.batch, args.prompt_len), device=device)
    token = torch.randint(cfg.vocab_size, (args.batch,), device=device)
    for _ in range(args.warmup):
        backend.prefill(prompt); backend.decode(token)
    torch.cuda.synchronize()
    prefill = time_ms(lambda: backend.prefill(prompt), args.repeats)
    # Rebuild the same prompt cache before every decode sample.  Prefill is
    # deliberately outside the CUDA Event interval: this measures one decode
    # token at one fixed context length, not an ever-growing sequence.
    decode = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.repeats):
        backend.prefill(prompt)
        start.record(); backend.decode(token); end.record(); end.synchronize()
        decode.append(start.elapsed_time(end))
    print(f"backend={backend.name} batch={args.batch} prompt={args.prompt_len}")
    print(f"prefill_ms median={statistics.median(prefill):.3f} mean={statistics.mean(prefill):.3f}")
    print(f"decode_ms  median={statistics.median(decode):.3f} mean={statistics.mean(decode):.3f}")


if __name__ == "__main__":
    main()
