"""Run correctness and CUDA-event timing for the first benchmark backend.

Usage from repository root:
  python benchmark/qwen3_contiguous/run_pytorch.py --batch 2 --prompt-len 128 --decode-steps 32
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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


def describe(values: list[float]) -> str:
    ordered = sorted(values)
    p90 = ordered[round(0.90 * (len(ordered) - 1))]
    return (
        f"min={ordered[0]:.3f} p50={statistics.median(ordered):.3f} "
        f"p90={p90:.3f} mean={statistics.mean(ordered):.3f} "
        f"std={statistics.pstdev(ordered):.3f} max={ordered[-1]:.3f}"
    )


class CudaBreakdown:
    """Event-based logical-stage profiler; use only outside the main timing run."""

    def __init__(self) -> None:
        self.events = []

    def __call__(self, name, fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self.events.append((name, start, end))
        return result

    def report(self) -> None:
        if not self.events:
            return
        self.events[-1][2].synchronize()
        totals = defaultdict(float)
        for name, start, end in self.events:
            totals[name] += start.elapsed_time(end)
        print("logical_breakdown_ms (one representative run; profiling overhead excluded from headline timing):")
        for name, elapsed in sorted(totals.items()):
            print(f"  {name:<34} {elapsed:8.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--preset", choices=("tiny", "qwen3-8b-core"), default="tiny")
    p.add_argument("--layers", type=int, default=None, help="Override the preset layer count.")
    p.add_argument("--vocab-size", type=int, default=4096,
                   help="LM vocabulary; use 151936 to include full Qwen3-8B LM-head cost.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--breakdown", action="store_true", help="Print one CUDA-Event logical operator breakdown.")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    max_seq_len = args.prompt_len + args.decode_steps
    if args.preset == "tiny":
        cfg = Qwen3BenchmarkConfig(
            vocab_size=args.vocab_size,
            num_layers=args.layers if args.layers is not None else 4,
            max_seq_len=max_seq_len,
        )
    else:
        cfg = Qwen3BenchmarkConfig.qwen3_8b_core(
            num_layers=args.layers if args.layers is not None else 1,
            vocab_size=args.vocab_size,
            max_seq_len=max_seq_len,
        )
    device = torch.device("cuda")
    weights = Qwen3Weights(cfg, device, torch.bfloat16, seed=1234)
    backend = PyTorchBackend(cfg, weights, args.batch)
    prompt = torch.randint(cfg.vocab_size, (args.batch, args.prompt_len), device=device)
    token = torch.randint(cfg.vocab_size, (args.batch,), device=device)
    for _ in range(args.warmup):
        backend.argmax(backend.prefill(prompt))
        backend.argmax(backend.decode(token))
    torch.cuda.synchronize()
    prefill = time_ms(lambda: backend.argmax(backend.prefill(prompt)), args.repeats)
    # Rebuild the same prompt cache before every decode sample.  Prefill is
    # deliberately outside the CUDA Event interval: this measures one decode
    # token at one fixed context length, not an ever-growing sequence.
    decode = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.repeats):
        backend.prefill(prompt)
        start.record(); backend.argmax(backend.decode(token)); end.record(); end.synchronize()
        decode.append(start.elapsed_time(end))
    print(
        f"backend={backend.name} preset={args.preset} layers={cfg.num_layers} "
        f"B={args.batch} S={args.prompt_len} H={cfg.hidden_size} I={cfg.intermediate_size} "
        f"vocab={cfg.vocab_size}"
    )
    print(f"prefill_ms {describe(prefill)}")
    print(f"decode_ms  {describe(decode)}")
    if args.breakdown:
        profiler = CudaBreakdown()
        profiled = PyTorchBackend(cfg, weights, args.batch, profiler=profiler)
        profiled.argmax(profiled.prefill(prompt))
        profiled.argmax(profiled.decode(token))
        profiler.report()


if __name__ == "__main__":
    main()
