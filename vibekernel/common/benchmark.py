from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics

import torch


@dataclass
class BenchmarkResult:
    backend: str
    median_us: float
    mean_us: float
    p10_us: float
    p90_us: float
    tflops: float
    samples: int
    repeats_per_sample: int
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


@torch.no_grad()
def measure_backend(
    backend,
    total_flops: int,
    warmup: int,
    iterations: int,
    repeats_per_sample: int,
) -> BenchmarkResult:
    if warmup < 1 or iterations < 1 or repeats_per_sample < 1:
        raise ValueError("warmup, iterations, and repeats_per_sample must be positive")

    for _ in range(warmup):
        backend.run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    latencies_us: list[float] = []

    for start, end in zip(starts, ends, strict=True):
        start.record()
        for _ in range(repeats_per_sample):
            backend.run()
        end.record()
        end.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0 / repeats_per_sample)

    median_us = statistics.median(latencies_us)
    return BenchmarkResult(
        backend=backend.name,
        median_us=median_us,
        mean_us=statistics.fmean(latencies_us),
        p10_us=_percentile(latencies_us, 0.10),
        p90_us=_percentile(latencies_us, 0.90),
        tflops=total_flops / (median_us * 1.0e6),
        samples=iterations,
        repeats_per_sample=repeats_per_sample,
        metadata=backend.metadata(),
    )
