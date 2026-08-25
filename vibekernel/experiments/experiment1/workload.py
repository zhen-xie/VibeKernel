from __future__ import annotations

from dataclasses import dataclass

import torch

M_VALUES = (4, 8, 16, 32, 4, 48, 8, 64)
N = 4096
K = 4096


@dataclass(frozen=True)
class GemmProblem:
    m: int
    n: int
    k: int
    a: torch.Tensor
    b: torch.Tensor


def create_workload(seed: int = 0, device: str = "cuda") -> list[GemmProblem]:
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 1 requires a CUDA GPU")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    problems = []
    for m in M_VALUES:
        a = torch.randn((m, K), device=device, dtype=torch.bfloat16, generator=generator)
        b = torch.randn((K, N), device=device, dtype=torch.bfloat16, generator=generator)
        problems.append(GemmProblem(m=m, n=N, k=K, a=a, b=b))
    return problems


def total_flops(problems: list[GemmProblem]) -> int:
    return sum(2 * p.m * p.n * p.k for p in problems)


def description(problems: list[GemmProblem]) -> dict[str, object]:
    return {
        "name": "eight_irregular_gemms",
        "m": [p.m for p in problems],
        "n": N,
        "k": K,
        "input_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "total_flops": total_flops(problems),
    }
