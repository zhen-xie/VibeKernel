from __future__ import annotations

from dataclasses import dataclass

import torch

# The original eight-problem irregular GEMM microbenchmark.  The small and
# unequal M dimensions expose launch and scheduling costs independently of a
# full MoE model's routing and weight-memory footprint.
EXPERT_TOKEN_COUNTS = (4, 8, 16, 32, 4, 48, 8, 64)
NUM_EXPERTS = len(EXPERT_TOKEN_COUNTS)
K = 4096
N = 4096


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
    for m in EXPERT_TOKEN_COUNTS:
        a = torch.randn((m, K), device=device, dtype=torch.bfloat16, generator=generator)
        b = torch.randn((K, N), device=device, dtype=torch.bfloat16, generator=generator)
        problems.append(GemmProblem(m=m, n=N, k=K, a=a, b=b))
    return problems


def total_flops(problems: list[GemmProblem]) -> int:
    return sum(2 * p.m * p.n * p.k for p in problems)


def description(problems: list[GemmProblem]) -> dict[str, object]:
    return {
        "name": "eight_irregular_gemms",
        "num_experts": NUM_EXPERTS,
        "expert_token_counts": [p.m for p in problems],
        "total_expert_token_assignments": sum(p.m for p in problems),
        "m": [p.m for p in problems],
        "n": N,
        "k": K,
        "input_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "total_flops": total_flops(problems),
    }
