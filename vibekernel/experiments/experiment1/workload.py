from __future__ import annotations

from dataclasses import dataclass

import torch

# One MoE FFN "up" projection. A batch of 512 tokens with top-2 routing
# creates 1,024 expert-token assignments. This 32-expert distribution is both
# realistic for sparse routing and deliberately skewed: its 4..112 token
# range supplies many short GEMMs for a persistent grouped scheduler.
EXPERT_TOKEN_COUNTS = (
    4, 4, 4, 4,
    8, 8, 8, 8,
    12, 12, 12, 12,
    16, 16, 16, 16,
    24, 24, 24, 24,
    32, 32, 32, 32,
    48, 48, 48, 48,
    112, 112, 112, 112,
)
NUM_INPUT_TOKENS = 512
TOP_K = 2
NUM_EXPERTS = len(EXPERT_TOKEN_COUNTS)
K = 4096       # model hidden size
N = 14336      # MoE FFN expansion size


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
        "name": "moe_ffn_up_projection_grouped_gemm",
        "num_experts": NUM_EXPERTS,
        "input_tokens": NUM_INPUT_TOKENS,
        "top_k": TOP_K,
        "expert_token_counts": [p.m for p in problems],
        "total_expert_token_assignments": sum(p.m for p in problems),
        "m": [p.m for p in problems],
        "n": N,
        "k": K,
        "input_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "total_flops": total_flops(problems),
    }
