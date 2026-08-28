from dataclasses import dataclass
import torch

SHAPES = ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 4096, 4096), (4096, 16384, 4096))

@dataclass(frozen=True)
class GemmProblem:
    m: int; n: int; k: int; a: torch.Tensor; b: torch.Tensor

def create_workload(shape, seed=0, device="cuda"):
    m, n, k = shape
    g = torch.Generator(device=device); g.manual_seed(seed)
    return [GemmProblem(m, n, k, torch.randn((m,k), device=device, dtype=torch.bfloat16, generator=g), torch.randn((k,n), device=device, dtype=torch.bfloat16, generator=g))]

def total_flops(problems): return sum(2 * p.m * p.n * p.k for p in problems)
