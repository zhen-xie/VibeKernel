from __future__ import annotations

import math

import torch


class TritonGroupedBackend:
    name = "triton_grouped"

    def __init__(self, block_m: int = 16, block_n: int = 64, block_k: int = 32, num_warps: int = 4) -> None:
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self._outputs: list[torch.Tensor] = []
        self._args: dict[str, torch.Tensor | int] = {}
        self._task_count = 0
        self._problems = []

    def prepare(self, problems) -> None:
        try:
            from .kernel import grouped_gemm_kernel
        except ImportError as error:
            raise RuntimeError("Triton is required for the triton_grouped backend") from error

        device = problems[0].a.device
        self._problems = list(problems)
        self._outputs = [torch.empty((p.m, p.n), device=device, dtype=torch.bfloat16) for p in problems]

        gemm_ids, tile_ms, tile_ns = [], [], []
        for gemm_id, p in enumerate(problems):
            for tm in range(math.ceil(p.m / self.block_m)):
                for tn in range(math.ceil(p.n / self.block_n)):
                    gemm_ids.append(gemm_id)
                    tile_ms.append(tm)
                    tile_ns.append(tn)

        self._task_count = len(gemm_ids)
        self._args = {
            "a_ptrs": torch.tensor([p.a.data_ptr() for p in problems], device=device, dtype=torch.int64),
            "b_ptrs": torch.tensor([p.b.data_ptr() for p in problems], device=device, dtype=torch.int64),
            "c_ptrs": torch.tensor([c.data_ptr() for c in self._outputs], device=device, dtype=torch.int64),
            "m_sizes": torch.tensor([p.m for p in problems], device=device, dtype=torch.int32),
            "task_gemm_ids": torch.tensor(gemm_ids, device=device, dtype=torch.int32),
            "task_tile_ms": torch.tensor(tile_ms, device=device, dtype=torch.int32),
            "task_tile_ns": torch.tensor(tile_ns, device=device, dtype=torch.int32),
            "n": problems[0].n,
            "k": problems[0].k,
        }
        self._kernel = grouped_gemm_kernel
        self.run()
        torch.cuda.synchronize()

    def run(self) -> list[torch.Tensor]:
        if not self._args:
            raise RuntimeError("backend has not been prepared")
        a = self._args
        self._kernel[(self._task_count,)](
            a["a_ptrs"], a["b_ptrs"], a["c_ptrs"], a["m_sizes"],
            a["task_gemm_ids"], a["task_tile_ms"], a["task_tile_ns"],
            N=a["n"], K=a["k"], BLOCK_M=self.block_m,
            BLOCK_N=self.block_n, BLOCK_K=self.block_k, num_warps=self.num_warps,
        )
        return self._outputs

    def metadata(self) -> dict[str, object]:
        return {
            "implementation": "single Triton grouped GEMM kernel",
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
            "task_count": self._task_count,
        }

    def close(self) -> None:
        self._args.clear()
        self._outputs.clear()
        self._problems.clear()
