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
        self._a_storage: torch.Tensor | None = None
        self._b_storage: torch.Tensor | None = None
        self._c_storage: torch.Tensor | None = None

    def prepare(self, problems) -> None:
        try:
            from .kernel import grouped_gemm_kernel
        except ImportError as error:
            raise RuntimeError("Triton is required for the triton_grouped backend") from error

        device = problems[0].a.device
        self._problems = list(problems)
        # Triton cannot dereference an int64 tensor of arbitrary device pointers.
        # Pack each operand family into one typed allocation and index group-local
        # matrices with element offsets from that allocation.
        self._a_storage = torch.cat([p.a.reshape(-1) for p in problems])
        self._b_storage = torch.cat([p.b.reshape(-1) for p in problems])
        c_elements = [p.m * p.n for p in problems]
        self._c_storage = torch.empty(sum(c_elements), device=device, dtype=torch.bfloat16)
        c_chunks = self._c_storage.split(c_elements)
        self._outputs = [chunk.view(p.m, p.n) for chunk, p in zip(c_chunks, problems, strict=True)]

        a_offsets, b_offsets, c_offsets = [], [], []
        a_cursor = b_cursor = c_cursor = 0
        for p, c_count in zip(problems, c_elements, strict=True):
            a_offsets.append(a_cursor)
            b_offsets.append(b_cursor)
            c_offsets.append(c_cursor)
            a_cursor += p.m * p.k
            b_cursor += p.k * p.n
            c_cursor += c_count

        gemm_ids, tile_ms, tile_ns = [], [], []
        for gemm_id, p in enumerate(problems):
            for tm in range(math.ceil(p.m / self.block_m)):
                for tn in range(math.ceil(p.n / self.block_n)):
                    gemm_ids.append(gemm_id)
                    tile_ms.append(tm)
                    tile_ns.append(tn)

        self._task_count = len(gemm_ids)
        self._args = {
            "a_group_offsets": torch.tensor(a_offsets, device=device, dtype=torch.int64),
            "b_group_offsets": torch.tensor(b_offsets, device=device, dtype=torch.int64),
            "c_group_offsets": torch.tensor(c_offsets, device=device, dtype=torch.int64),
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
            self._a_storage, self._b_storage, self._c_storage,
            a["a_group_offsets"], a["b_group_offsets"], a["c_group_offsets"], a["m_sizes"],
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
            "operand_storage": "packed BF16 buffers with per-group element offsets",
        }

    def close(self) -> None:
        self._args.clear()
        self._outputs.clear()
        self._problems.clear()
        self._a_storage = None
        self._b_storage = None
        self._c_storage = None
