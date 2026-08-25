from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def grouped_gemm_kernel(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    m_sizes,
    task_gemm_ids,
    task_tile_ms,
    task_tile_ns,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    task_id = tl.program_id(0)
    gemm_id = tl.load(task_gemm_ids + task_id)
    tile_m = tl.load(task_tile_ms + task_id)
    tile_n = tl.load(task_tile_ns + task_id)
    m = tl.load(m_sizes + gemm_id)

    a_base = tl.load(a_ptrs + gemm_id)
    b_base = tl.load(b_ptrs + gemm_id)
    c_base = tl.load(c_ptrs + gemm_id)

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + ks
        a_offsets = rows[:, None] * K + k_offsets[None, :]
        b_offsets = k_offsets[:, None] * N + cols[None, :]
        a = tl.load(a_base + a_offsets, mask=(rows[:, None] < m) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(b_base + b_offsets, mask=(k_offsets[:, None] < K) & (cols[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)

    c_offsets = rows[:, None] * N + cols[None, :]
    tl.store(c_base + c_offsets, accumulator, mask=(rows[:, None] < m) & (cols[None, :] < N))
