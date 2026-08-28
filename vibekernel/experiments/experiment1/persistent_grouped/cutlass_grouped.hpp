#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

struct CutlassGroupedRunnerHandle;

CutlassGroupedRunnerHandle* create_cutlass_grouped_runner(
    const void* const* a_ptrs,
    const void* const* b_ptrs,
    void* const* c_ptrs,
    const int64_t* mnk_shapes,
    int group_count,
    int device_index,
    cudaStream_t stream);

void destroy_cutlass_grouped_runner(CutlassGroupedRunnerHandle* runner);
void run_cutlass_grouped_runner(CutlassGroupedRunnerHandle* runner, cudaStream_t stream);
size_t cutlass_grouped_workspace_bytes(const CutlassGroupedRunnerHandle* runner);
