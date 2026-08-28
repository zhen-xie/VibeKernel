#include "cutlass_grouped.hpp"

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using TileShape = Shape<_64, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentC,
    ElementC, LayoutC*, AlignmentC,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA*, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideC = typename GemmKernel::InternalStrideC;
using StrideD = typename GemmKernel::InternalStrideD;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cutlassGetStatusString(status));
  }
}

}  // namespace

struct CutlassGroupedRunnerHandle {
  CutlassGroupedRunnerHandle(
      const void* const* a_ptrs,
      const void* const* b_ptrs,
      void* const* c_ptrs,
      const int64_t* mnk_shapes,
      int count,
      int device,
      cudaStream_t stream)
      : group_count(count), device_index(device) {
    if (group_count <= 0) {
      throw std::invalid_argument("grouped GEMM requires at least one problem");
    }

    std::vector<const ElementA*> a_host;
    std::vector<const ElementB*> b_host;
    std::vector<const ElementC*> c_host;
    std::vector<ElementC*> d_host;
    std::vector<StrideA> stride_a_host;
    std::vector<StrideB> stride_b_host;
    std::vector<StrideC> stride_c_host;
    std::vector<StrideD> stride_d_host;
    problem_shapes_host.reserve(group_count);
    a_host.reserve(group_count);
    b_host.reserve(group_count);
    c_host.reserve(group_count);
    d_host.reserve(group_count);

    for (int i = 0; i < group_count; ++i) {
      const int m = static_cast<int>(mnk_shapes[3 * i]);
      const int n = static_cast<int>(mnk_shapes[3 * i + 1]);
      const int k = static_cast<int>(mnk_shapes[3 * i + 2]);
      if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("all grouped GEMM dimensions must be positive");
      }
      problem_shapes_host.push_back({m, n, k});
      a_host.push_back(reinterpret_cast<const ElementA*>(a_ptrs[i]));
      b_host.push_back(reinterpret_cast<const ElementB*>(b_ptrs[i]));
      c_host.push_back(reinterpret_cast<const ElementC*>(c_ptrs[i]));
      d_host.push_back(reinterpret_cast<ElementC*>(c_ptrs[i]));
      stride_a_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
      // CUTLASS represents B's logical coordinates as (N, K, L).
      stride_b_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
      stride_c_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
      stride_d_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));
    }

    problem_shapes.reset(group_count);
    ptr_a.reset(group_count);
    ptr_b.reset(group_count);
    ptr_c.reset(group_count);
    ptr_d.reset(group_count);
    stride_a.reset(group_count);
    stride_b.reset(group_count);
    stride_c.reset(group_count);
    stride_d.reset(group_count);
    problem_shapes.copy_from_host(problem_shapes_host.data());
    ptr_a.copy_from_host(a_host.data());
    ptr_b.copy_from_host(b_host.data());
    ptr_c.copy_from_host(c_host.data());
    ptr_d.copy_from_host(d_host.data());
    stride_a.copy_from_host(stride_a_host.data());
    stride_b.copy_from_host(stride_b_host.data());
    stride_c.copy_from_host(stride_c_host.data());
    stride_d.copy_from_host(stride_d_host.data());

    auto hardware = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_index);
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion;
    fusion.alpha = 1.0f;
    fusion.beta = 0.0f;
    fusion.alpha_ptr = nullptr;
    fusion.beta_ptr = nullptr;
    fusion.alpha_ptr_array = nullptr;
    fusion.beta_ptr_array = nullptr;
    fusion.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion.dBeta = {cute::_0{}, cute::_0{}, 0};

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {group_count, problem_shapes.get(), problem_shapes_host.data()},
        {ptr_a.get(), stride_a.get(), ptr_b.get(), stride_b.get()},
        {fusion, ptr_c.get(), stride_c.get(), ptr_d.get(), stride_d.get()},
        hardware};

    workspace_size = Gemm::get_workspace_size(arguments);
    workspace.reset(workspace_size);
    check_cutlass(gemm.can_implement(arguments), "CUTLASS can_implement");
    check_cutlass(gemm.initialize(arguments, workspace.get(), stream), "CUTLASS initialize");
  }

  int group_count = 0;
  int device_index = 0;
  size_t workspace_size = 0;
  std::vector<UnderlyingProblemShape> problem_shapes_host;
  cutlass::DeviceAllocation<UnderlyingProblemShape> problem_shapes;
  cutlass::DeviceAllocation<const ElementA*> ptr_a;
  cutlass::DeviceAllocation<const ElementB*> ptr_b;
  cutlass::DeviceAllocation<const ElementC*> ptr_c;
  cutlass::DeviceAllocation<ElementC*> ptr_d;
  cutlass::DeviceAllocation<StrideA> stride_a;
  cutlass::DeviceAllocation<StrideB> stride_b;
  cutlass::DeviceAllocation<StrideC> stride_c;
  cutlass::DeviceAllocation<StrideD> stride_d;
  cutlass::DeviceAllocation<uint8_t> workspace;
  Gemm gemm;
};

CutlassGroupedRunnerHandle* create_cutlass_grouped_runner(
    const void* const* a_ptrs,
    const void* const* b_ptrs,
    void* const* c_ptrs,
    const int64_t* mnk_shapes,
    int group_count,
    int device_index,
    cudaStream_t stream) {
  return new CutlassGroupedRunnerHandle(
      a_ptrs, b_ptrs, c_ptrs, mnk_shapes, group_count, device_index, stream);
}

void destroy_cutlass_grouped_runner(CutlassGroupedRunnerHandle* runner) {
  delete runner;
}

void run_cutlass_grouped_runner(CutlassGroupedRunnerHandle* runner, cudaStream_t stream) {
  if (runner == nullptr) {
    throw std::invalid_argument("CUTLASS grouped GEMM runner is null");
  }
  check_cutlass(runner->gemm.run(stream), "CUTLASS grouped GEMM run");
}

size_t cutlass_grouped_workspace_bytes(const CutlassGroupedRunnerHandle* runner) {
  return runner == nullptr ? 0 : runner->workspace_size;
}
