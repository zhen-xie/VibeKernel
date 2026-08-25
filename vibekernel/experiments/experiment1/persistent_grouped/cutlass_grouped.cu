#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

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

#include <cstdint>
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

// M is intentionally the smallest Hopper grouped-GEMM CTA extent supported by
// this configuration. Tiny M problems are predicated within the 64-row tile.
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
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              operation, " failed: ", cutlassGetStatusString(status));
}

void check_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous row-major");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a matrix");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              name, " pointer must be 16-byte aligned for TMA");
}

class CutlassGroupedRunner {
 public:
  CutlassGroupedRunner(
      const std::vector<torch::Tensor>& inputs_a,
      const std::vector<torch::Tensor>& inputs_b,
      const std::vector<torch::Tensor>& outputs)
      : inputs_a_(inputs_a), inputs_b_(inputs_b), outputs_(outputs) {
    TORCH_CHECK(!inputs_a.empty(), "grouped GEMM requires at least one problem");
    TORCH_CHECK(inputs_a.size() == inputs_b.size() && inputs_a.size() == outputs.size(),
                "A, B, and output lists must have equal lengths");
    group_count_ = static_cast<int>(inputs_a.size());
    device_index_ = inputs_a.front().get_device();
    c10::cuda::CUDAGuard guard(inputs_a.front().device());

    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index_));
    TORCH_CHECK(properties.major == 9 && properties.minor == 0,
                "CUTLASS grouped runner requires an SM90 GPU");

    std::vector<const ElementA*> ptr_a_host;
    std::vector<const ElementB*> ptr_b_host;
    std::vector<const ElementC*> ptr_c_host;
    std::vector<ElementC*> ptr_d_host;
    std::vector<StrideA> stride_a_host;
    std::vector<StrideB> stride_b_host;
    std::vector<StrideC> stride_c_host;
    std::vector<StrideD> stride_d_host;
    problem_shapes_host_.reserve(group_count_);
    ptr_a_host.reserve(group_count_);
    ptr_b_host.reserve(group_count_);
    ptr_c_host.reserve(group_count_);
    ptr_d_host.reserve(group_count_);

    for (int i = 0; i < group_count_; ++i) {
      const auto& a = inputs_a.at(i);
      const auto& b = inputs_b.at(i);
      const auto& d = outputs.at(i);
      check_matrix(a, "A");
      check_matrix(b, "B");
      check_matrix(d, "D");
      TORCH_CHECK(a.get_device() == device_index_ && b.get_device() == device_index_ &&
                  d.get_device() == device_index_, "all grouped tensors must use one GPU");
      TORCH_CHECK(a.size(1) == b.size(0), "A.K must equal B.K");
      TORCH_CHECK(d.size(0) == a.size(0) && d.size(1) == b.size(1),
                  "output has an incorrect shape");

      const int m = static_cast<int>(a.size(0));
      const int n = static_cast<int>(b.size(1));
      const int k = static_cast<int>(a.size(1));
      problem_shapes_host_.push_back({m, n, k});
      ptr_a_host.push_back(reinterpret_cast<const ElementA*>(a.data_ptr<at::BFloat16>()));
      ptr_b_host.push_back(reinterpret_cast<const ElementB*>(b.data_ptr<at::BFloat16>()));
      // beta=0, but a valid C pointer keeps the epilogue descriptor uniform.
      ptr_c_host.push_back(reinterpret_cast<const ElementC*>(d.data_ptr<at::BFloat16>()));
      ptr_d_host.push_back(reinterpret_cast<ElementC*>(d.data_ptr<at::BFloat16>()));
      stride_a_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
      // CUTLASS expresses logical B strides in (N,K,L) coordinate order.
      stride_b_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
      stride_c_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
      stride_d_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));
    }

    problem_shapes_.reset(group_count_);
    ptr_a_.reset(group_count_);
    ptr_b_.reset(group_count_);
    ptr_c_.reset(group_count_);
    ptr_d_.reset(group_count_);
    stride_a_.reset(group_count_);
    stride_b_.reset(group_count_);
    stride_c_.reset(group_count_);
    stride_d_.reset(group_count_);
    problem_shapes_.copy_from_host(problem_shapes_host_.data());
    ptr_a_.copy_from_host(ptr_a_host.data());
    ptr_b_.copy_from_host(ptr_b_host.data());
    ptr_c_.copy_from_host(ptr_c_host.data());
    ptr_d_.copy_from_host(ptr_d_host.data());
    stride_a_.copy_from_host(stride_a_host.data());
    stride_b_.copy_from_host(stride_b_host.data());
    stride_c_.copy_from_host(stride_c_host.data());
    stride_d_.copy_from_host(stride_d_host.data());
    initialize();
  }

  void run() {
    c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, device_index_));
    const auto stream = at::cuda::getCurrentCUDAStream(device_index_);
    check_cutlass(gemm_.run(stream.stream()), "CUTLASS grouped GEMM run");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  pybind11::dict metadata() const {
    pybind11::dict result;
    result["group_count"] = group_count_;
    result["tile_shape_mnk"] = "64x128x64";
    result["cluster_shape_mnk"] = "1x1x1";
    result["kernel_schedule"] = "PtrArrayTmaWarpSpecializedPingpong";
    result["workspace_bytes"] = workspace_size_;
    return result;
  }

 private:
  void initialize() {
    const auto stream = at::cuda::getCurrentCUDAStream(device_index_);
    auto hardware = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_index_);
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
        {group_count_, problem_shapes_.get(), problem_shapes_host_.data()},
        {ptr_a_.get(), stride_a_.get(), ptr_b_.get(), stride_b_.get()},
        {fusion, ptr_c_.get(), stride_c_.get(), ptr_d_.get(), stride_d_.get()},
        hardware};

    workspace_size_ = Gemm::get_workspace_size(arguments);
    workspace_.reset(workspace_size_);
    check_cutlass(gemm_.can_implement(arguments), "CUTLASS can_implement");
    check_cutlass(gemm_.initialize(arguments, workspace_.get(), stream.stream()),
                  "CUTLASS initialize");
  }

  int group_count_ = 0;
  int device_index_ = 0;
  size_t workspace_size_ = 0;
  std::vector<torch::Tensor> inputs_a_;
  std::vector<torch::Tensor> inputs_b_;
  std::vector<torch::Tensor> outputs_;
  std::vector<UnderlyingProblemShape> problem_shapes_host_;
  cutlass::DeviceAllocation<UnderlyingProblemShape> problem_shapes_;
  cutlass::DeviceAllocation<const ElementA*> ptr_a_;
  cutlass::DeviceAllocation<const ElementB*> ptr_b_;
  cutlass::DeviceAllocation<const ElementC*> ptr_c_;
  cutlass::DeviceAllocation<ElementC*> ptr_d_;
  cutlass::DeviceAllocation<StrideA> stride_a_;
  cutlass::DeviceAllocation<StrideB> stride_b_;
  cutlass::DeviceAllocation<StrideC> stride_c_;
  cutlass::DeviceAllocation<StrideD> stride_d_;
  cutlass::DeviceAllocation<uint8_t> workspace_;
  Gemm gemm_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<CutlassGroupedRunner>(module, "CutlassGroupedRunner")
      .def(pybind11::init<const std::vector<torch::Tensor>&,
                          const std::vector<torch::Tensor>&,
                          const std::vector<torch::Tensor>&>())
      .def("run", &CutlassGroupedRunner::run)
      .def("metadata", &CutlassGroupedRunner::metadata);
}
