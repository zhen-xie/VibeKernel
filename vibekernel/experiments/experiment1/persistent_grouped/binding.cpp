#include "cutlass_grouped.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <memory>
#include <vector>

namespace {

void check_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous row-major");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a matrix");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              name, " pointer must be 16-byte aligned for TMA");
}

struct RunnerDeleter {
  void operator()(CutlassGroupedRunnerHandle* runner) const {
    destroy_cutlass_grouped_runner(runner);
  }
};

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

    std::vector<const void*> a_ptrs;
    std::vector<const void*> b_ptrs;
    std::vector<void*> c_ptrs;
    std::vector<int64_t> mnk_shapes;
    a_ptrs.reserve(group_count_);
    b_ptrs.reserve(group_count_);
    c_ptrs.reserve(group_count_);
    mnk_shapes.reserve(3 * group_count_);

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

      a_ptrs.push_back(a.data_ptr());
      b_ptrs.push_back(b.data_ptr());
      c_ptrs.push_back(d.data_ptr());
      mnk_shapes.push_back(a.size(0));
      mnk_shapes.push_back(b.size(1));
      mnk_shapes.push_back(a.size(1));
    }

    const auto stream = at::cuda::getCurrentCUDAStream(device_index_);
    runner_.reset(create_cutlass_grouped_runner(
        a_ptrs.data(), b_ptrs.data(), c_ptrs.data(), mnk_shapes.data(), group_count_,
        device_index_, stream.stream()));
    TORCH_CHECK(runner_, "failed to create CUTLASS grouped GEMM runner");
  }

  void run() {
    c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, device_index_));
    const auto stream = at::cuda::getCurrentCUDAStream(device_index_);
    run_cutlass_grouped_runner(runner_.get(), stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  pybind11::dict metadata() const {
    pybind11::dict result;
    result["group_count"] = group_count_;
    result["tile_shape_mnk"] = "64x128x64";
    result["cluster_shape_mnk"] = "1x1x1";
    result["kernel_schedule"] = "PtrArrayTmaWarpSpecializedPingpong";
    result["workspace_bytes"] = cutlass_grouped_workspace_bytes(runner_.get());
    return result;
  }

 private:
  int group_count_ = 0;
  int device_index_ = 0;
  std::vector<torch::Tensor> inputs_a_;
  std::vector<torch::Tensor> inputs_b_;
  std::vector<torch::Tensor> outputs_;
  std::unique_ptr<CutlassGroupedRunnerHandle, RunnerDeleter> runner_;
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
