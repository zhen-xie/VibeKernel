#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t const err = (expr);                                            \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#expr) + ": " +                    \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// This is deliberately small.  Every backend invokes exactly this operation
// for each task in the chain; only the dispatch mechanism changes.
__device__ __forceinline__ float task_op(float x, int stage) {
  float const scale = 1.0001f + 0.00001f * static_cast<float>(stage);
  float const bias = 0.0003f * static_cast<float>(stage + 1);
  return tanhf(fmaf(x, scale, bias));
}

__global__ void task_kernel(float const* input, float* output, int n, int stage) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += blockDim.x * gridDim.x) {
    output[i] = task_op(input[i], stage);
  }
}

// MPK-v0: a single persistent CTA owns a fixed, dependent task chain.  This
// is intentionally not Mirage's scheduler.  It isolates the effect of doing
// the task transitions on device instead of re-launching CUDA kernels.
__global__ void mpk_persistent_kernel(float const* input, float* ping,
                                      float* pong, int n, int tasks) {
  float const* src = input;
  float* dst = ping;
  for (int stage = 0; stage < tasks; ++stage) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      dst[i] = task_op(src[i], stage);
    }
    __syncthreads();
    src = dst;
    dst = (dst == ping) ? pong : ping;
  }
}

struct Options {
  int elements = 4096;
  int tasks = 20;
  int warmup = 100;
  int repeats = 1000;
  int block = 256;
};

Options parse_options(int argc, char** argv) {
  Options options;
  auto require_value = [&](int& i, char const* name) {
    if (++i == argc) throw std::runtime_error(std::string("missing value for ") + name);
    return std::atoi(argv[i]);
  };
  for (int i = 1; i < argc; ++i) {
    std::string const arg(argv[i]);
    if (arg == "--elements") options.elements = require_value(i, "--elements");
    else if (arg == "--tasks") options.tasks = require_value(i, "--tasks");
    else if (arg == "--warmup") options.warmup = require_value(i, "--warmup");
    else if (arg == "--repeats") options.repeats = require_value(i, "--repeats");
    else if (arg == "--block") options.block = require_value(i, "--block");
    else if (arg == "--help") {
      std::printf("Usage: launch_overhead [--elements N] [--tasks N] [--warmup N] [--repeats N] [--block N]\n");
      std::exit(0);
    } else throw std::runtime_error("unknown option: " + arg);
  }
  if (options.elements <= 0 || options.tasks <= 0 || options.warmup < 0 ||
      options.repeats <= 0 || options.block <= 0 || options.block > 1024)
    throw std::runtime_error("all sizes must be positive; --block must be <= 1024");
  return options;
}

using Runner = void (*)();

void run_baseline(float const* input, float* ping, float* pong, Options const& o) {
  int const grid = (o.elements + o.block - 1) / o.block;
  float const* src = input;
  float* dst = ping;
  for (int stage = 0; stage < o.tasks; ++stage) {
    task_kernel<<<grid, o.block>>>(src, dst, o.elements, stage);
    src = dst;
    dst = (dst == ping) ? pong : ping;
  }
  CUDA_CHECK(cudaGetLastError());
}

cudaGraphExec_t build_graph(float const* input, float* ping, float* pong, Options const& o) {
  cudaGraph_t graph{};
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  std::vector<cudaGraphNode_t> nodes;
  nodes.reserve(o.tasks);
  // cudaKernelNodeParams::kernelParams is void**.  Keep graph argument
  // storage non-const even though the value originates in const Options.
  int n = o.elements;
  float const* src = input;
  float* dst = ping;
  for (int stage = 0; stage < o.tasks; ++stage) {
    void* args[] = {&src, &dst, &n, &stage};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(task_kernel);
    params.gridDim = dim3((o.elements + o.block - 1) / o.block);
    params.blockDim = dim3(o.block);
    params.kernelParams = args;
    cudaGraphNode_t node{};
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph,
        nodes.empty() ? nullptr : &nodes.back(), nodes.empty() ? 0 : 1, &params));
    nodes.push_back(node);
    src = dst;
    dst = (dst == ping) ? pong : ping;
  }
  cudaGraphExec_t executable{};
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return executable;
}

template <typename F>
std::vector<float> benchmark(F&& fn, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> samples;
  samples.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    samples.push_back(milliseconds * 1000.0f);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return samples;
}

float percentile(std::vector<float> values, float p) {
  std::sort(values.begin(), values.end());
  size_t const index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

void report(char const* name, std::vector<float> const& samples) {
  float sum = 0.0f;
  for (float x : samples) sum += x;
  std::printf("%-18s p50=%8.2f us  p90=%8.2f us  mean=%8.2f us\n", name,
              percentile(samples, 0.50f), percentile(samples, 0.90f),
              sum / samples.size());
}

float max_abs_difference(float const* a, float const* b, int n) {
  float maximum = 0.0f;
  for (int i = 0; i < n; ++i) maximum = std::max(maximum, std::abs(a[i] - b[i]));
  return maximum;
}

int main(int argc, char** argv) {
  try {
    Options const o = parse_options(argc, argv);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (o.elements > 1 << 20)
      throw std::runtime_error("MPK-v0 uses one CTA; keep --elements <= 1048576");
    std::printf("device=%s sm_%d%d, elements=%d tasks=%d block=%d\n", prop.name,
                prop.major, prop.minor, o.elements, o.tasks, o.block);

    std::vector<float> host_input(o.elements);
    for (int i = 0; i < o.elements; ++i) host_input[i] = 0.001f * (i % 113 - 56);
    float *input{}, *baseline_ping{}, *baseline_pong{}, *graph_ping{}, *graph_pong{}, *mpk_ping{}, *mpk_pong{};
    size_t const bytes = sizeof(float) * o.elements;
    CUDA_CHECK(cudaMalloc(&input, bytes)); CUDA_CHECK(cudaMalloc(&baseline_ping, bytes)); CUDA_CHECK(cudaMalloc(&baseline_pong, bytes));
    CUDA_CHECK(cudaMalloc(&graph_ping, bytes)); CUDA_CHECK(cudaMalloc(&graph_pong, bytes));
    CUDA_CHECK(cudaMalloc(&mpk_ping, bytes)); CUDA_CHECK(cudaMalloc(&mpk_pong, bytes));
    CUDA_CHECK(cudaMemcpy(input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    cudaGraphExec_t graph = build_graph(input, graph_ping, graph_pong, o);
    auto baseline = [&] { run_baseline(input, baseline_ping, baseline_pong, o); };
    auto graph_launch = [&] { CUDA_CHECK(cudaGraphLaunch(graph, 0)); };
    auto mpk = [&] { mpk_persistent_kernel<<<1, o.block>>>(input, mpk_ping, mpk_pong, o.elements, o.tasks); CUDA_CHECK(cudaGetLastError()); };

    baseline(); graph_launch(); mpk(); CUDA_CHECK(cudaDeviceSynchronize());
    float const* baseline_output = (o.tasks % 2) ? baseline_ping : baseline_pong;
    float const* graph_output = (o.tasks % 2) ? graph_ping : graph_pong;
    float const* mpk_output = (o.tasks % 2) ? mpk_ping : mpk_pong;
    std::vector<float> baseline_host(o.elements), graph_host(o.elements), mpk_host(o.elements);
    CUDA_CHECK(cudaMemcpy(baseline_host.data(), baseline_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(graph_host.data(), graph_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mpk_host.data(), mpk_output, bytes, cudaMemcpyDeviceToHost));
    float const graph_error = max_abs_difference(baseline_host.data(), graph_host.data(), o.elements);
    float const mpk_error = max_abs_difference(baseline_host.data(), mpk_host.data(), o.elements);
    std::printf("correctness: graph max_abs=%g, mpk max_abs=%g\n", graph_error, mpk_error);
    if (graph_error != 0.0f || mpk_error != 0.0f) throw std::runtime_error("backend outputs differ");

    auto const baseline_samples = benchmark(baseline, o.warmup, o.repeats);
    auto const graph_samples = benchmark(graph_launch, o.warmup, o.repeats);
    auto const mpk_samples = benchmark(mpk, o.warmup, o.repeats);
    report("baseline <<<>>>", baseline_samples);
    report("cudaGraphLaunch", graph_samples);
    report("mpk persistent", mpk_samples);
    std::printf("Note: MPK-v0 is a static single-CTA chain, not a general task scheduler.\n");
    CUDA_CHECK(cudaGraphExecDestroy(graph));
    CUDA_CHECK(cudaFree(input)); CUDA_CHECK(cudaFree(baseline_ping)); CUDA_CHECK(cudaFree(baseline_pong));
    CUDA_CHECK(cudaFree(graph_ping)); CUDA_CHECK(cudaFree(graph_pong)); CUDA_CHECK(cudaFree(mpk_ping)); CUDA_CHECK(cudaFree(mpk_pong));
  } catch (std::exception const& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
