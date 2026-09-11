#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t const e = (expr); if (e != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e)); } while (0)

__device__ __forceinline__ float task_op(float x, int task_id) {
  return tanhf(fmaf(x, 1.0001f + 0.00001f * task_id, 0.0003f * (task_id + 1)));
}

// One independent logical task occupies one CTA and writes a disjoint output
// slice. Baseline, Graph, and MPK invoke exactly this task body.
__global__ void independent_task_kernel(float const* input, float* output,
                                        int elements_per_task, int task_id) {
  float* task_output = output + static_cast<size_t>(task_id) * elements_per_task;
  for (int i = threadIdx.x; i < elements_per_task; i += blockDim.x)
    task_output[i] = task_op(input[i], task_id);
}

struct QueueState { int next_task; };

// No dependency tracking: workers simply claim independent logical tasks.
__global__ void mpk_independent_kernel(float const* input, float* output,
                                       int elements_per_task, int task_count,
                                       QueueState* queue) {
  while (true) {
    int const task_id = atomicAdd(&queue->next_task, 1);
    if (task_id >= task_count) return;
    float* task_output = output + static_cast<size_t>(task_id) * elements_per_task;
    for (int i = threadIdx.x; i < elements_per_task; i += blockDim.x)
      task_output[i] = task_op(input[i], task_id);
    __syncthreads();
  }
}

struct Options { int elements = 4096, tasks = 64, warmup = 100, repeats = 1000, block = 256, workers = 0; };

Options parse_options(int argc, char** argv) {
  Options o;
  auto value = [&](int& i, char const* name) { if (++i == argc) throw std::runtime_error(std::string("missing value for ") + name); return std::atoi(argv[i]); };
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == "--elements") o.elements = value(i, "--elements");
    else if (a == "--tasks") o.tasks = value(i, "--tasks");
    else if (a == "--warmup") o.warmup = value(i, "--warmup");
    else if (a == "--repeats") o.repeats = value(i, "--repeats");
    else if (a == "--block") o.block = value(i, "--block");
    else if (a == "--workers") o.workers = value(i, "--workers");
    else if (a == "--help") { std::printf("Usage: independent_tasks [--elements N] [--tasks N] [--warmup N] [--repeats N] [--block N] [--workers N]\n"); std::exit(0); }
    else throw std::runtime_error("unknown option: " + a);
  }
  if (o.elements <= 0 || o.tasks <= 0 || o.warmup < 0 || o.repeats <= 0 || o.block <= 0 || o.block > 1024 || o.workers < 0) throw std::runtime_error("invalid size");
  return o;
}

void run_baseline(float const* input, float* output, Options const& o) {
  for (int task = 0; task < o.tasks; ++task)
    independent_task_kernel<<<1, o.block>>>(input, output, o.elements, task);
  CUDA_CHECK(cudaGetLastError());
}

cudaGraphExec_t build_graph(float const* input, float* output, Options const& o) {
  cudaGraph_t graph{}; CUDA_CHECK(cudaGraphCreate(&graph, 0));
  int elements = o.elements;
  for (int task = 0; task < o.tasks; ++task) {
    void* args[] = {&input, &output, &elements, &task};
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(independent_task_kernel);
    p.gridDim = dim3(1); p.blockDim = dim3(o.block); p.kernelParams = args;
    cudaGraphNode_t node{};
    // No predecessor: these are intentionally independent graph nodes.
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p));
  }
  cudaGraphExec_t exec{}; CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)); CUDA_CHECK(cudaGraphDestroy(graph)); return exec;
}

template <typename F> std::vector<float> benchmark(F&& f, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> samples; samples.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start)); f(); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); samples.push_back(ms * 1000.0f);
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); return samples;
}

// Reset the device queue before the start event. The reset is necessary setup
// for each launch but is deliberately excluded from the measured kernel time.
template <typename F> std::vector<float> benchmark_mpk(F&& f, QueueState* queue,
                                                       int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemsetAsync(queue, 0, sizeof(QueueState), 0));
    f();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> samples; samples.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaMemsetAsync(queue, 0, sizeof(QueueState), 0));
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); samples.push_back(ms * 1000.0f);
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); return samples;
}

float p50(std::vector<float> x) { std::sort(x.begin(), x.end()); return x[(x.size() - 1) / 2]; }
void report(char const* name, std::vector<float> const& x) { float sum = 0; for (float v : x) sum += v; std::printf("%-18s p50=%8.2f us  mean=%8.2f us\n", name, p50(x), sum / x.size()); }
float max_abs(float const* a, float const* b, size_t n) { float m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

int main(int argc, char** argv) {
  try {
    Options const o = parse_options(argc, argv); cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int const workers = o.workers == 0 ? std::min(o.tasks, prop.multiProcessorCount) : o.workers;
    if (workers <= 0 || workers > prop.multiProcessorCount) throw std::runtime_error("--workers must be in [1, SM count]");
    std::printf("device=%s sm_%d%d, elements/task=%d independent_tasks=%d block=%d workers=%d\n", prop.name, prop.major, prop.minor, o.elements, o.tasks, o.block, workers);
    size_t const input_bytes = sizeof(float) * o.elements, output_count = static_cast<size_t>(o.elements) * o.tasks, output_bytes = sizeof(float) * output_count;
    std::vector<float> host_input(o.elements); for (int i = 0; i < o.elements; ++i) host_input[i] = .001f * (i % 113 - 56);
    float *input{}, *baseline_out{}, *graph_out{}, *mpk_out{}; QueueState* queue{};
    CUDA_CHECK(cudaMalloc(&input, input_bytes)); CUDA_CHECK(cudaMalloc(&baseline_out, output_bytes)); CUDA_CHECK(cudaMalloc(&graph_out, output_bytes)); CUDA_CHECK(cudaMalloc(&mpk_out, output_bytes)); CUDA_CHECK(cudaMalloc(&queue, sizeof(QueueState)));
    CUDA_CHECK(cudaMemset(queue, 0, sizeof(QueueState))); CUDA_CHECK(cudaMemcpy(input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
    cudaGraphExec_t graph = build_graph(input, graph_out, o);
    auto baseline = [&] { run_baseline(input, baseline_out, o); };
    auto graph_launch = [&] { CUDA_CHECK(cudaGraphLaunch(graph, 0)); };
    auto mpk = [&] { mpk_independent_kernel<<<workers, o.block>>>(input, mpk_out, o.elements, o.tasks, queue); CUDA_CHECK(cudaGetLastError()); };
    baseline(); graph_launch(); mpk(); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> baseline_host(output_count), graph_host(output_count), mpk_host(output_count);
    CUDA_CHECK(cudaMemcpy(baseline_host.data(), baseline_out, output_bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(graph_host.data(), graph_out, output_bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(mpk_host.data(), mpk_out, output_bytes, cudaMemcpyDeviceToHost));
    float const graph_error = max_abs(baseline_host.data(), graph_host.data(), output_count), mpk_error = max_abs(baseline_host.data(), mpk_host.data(), output_count);
    std::printf("correctness: graph max_abs=%g, mpk max_abs=%g\n", graph_error, mpk_error); if (graph_error != 0 || mpk_error != 0) throw std::runtime_error("backend outputs differ");
    report("baseline <<<>>>", benchmark(baseline, o.warmup, o.repeats)); report("cudaGraphLaunch", benchmark(graph_launch, o.warmup, o.repeats)); report("mpk persistent", benchmark_mpk(mpk, queue, o.warmup, o.repeats));
    std::printf("Note: MPK uses only an independent-task atomic queue; no dependency counters or stage barriers.\n");
    CUDA_CHECK(cudaGraphExecDestroy(graph)); CUDA_CHECK(cudaFree(input)); CUDA_CHECK(cudaFree(baseline_out)); CUDA_CHECK(cudaFree(graph_out)); CUDA_CHECK(cudaFree(mpk_out)); CUDA_CHECK(cudaFree(queue));
  } catch (std::exception const& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
