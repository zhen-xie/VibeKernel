#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t const e = (expr); if (e != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e)); } while (0)
#define CUBLAS_CHECK(expr) do { cublasStatus_t const s = (expr); if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#expr) + " failed with status " + std::to_string(static_cast<int>(s))); } while (0)

// One CTA computes one independent C_task[M, N] = A_task[M, K] * B[K, N].
// The simple FP32 body is intentionally shared verbatim by all three runners.
__global__ void gemm_task_kernel(float const* a, float const* b, float* c,
                                 int m, int n, int k, int task_id) {
  int const linear = threadIdx.x;
  int const row = linear / n;
  int const col = linear % n;
  if (row >= m || col >= n) return;
  float const* a_task = a + static_cast<size_t>(task_id) * m * k;
  float sum = 0.0f;
  for (int kk = 0; kk < k; ++kk)
    sum = fmaf(a_task[row * k + kk], b[kk * n + col], sum);
  c[static_cast<size_t>(task_id) * m * n + row * n + col] = sum;
}

struct QueueState { int next_task; };

__global__ void mpk_independent_gemm_kernel(float const* a, float const* b,
                                             float* c, int m, int n, int k,
                                             int task_count, QueueState* queue) {
  __shared__ int task_id;
  while (true) {
    if (threadIdx.x == 0) task_id = atomicAdd(&queue->next_task, 1);
    __syncthreads();
    if (task_id >= task_count) return;
    int const linear = threadIdx.x;
    int const row = linear / n;
    int const col = linear % n;
    if (row < m && col < n) {
      float const* a_task = a + static_cast<size_t>(task_id) * m * k;
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk)
        sum = fmaf(a_task[row * k + kk], b[kk * n + col], sum);
      c[static_cast<size_t>(task_id) * m * n + row * n + col] = sum;
    }
    __syncthreads();
  }
}

struct Options { int m = 16, n = 16, k = 256, tasks = 64, warmup = 100, repeats = 1000, workers = 0; };

Options parse_options(int argc, char** argv) {
  Options o;
  auto value = [&](int& i, char const* name) { if (++i == argc) throw std::runtime_error(std::string("missing value for ") + name); return std::atoi(argv[i]); };
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == "--m") o.m = value(i, "--m"); else if (a == "--n") o.n = value(i, "--n"); else if (a == "--k") o.k = value(i, "--k");
    else if (a == "--tasks") o.tasks = value(i, "--tasks"); else if (a == "--warmup") o.warmup = value(i, "--warmup"); else if (a == "--repeats") o.repeats = value(i, "--repeats"); else if (a == "--workers") o.workers = value(i, "--workers");
    else if (a == "--help") { std::printf("Usage: independent_gemm_tasks [--m M] [--n N] [--k K] [--tasks T] [--workers W] [--warmup W] [--repeats R]\n"); std::exit(0); }
    else throw std::runtime_error("unknown option: " + a);
  }
  if (o.m <= 0 || o.n <= 0 || o.k <= 0 || o.m * o.n > 1024 || o.tasks <= 0 || o.warmup < 0 || o.repeats <= 0 || o.workers < 0) throw std::runtime_error("invalid size; require M*N <= 1024");
  return o;
}

void run_baseline(float const* a, float const* b, float* c, Options const& o) {
  for (int task = 0; task < o.tasks; ++task) gemm_task_kernel<<<1, o.m * o.n>>>(a, b, c, o.m, o.n, o.k, task);
  CUDA_CHECK(cudaGetLastError());
}

// cuBLAS uses column-major storage.  Treating row-major C=A*B as the
// column-major transpose C^T=B^T*A^T gives the argument order below.
void run_cublas_loop(cublasHandle_t handle, float const* a, float const* b,
                     float* c, Options const& o) {
  float const alpha = 1.0f, beta = 0.0f;
  for (int task = 0; task < o.tasks; ++task) {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, o.n, o.m, o.k,
        &alpha, b, o.n, a + static_cast<size_t>(task) * o.m * o.k, o.k,
        &beta, c + static_cast<size_t>(task) * o.m * o.n, o.n));
  }
}

void run_cublas_batched(cublasHandle_t handle, float const* const* a_array,
                        float const* const* b_array, float* const* c_array,
                        Options const& o) {
  float const alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      o.n, o.m, o.k, &alpha, b_array, o.n, a_array, o.k, &beta,
      c_array, o.n, o.tasks));
}

cudaGraphExec_t build_graph(float const* a, float const* b, float* c, Options const& o) {
  cudaGraph_t graph{}; CUDA_CHECK(cudaGraphCreate(&graph, 0));
  int m = o.m, n = o.n, k = o.k;
  for (int task = 0; task < o.tasks; ++task) {
    void* args[] = {&a, &b, &c, &m, &n, &k, &task};
    cudaKernelNodeParams p{}; p.func = reinterpret_cast<void*>(gemm_task_kernel); p.gridDim = dim3(1); p.blockDim = dim3(o.m * o.n); p.kernelParams = args;
    cudaGraphNode_t node{}; CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p));
  }
  cudaGraphExec_t exec{}; CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)); CUDA_CHECK(cudaGraphDestroy(graph)); return exec;
}

template <typename F> std::vector<float> benchmark(F&& f, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) f(); CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); std::vector<float> x; x.reserve(repeats);
  for (int i = 0; i < repeats; ++i) { CUDA_CHECK(cudaEventRecord(start)); f(); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); x.push_back(ms * 1000.0f); }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); return x;
}
template <typename F> std::vector<float> benchmark_mpk(F&& f, QueueState* q, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) { CUDA_CHECK(cudaMemsetAsync(q, 0, sizeof(QueueState), 0)); f(); } CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); std::vector<float> x; x.reserve(repeats);
  for (int i = 0; i < repeats; ++i) { CUDA_CHECK(cudaMemsetAsync(q, 0, sizeof(QueueState), 0)); CUDA_CHECK(cudaEventRecord(start)); f(); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); x.push_back(ms * 1000.0f); }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); return x;
}
float p50(std::vector<float> x) { std::sort(x.begin(), x.end()); return x[(x.size() - 1) / 2]; }
void report(char const* name, std::vector<float> const& x) { float sum = 0; for (float v : x) sum += v; std::printf("%-18s p50=%8.2f us  mean=%8.2f us\n", name, p50(x), sum / x.size()); }
float max_abs(float const* a, float const* b, size_t n) { float result = 0; for (size_t i = 0; i < n; ++i) result = std::max(result, std::abs(a[i] - b[i])); return result; }

int main(int argc, char** argv) {
  try {
    Options const o = parse_options(argc, argv); cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int const workers = o.workers == 0 ? std::min(o.tasks, prop.multiProcessorCount) : o.workers;
    if (workers <= 0 || workers > prop.multiProcessorCount) throw std::runtime_error("--workers must be in [1, SM count]");
    std::printf("device=%s sm_%d%d, GEMM/task=[%d,%d]x[%d,%d], tasks=%d workers=%d\n", prop.name, prop.major, prop.minor, o.m, o.k, o.k, o.n, o.tasks, workers);
    size_t const a_count = static_cast<size_t>(o.tasks) * o.m * o.k, b_count = static_cast<size_t>(o.k) * o.n, c_count = static_cast<size_t>(o.tasks) * o.m * o.n;
    std::vector<float> ha(a_count), hb(b_count); for (size_t i = 0; i < a_count; ++i) ha[i] = .001f * (static_cast<int>(i % 97) - 48); for (size_t i = 0; i < b_count; ++i) hb[i] = .001f * (static_cast<int>(i % 83) - 41);
    float *a{}, *b{}, *bc{}, *gc{}, *mc{}, *clc{}, *cbc{}; QueueState* q{};
    float const** a_array{}; float const** b_array{}; float** c_array{};
    CUDA_CHECK(cudaMalloc(&a, a_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&b, b_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&bc, c_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&gc, c_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&mc, c_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&clc, c_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&cbc, c_count * sizeof(float))); CUDA_CHECK(cudaMalloc(&q, sizeof(QueueState)));
    CUDA_CHECK(cudaMalloc(&a_array, o.tasks * sizeof(float const*))); CUDA_CHECK(cudaMalloc(&b_array, o.tasks * sizeof(float const*))); CUDA_CHECK(cudaMalloc(&c_array, o.tasks * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(a, ha.data(), a_count * sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(b, hb.data(), b_count * sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemset(q, 0, sizeof(QueueState)));
    std::vector<float const*> h_a_array(o.tasks), h_b_array(o.tasks); std::vector<float*> h_c_array(o.tasks);
    for (int task = 0; task < o.tasks; ++task) { h_a_array[task] = a + static_cast<size_t>(task) * o.m * o.k; h_b_array[task] = b; h_c_array[task] = cbc + static_cast<size_t>(task) * o.m * o.n; }
    CUDA_CHECK(cudaMemcpy(a_array, h_a_array.data(), o.tasks * sizeof(float const*), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(b_array, h_b_array.data(), o.tasks * sizeof(float const*), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(c_array, h_c_array.data(), o.tasks * sizeof(float*), cudaMemcpyHostToDevice));
    cublasHandle_t handle{}; CUBLAS_CHECK(cublasCreate(&handle)); CUBLAS_CHECK(cublasSetStream(handle, 0));
    cudaGraphExec_t graph = build_graph(a, b, gc, o); auto baseline = [&] { run_baseline(a, b, bc, o); }; auto graph_launch = [&] { CUDA_CHECK(cudaGraphLaunch(graph, 0)); }; auto mpk = [&] { mpk_independent_gemm_kernel<<<workers, o.m * o.n>>>(a, b, mc, o.m, o.n, o.k, o.tasks, q); CUDA_CHECK(cudaGetLastError()); }; auto cublas_loop = [&] { run_cublas_loop(handle, a, b, clc, o); }; auto cublas_batched = [&] { run_cublas_batched(handle, a_array, b_array, c_array, o); };
    baseline(); graph_launch(); mpk(); cublas_loop(); cublas_batched(); CUDA_CHECK(cudaDeviceSynchronize()); std::vector<float> hbc(c_count), hgc(c_count), hmc(c_count), hclc(c_count), hcbc(c_count); CUDA_CHECK(cudaMemcpy(hbc.data(), bc, c_count * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hgc.data(), gc, c_count * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hmc.data(), mc, c_count * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hclc.data(), clc, c_count * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hcbc.data(), cbc, c_count * sizeof(float), cudaMemcpyDeviceToHost)); float const ge = max_abs(hbc.data(), hgc.data(), c_count), me = max_abs(hbc.data(), hmc.data(), c_count), le = max_abs(hbc.data(), hclc.data(), c_count), be = max_abs(hbc.data(), hcbc.data(), c_count); std::printf("correctness: graph=%g mpk=%g cublas_loop=%g cublas_batched=%g\n", ge, me, le, be); if (ge != 0 || me != 0 || le > 1e-4f || be > 1e-4f) throw std::runtime_error("backend outputs differ");
    report("baseline <<<>>>", benchmark(baseline, o.warmup, o.repeats)); report("cudaGraphLaunch", benchmark(graph_launch, o.warmup, o.repeats)); report("mpk persistent", benchmark_mpk(mpk, q, o.warmup, o.repeats)); report("cuBLAS loop", benchmark(cublas_loop, o.warmup, o.repeats)); report("cuBLAS batched", benchmark(cublas_batched, o.warmup, o.repeats));
    std::printf("Note: cuBLAS uses the column-major transpose formulation; MPK uses only an independent-task queue.\n"); CUBLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaGraphExecDestroy(graph)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(bc)); CUDA_CHECK(cudaFree(gc)); CUDA_CHECK(cudaFree(mc)); CUDA_CHECK(cudaFree(clc)); CUDA_CHECK(cudaFree(cbc)); CUDA_CHECK(cudaFree(q)); CUDA_CHECK(cudaFree(a_array)); CUDA_CHECK(cudaFree(b_array)); CUDA_CHECK(cudaFree(c_array));
  } catch (std::exception const& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
