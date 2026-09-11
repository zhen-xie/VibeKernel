#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t const e = (expr); if (e != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e)); } while (0)

__device__ __forceinline__ float task_op(float x, int stage) {
  return tanhf(fmaf(x, 1.0001f + 0.00001f * stage, 0.0003f * (stage + 1)));
}

// All three backends use this exact tile implementation.
__global__ void task_kernel(float const* input, float* output, int n, int stage) {
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) output[i] = task_op(input[i], stage);
}

// ticket = (stage << 32) | next_tile.  Keeping these two values in one atomic
// word prevents a worker from claiming a tile for a stage that just ended.
struct PersistentState { unsigned long long ticket; int completed_tiles, epoch; };

// Minimal multi-CTA device scheduler. Workers claim tiles of the current
// logical task; its last tile opens the next dependent task.
__global__ void mpk_persistent_kernel(float const* input, float* ping, float* pong,
                                      int n, int tasks, int tiles,
                                      PersistentState* state, int launch_epoch) {
  __shared__ int local_stage, local_tile;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    state->ticket = 0; state->completed_tiles = 0;
    __threadfence();
    atomicExch(&state->epoch, launch_epoch);
  }
  while (atomicAdd(&state->epoch, 0) != launch_epoch) {}

  while (true) {
    if (threadIdx.x == 0) {
      // Atomically claim (stage, tile) with a CAS.  A separate stage load and
      // next_tile increment would permit a stale worker to corrupt a later
      // stage after its predecessor completed.
      while (true) {
        unsigned long long const old = atomicCAS(&state->ticket, 0ULL, 0ULL);
        int const stage = static_cast<int>(old >> 32);
        int const tile = static_cast<int>(old & 0xffffffffULL);
        if (stage >= tasks) { local_stage = stage; local_tile = -1; break; }
        if (tile >= tiles) {
          while (static_cast<int>(atomicCAS(&state->ticket, 0ULL, 0ULL) >> 32) == stage) {}
          continue;
        }
        unsigned long long const desired =
            (static_cast<unsigned long long>(stage) << 32) | static_cast<unsigned int>(tile + 1);
        if (atomicCAS(&state->ticket, old, desired) == old) {
          local_stage = stage; local_tile = tile; break;
        }
      }
    }
    __syncthreads();
    if (local_stage >= tasks) return;
    // This data crosses CTA boundaries at every stage.  Volatile prevents a
    // worker from reusing a stale L1 value after the device-side stage barrier;
    // __threadfence below publishes the producer CTA's writes first.
    float const volatile* src = (local_stage & 1) ? ping : input;
    float volatile* dst = (local_stage & 1) ? pong : ping;
    int const i = local_tile * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = task_op(src[i], local_stage);
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      if (atomicAdd(&state->completed_tiles, 1) == tiles - 1) {
        state->completed_tiles = 0;
        __threadfence();
        atomicExch(&state->ticket,
                   static_cast<unsigned long long>(local_stage + 1) << 32);
      }
    }
    __syncthreads();
  }
}

struct Options { int elements = 4096, tasks = 20, warmup = 100, repeats = 1000, block = 256, workers = 0; };

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
    else if (a == "--help") { std::printf("Usage: launch_overhead [--elements N] [--tasks N] [--warmup N] [--repeats N] [--block N] [--workers N]\n"); std::exit(0); }
    else throw std::runtime_error("unknown option: " + a);
  }
  if (o.elements <= 0 || o.tasks <= 0 || o.warmup < 0 || o.repeats <= 0 || o.block <= 0 || o.block > 1024 || o.workers < 0) throw std::runtime_error("invalid size");
  return o;
}

void run_baseline(float const* input, float* ping, float* pong, Options const& o) {
  int const grid = (o.elements + o.block - 1) / o.block;
  float const* src = input; float* dst = ping;
  for (int stage = 0; stage < o.tasks; ++stage) {
    task_kernel<<<grid, o.block>>>(src, dst, o.elements, stage);
    src = dst; dst = (dst == ping) ? pong : ping;
  }
  CUDA_CHECK(cudaGetLastError());
}

cudaGraphExec_t build_graph(float const* input, float* ping, float* pong, Options const& o) {
  cudaGraph_t graph{}; CUDA_CHECK(cudaGraphCreate(&graph, 0));
  std::vector<cudaGraphNode_t> nodes; nodes.reserve(o.tasks);
  int n = o.elements; float const* src = input; float* dst = ping;
  for (int stage = 0; stage < o.tasks; ++stage) {
    void* args[] = {&src, &dst, &n, &stage};
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(task_kernel);
    p.gridDim = dim3((o.elements + o.block - 1) / o.block); p.blockDim = dim3(o.block); p.kernelParams = args;
    cudaGraphNode_t node{};
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nodes.empty() ? nullptr : &nodes.back(), nodes.empty() ? 0 : 1, &p));
    nodes.push_back(node); src = dst; dst = (dst == ping) ? pong : ping;
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

float percentile(std::vector<float> x, float p) { std::sort(x.begin(), x.end()); return x[std::min(static_cast<size_t>(std::ceil(p * x.size())) - 1, x.size() - 1)]; }
void report(char const* name, std::vector<float> const& x) { float sum = 0; for (float v : x) sum += v; std::printf("%-18s p50=%8.2f us  p90=%8.2f us  mean=%8.2f us\n", name, percentile(x, .5f), percentile(x, .9f), sum / x.size()); }
float max_abs_difference(float const* a, float const* b, int n) { float m = 0; for (int i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

int main(int argc, char** argv) {
  try {
    Options const o = parse_options(argc, argv); cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int const tiles = (o.elements + o.block - 1) / o.block;
    int const workers = o.workers == 0 ? std::min(tiles, prop.multiProcessorCount) : o.workers;
    if (workers <= 0 || workers > prop.multiProcessorCount) throw std::runtime_error("--workers must be in [1, SM count]");
    std::printf("device=%s sm_%d%d, elements=%d tasks=%d block=%d tiles=%d workers=%d\n", prop.name, prop.major, prop.minor, o.elements, o.tasks, o.block, tiles, workers);
    std::vector<float> host_input(o.elements); for (int i = 0; i < o.elements; ++i) host_input[i] = .001f * (i % 113 - 56);
    float *input{}, *bp{}, *bq{}, *gp{}, *gq{}, *mp{}, *mq{}; PersistentState* state{}; size_t const bytes = sizeof(float) * o.elements;
    CUDA_CHECK(cudaMalloc(&input, bytes)); CUDA_CHECK(cudaMalloc(&bp, bytes)); CUDA_CHECK(cudaMalloc(&bq, bytes)); CUDA_CHECK(cudaMalloc(&gp, bytes)); CUDA_CHECK(cudaMalloc(&gq, bytes)); CUDA_CHECK(cudaMalloc(&mp, bytes)); CUDA_CHECK(cudaMalloc(&mq, bytes)); CUDA_CHECK(cudaMalloc(&state, sizeof(PersistentState)));
    CUDA_CHECK(cudaMemset(state, 0, sizeof(PersistentState))); CUDA_CHECK(cudaMemcpy(input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    cudaGraphExec_t graph = build_graph(input, gp, gq, o);
    auto baseline = [&] { run_baseline(input, bp, bq, o); };
    auto graph_launch = [&] { CUDA_CHECK(cudaGraphLaunch(graph, 0)); };
    int epoch = 0;
    auto mpk = [&] { mpk_persistent_kernel<<<workers, o.block>>>(input, mp, mq, o.elements, o.tasks, tiles, state, ++epoch); CUDA_CHECK(cudaGetLastError()); };
    baseline(); graph_launch(); mpk(); CUDA_CHECK(cudaDeviceSynchronize());
    float const* bo = (o.tasks & 1) ? bp : bq; float const* go = (o.tasks & 1) ? gp : gq; float const* mo = (o.tasks & 1) ? mp : mq;
    std::vector<float> bh(o.elements), gh(o.elements), mh(o.elements); CUDA_CHECK(cudaMemcpy(bh.data(), bo, bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(gh.data(), go, bytes, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(mh.data(), mo, bytes, cudaMemcpyDeviceToHost));
    float const ge = max_abs_difference(bh.data(), gh.data(), o.elements), me = max_abs_difference(bh.data(), mh.data(), o.elements); std::printf("correctness: graph max_abs=%g, mpk max_abs=%g\n", ge, me); if (ge != 0 || me != 0) throw std::runtime_error("backend outputs differ");
    report("baseline <<<>>>", benchmark(baseline, o.warmup, o.repeats)); report("cudaGraphLaunch", benchmark(graph_launch, o.warmup, o.repeats)); report("mpk persistent", benchmark(mpk, o.warmup, o.repeats));
    std::printf("Note: MPK-v1 is a minimal multi-CTA tile scheduler, not Mirage's full runtime.\n");
    CUDA_CHECK(cudaGraphExecDestroy(graph)); CUDA_CHECK(cudaFree(input)); CUDA_CHECK(cudaFree(bp)); CUDA_CHECK(cudaFree(bq)); CUDA_CHECK(cudaFree(gp)); CUDA_CHECK(cudaFree(gq)); CUDA_CHECK(cudaFree(mp)); CUDA_CHECK(cudaFree(mq)); CUDA_CHECK(cudaFree(state));
  } catch (std::exception const& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
