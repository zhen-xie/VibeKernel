# Independent CUDA dispatch experiments

This directory deliberately does not include or link Mirage/MPK.  The first
experiment compares three ways to execute an identical dependent task chain:

1. ordinary CUDA launches (`baseline<<<...>>>` once per task),
2. a CUDA Graph containing those same task kernels, and
3. one standalone persistent CUDA kernel (`MPK-v1`).

`MPK-v1` starts multiple persistent worker CTAs. Workers obtain tiles of the
current task through a device-resident counter; completion of all tiles opens
the next dependent task. It is an isolation test, not a reproduction of
Mirage's richer multi-worker dynamic scheduler.

## Build

```bash
cmake -S iso_tests -B build/iso_tests -DCMAKE_BUILD_TYPE=Release
cmake --build build/iso_tests -j
```

## Run

```bash
CUDA_VISIBLE_DEVICES=0 ./build/iso_tests/01_launch_overhead/launch_overhead \
  --elements 4096 --tasks 20 --warmup 100 --repeats 1000
```

Use small per-task work (for example, 256--16384 elements) and progressively
increase `--tasks` to expose dispatch overhead.  The next experiment should
replace this elementwise chain with a transformer-block DAG while retaining
the same three runners.

## 02: independent task queue

`02_independent_tasks` removes all data dependencies.  Each logical task is a
single CTA that reads the same input and writes a disjoint output slice.  The
persistent implementation therefore performs only `atomicAdd(next_task)` task
claims—no stage barrier, dependency counter, or worker spin wait.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/iso_tests/02_independent_tasks/independent_tasks \
  --elements 4096 --tasks 64 --warmup 100 --repeats 1000
```

## 03: independent GEMM tasks

Each task is a one-CTA FP32 GEMM, `C_task[M,N] = A_task[M,K] × B[K,N]`. Tasks
share B but write disjoint C slices, so they have no data dependency. In
addition to the hand-written CUDA/Graph/MPK runners, this test reports a loop
of `cublasSgemm` calls and one `cublasSgemmBatched` call.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/iso_tests/03_independent_gemm_tasks/independent_gemm_tasks \
  --m 16 --n 16 --k 256 --tasks 64 --warmup 100 --repeats 1000
```
