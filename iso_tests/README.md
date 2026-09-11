# Independent CUDA dispatch experiments

This directory deliberately does not include or link Mirage/MPK.  The first
experiment compares three ways to execute an identical dependent task chain:

1. ordinary CUDA launches (`baseline<<<...>>>` once per task),
2. a CUDA Graph containing those same task kernels, and
3. one standalone persistent CUDA kernel (`MPK-v0`).

`MPK-v0` uses one CTA and a statically known chain.  It is an isolation test
for launch elimination and device-side transitions, not a reproduction of
Mirage's multi-worker dynamic scheduler.

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
