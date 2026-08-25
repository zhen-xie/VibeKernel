#!/usr/bin/env bash
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
python -m vibekernel.experiments.experiment1.run \
  --backend all \
  --warmup "${WARMUP:-100}" \
  --iterations "${ITERATIONS:-1000}" \
  --repeats-per-sample "${REPEATS_PER_SAMPLE:-1}" \
  "$@"
