#!/usr/bin/env bash
set -euo pipefail

backend="${1:-persistent_grouped}"
mkdir -p profiling/nsys/experiment1
nsys profile \
  --force-overwrite=true \
  --sample=none \
  --trace=cuda,nvtx,cublas \
  --output="profiling/nsys/experiment1/${backend}" \
  python -m vibekernel.experiments.experiment1.run \
    --backend "${backend}" --profile
