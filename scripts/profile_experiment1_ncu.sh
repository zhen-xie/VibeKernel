#!/usr/bin/env bash
set -euo pipefail

backend="${1:-persistent_grouped}"
mkdir -p profiling/ncu/experiment1
ncu \
  --force-overwrite \
  --target-processes all \
  --set full \
  --nvtx \
  --nvtx-include "experiment1/${backend}/profile_region/" \
  --export "profiling/ncu/experiment1/${backend}" \
  python -m vibekernel.experiments.experiment1.run \
    --backend "${backend}" --profile

ncu --import "profiling/ncu/experiment1/${backend}.ncu-rep" \
  --csv --page raw > "profiling/ncu/experiment1/${backend}.csv"
