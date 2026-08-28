# VibeKernel

Experiment 1 compares three complete execution strategies for a BF16 MoE FFN
up-projection grouped GEMM on an NVIDIA H100:

1. cuBLASLt-backed PyTorch matmuls captured in a multi-stream CUDA Graph.
2. A single Triton grouped-GEMM kernel.
3. CUTLASS 3.x Hopper Persistent Grouped GEMM with device-side scheduling.

The workload models 512 input tokens routed top-2 to 32 experts (1,024
expert-token assignments). The intentionally skewed expert loads are
`M=[4,4,4,4,8,8,8,8,12,12,12,12,16,16,16,16,24,24,24,24,32,32,32,32,48,48,48,48,112,112,112,112]`;
each expert computes
`X_e[M,4096] @ W_e[4096,14336]`. This represents the first, expanding FFN
projection of an MoE block. Inputs and outputs are BF16 and accumulation is
FP32.

## H100 setup

Use Linux, CUDA 12.x, Python 3.10+, and a CUDA-enabled PyTorch build.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
export CUTLASS_PATH="$PWD/third_party/cutlass"
export TORCH_CUDA_ARCH_LIST=9.0a
python -m vibekernel.experiments.experiment1.run --backend all
```

The persistent CUDA extension is JIT-compiled on first use. Build/compile time,
CUDA Graph capture, tensor allocation, and correctness validation are outside
the steady-state timing region.

## Profiling

```bash
bash scripts/profile_experiment1_nsys.sh persistent_grouped
bash scripts/profile_experiment1_ncu.sh persistent_grouped
python scripts/plot_experiment1.py results/experiment1/latest.json
```

Profiler runs are separate from latency runs. Nsight overhead must not be used
as benchmark latency. Access to hardware counters may require administrator
configuration on the H100 host.

## Current implementation status

The persistent backend uses CUTLASS 3.x Hopper Grouped GEMM with BF16 inputs,
FP32 accumulation, a TMA+GMMA warp-specialized mainloop, and CUTLASS's
device-side persistent grouped scheduler. It requires a CUTLASS checkout via
`CUTLASS_PATH`, CUDA 12.3 or newer, and an SM90 H100 GPU.

The library backend requests PyTorch's cuBLASLt preference and captures the
matmuls in a CUDA Graph. Confirm the actual selected kernels in Nsight Systems;
the final production version may replace this wrapper with direct cuBLASLt API
calls if the target PyTorch build does not honor the preference for a shape.
