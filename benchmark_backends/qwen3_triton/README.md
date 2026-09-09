# Qwen3 pure-Triton backend

This directory contains only Triton compute code.  It reuses the deterministic
weights, Qwen3 tensor dimensions, and contiguous KV-cache layout defined by
`../qwen3_contiguous/common.py`, so outputs and timings stay comparable.

Run the current operator correctness gate on CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_backends/qwen3_triton/test_triton_operators.py
```

RMSNorm, bias-free BF16 linear/GEMM, RoPE, and contiguous KV writes have
correctness gates.  Causal grouped-query attention, elementwise MLP /
residual, and the full `run_triton.py` timing entrypoint follow in that order.
