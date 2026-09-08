# Qwen3 contiguous-KV backend benchmark

This directory is intentionally separate from `qwen3_compute_sim.py`.  The
simulator remains the readable semantic reference; this benchmark uses a
preallocated contiguous KV cache and never materializes repeated GQA K/V.

The initial executable backend is `pytorch_backend.py`.  It establishes the
shared contract for the planned Triton and Mirage-MPK backends:

- BF16, deterministic shared weights;
- fused QKV and Gate+Up projections;
- cache layout `[layer, K/V, batch, KV heads, sequence, head dim]`;
- in-place cache writes; and
- RoPE table precomputed once per backend.

Run on CUDA:

```bash
python benchmark/qwen3_contiguous/run_pytorch.py --batch 2 --prompt-len 128 --decode-steps 32
```

The next implementations use the same `prefill`, `decode`, and `reset_cache`
interface.  Mirage-MPK needs a dedicated contiguous-GQA attention task or
adapter: the production Qwen3 builder is paged-KV based and is deliberately
not used as an apples-to-apples substitute here.
