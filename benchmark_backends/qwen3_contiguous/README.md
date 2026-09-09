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
python benchmark_backends/qwen3_contiguous/run_pytorch.py --batch 2 --prompt-len 128 --decode-steps 32
```

Use Qwen3-8B Transformer dimensions before attempting the full model:

```bash
python benchmark_backends/qwen3_contiguous/run_pytorch.py \
  --preset qwen3-8b-core --layers 1 --batch 1 --prompt-len 128 \
  --decode-steps 32 --warmup 30 --repeats 200
```

`qwen3-8b-core` defaults to vocabulary size 4096 so that Transformer timing is
not dominated by the LM head. Add `--vocab-size 151936` to include the full
Qwen3-8B LM-head cost.

Append `--breakdown` to print CUDA-Event timings for the logical stages of one
prefill and one decode. This is diagnostic output, not the headline benchmark:
the main p50/p90 timing is measured separately without profiler events.

The next implementations use the same `prefill`, `decode`, and `reset_cache`
interface.  Mirage-MPK needs a dedicated contiguous-GQA attention task or
adapter: the production Qwen3 builder is paged-KV based and is deliberately
not used as an apples-to-apples substitute here.
