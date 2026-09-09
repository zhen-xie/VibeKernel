# Qwen3 contiguous-KV MPK backend

This is a benchmark-local MPK extension.  It must keep the upstream paged-KV
Qwen3 builder unchanged while adding a continuous-KV attention task for an
apples-to-apples comparison with `qwen3_contiguous` and `qwen3_triton`.

Planned execution path:

```text
benchmark-local builder
  -> existing PersistentKernel scheduler/codegen
  -> benchmark-local contiguous_attention_hopper task
  -> generated test.cu / nvcc / MPK launcher
```

The extension owns the task-specific graph binding and CUDA code; it does not
replace the MPK scheduler, task graph JSON format, or existing paged tasks.
