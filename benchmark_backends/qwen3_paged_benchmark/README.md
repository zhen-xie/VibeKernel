# Real Qwen3 paged-KV benchmark

This benchmark compares Mirage-MPK, pure PyTorch, and pure Triton under one
contract: the same Qwen3 checkpoint, tokenizer-generated prompt tokens, paged
KV layout, page table, warmup, repeats, and JSON result schema.

Implementation order is MPK → PyTorch → Triton.  MPK reuses the repository's
existing Qwen3Builder and Hopper paged-attention task; no custom continuous-KV
task is introduced.
