# Qwen3 paged-KV benchmark

This directory changes the comparison contract from contiguous KV to Mirage
MPK's native paged KV representation:

```text
[layer, K/V, page, token-in-page, KV-head, head-dim]
```

The page-table metadata matches MPK Qwen3Builder: `indptr`, `indices`, and
`last_page_len`.  PyTorch and Triton implementations will consume this exact
layout before the existing MPK Qwen3 builder is benchmarked.
