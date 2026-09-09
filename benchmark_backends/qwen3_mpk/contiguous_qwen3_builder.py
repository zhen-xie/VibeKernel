"""Local Qwen3 MPK builder for contiguous KV-cache benchmarking.

This module intentionally starts as an integration boundary: it will reuse
the upstream Qwen3 GEMM/RMSNorm task construction while replacing only its
paged attention/cache binding with ``contiguous_attention_hopper``.
"""


class ContiguousQwen3MPKBuilder:
    """Reserved local builder; no upstream Mirage source is modified."""

    task_name = "contiguous_attention_hopper"
