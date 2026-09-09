"""PyTorch semantic reference using Mirage-compatible paged KV storage."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

REFERENCE = Path(__file__).resolve().parents[1] / "qwen3_contiguous"
sys.path.insert(0, str(REFERENCE))
from pytorch_backend import PyTorchBackend  # noqa: E402
from common import Qwen3BenchmarkConfig, Qwen3Weights  # noqa: E402
from paged_cache import PagedKVCache


class PagedPyTorchBackend(PyTorchBackend):
    name = "pytorch-eager-paged"

    def __init__(self, cfg: Qwen3BenchmarkConfig, weights: Qwen3Weights, batch: int, profiler=None, page_size: int = 64):
        if batch != 1:
            raise ValueError("initial paged reference is single-request; multi-request page tables are next")
        super().__init__(cfg, weights, batch, profiler)
        pages = (cfg.max_seq_len + page_size - 1) // page_size
        self.cache = PagedKVCache(cfg.num_layers, pages, page_size, cfg.num_key_value_heads,
                                  cfg.head_dim, weights.embedding.device, weights.embedding.dtype)
