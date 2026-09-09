"""Paged KV-cache contract matching Mirage-MPK's Qwen3Builder layout."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PagedKVMetadata:
    """One-request page table, compatible with MPK's three paging buffers."""
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor


class PagedKVCache:
    """[layer, page, token-in-page, KVH, D] BF16 cache shared by all backends."""

    def __init__(self, layers: int, pages: int, page_size: int, kv_heads: int, head_dim: int, device, dtype):
        self.storage = torch.empty((layers, 2, pages, page_size, kv_heads, head_dim), device=device, dtype=dtype)
        self.page_size, self.pages, self.length = page_size, pages, 0
        self.metadata = PagedKVMetadata(
            indptr=torch.tensor([0, pages], device=device, dtype=torch.int32),
            indices=torch.arange(pages, device=device, dtype=torch.int32),
            last_page_len=torch.zeros(1, device=device, dtype=torch.int32),
        )

    def reset(self) -> None:
        self.length = 0
        self.metadata.last_page_len.zero_()

    def finish_layer_stack(self, count: int) -> None:
        self.length += count
        self.metadata.last_page_len.fill_(((self.length - 1) % self.page_size) + 1)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Write ``[B, KVH, S, D]`` into physical pages and return history.

        The reference intentionally gathers history into logical token order;
        later Triton/MPK paths will consume page-table indirection directly.
        """
        b, h, count, d = k.shape
        if b != 1:
            raise ValueError("initial paged reference supports one request; batch paging follows next")
        start, end = self.length, self.length + count
        if end > self.pages * self.page_size:
            raise ValueError("paged KV cache overflow")
        # One indexed GPU copy, not a Python token loop.  Calling ``.item()``
        # per token would synchronize the CPU and turn this reference into a
        # benchmark of Python overhead rather than paged KV storage.
        write_positions = torch.arange(start, end, device=k.device)
        write_slots, write_in_page = (torch.div(write_positions, self.page_size, rounding_mode="floor"),
                                      write_positions % self.page_size)
        write_pages = self.metadata.indices[write_slots].long()
        self.storage[layer, 0, write_pages, write_in_page].copy_(k[0].permute(1, 0, 2))
        self.storage[layer, 1, write_pages, write_in_page].copy_(v[0].permute(1, 0, 2))
        positions = torch.arange(end, device=k.device)
        page_slots, in_page = torch.div(positions, self.page_size, rounding_mode="floor"), positions % self.page_size
        pages = self.metadata.indices[page_slots].long()
        kh = self.storage[layer, 0, pages, in_page].permute(1, 0, 2).unsqueeze(0)
        vh = self.storage[layer, 1, pages, in_page].permute(1, 0, 2).unsqueeze(0)
        return kh, vh, start
