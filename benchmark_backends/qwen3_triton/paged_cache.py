"""Triton page-table KV write/gather kernels for the paged benchmark."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _write(src, indices, cache, start, n, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); d = p % D; t = p // D
    token = t % S; h = t // S; logical = start + token
    physical = tl.load(indices + logical // PAGE); dst = ((physical * PAGE + logical % PAGE) * H + h) * D + d
    tl.store(cache + dst, tl.load(src + p, mask=p < n), mask=p < n)

@triton.jit
def _gather(cache, indices, out, length, H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); d = p % D; t = p // D
    token = t % length; h = t // length; physical = tl.load(indices + token // PAGE)
    src = ((physical * PAGE + token % PAGE) * H + h) * D + d
    tl.store(out + p, tl.load(cache + src, mask=p < length * H * D), mask=p < length * H * D)

def paged_write(src, cache, indices, start: int) -> None:
    _, h, s, d = src.shape; n = src.numel()
    _write[(triton.cdiv(n, 256),)](src, indices, cache, start, n, S=s, H=h, D=d, PAGE=cache.shape[1], BLOCK=256)

def paged_gather(cache, indices, length: int) -> torch.Tensor:
    _, _, h, d = cache.shape; out = torch.empty((1, h, length, d), device=cache.device, dtype=cache.dtype)
    _gather[(triton.cdiv(out.numel(), 256),)](cache, indices, out, length, H=h, D=d, PAGE=cache.shape[1], BLOCK=256)
    return out
