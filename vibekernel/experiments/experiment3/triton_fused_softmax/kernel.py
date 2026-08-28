import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(x, y, stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0); col = tl.arange(0, BLOCK)
    value = tl.load(x + row * stride + col, mask=col < N, other=-float("inf"))
    value = tl.exp(value - tl.max(value, axis=0))
    tl.store(y + row * stride + col, value / tl.sum(value, axis=0), mask=col < N)
