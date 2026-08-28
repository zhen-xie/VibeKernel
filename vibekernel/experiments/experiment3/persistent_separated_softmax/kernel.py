import triton
import triton.language as tl

@triton.jit
def persistent_softmax_kernel(x,y,next_task,stride,ROWS:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    col=tl.arange(0,BLOCK); row=tl.atomic_add(next_task,1)
    while row < ROWS:
        value=tl.load(x+row*stride+col,mask=col<N,other=-float("inf")); value=tl.exp(value-tl.max(value,axis=0))
        tl.store(y+row*stride+col,value/tl.sum(value,axis=0),mask=col<N); row=tl.atomic_add(next_task,1)
