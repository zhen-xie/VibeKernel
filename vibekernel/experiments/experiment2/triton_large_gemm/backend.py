from vibekernel.experiments.experiment1.triton_grouped.backend import TritonGroupedBackend
class TritonLargeBackend(TritonGroupedBackend):
 name='triton_large_gemm'
 def __init__(self):super().__init__(block_m=64,block_n=128,block_k=32,num_warps=8)
