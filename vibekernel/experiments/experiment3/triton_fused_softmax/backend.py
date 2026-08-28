import torch
from .kernel import fused_softmax_kernel

class TritonFusedSoftmaxBackend:
    name = "triton_fused_softmax"
    def prepare(self, x): self.x=x; self.y=torch.empty_like(x); self.run(); torch.cuda.synchronize()
    def run(self):
        fused_softmax_kernel[(self.x.shape[0],)](self.x,self.y,self.x.stride(0),N=self.x.shape[1],BLOCK=4096,num_warps=8); return [self.y]
    def metadata(self): return {"implementation":"one fused Triton softmax kernel","kernel_count":1}
    def close(self): pass
