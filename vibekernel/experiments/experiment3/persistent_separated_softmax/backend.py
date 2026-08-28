import torch
from .kernel import persistent_softmax_kernel

class PersistentSeparatedSoftmaxBackend:
    name="persistent_separated_softmax"
    def prepare(self,x): self.x=x; self.y=torch.empty_like(x); self.queue=torch.zeros((),device=x.device,dtype=torch.int32); self.run(); torch.cuda.synchronize()
    def run(self):
        self.queue.zero_(); sm=torch.cuda.get_device_properties(self.x.device).multi_processor_count
        persistent_softmax_kernel[(sm,)](self.x,self.y,self.queue,self.x.stride(0),ROWS=self.x.shape[0],N=self.x.shape[1],BLOCK=4096,num_warps=8); return [self.y]
    def metadata(self): return {"implementation":"persistent Triton CTA worker pool; per-row softmax","kernel_count":1,"scheduler":"atomic device task queue"}
    def close(self): pass
