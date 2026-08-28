import argparse
from datetime import datetime
from pathlib import Path
import torch
import triton
import triton.language as tl
from vibekernel.common.benchmark import measure_backend
from vibekernel.common.results import write_results

@triton.jit
def softmax_kernel(x, y, stride, N: tl.constexpr, BLOCK: tl.constexpr):
 r=tl.program_id(0); c=tl.arange(0,BLOCK); v=tl.load(x+r*stride+c, mask=c<N, other=-float('inf')); v=tl.exp(v-tl.max(v,axis=0)); tl.store(y+r*stride+c,v/tl.sum(v,axis=0),mask=c<N)
@triton.jit
def persistent_kernel(x,y,next_task,stride,ROWS:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
 c=tl.arange(0,BLOCK); r=tl.atomic_add(next_task,1)
 while r<ROWS:
  v=tl.load(x+r*stride+c,mask=c<N,other=-float('inf')); v=tl.exp(v-tl.max(v,axis=0)); tl.store(y+r*stride+c,v/tl.sum(v,axis=0),mask=c<N); r=tl.atomic_add(next_task,1)

class Separated:
 name='separated_softmax'
 def prepare(self,x): self.x=x; self.y=torch.empty_like(x); self.run(); torch.cuda.synchronize()
 def run(self):
  m=self.x.max(dim=1,keepdim=True).values; e=torch.exp(self.x-m); self.y.copy_(e/e.sum(dim=1,keepdim=True)); return [self.y]
 def metadata(self): return {'implementation':'separate Max/Sub/Exp/Sum/Div tensor operations','kernel_count':5}
 def close(self): pass
class TritonFused:
 name='triton_fused_softmax'
 def prepare(self,x): self.x=x; self.y=torch.empty_like(x); self.run(); torch.cuda.synchronize()
 def run(self): softmax_kernel[(self.x.shape[0],)](self.x,self.y,self.x.stride(0),N=self.x.shape[1],BLOCK=4096,num_warps=8); return [self.y]
 def metadata(self): return {'implementation':'one fused Triton softmax kernel','kernel_count':1}
 def close(self): pass
class Persistent(TritonFused):
 name='persistent_separated_softmax'
 def prepare(self,x): self.x=x; self.y=torch.empty_like(x); self.queue=torch.zeros((),device=x.device,dtype=torch.int32); self.run(); torch.cuda.synchronize()
 def run(self): self.queue.zero_(); persistent_kernel[(torch.cuda.get_device_properties(self.x.device).multi_processor_count,)](self.x,self.y,self.queue,self.x.stride(0),ROWS=self.x.shape[0],N=self.x.shape[1],BLOCK=4096,num_warps=8); return [self.y]
 def metadata(self): return {'implementation':'persistent Triton CTA worker pool; per-row Max/Sub/Exp/Sum/Div','kernel_count':1,'scheduler':'atomic device task queue'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--backend',choices=('all','separated_softmax','triton_fused_softmax','persistent_separated_softmax'),default='all'); p.add_argument('--warmup',type=int,default=100); p.add_argument('--iterations',type=int,default=1000); p.add_argument('--output',type=Path); a=p.parse_args(); x=torch.randn((4096,4096),device='cuda',dtype=torch.float32); expected=torch.softmax(x,dim=1); choices={'separated_softmax':Separated,'triton_fused_softmax':TritonFused,'persistent_separated_softmax':Persistent}; names=choices if a.backend=='all' else (a.backend,); results=[]
 for n in names:
  b=choices[n](); b.prepare(x); got=b.run()[0]; torch.cuda.synchronize(); ok=bool(torch.allclose(got,expected,atol=2e-5,rtol=2e-5)); assert ok,n; m=measure_backend(b,0,a.warmup,a.iterations); d=m.to_dict(); d.update(backend=n,correctness={'correct':ok},metadata=b.metadata()); results.append(d); print(f'{n:30s} median={m.median_us:.3f} us')
 path=write_results({'experiment':{'name':'softmax_fusion','shape':[4096,4096],'dtype':'float32'},'results':results},a.output or Path(f'results/experiment3/{datetime.now():%Y%m%d-%H%M%S}.json')); print(f'results: {path}')
if __name__=='__main__': main()
