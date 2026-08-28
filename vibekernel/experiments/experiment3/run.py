import argparse
from datetime import datetime
from pathlib import Path
import torch
from vibekernel.common.benchmark import measure_backend
from vibekernel.common.results import write_results
from .separated_softmax import SeparatedSoftmaxBackend
from .triton_fused_softmax import TritonFusedSoftmaxBackend
from .persistent_separated_softmax import PersistentSeparatedSoftmaxBackend

def main():
 p=argparse.ArgumentParser(); p.add_argument('--backend',choices=('all','separated_softmax','triton_fused_softmax','persistent_separated_softmax'),default='all'); p.add_argument('--warmup',type=int,default=100); p.add_argument('--iterations',type=int,default=1000); p.add_argument('--repeats-per-sample',type=int,default=1); p.add_argument('--output',type=Path); a=p.parse_args()
 x=torch.randn((4096,4096),device='cuda',dtype=torch.float32); expected=torch.softmax(x,dim=1)
 choices={'separated_softmax':SeparatedSoftmaxBackend,'triton_fused_softmax':TritonFusedSoftmaxBackend,'persistent_separated_softmax':PersistentSeparatedSoftmaxBackend}; names=choices if a.backend=='all' else (a.backend,); results=[]
 for name in names:
  b=choices[name](); b.prepare(x); got=b.run()[0]; torch.cuda.synchronize(); ok=bool(torch.allclose(got,expected,atol=2e-5,rtol=2e-5)); assert ok,name
  m=measure_backend(b,0,a.warmup,a.iterations,a.repeats_per_sample)
  d=m.to_dict(); d.update(backend=name,correctness={'correct':ok},metadata=b.metadata()); results.append(d); print(f'{name:30s} median={m.median_us:.3f} us')
 path=write_results({'experiment':{'name':'softmax_fusion','shape':[4096,4096],'dtype':'float32'},'results':results},a.output or Path(f'results/experiment3/{datetime.now():%Y%m%d-%H%M%S}.json')); print(f'results: {path}')
if __name__=='__main__': main()
