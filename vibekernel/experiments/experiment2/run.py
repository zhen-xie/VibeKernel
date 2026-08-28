import argparse, json
from datetime import datetime
from pathlib import Path
import torch
from vibekernel.common.benchmark import measure_backend
from vibekernel.common.results import write_results
from vibekernel.common.validation import reference_outputs, validate_outputs
from .workload import SHAPES, create_workload, total_flops
from .cublaslt_gemm import CublasLtBackend
from .triton_large_gemm import TritonLargeBackend
from .persistent_gemm import PersistentLargeBackend

def main():
 p=argparse.ArgumentParser(); p.add_argument('--backend', choices=('all','cublaslt_gemm','triton_large_gemm','persistent_gemm'),default='all'); p.add_argument('--shape-index',type=int,default=1); p.add_argument('--warmup',type=int,default=100); p.add_argument('--iterations',type=int,default=1000); p.add_argument('--repeats-per-sample',type=int,default=1); p.add_argument('--cutlass-path'); p.add_argument('--output',type=Path); a=p.parse_args()
 problems=create_workload(SHAPES[a.shape_index]); expected=reference_outputs(problems); names=('cublaslt_gemm','triton_large_gemm','persistent_gemm') if a.backend=='all' else (a.backend,); out=[]
 for name in names:
  b={'cublaslt_gemm':CublasLtBackend,'triton_large_gemm':TritonLargeBackend,'persistent_gemm':lambda:PersistentLargeBackend(a.cutlass_path)}[name]()
  try:
   b.prepare(problems); correct=validate_outputs(b.run(),expected); assert correct['correct'], json.dumps(correct); x=measure_backend(b,total_flops(problems),a.warmup,a.iterations,a.repeats_per_sample); d=x.to_dict(); d.update(backend=name,correctness=correct,metadata=b.metadata()); out.append(d); print(f'{name:20s} median={x.median_us:.3f} us TFLOPS={x.tflops:.3f}')
  finally: b.close()
 path=write_results({'experiment':{'name':'large_bf16_gemm','shape_mnk':SHAPES[a.shape_index]},'results':out}, a.output or Path(f'results/experiment2/{datetime.now():%Y%m%d-%H%M%S}.json')); print(f'results: {path}')
if __name__=='__main__': main()
