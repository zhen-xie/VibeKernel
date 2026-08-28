import torch
class CublasLtBackend:
 name='cublaslt_gemm'
 def prepare(self,problems):
  f=getattr(torch.backends.cuda,'preferred_blas_library',None)
  if f:f('cublaslt')
  self.problems=list(problems); self.outputs=[torch.empty((p.m,p.n),device=p.a.device,dtype=torch.bfloat16) for p in problems]; self.run(); torch.cuda.synchronize()
 def run(self):
  for p,o in zip(self.problems,self.outputs,strict=True):torch.mm(p.a,p.b,out=o)
  return self.outputs
 def metadata(self):return {'implementation':'cuBLASLt-backed PyTorch GEMM','graph_replay':False}
 def close(self):self.problems=[];self.outputs=[]
