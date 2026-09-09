"""Real Qwen3 paged-attention MPK benchmark, initially one request."""
import argparse, time
from pathlib import Path
import torch
import mirage as mi
from mirage.mpk.mpk import MPK, MPKMetadata, MirageModelConfig

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--prompt",default="Explain paged attention.")
    p.add_argument("--prompt-len",type=int,default=512); p.add_argument("--decode-steps",type=int,default=32); p.add_argument("--page-size",type=int,default=64); p.add_argument("--output-dir",default="benchmark_backends/qwen3_paged_benchmark/results/mpk"); p.add_argument("--profiling",action="store_true")
    a=p.parse_args(); max_seq=a.prompt_len+a.decode_steps; pages=(max_seq+a.page_size-1)//a.page_size; d="cuda"
    tokens=torch.zeros((1,max_seq),device=d,dtype=torch.long); inp=torch.zeros((1,1),device=d,dtype=torch.long); out=torch.zeros_like(inp)
    step=torch.zeros(1,device=d,dtype=torch.int32); new=torch.full((1,),a.decode_steps,device=d,dtype=torch.int32); length=torch.zeros(1,device=d,dtype=torch.int32)
    z=lambda n:torch.zeros(n,device=d,dtype=torch.int32); prof=torch.zeros(3000*128,device=d,dtype=torch.uint64) if a.profiling else None
    cap = 8
    ring = dict(
        paged_kv_indices_snapshot=z(pages),
        pinned_req_ready=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_req_request_id=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_req_prompt_len=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_req_initial_step=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_comp_ready=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_comp_request_id=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_comp_buffer_row=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_comp_final_step=torch.zeros(cap, dtype=torch.int32).pin_memory(),
        pinned_shutdown=torch.zeros(1, dtype=torch.int32).pin_memory(),
        pinned_step=torch.zeros(1, dtype=torch.int32).pin_memory(),
        pinned_inbox_tokens=torch.zeros(cap, max_seq, dtype=torch.int64).pin_memory(),
        pinned_rid_at_row=torch.full((1,), -1, dtype=torch.int32).pin_memory(),
    )
    meta=MPKMetadata(mode="offline",total_num_requests=1,num_remote_schedulers=0,max_seq_length=max_seq,max_num_batched_requests=1,max_num_batched_tokens=1,max_num_pages=pages,page_size=a.page_size,weight_from_model=True,model_name=a.model,step=step,tokens=tokens,input_tokens=inp,output_tokens=out,num_new_tokens=new,prompt_lengths=length,qo_indptr_buffer=z(2),paged_kv_indptr_buffer=z(2),paged_kv_indices_buffer=z(pages),paged_kv_last_page_len_buffer=z(1),model_config=MirageModelConfig(with_lm_head=True),profiling=a.profiling,profiler_tensor=prof,trace_name="qwen3_paged_benchmark",spec_decode=None,spec_decode_config=mi.mpk.speculative.spec_decode_class(None),use_cutlass_kernel=True,**ring)
    mpk=MPK(meta); t=time.perf_counter(); mpk.build(); mpk.compile(output_dir=a.output_dir); compile_ms=(time.perf_counter()-t)*1e3
    mpk.load_new_request(a.prompt); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record(); mpk(); e.record(); e.synchronize(); run_ms=s.elapsed_time(e)
    result={"backend":"mirage-mpk-paged","compile_ms":compile_ms,"offline_generation_ms":run_ms,"requested_decode_tokens":a.decode_steps,"ms_per_requested_token":run_ms/a.decode_steps,"profiling":a.profiling}; print(result)
    path=Path(a.output_dir)/"result.json"; path.parent.mkdir(parents=True,exist_ok=True); import json; path.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()
