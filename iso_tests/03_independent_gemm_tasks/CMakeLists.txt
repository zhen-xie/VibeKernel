add_executable(independent_gemm_tasks main.cu)
target_compile_options(independent_gemm_tasks PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
