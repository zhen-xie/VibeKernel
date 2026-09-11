add_executable(independent_tasks main.cu)
target_compile_options(independent_tasks PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
