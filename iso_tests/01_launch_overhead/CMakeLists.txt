add_executable(launch_overhead main.cu)
target_compile_options(launch_overhead PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
