add_executable(independent_gemm_tasks main.cu)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(independent_gemm_tasks PRIVATE CUDA::cublas)
target_compile_options(independent_gemm_tasks PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
