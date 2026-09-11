cmake_minimum_required(VERSION 3.24)
project(iso_tests LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_subdirectory(01_launch_overhead)
add_subdirectory(02_independent_tasks)
