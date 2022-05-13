cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# https://developer.nvidia.com/blog/building-cuda-applications-cmake/

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# find_package(CUDA QUIET REQUIRED)
# enable_language(CUDA)
project(trt_engine 
  VERSION 0.1
  DESCRIPTION "TensorRT server"
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA QUIET)
if(${CUDA_FOUND})
  enable_language(CUDA)
  include(cuda)
endif()

# include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_INCLUDE_DIR: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
add_library(trt_engine trt_engine.cc)

# target_link_libraries(trt_engine nvinfer)
# add_executable(sample main.cu square_sum.cu matmul.cu kernel.cu)
# set_target_properties(trt_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
# set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

enable_testing()
add_subdirectory(test)