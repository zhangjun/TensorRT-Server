cmake_minimum_required(VERSION 3.10)
project(tensorrt_cpp_api)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_PREFIX_PATH  "/zhangjun/code/libtorch/libtorch")
find_package(Torch REQUIRED)
# target_link_libraries(main "${TORCH_LIBRARIES}")
# set_property(TARGET main PROPERTY CXX_STANDARD 14)

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /zhangjun/software/TensorRT-8.6.0.12/)

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} "${TORCH_LIBRARIES}")

# add_executable(driver src/main.cpp)
# target_link_libraries(driver tensorrt_cpp_api)