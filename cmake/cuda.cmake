include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")

list(APPEND CUDA_NVCC_FLAGS -w -Wno-deprecated-gpu-targets) 
if (CUDA_VERSION VERSION_GREATER_EQUAL "11.4")
  set(DEFAULT_CUDA_NVCC_GENCODES
      "arch=compute_61,code=sm_61"
      "arch=compute_61,code=compute_61"
      "arch=compute_75,code=sm_75"
      "arch=compute_75,code=compute_75")
endif()