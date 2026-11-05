cmake_minimum_required(VERSION 3.24)
project(inferop VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(INFEROP_BUILD_CUDA "Build the CUDA kernel library" ON)
option(INFEROP_BUILD_TESTS "Build standalone tests" ON)
option(INFEROP_BUILD_BENCH "Build the GPU benchmark" ON)
set(INFEROP_CUDA_ARCHES "80;86;89;90;100;120" CACHE STRING
    "Target SM architectures (semicolon list, dropped automatically when the nvcc in use is too old)")

# ---------------------------------------------------------------------------
# CPU golden reference (always built, no CUDA dependency)
# ---------------------------------------------------------------------------
add_library(inferop_reference STATIC reference/reference.cpp)
target_include_directories(inferop_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(inferop_reference PRIVATE -Wall -Wextra -Wno-unused-parameter)

set(INFEROP_CUDA_ENABLED OFF)
if(INFEROP_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # CUDA::cuda_runtime imported target
    set(INFEROP_CUDA_ENABLED ON)

    # Drop architectures the detected nvcc does not know about.
    set(_supported "")
    foreach(a ${INFEROP_CUDA_ARCHES})
      set(_ok TRUE)
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8 AND a MATCHES "^(89|90)$")
        set(_ok FALSE)
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8 AND a MATCHES "^(100|101|103|110|120)$")
        set(_ok FALSE)
      endif()
      if(_ok)
        list(APPEND _supported ${a})
      else()
        message(STATUS "inferop: dropping sm_${a} (nvcc ${CMAKE_CUDA_COMPILER_VERSION} too old)")
      endif()
    endforeach()
    if(NOT _supported)
      message(FATAL_ERROR "inferop: no supported CUDA architectures left for nvcc ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_supported})
    message(STATUS "inferop: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    add_library(inferop_kernels STATIC
      kernels/api.cu
      kernels/rmsnorm.cu
      kernels/rope.cu
      kernels/silu_mul.cu
      kernels/quant.cu
      kernels/gemm/w8a8_block_fp8.cu
      kernels/attention/paged_attn_decode.cu
      kernels/moe/topk_softmax.cu)
    set_target_properties(inferop_kernels PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    target_include_directories(inferop_kernels PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
    target_compile_options(inferop_kernels PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  else()
    message(STATUS "inferop: no CUDA compiler found, building CPU-only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Standalone tests: CPU golden always; GPU cross-checks when kernels built.
# ---------------------------------------------------------------------------
if(INFEROP_BUILD_TESTS)
  enable_testing()
  set(_tests fp8 rmsnorm rope silu_mul quant gemm paged_attn topk)
  foreach(t IN LISTS _tests)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_${t} PRIVATE inferop_reference)
    if(INFEROP_CUDA_ENABLED)
      target_link_libraries(test_${t} PRIVATE inferop_kernels CUDA::cuda_runtime)
      target_compile_definitions(test_${t} PRIVATE INFEROP_HAVE_CUDA)
    endif()
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
endif()

if(INFEROP_BUILD_BENCH)
  add_executable(bench_kernels bench/bench_kernels.cpp)
  target_include_directories(bench_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  if(INFEROP_CUDA_ENABLED)
    target_link_libraries(bench_kernels PRIVATE inferop_reference inferop_kernels
                                                CUDA::cuda_runtime)
    target_compile_definitions(bench_kernels PRIVATE INFEROP_HAVE_CUDA)
  else()
    target_link_libraries(bench_kernels PRIVATE inferop_reference)
  endif()
endif()
