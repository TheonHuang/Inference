// inferop : small CUDA device-side helpers shared by all kernels.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "inferop/fp8.h"

namespace inferop {
namespace gpu {

constexpr int kWarpSize = 32;

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f32<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float from_f32<float>(float v) {
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

// Block-wide sum reduction. blockDim must be a multiple of 32.
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[kWarpSize];
  __shared__ float broadcast;
  int lane = threadIdx.x % kWarpSize;
  int warp = threadIdx.x / kWarpSize;
  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  int num_warps = blockDim.x / kWarpSize;
  if (warp == 0) {
    float w = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    w = warp_reduce_sum(w);
    if (lane == 0) broadcast = w;
  }
  __syncthreads();
  return broadcast;
}

// Block-wide max reduction. blockDim must be a multiple of 32.
__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float warp_max[kWarpSize];
  __shared__ float broadcast;
  int lane = threadIdx.x % kWarpSize;
  int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  if (lane == 0) warp_max[warp] = v;
  __syncthreads();
  int num_warps = blockDim.x / kWarpSize;
  if (warp == 0) {
    float w = (lane < num_warps) ? warp_max[lane] : -INFINITY;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      w = fmaxf(w, __shfl_xor_sync(0xFFFFFFFFu, w, offset));
    }
    if (lane == 0) broadcast = w;
  }
  __syncthreads();
  return broadcast;
}

struct FV {  // float + index, for argmax reductions (ties -> lower index wins)
  float v;
  int i;
};

__device__ __forceinline__ FV warp_reduce_argmax(FV a) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    FV b;
    b.v = __shfl_xor_sync(0xFFFFFFFFu, a.v, offset);
    b.i = __shfl_xor_sync(0xFFFFFFFFu, a.i, offset);
    if (b.v > a.v || (b.v == a.v && b.i < a.i)) a = b;
  }
  return a;
}

}  // namespace gpu
}  // namespace inferop
