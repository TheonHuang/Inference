// inferop : per-token-group FP8 (E4M3) quantization, DeepSeek style.
//   scale(group) = absmax(group) / 448 ;  q = e4m3_rn_sat(x / scale)
// One block per (row, group). Works on every SM (software fp8 conversion).
#include <cuda_runtime.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {

constexpr int kBlockThreads = 256;

template <typename T>
__global__ void quant_fp8_kernel(uint8_t* __restrict__ q, float* __restrict__ scales,
                                 const T* __restrict__ x, int cols, int groups_per_row,
                                 int group_size) {
  int64_t row = blockIdx.x;
  int group = blockIdx.y;
  int64_t base = row * cols + int64_t(group) * group_size;

  float local_max = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += kBlockThreads) {
    local_max = fmaxf(local_max, fabsf(to_f32(x[base + i])));
  }
  float absmax = block_reduce_max(local_max);
  float scale = (absmax == 0.0f) ? 1.0f : absmax * (1.0f / 448.0f);
  if (threadIdx.x == 0) scales[row * groups_per_row + group] = scale;
  float inv = 1.0f / scale;
  for (int i = threadIdx.x; i < group_size; i += kBlockThreads) {
    q[base + i] = fp8::float_to_e4m3_rn_sat(to_f32(x[base + i]) * inv);
  }
}

template <typename T>
__global__ void dequant_fp8_kernel(T* __restrict__ x, const uint8_t* __restrict__ q,
                                   const float* __restrict__ scales, int64_t total, int cols,
                                   int group_size) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int64_t row = i / cols;
  int64_t g = (i % cols) / group_size;
  x[i] = from_f32<T>(fp8::e4m3_to_float(q[i]) * scales[row * (cols / group_size) + g]);
}

}  // namespace

void quant_fp8_per_token_group_impl(void* q, void* scales, const void* x, dtype dt,
                                    int64_t rows, int64_t cols, int group_size, void* stream) {
  if (rows <= 0 || cols <= 0 || group_size <= 0 || cols % group_size != 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  // rows in gridDim.x (up to 2^31), groups in gridDim.y (<= 65535)
  dim3 grid(unsigned(rows), unsigned(cols / group_size));
  switch (dt) {
    case dtype::fp32:
      quant_fp8_kernel<float><<<grid, kBlockThreads, 0, s>>>(
          static_cast<uint8_t*>(q), static_cast<float*>(scales),
          static_cast<const float*>(x), int(cols), int(cols / group_size), group_size);
      break;
    case dtype::fp16:
      quant_fp8_kernel<__half><<<grid, kBlockThreads, 0, s>>>(
          static_cast<uint8_t*>(q), static_cast<float*>(scales),
          static_cast<const __half*>(x), int(cols), int(cols / group_size), group_size);
      break;
    case dtype::bf16:
      quant_fp8_kernel<__nv_bfloat16><<<grid, kBlockThreads, 0, s>>>(
          static_cast<uint8_t*>(q), static_cast<float*>(scales),
          static_cast<const __nv_bfloat16*>(x), int(cols), int(cols / group_size), group_size);
      break;
    default:
      break;
  }
}

void dequant_fp8_per_token_group_impl(void* x, dtype dt, const void* q, const void* scales,
                                      int64_t rows, int64_t cols, int group_size, void* stream) {
  if (rows <= 0 || cols <= 0 || group_size <= 0 || cols % group_size != 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  int64_t total = rows * cols;
  int64_t blocks = (total + kBlockThreads - 1) / kBlockThreads;
  dim3 grid(unsigned(blocks));
  switch (dt) {
    case dtype::fp32:
      dequant_fp8_kernel<float><<<grid, kBlockThreads, 0, s>>>(
          static_cast<float*>(x), static_cast<const uint8_t*>(q),
          static_cast<const float*>(scales), total, int(cols), group_size);
      break;
    case dtype::fp16:
      dequant_fp8_kernel<__half><<<grid, kBlockThreads, 0, s>>>(
          static_cast<__half*>(x), static_cast<const uint8_t*>(q),
          static_cast<const float*>(scales), total, int(cols), group_size);
      break;
    case dtype::bf16:
      dequant_fp8_kernel<__nv_bfloat16><<<grid, kBlockThreads, 0, s>>>(
          static_cast<__nv_bfloat16*>(x), static_cast<const uint8_t*>(q),
          static_cast<const float*>(scales), total, int(cols), group_size);
      break;
    default:
      break;
  }
}

}  // namespace kernels
}  // namespace inferop
