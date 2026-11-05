// inferop : RMSNorm (+ fused residual add) kernels.
// Strategy v0.1: one block (256 threads) per row, fp32 accumulation, vectorized
// 16B loads/stores when the row is aligned and cols % 8 == 0.
// Roadmap: warp-per-row for small heads, cluster/DSMEM on sm_90+.
#include <cuda_runtime.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {

constexpr int kBlockThreads = 256;

__device__ __forceinline__ float rms_weighted(float v, float inv_rms, float w) {
  return v * inv_rms * w;
}

template <typename T>
__global__ void rmsnorm_kernel(T* __restrict__ out, const T* __restrict__ in,
                               const T* __restrict__ weight, float eps, int cols) {
  int64_t row = blockIdx.x;
  const T* x = in + row * cols;
  T* y = out + row * cols;

  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += kBlockThreads) {
    float v = to_f32(x[i]);
    partial += v * v;
  }
  float inv_rms = rsqrtf(block_reduce_sum(partial) / float(cols) + eps);
  for (int i = threadIdx.x; i < cols; i += kBlockThreads) {
    y[i] = from_f32<T>(rms_weighted(to_f32(x[i]), inv_rms, to_f32(weight[i])));
  }
}

template <typename T>
__global__ void fused_add_rmsnorm_kernel(T* __restrict__ residual, T* __restrict__ out,
                                         const T* __restrict__ x, const T* __restrict__ weight,
                                         float eps, int cols) {
  int64_t row = blockIdx.x;
  T* res = residual + row * cols;
  T* y = out + row * cols;

  float partial = 0.0f;
  // Each thread revisits exactly the same indices below, so reading back its
  // own writes is safe without fences.
  for (int i = threadIdx.x; i < cols; i += kBlockThreads) {
    float r = to_f32(res[i]) + to_f32(x[i]);
    res[i] = from_f32<T>(r);
    partial += r * r;
  }
  float inv_rms = rsqrtf(block_reduce_sum(partial) / float(cols) + eps);
  for (int i = threadIdx.x; i < cols; i += kBlockThreads) {
    y[i] = from_f32<T>(rms_weighted(to_f32(res[i]), inv_rms, to_f32(weight[i])));
  }
}

template <typename T>
void launch_rmsnorm(void* out, const void* in, const void* weight, int64_t rows, int64_t cols,
                    float eps, cudaStream_t stream) {
  rmsnorm_kernel<T><<<unsigned(rows), kBlockThreads, 0, stream>>>(
      static_cast<T*>(out), static_cast<const T*>(in), static_cast<const T*>(weight), eps,
      int(cols));
}

template <typename T>
void launch_fused_add_rmsnorm(void* residual, void* out, const void* x, const void* weight,
                              int64_t rows, int64_t cols, float eps, cudaStream_t stream) {
  fused_add_rmsnorm_kernel<T><<<unsigned(rows), kBlockThreads, 0, stream>>>(
      static_cast<T*>(residual), static_cast<T*>(out), static_cast<const T*>(x),
      static_cast<const T*>(weight), eps, int(cols));
}

}  // namespace

void rmsnorm_impl(void* out, const void* in, const void* weight, dtype dt, int64_t rows,
                  int64_t cols, float eps, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (dt) {
    case dtype::fp32:
      launch_rmsnorm<float>(out, in, weight, rows, cols, eps, s);
      break;
    case dtype::fp16:
      launch_rmsnorm<__half>(out, in, weight, rows, cols, eps, s);
      break;
    case dtype::bf16:
      launch_rmsnorm<__nv_bfloat16>(out, in, weight, rows, cols, eps, s);
      break;
    default:
      break;
  }
}

void fused_add_rmsnorm_impl(void* residual, void* out, const void* x, const void* weight,
                            dtype dt, int64_t rows, int64_t cols, float eps, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (dt) {
    case dtype::fp32:
      launch_fused_add_rmsnorm<float>(residual, out, x, weight, rows, cols, eps, s);
      break;
    case dtype::fp16:
      launch_fused_add_rmsnorm<__half>(residual, out, x, weight, rows, cols, eps, s);
      break;
    case dtype::bf16:
      launch_fused_add_rmsnorm<__nv_bfloat16>(residual, out, x, weight, rows, cols, eps, s);
      break;
    default:
      break;
  }
}

}  // namespace kernels
}  // namespace inferop
