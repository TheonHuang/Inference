// inferop : SiLU-and-mul (SwiGLU activation, gate_up fused layout).
//   gate_up: [tokens, 2*d] ; out: [tokens, d]
//   out[t, i] = silu(gate_up[t, i]) * gate_up[t, d + i]
#include <cuda_runtime.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {

constexpr int kBlockThreads = 256;

template <typename T>
__global__ void silu_and_mul_kernel(T* __restrict__ out, const T* __restrict__ gate_up,
                                    int64_t n, int64_t d) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t t = i / d;
  int64_t col = i % d;
  const T* row = gate_up + t * 2 * d;
  float gate = to_f32(row[col]);
  float up = to_f32(row[d + col]);
  float silu = gate / (1.0f + __expf(-gate));
  out[i] = from_f32<T>(silu * up);
}

}  // namespace

void silu_and_mul_impl(void* out, const void* gate_up, dtype dt, int64_t tokens, int64_t d,
                       void* stream) {
  if (tokens <= 0 || d <= 0) return;
  int64_t n = tokens * d;
  int64_t blocks = (n + kBlockThreads - 1) / kBlockThreads;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  // Grid-dim limit is 2^31-1; typical shapes stay far below. Chunk if needed.
  if (blocks > int64_t(2147483647)) blocks = int64_t(2147483647);
  dim3 grid(unsigned(blocks));
  switch (dt) {
    case dtype::fp32:
      silu_and_mul_kernel<float><<<grid, kBlockThreads, 0, s>>>(
          static_cast<float*>(out), static_cast<const float*>(gate_up), n, d);
      break;
    case dtype::fp16:
      silu_and_mul_kernel<__half><<<grid, kBlockThreads, 0, s>>>(
          static_cast<__half*>(out), static_cast<const __half*>(gate_up), n, d);
      break;
    case dtype::bf16:
      silu_and_mul_kernel<__nv_bfloat16><<<grid, kBlockThreads, 0, s>>>(
          static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(gate_up), n, d);
      break;
    default:
      break;
  }
}

}  // namespace kernels
}  // namespace inferop
