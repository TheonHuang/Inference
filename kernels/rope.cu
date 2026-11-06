// inferop : RoPE (GPT-NeoX style, half-dim rotation, in-place).
// Thread handles one (token, head, i) pair with i < rotary_dim/2:
//   x_i' = x_i * cos(theta) - x_{i+rd/2} * sin(theta)
//   x_{i+rd/2}' = x_{i+rd/2} * cos(theta) + x_i * sin(theta)
//   theta = pos * base^(-2i/rd) * yarn_factor
// Roadmap: vectorized half2 pairs, batched qkv fused layout, YaRN NTK-by-parts.
#include <cuda_runtime.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {
using namespace inferop::gpu;

constexpr int kBlockThreads = 256;

template <typename T>
__global__ void rope_neox_kernel(T* __restrict__ qkv, const int64_t* __restrict__ positions,
                                 int64_t total_pairs, int heads, int dim, int rotary_dim,
                                 float base, float yarn) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_pairs) return;
  int half_rd = rotary_dim >> 1;
  int i = int(idx % half_rd);
  int64_t rest = idx / half_rd;
  int h = int(rest % heads);
  int64_t t = rest / heads;

  float freq = powf(base, -2.0f * float(i) / float(rotary_dim)) * yarn;
  float theta = float(positions[t]) * freq;
  float sin_t, cos_t;
  sincosf(theta, &sin_t, &cos_t);

  T* p = qkv + (t * heads + h) * dim;
  float x0 = to_f32(p[i]);
  float x1 = to_f32(p[i + half_rd]);
  p[i] = from_f32<T>(x0 * cos_t - x1 * sin_t);
  p[i + half_rd] = from_f32<T>(x1 * cos_t + x0 * sin_t);
}

}  // namespace

void rope_neox_impl(void* qkv, const int64_t* positions, dtype dt, int64_t tokens, int heads,
                    int head_dim, int rotary_dim, float base, float yarn_factor, void* stream) {
  if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) != 0) return;
  int64_t pairs = int64_t(tokens) * heads * (rotary_dim / 2);
  int64_t blocks = (pairs + kBlockThreads - 1) / kBlockThreads;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  dim3 grid{unsigned(blocks)};
  switch (dt) {
    case dtype::fp32:
      rope_neox_kernel<float><<<grid, kBlockThreads, 0, s>>>(
          static_cast<float*>(qkv), positions, pairs, heads, head_dim, rotary_dim, base,
          yarn_factor);
      break;
    case dtype::fp16:
      rope_neox_kernel<__half><<<grid, kBlockThreads, 0, s>>>(
          static_cast<__half*>(qkv), positions, pairs, heads, head_dim, rotary_dim, base,
          yarn_factor);
      break;
    case dtype::bf16:
      rope_neox_kernel<__nv_bfloat16><<<grid, kBlockThreads, 0, s>>>(
          static_cast<__nv_bfloat16*>(qkv), positions, pairs, heads, head_dim, rotary_dim, base,
          yarn_factor);
      break;
    default:
      break;
  }
}

}  // namespace kernels
}  // namespace inferop
