// inferop : W8A8 blockwise-FP8 GEMM (DeepSeek V3/R1 style weight-only-serving
// recipe): activations quantized per 128-token-group, weights per 128x128
// block, fp32 accumulate on dequantized values.
//
//   C[m, n] = sum_k Aq[m,k] * As[m, k/128] * Bq[n,k] * Bs[n/128, k/128]
//
// v0.1 = correctness-first CUDA-core kernel, compile-time tile variants picked
// at runtime by SM version:
//   sm_80/86/89 : 64x128x128 tile (16KB smem)
//   sm_90+      : 128x128... (128x64 tile, 24KB smem)
// Roadmap (docs/ROADMAP): sm_90 wgmma + TMA path, sm_100 tcgen05 + NVFP4.
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {
using namespace inferop::gpu;

constexpr int kBlockThreads = 256;  // 16 (N) x 16 (M) thread tile
constexpr int kBK = 128;            // == group size: one k-group per iteration
constexpr int kPad = 4;             // smem row padding (bytes)

template <int BM, int BN, typename OUT_T>
__global__ void w8a8_block_gemm_kernel(OUT_T* __restrict__ C, const uint8_t* __restrict__ Aq,
                                       const float* __restrict__ As,
                                       const uint8_t* __restrict__ Bq,
                                       const float* __restrict__ Bs, int M, int N, int K) {
  static_assert(BM % 16 == 0 && BN % 16 == 0, "thread grid is 16x16");
  constexpr int TM = BM / 16;
  constexpr int TN = BN / 16;
  constexpr int ROW_U32 = (kBK + kPad) / 4;  // 33 u32 per smem row (incl. pad)

  int tile_n = blockIdx.x;
  int tile_m = blockIdx.y;
  int tn = threadIdx.x % 16;
  int tm = threadIdx.x / 16;
  int m0 = tile_m * BM + tm * TM;
  int n0 = tile_n * BN + tn * TN;

  __shared__ __align__(16) uint8_t sA[BM][kBK + kPad];
  __shared__ __align__(16) uint8_t sB[BN][kBK + kPad];

  float acc[TM][TN];
#pragma unroll
  for (int r = 0; r < TM; ++r)
#pragma unroll
    for (int c = 0; c < TN; ++c) acc[r][c] = 0.0f;

  const int k_groups = K / kBK;

  for (int kt = 0; kt < k_groups; ++kt) {
    // ---- stage A tile [BM, BK] (u32 view, pad lanes get 0) ----
    for (int u = threadIdx.x; u < BM * ROW_U32; u += kBlockThreads) {
      int r = u / ROW_U32;
      int c4 = u % ROW_U32;
      uint32_t w = 0;
      if (c4 < kBK / 4) {
        int64_t gr = int64_t(tile_m) * BM + r;
        if (gr < M) {
          const uint32_t* src = reinterpret_cast<const uint32_t*>(Aq + gr * K + int64_t(kt) * kBK);
          w = src[c4];
        }
      }
      *reinterpret_cast<uint32_t*>(&sA[r][0] + c4 * 4) = w;
    }
    // ---- stage B tile [BN, BK] ----
    for (int u = threadIdx.x; u < BN * ROW_U32; u += kBlockThreads) {
      int r = u / ROW_U32;
      int c4 = u % ROW_U32;
      uint32_t w = 0;
      if (c4 < kBK / 4) {
        int64_t gr = int64_t(tile_n) * BN + r;
        if (gr < N) {
          const uint32_t* src = reinterpret_cast<const uint32_t*>(Bq + gr * K + int64_t(kt) * kBK);
          w = src[c4];
        }
      }
      *reinterpret_cast<uint32_t*>(&sB[r][0] + c4 * 4) = w;
    }
    __syncthreads();

    // ---- hoist per-k-group scales (group == k tile == 128) ----
    float sa[TM], sb[TN];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
      int64_t gm = int64_t(m0) + r;
      sa[r] = (gm < M) ? As[gm * k_groups + kt] : 0.0f;
    }
#pragma unroll
    for (int c = 0; c < TN; ++c) {
      int64_t gn = int64_t(n0) + c;
      sb[c] = (gn < N) ? Bs[(gn / 128) * k_groups + kt] : 0.0f;
    }

    // ---- compute on dequantized values ----
    for (int kk = 0; kk < kBK; ++kk) {
      float av[TM], bv[TN];
#pragma unroll
      for (int r = 0; r < TM; ++r) {
        av[r] = fp8::e4m3_to_float(sA[tm * TM + r][kk]) * sa[r];
      }
#pragma unroll
      for (int c = 0; c < TN; ++c) {
        bv[c] = fp8::e4m3_to_float(sB[tn * TN + c][kk]) * sb[c];
      }
#pragma unroll
      for (int r = 0; r < TM; ++r)
#pragma unroll
        for (int c = 0; c < TN; ++c) acc[r][c] += av[r] * bv[c];
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < TM; ++r)
#pragma unroll
    for (int c = 0; c < TN; ++c) {
      int64_t gm = int64_t(m0) + r;
      int64_t gn = int64_t(n0) + c;
      if (gm < M && gn < N) C[gm * N + gn] = from_f32<OUT_T>(acc[r][c]);
    }
}

int current_sm() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 80;
  static int cached_sm = -1;
  static int cached_dev = -1;
  if (cached_sm < 0 || cached_dev != dev) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 80;
    cached_sm = prop.major * 10 + prop.minor;
    cached_dev = dev;
  }
  return cached_sm;
}

template <typename OUT_T>
void launch_gemm(void* c, const void* a_q, const void* a_scales, const void* b_q,
                 const void* b_scales, int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  int sm = current_sm();
  if (sm >= 90 && m >= 128) {
    dim3 grid{unsigned((n + 63) / 64), unsigned((m + 127) / 128)};
    w8a8_block_gemm_kernel<128, 64, OUT_T><<<grid, kBlockThreads, 0, stream>>>(
        static_cast<OUT_T*>(c), static_cast<const uint8_t*>(a_q),
        static_cast<const float*>(a_scales), static_cast<const uint8_t*>(b_q),
        static_cast<const float*>(b_scales), int(m), int(n), int(k));
  } else {
    dim3 grid{unsigned((n + 63) / 64), unsigned((m + 63) / 64)};
    w8a8_block_gemm_kernel<64, 64, OUT_T><<<grid, kBlockThreads, 0, stream>>>(
        static_cast<OUT_T*>(c), static_cast<const uint8_t*>(a_q),
        static_cast<const float*>(a_scales), static_cast<const uint8_t*>(b_q),
        static_cast<const float*>(b_scales), int(m), int(n), int(k));
  }
}

}  // namespace

void w8a8_block_fp8_gemm_impl(void* c, const void* a_q, const void* a_scales, const void* b_q,
                              const void* b_scales, int64_t m, int64_t n, int64_t k,
                              int group_size, int block_n, dtype out_dt, void* stream) {
  // v0.1 contract: group_size == block_n == BK == 128, K % 128 == 0.
  if (group_size != 128 || block_n != 128 || k % 128 != 0 || m <= 0 || n <= 0) {
    fprintf(stderr,
            "[inferop] w8a8_block_fp8_gemm: unsupported shape (m=%lld n=%lld k=%lld gs=%d "
            "bn=%d); v0.1 requires gs=bn=128, k%%128==0\n",
            (long long)m, (long long)n, (long long)k, group_size, block_n);
    return;
  }
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (out_dt) {
    case dtype::fp32:
      launch_gemm<float>(c, a_q, a_scales, b_q, b_scales, m, n, k, s);
      break;
    case dtype::fp16:
      launch_gemm<__half>(c, a_q, a_scales, b_q, b_scales, m, n, k, s);
      break;
    case dtype::bf16:
      launch_gemm<__nv_bfloat16>(c, a_q, a_scales, b_q, b_scales, m, n, k, s);
      break;
    default:
      break;
  }
}

}  // namespace kernels
}  // namespace inferop
