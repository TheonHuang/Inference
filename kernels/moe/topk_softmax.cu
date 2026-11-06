// inferop : MoE routing - softmax over expert logits + iterative top-k
// selection (vLLM fused_moe style). One block per token.
//   logits (fp32): [tokens, num_experts], num_experts <= 1024
//   out: weights fp32 [tokens, topk], ids int32 [tokens, topk]
#include <cuda_runtime.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {
using namespace inferop::gpu;

constexpr int kBlockThreads = 256;
constexpr int kMaxExperts = 1024;

__global__ void topk_softmax_kernel(int32_t* __restrict__ ids, float* __restrict__ weights,
                                    const float* __restrict__ logits, int num_experts, int topk,
                                    bool renormalize) {
  int t = blockIdx.x;
  __shared__ float probs[kMaxExperts];
  __shared__ uint32_t chosen[kMaxExperts / 32];
  __shared__ FV cand[kWarpSize];

  const float* row = logits + int64_t(t) * num_experts;
  for (int i = threadIdx.x; i < num_experts; i += kBlockThreads) probs[i] = row[i];
  for (int i = threadIdx.x; i < (num_experts + 31) / 32; i += kBlockThreads) chosen[i] = 0u;
  __syncthreads();

  // softmax numerics
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < num_experts; i += kBlockThreads) {
    mx = fmaxf(mx, probs[i]);
  }
  mx = block_reduce_max(mx);
  float lsum = 0.0f;
  for (int i = threadIdx.x; i < num_experts; i += kBlockThreads) {
    float e = __expf(probs[i] - mx);
    probs[i] = e;
    lsum += e;
  }
  lsum = block_reduce_sum(lsum);

  const int num_warps = kBlockThreads / kWarpSize;
  for (int k = 0; k < topk; ++k) {
    // per-thread best over unchosen experts (ascending scan -> lower idx wins ties)
    FV best;
    best.v = -INFINITY;
    best.i = 0;
    for (int i = threadIdx.x; i < num_experts; i += kBlockThreads) {
      if (((chosen[i >> 5] >> (i & 31)) & 1u) == 0u && probs[i] > best.v) {
        best.v = probs[i];
        best.i = i;
      }
    }
    best = warp_reduce_argmax(best);
    int warp = threadIdx.x / kWarpSize;
    int lane = threadIdx.x % kWarpSize;
    if (lane == 0) cand[warp] = best;
    __syncthreads();
    if (warp == 0) {
      FV b;
      b.v = (lane < num_warps) ? cand[lane].v : -INFINITY;
      b.i = (lane < num_warps) ? cand[lane].i : 0;
      b = warp_reduce_argmax(b);
      if (lane == 0) {
        ids[t * topk + k] = b.i;
        weights[t * topk + k] = b.v / lsum;
        atomicOr(&chosen[b.i >> 5], 1u << (b.i & 31));
      }
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    float s = 0.0f;
    for (int k = 0; k < topk; ++k) s += weights[t * topk + k];
    if (s > 0.0f) {
      for (int k = 0; k < topk; ++k) weights[t * topk + k] /= s;
    }
  }
}

}  // namespace

void topk_softmax_impl(int32_t* topk_ids, void* topk_weights, const void* logits, int tokens,
                       int num_experts, int topk, bool renormalize, void* stream) {
  if (tokens <= 0 || num_experts <= 0 || num_experts > kMaxExperts || topk <= 0 ||
      topk > num_experts) {
    return;
  }
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  topk_softmax_kernel<<<unsigned(tokens), kBlockThreads, 0, s>>>(
      topk_ids, static_cast<float*>(topk_weights), static_cast<const float*>(logits),
      num_experts, topk, renormalize);
}

}  // namespace kernels
}  // namespace inferop
