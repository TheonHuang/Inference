// inferop : paged attention, single-token decode with GQA (vLLM-style block
// tables). FMA CUDA-core path that runs on every arch; tensor-core variants
// land per-arch later (roadmap).
//
//   q       : [batch, num_heads, head_dim]
//   k/v     : [num_blocks, block_size, num_kv_heads, head_dim]
//   tables  : int32 [batch, max_blocks_per_seq]
//   out     : [batch, num_heads, head_dim]
//
// Kernel structure (one block per (head, seq)):
//   phase A - 4 lanes cooperate per key for the q*k dot (warp-shuffle reduce),
//             8 keys per warp, 32 keys per iteration for 128 threads.
//   phase B - block-uniform online-softmax update (m, l).
//   phase C - every thread owns `DPT` head-dim columns and accumulates
//             weighted V rows (coalesced 32-thread reads per key).
#include <cuda_runtime.h>

#include "cuda_math.cuh"
#include "internal.h"

namespace inferop {
namespace kernels {
namespace {
using namespace inferop::gpu;

template <typename T, int HEAD_DIM, int BLOCK_THREADS>
__global__ void paged_attn_decode_kernel(T* __restrict__ out, const T* __restrict__ q,
                                         const T* __restrict__ k_cache,
                                         const T* __restrict__ v_cache,
                                         const int32_t* __restrict__ block_tables,
                                         const int32_t* __restrict__ seq_lens, int num_heads,
                                         int num_kv_heads, int block_size,
                                         int max_blocks_per_seq, float scale) {
  static_assert(HEAD_DIM % 32 == 0, "head_dim must be a multiple of 32");
  constexpr int DPT = HEAD_DIM / BLOCK_THREADS;  // head-dim elements per thread
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  constexpr int KEYS_PER_ITER = NUM_WARPS * 8;

  int head = blockIdx.x;
  int seq = blockIdx.y;
  int kvh = head / (num_heads / num_kv_heads);

  const T* q_row = q + (int64_t(seq) * num_heads + head) * HEAD_DIM;
  __shared__ float q_s[HEAD_DIM];
  __shared__ float qk_s[KEYS_PER_ITER];
  for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_THREADS) q_s[d] = to_f32(q_row[d]);
  for (int j = threadIdx.x; j < KEYS_PER_ITER; j += BLOCK_THREADS) qk_s[j] = -INFINITY;
  __syncthreads();

  float acc[DPT];
#pragma unroll
  for (int d = 0; d < DPT; ++d) acc[d] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  const int len = seq_lens[seq];
  const int32_t* bt = block_tables + int64_t(seq) * max_blocks_per_seq;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  for (int base = 0; base < len; base += KEYS_PER_ITER) {
    // ---- phase A: q*k dots for keys [base, base + KEYS_PER_ITER) ----
    int kloc = lane >> 2;  // 0..7 key within this warp
    int sub = lane & 3;    // which of 4 cooperating lanes
    int key = base + warp * 8 + kloc;
    bool valid = key < len;
    float part = 0.0f;
    if (valid) {
      int blk = bt[key / block_size];
      int off = key % block_size;
      const T* krow =
          k_cache + ((int64_t(blk) * block_size + off) * num_kv_heads + kvh) * HEAD_DIM;
      for (int d = sub; d < HEAD_DIM; d += 4) part += q_s[d] * to_f32(krow[d]);
    }
    part += __shfl_xor_sync(0xFFFFFFFFu, part, 1);
    part += __shfl_xor_sync(0xFFFFFFFFu, part, 2);
    if (sub == 0) qk_s[warp * 8 + kloc] = valid ? part * scale : -INFINITY;
    __syncthreads();

    // ---- phase B: online softmax update (block-uniform) ----
    float m_iter = -INFINITY;
    for (int j = 0; j < KEYS_PER_ITER; ++j) m_iter = fmaxf(m_iter, qk_s[j]);
    float m_new = fmaxf(m, m_iter);
    float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
    float p_sum = 0.0f;
    for (int j = 0; j < KEYS_PER_ITER; ++j) p_sum += __expf(qk_s[j] - m_new);
    l = l * alpha + p_sum;
#pragma unroll
    for (int d = 0; d < DPT; ++d) acc[d] *= alpha;

    // ---- phase C: V accumulation for the head-dim slice this thread owns ----
    for (int j = 0; j < KEYS_PER_ITER; ++j) {
      int vkey = base + j;
      if (vkey >= len) continue;  // its qk is -inf -> weight 0
      float p = __expf(qk_s[j] - m_new);
      int blk = bt[vkey / block_size];
      int off = vkey % block_size;
      const T* vrow =
          v_cache + ((int64_t(blk) * block_size + off) * num_kv_heads + kvh) * HEAD_DIM;
#pragma unroll
      for (int d = 0; d < DPT; ++d) {
        int gd = threadIdx.x + d * BLOCK_THREADS;
        acc[d] += p * to_f32(vrow[gd]);
      }
    }
    m = m_new;
    __syncthreads();  // qk_s reused next iteration
  }

  T* orow = out + (int64_t(seq) * num_heads + head) * HEAD_DIM;
  float inv_l = (l == 0.0f) ? 0.0f : 1.0f / l;
#pragma unroll
  for (int d = 0; d < DPT; ++d) {
    int gd = threadIdx.x + d * BLOCK_THREADS;
    orow[gd] = from_f32<T>(acc[d] * inv_l);
  }
}

template <typename T>
void launch_paged_attn(void* out, const void* q, const void* k_cache, const void* v_cache,
                       const int32_t* block_tables, const int32_t* seq_lens, int batch,
                       int num_heads, int num_kv_heads, int head_dim, int block_size,
                       int max_blocks_per_seq, float scale, cudaStream_t stream) {
  dim3 grid{unsigned(num_heads), unsigned(batch)};
  switch (head_dim) {
    case 64:
      paged_attn_decode_kernel<T, 64, 64><<<grid, 64, 0, stream>>>(
          static_cast<T*>(out), static_cast<const T*>(q), static_cast<const T*>(k_cache),
          static_cast<const T*>(v_cache), block_tables, seq_lens, num_heads, num_kv_heads,
          block_size, max_blocks_per_seq, scale);
      break;
    case 128:
      paged_attn_decode_kernel<T, 128, 128><<<grid, 128, 0, stream>>>(
          static_cast<T*>(out), static_cast<const T*>(q), static_cast<const T*>(k_cache),
          static_cast<const T*>(v_cache), block_tables, seq_lens, num_heads, num_kv_heads,
          block_size, max_blocks_per_seq, scale);
      break;
    case 256:
      paged_attn_decode_kernel<T, 256, 128><<<grid, 128, 0, stream>>>(
          static_cast<T*>(out), static_cast<const T*>(q), static_cast<const T*>(k_cache),
          static_cast<const T*>(v_cache), block_tables, seq_lens, num_heads, num_kv_heads,
          block_size, max_blocks_per_seq, scale);
      break;
    default:
      break;
  }
}

}  // namespace

void paged_attn_decode_impl(void* out, const void* q, const void* k_cache, const void* v_cache,
                            const int32_t* block_tables, const int32_t* seq_lens, dtype dt,
                            int batch, int num_heads, int num_kv_heads, int head_dim,
                            int block_size, int max_blocks_per_seq, float scale, void* stream) {
  if (batch <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
      num_heads % num_kv_heads != 0 || head_dim % 32 != 0) {
    return;
  }
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (dt) {
    case dtype::fp32:
      launch_paged_attn<float>(out, q, k_cache, v_cache, block_tables, seq_lens, batch,
                               num_heads, num_kv_heads, head_dim, block_size,
                               max_blocks_per_seq, scale, s);
      break;
    case dtype::fp16:
      launch_paged_attn<__half>(out, q, k_cache, v_cache, block_tables, seq_lens, batch,
                                num_heads, num_kv_heads, head_dim, block_size,
                                max_blocks_per_seq, scale, s);
      break;
    case dtype::bf16:
      launch_paged_attn<__nv_bfloat16>(out, q, k_cache, v_cache, block_tables, seq_lens, batch,
                                       num_heads, num_kv_heads, head_dim, block_size,
                                       max_blocks_per_seq, scale, s);
      break;
    default:
      break;
  }
}

}  // namespace kernels
}  // namespace inferop
