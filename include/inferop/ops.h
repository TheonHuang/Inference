// inferop : public host API. All tensors are device pointers (except where
// noted), passed as void* + inferop::dtype. Streams are opaque (cudaStream_t
// under the hood, nullptr = default stream).
//
// The library targets multiple NVIDIA GPU architectures:
//   sm_80/sm_86 (A100/A10/30xx)  - Ampere baseline, mma-free fast paths
//   sm_89 (L4/4090)             - fp8 storage support
//   sm_90 (H100/H200)           - Hopper paths (wgmma/TMA in roadmap)
//   sm_100/sm_120 (B200/DGX Spark) - Blackwell (tcgen05/NVFP4 in roadmap)
// Kernels are compiled for all requested arches; tile-size selection happens
// at runtime based on the actual SM version.
#pragma once

#include "inferop/common.h"

namespace inferop {

// True if at least one CUDA device is visible and the kernels were built in.
bool has_cuda();
// Number of CUDA devices visible to this process (0 when built without CUDA).
int device_count();
// e.g. "NVIDIA GeForce RTX 4090", "-" when built without CUDA.
const char* device_name(int ordinal = 0);
// SM version of a device as major*10+minor (e.g. 89), -1 when no CUDA.
int device_sm_version(int ordinal = 0);

// ---------------------------------------------------------------------------
// rmsnorm:  out[r, c] = w[c] * in[r, c] / sqrt(mean(in[r,:]^2) + eps)
// ---------------------------------------------------------------------------
void rmsnorm(void* out, const void* in, const void* weight, dtype dt, int64_t rows,
             int64_t cols, float eps = 1e-6f, void* stream = nullptr);

// fused residual add + rmsnorm (in-place on residual):
//   residual[r, c] += x[r, c];  out[r, c] = w[c] * residual[r, c] / rms(...)
void fused_add_rmsnorm(void* residual, void* out, const void* x, const void* weight,
                       dtype dt, int64_t rows, int64_t cols, float eps = 1e-6f,
                       void* stream = nullptr);

// ---------------------------------------------------------------------------
// Rotary position embedding (GPT-NeoX style, half-dim rotation, in-place).
// qkv: [tokens, num_heads, head_dim] (only the first num_rope_heads? no:
// applies to all `heads` heads of this tensor).
// positions: int64[tokens]. rotary_dim <= head_dim, multiple of 2.
// yarn_factor: simple linear scaling (full YaRN/NTK-by-parts in roadmap).
// ---------------------------------------------------------------------------
void rope_neox(void* qkv, const int64_t* positions, dtype dt, int64_t tokens, int heads,
               int head_dim, int rotary_dim, float base = 10000.0f, float yarn_factor = 1.0f,
               void* stream = nullptr);

// ---------------------------------------------------------------------------
// silu_and_mul: out[t, d] = silu(gate_up[t, d]) * gate_up[t, d + d]
// gate_up: [tokens, 2*d]
// ---------------------------------------------------------------------------
void silu_and_mul(void* out, const void* gate_up, dtype dt, int64_t tokens, int64_t d,
                  void* stream = nullptr);

// ---------------------------------------------------------------------------
// Per-token-group FP8 (E4M3) quantization, DeepSeek style.
//   x: [rows, cols] in dt (fp16/bf16/fp32)
//   q: [rows, cols] uint8 (fp8_e4m3)
//   scales: [rows, cols/group_size] fp32
//   scale = absmax(group) / 448 ; q = round_e4m3_rn(x / scale)
// ---------------------------------------------------------------------------
void quant_fp8_per_token_group(void* q, void* scales, const void* x, dtype dt, int64_t rows,
                               int64_t cols, int group_size = 128, void* stream = nullptr);

// Dequantization counterpart (used by tests and MoE gather paths).
void dequant_fp8_per_token_group(void* x, dtype dt, const void* q, const void* scales,
                                 int64_t rows, int64_t cols, int group_size = 128,
                                 void* stream = nullptr);

// ---------------------------------------------------------------------------
// W8A8 blockwise-FP8 GEMM (DeepSeek V3/R1 style):
//   C[M,N] = sum_k A_q[m,k] * sa[m, k/GS] * B_q[n,k] * sb[n/BN, k/GS]
//   A_q: [M,K] fp8_e4m3, sa: [M, K/GS] fp32   (per-token-group  activations)
//   B_q: [N,K] fp8_e4m3, sb: [N/BN, K/GS] fp32 (per 128x128 weight block)
//   C: [M,N] in out_dt (fp32/fp16/bf16)
// Notes: K % GS == 0, N % BN == 0 required. v0.1 ships a correctness-first
// CUDA-core kernel (works on every arch); tensor-core paths land per-arch.
// ---------------------------------------------------------------------------
void w8a8_block_fp8_gemm(void* c, const void* a_q, const void* a_scales, const void* b_q,
                         const void* b_scales, int64_t m, int64_t n, int64_t k,
                         int group_size = 128, int block_n = 128, dtype out_dt = dtype::bf16,
                         void* stream = nullptr);

// ---------------------------------------------------------------------------
// Paged attention (single-token decode, GQA, no KV quant in v0.1).
//   q:        [batch, num_heads, head_dim]                       (dt)
//   k_cache:  [num_blocks, block_size, num_kv_heads, head_dim]   (dt)
//   v_cache:  [num_blocks, block_size, num_kv_heads, head_dim]   (dt)
//   block_tables: int32 [batch, max_blocks_per_seq]
//   seq_lens: int32 [batch]
//   out:      [batch, num_heads, head_dim]                       (dt)
// head_dim in {64, 128, 256}. num_heads % num_kv_heads == 0.
// ---------------------------------------------------------------------------
void paged_attn_decode(void* out, const void* q, const void* k_cache, const void* v_cache,
                       const int32_t* block_tables, const int32_t* seq_lens, dtype dt,
                       int batch, int num_heads, int num_kv_heads, int head_dim,
                       int block_size, int max_blocks_per_seq, float scale,
                       void* stream = nullptr);

// ---------------------------------------------------------------------------
// MoE routing: softmax over expert logits + top-k selection.
//   logits: [tokens, num_experts] (fp32)
//   topk_weights: fp32 [tokens, topk]  (softmax prob if !renormalize)
//   topk_ids:     int32 [tokens, topk]
// ---------------------------------------------------------------------------
void topk_softmax(int32_t* topk_ids, void* topk_weights, const void* logits, int tokens,
                  int num_experts, int topk, bool renormalize = true,
                  void* stream = nullptr);

}  // namespace inferop
