// inferop : CPU golden reference implementations (no CUDA dependency).
// Every op has an fp32 reference used by the standalone tests to validate the
// GPU kernels, plus fp8 helpers shared bit-exactly with the kernels.
#pragma once

#include <cstdint>

#include "inferop/common.h"
#include "inferop/fp8.h"

namespace inferop {
namespace ref {

void rmsnorm(float* out, const float* in, const float* weight, int64_t rows, int64_t cols,
             float eps);

void fused_add_rmsnorm(float* residual, float* out, const float* x, const float* weight,
                       int64_t rows, int64_t cols, float eps);

// qkv [tokens, heads, dim] modified in place.
void rope_neox(float* qkv, const int64_t* positions, int64_t tokens, int heads, int head_dim,
               int rotary_dim, float base, float yarn_factor);

void silu_and_mul(float* out, const float* gate_up, int64_t tokens, int64_t d);

void quant_fp8_per_token_group(uint8_t* q, float* scales, const float* x, int64_t rows,
                               int64_t cols, int group_size);

void dequant_fp8_per_token_group(float* x, const uint8_t* q, const float* scales, int64_t rows,
                                 int64_t cols, int group_size);

// sb is [ceil(n / block_n), k / group_size] following the kernel contract.
void w8a8_block_fp8_gemm(float* c, const uint8_t* a_q, const float* a_scales, const uint8_t* b_q,
                         const float* b_scales, int64_t m, int64_t n, int64_t k, int group_size,
                         int block_n);

void paged_attn_decode(float* out, const float* q, const float* k_cache, const float* v_cache,
                       const int32_t* block_tables, const int32_t* seq_lens, int batch,
                       int num_heads, int num_kv_heads, int head_dim, int block_size,
                       int max_blocks_per_seq, float scale);

void topk_softmax(int32_t* topk_ids, float* topk_weights, const float* logits, int64_t tokens,
                  int num_experts, int topk, bool renormalize);

}  // namespace ref
}  // namespace inferop
