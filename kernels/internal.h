// inferop : internal dispatch entries implemented by each kernel translation
// unit (dtype switch happens here, per op).
#pragma once

#include "inferop/common.h"

namespace inferop {
namespace kernels {

void rmsnorm_impl(void* out, const void* in, const void* weight, dtype dt, int64_t rows,
                  int64_t cols, float eps, void* stream);

void fused_add_rmsnorm_impl(void* residual, void* out, const void* x, const void* weight,
                            dtype dt, int64_t rows, int64_t cols, float eps, void* stream);

void rope_neox_impl(void* qkv, const int64_t* positions, dtype dt, int64_t tokens, int heads,
                    int head_dim, int rotary_dim, float base, float yarn_factor, void* stream);

void silu_and_mul_impl(void* out, const void* gate_up, dtype dt, int64_t tokens, int64_t d,
                       void* stream);

void quant_fp8_per_token_group_impl(void* q, void* scales, const void* x, dtype dt,
                                    int64_t rows, int64_t cols, int group_size, void* stream);

void dequant_fp8_per_token_group_impl(void* x, dtype dt, const void* q, const void* scales,
                                      int64_t rows, int64_t cols, int group_size, void* stream);

void w8a8_block_fp8_gemm_impl(void* c, const void* a_q, const void* a_scales, const void* b_q,
                              const void* b_scales, int64_t m, int64_t n, int64_t k,
                              int group_size, int block_n, dtype out_dt, void* stream);

void paged_attn_decode_impl(void* out, const void* q, const void* k_cache, const void* v_cache,
                            const int32_t* block_tables, const int32_t* seq_lens, dtype dt,
                            int batch, int num_heads, int num_kv_heads, int head_dim,
                            int block_size, int max_blocks_per_seq, float scale, void* stream);

void topk_softmax_impl(int32_t* topk_ids, void* topk_weights, const void* logits, int tokens,
                       int num_experts, int topk, bool renormalize, void* stream);

}  // namespace kernels
}  // namespace inferop
