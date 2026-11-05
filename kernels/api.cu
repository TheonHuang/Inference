// inferop : public API implementation - device introspection + forwarding to
// kernel translation units.
#include <cuda_runtime.h>
#include <stdio.h>

#include "inferop/ops.h"
#include "internal.h"

namespace inferop {
namespace {

struct DeviceInfo {
  int count = 0;
  bool init = false;
  int sm[16] = {};
  char name[16][128] = {};
};

const DeviceInfo& devices() {
  static DeviceInfo info;
  if (!info.init) {
    if (cudaGetDeviceCount(&info.count) != cudaSuccess || info.count < 0) info.count = 0;
    int n = info.count < 16 ? info.count : 16;
    for (int i = 0; i < n; ++i) {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
        info.sm[i] = prop.major * 10 + prop.minor;
        snprintf(info.name[i], sizeof(info.name[i]), "%s", prop.name);
      }
    }
    info.init = true;
  }
  return info;
}

}  // namespace

bool has_cuda() { return devices().count > 0; }

int device_count() { return devices().count; }

const char* device_name(int ordinal) {
  if (ordinal < 0 || ordinal >= devices().count || ordinal >= 16) return "-";
  return devices().name[ordinal];
}

int device_sm_version(int ordinal) {
  if (ordinal < 0 || ordinal >= devices().count || ordinal >= 16) return -1;
  return devices().sm[ordinal];
}

void rmsnorm(void* out, const void* in, const void* weight, dtype dt, int64_t rows, int64_t cols,
             float eps, void* stream) {
  kernels::rmsnorm_impl(out, in, weight, dt, rows, cols, eps, stream);
}

void fused_add_rmsnorm(void* residual, void* out, const void* x, const void* weight, dtype dt,
                       int64_t rows, int64_t cols, float eps, void* stream) {
  kernels::fused_add_rmsnorm_impl(residual, out, x, weight, dt, rows, cols, eps, stream);
}

void rope_neox(void* qkv, const int64_t* positions, dtype dt, int64_t tokens, int heads,
               int head_dim, int rotary_dim, float base, float yarn_factor, void* stream) {
  kernels::rope_neox_impl(qkv, positions, dt, tokens, heads, head_dim, rotary_dim, base,
                          yarn_factor, stream);
}

void silu_and_mul(void* out, const void* gate_up, dtype dt, int64_t tokens, int64_t d,
                  void* stream) {
  kernels::silu_and_mul_impl(out, gate_up, dt, tokens, d, stream);
}

void quant_fp8_per_token_group(void* q, void* scales, const void* x, dtype dt, int64_t rows,
                               int64_t cols, int group_size, void* stream) {
  kernels::quant_fp8_per_token_group_impl(q, scales, x, dt, rows, cols, group_size, stream);
}

void dequant_fp8_per_token_group(void* x, dtype dt, const void* q, const void* scales,
                                 int64_t rows, int64_t cols, int group_size, void* stream) {
  kernels::dequant_fp8_per_token_group_impl(x, dt, q, scales, rows, cols, group_size, stream);
}

void w8a8_block_fp8_gemm(void* c, const void* a_q, const void* a_scales, const void* b_q,
                         const void* b_scales, int64_t m, int64_t n, int64_t k, int group_size,
                         int block_n, dtype out_dt, void* stream) {
  kernels::w8a8_block_fp8_gemm_impl(c, a_q, a_scales, b_q, b_scales, m, n, k, group_size, block_n,
                                    out_dt, stream);
}

void paged_attn_decode(void* out, const void* q, const void* k_cache, const void* v_cache,
                       const int32_t* block_tables, const int32_t* seq_lens, dtype dt, int batch,
                       int num_heads, int num_kv_heads, int head_dim, int block_size,
                       int max_blocks_per_seq, float scale, void* stream) {
  kernels::paged_attn_decode_impl(out, q, k_cache, v_cache, block_tables, seq_lens, dt, batch,
                                  num_heads, num_kv_heads, head_dim, block_size,
                                  max_blocks_per_seq, scale, stream);
}

void topk_softmax(int32_t* topk_ids, void* topk_weights, const void* logits, int tokens,
                  int num_experts, int topk, bool renormalize, void* stream) {
  kernels::topk_softmax_impl(topk_ids, topk_weights, logits, tokens, num_experts, topk,
                             renormalize, stream);
}

}  // namespace inferop
