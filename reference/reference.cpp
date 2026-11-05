#include "reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace inferop {
namespace ref {

void rmsnorm(float* out, const float* in, const float* weight, int64_t rows, int64_t cols,
             float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* x = in + r * cols;
    float* y = out + r * cols;
    float ss = 0.0f;
    for (int64_t c = 0; c < cols; ++c) ss += x[c] * x[c];
    float inv = 1.0f / std::sqrt(ss / float(cols) + eps);
    for (int64_t c = 0; c < cols; ++c) y[c] = x[c] * inv * weight[c];
  }
}

void fused_add_rmsnorm(float* residual, float* out, const float* x, const float* weight,
                       int64_t rows, int64_t cols, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    float* res = residual + r * cols;
    float* y = out + r * cols;
    const float* xr = x + r * cols;
    for (int64_t c = 0; c < cols; ++c) res[c] += xr[c];
    float ss = 0.0f;
    for (int64_t c = 0; c < cols; ++c) ss += res[c] * res[c];
    float inv = 1.0f / std::sqrt(ss / float(cols) + eps);
    for (int64_t c = 0; c < cols; ++c) y[c] = res[c] * inv * weight[c];
  }
}

void rope_neox(float* qkv, const int64_t* positions, int64_t tokens, int heads, int head_dim,
               int rotary_dim, float base, float yarn_factor) {
  const int half = rotary_dim / 2;
  for (int64_t t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      float* p = qkv + (t * heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        float freq = std::pow(base, -2.0f * float(i) / float(rotary_dim)) * yarn_factor;
        float theta = float(positions[t]) * freq;
        float c = std::cos(theta), s = std::sin(theta);
        float x0 = p[i], x1 = p[i + half];
        p[i] = x0 * c - x1 * s;
        p[i + half] = x1 * c + x0 * s;
      }
    }
  }
}

void silu_and_mul(float* out, const float* gate_up, int64_t tokens, int64_t d) {
  for (int64_t t = 0; t < tokens; ++t) {
    const float* row = gate_up + t * 2 * d;
    for (int64_t i = 0; i < d; ++i) {
      float g = row[i];
      out[t * d + i] = (g / (1.0f + std::exp(-g))) * row[d + i];
    }
  }
}

void quant_fp8_per_token_group(uint8_t* q, float* scales, const float* x, int64_t rows,
                               int64_t cols, int group_size) {
  int64_t groups_per_row = cols / group_size;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t g = 0; g < groups_per_row; ++g) {
      const float* gx = x + r * cols + g * group_size;
      float absmax = 0.0f;
      for (int i = 0; i < group_size; ++i) absmax = std::max(absmax, std::fabs(gx[i]));
      float scale = (absmax == 0.0f) ? 1.0f : absmax * (1.0f / 448.0f);
      scales[r * groups_per_row + g] = scale;
      for (int i = 0; i < group_size; ++i) {
        q[r * cols + g * group_size + i] = fp8::float_to_e4m3_rn_sat(gx[i] / scale);
      }
    }
  }
}

void dequant_fp8_per_token_group(float* x, const uint8_t* q, const float* scales, int64_t rows,
                                 int64_t cols, int group_size) {
  int64_t groups_per_row = cols / group_size;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      float s = scales[r * groups_per_row + c / group_size];
      x[r * cols + c] = fp8::e4m3_to_float(q[r * cols + c]) * s;
    }
  }
}

void w8a8_block_fp8_gemm(float* c, const uint8_t* a_q, const float* a_scales, const uint8_t* b_q,
                         const float* b_scales, int64_t m, int64_t n, int64_t k, int group_size,
                         int block_n) {
  int64_t k_groups = k / group_size;
  for (int64_t mi = 0; mi < m; ++mi) {
    for (int64_t ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (int64_t ki = 0; ki < k; ++ki) {
        float sa = a_scales[mi * k_groups + ki / group_size];
        float sb = b_scales[(ni / block_n) * k_groups + ki / group_size];
        float av = fp8::e4m3_to_float(a_q[mi * k + ki]) * sa;
        float bv = fp8::e4m3_to_float(b_q[ni * k + ki]) * sb;
        acc += double(av) * double(bv);
      }
      c[mi * n + ni] = float(acc);
    }
  }
}

void paged_attn_decode(float* out, const float* q, const float* k_cache, const float* v_cache,
                       const int32_t* block_tables, const int32_t* seq_lens, int batch,
                       int num_heads, int num_kv_heads, int head_dim, int block_size,
                       int max_blocks_per_seq, float scale) {
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      int kvh = h / (num_heads / num_kv_heads);
      const float* qv = q + (int64_t(b) * num_heads + h) * head_dim;
      int len = seq_lens[b];
      const int32_t* bt = block_tables + int64_t(b) * max_blocks_per_seq;
      std::vector<float> scores(len);
      float m = -std::numeric_limits<float>::infinity();
      for (int key = 0; key < len; ++key) {
        int blk = bt[key / block_size];
        int off = key % block_size;
        const float* krow =
            k_cache + ((int64_t(blk) * block_size + off) * num_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qv[d] * krow[d];
        scores[key] = dot * scale;
        m = std::max(m, scores[key]);
      }
      std::vector<float> acc(head_dim, 0.0f);
      float l = 0.0f;
      for (int key = 0; key < len; ++key) {
        float p = std::exp(scores[key] - m);
        l += p;
        int blk = bt[key / block_size];
        int off = key % block_size;
        const float* vrow =
            v_cache + ((int64_t(blk) * block_size + off) * num_kv_heads + kvh) * head_dim;
        for (int d = 0; d < head_dim; ++d) acc[d] += p * vrow[d];
      }
      float* o = out + (int64_t(b) * num_heads + h) * head_dim;
      float inv = (l == 0.0f) ? 0.0f : 1.0f / l;
      for (int d = 0; d < head_dim; ++d) o[d] = acc[d] * inv;
    }
  }
}

void topk_softmax(int32_t* topk_ids, float* topk_weights, const float* logits, int64_t tokens,
                  int num_experts, int topk, bool renormalize) {
  std::vector<float> probs(num_experts);
  std::vector<int> order(num_experts);
  for (int64_t t = 0; t < tokens; ++t) {
    const float* row = logits + t * num_experts;
    float mx = -std::numeric_limits<float>::infinity();
    for (int e = 0; e < num_experts; ++e) mx = std::max(mx, row[e]);
    float l = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
      probs[e] = std::exp(row[e] - mx);
      l += probs[e];
    }
    for (int e = 0; e < num_experts; ++e) order[e] = e;
    // sort by prob desc, index asc for deterministic tie-breaking (matches kernel)
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      if (probs[a] != probs[b]) return probs[a] > probs[b];
      return a < b;
    });
    float wsum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      topk_ids[t * topk + k] = order[k];
      topk_weights[t * topk + k] = probs[order[k]] / l;
      wsum += probs[order[k]] / l;
    }
    if (renormalize && wsum > 0.0f) {
      for (int k = 0; k < topk; ++k) topk_weights[t * topk + k] /= wsum;
    }
  }
}

}  // namespace ref
}  // namespace inferop
