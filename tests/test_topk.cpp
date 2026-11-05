// Standalone test : MoE topk_softmax routing (CPU golden + GPU cross-check).
#include "test_util.h"
#include "reference/reference.h"

namespace {

constexpr int64_t kTokens = 257;
constexpr int kExperts = 256;
constexpr int kTopk = 8;

void cpu_golden() {
  std::vector<float> logits(kTokens * kExperts);
  for (auto& v : logits) v = inferop::testutil::normal(0.f, 2.f);
  std::vector<int32_t> ids(kTokens * kTopk);
  std::vector<float> w(kTokens * kTopk);
  inferop::ref::topk_softmax(ids.data(), w.data(), logits.data(), kTokens, kExperts, kTopk, true);

  // independent properties
  bool weights_sum_one = true, ids_unique = true, weights_sorted = true;
  for (int64_t t = 0; t < kTokens; ++t) {
    float s = 0;
    for (int k = 0; k < kTopk; ++k) s += w[t * kTopk + k];
    weights_sum_one = weights_sum_one && std::fabs(s - 1.f) < 1e-4;
    for (int k = 0; k + 1 < kTopk; ++k)
      weights_sorted = weights_sorted && w[t * kTopk + k] >= w[t * kTopk + k + 1];
    for (int a = 0; a < kTopk; ++a)
      for (int b = a + 1; b < kTopk; ++b)
        ids_unique = ids_unique && ids[t * kTopk + a] != ids[t * kTopk + b];
  }
  inferop::testutil::check(weights_sum_one, "renormalized weights sum to 1");
  inferop::testutil::check(ids_unique, "expert ids unique");
  inferop::testutil::check(weights_sorted, "weights sorted desc");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(int64_t tokens, int experts, int topk) {
  using namespace inferop;
  std::vector<float> logits(tokens * experts);
  for (auto& v : logits) v = testutil::normal(0.f, 2.f);
  std::vector<int32_t> ids_ref(tokens * topk), ids_got(tokens * topk);
  std::vector<float> w_ref(tokens * topk), w_got(tokens * topk);
  ref::topk_softmax(ids_ref.data(), w_ref.data(), logits.data(), tokens, experts, topk, true);

  testutil::GpuBuf d_ids(ids_got.size() * 4), d_w(w_got.size() * 4),
      d_logits(logits.size() * 4);
  testutil::h2d(d_logits.p, logits.data(), logits.size() * 4);
  topk_softmax(static_cast<int32_t*>(d_ids.p), d_w.p, d_logits.p, int(tokens), experts, topk,
               true);
  testutil::d2h(ids_got.data(), d_ids.p, ids_got.size() * 4);
  testutil::d2h(w_got.data(), d_w.p, w_got.size() * 4);
  bool ids_match = true;
  for (size_t i = 0; i < ids_got.size(); ++i) ids_match = ids_match && ids_got[i] == ids_ref[i];
  inferop::testutil::check(ids_match, "gpu topk ids == ref");
  inferop::testutil::all_close(w_got.data(), w_ref.data(), w_got.size(), 1e-4f, 1e-5f,
                               "gpu topk weights vs ref");
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check(kTokens, kExperts, kTopk);
    gpu_cross_check(3, 8, 2);        // tiny
    gpu_cross_check(17, 1024, 16);   // max experts
  });
  return inferop::testutil::finish("test_topk");
}
