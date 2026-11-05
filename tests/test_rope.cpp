// Standalone test : RoPE NeoX (CPU golden + GPU cross-check).
#include "test_util.h"
#include "reference/reference.h"

namespace {

constexpr int64_t kTokens = 64;
constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kRotary = 64;

void cpu_golden() {
  std::vector<float> q(kTokens * kHeads * kDim), orig;
  std::vector<int64_t> pos(kTokens);
  for (auto& v : q) v = inferop::testutil::normal(0.f, 1.f);
  for (int64_t t = 0; t < kTokens; ++t) pos[t] = t * 17 + (t % 3);
  orig = q;

  inferop::ref::rope_neox(q.data(), pos.data(), kTokens, kHeads, kDim, kRotary, 10000.f, 1.f);

  // property 1: pair norms preserved
  double max_delta = 0;
  int half = kRotary / 2;
  for (int64_t t = 0; t < kTokens; ++t)
    for (int h = 0; h < kHeads; ++h)
      for (int i = 0; i < half; ++i) {
        float n0 = orig[(t * kHeads + h) * kDim + i], n1 = orig[(t * kHeads + h) * kDim + i + half];
        float r0 = q[(t * kHeads + h) * kDim + i], r1 = q[(t * kHeads + h) * kDim + i + half];
        max_delta = std::max(max_delta,
                             std::fabs(std::sqrt(double(n0 * n0 + n1 * n1)) -
                                       std::sqrt(double(r0 * r0 + r1 * r1))));
      }
  inferop::testutil::check(max_delta < 1e-4, "pair norms preserved");

  // property 2: positions all zero -> identity
  std::vector<float> q2(orig);
  std::vector<int64_t> zeros(kTokens, 0);
  inferop::ref::rope_neox(q2.data(), zeros.data(), kTokens, kHeads, kDim, kRotary, 10000.f, 1.f);
  inferop::testutil::all_close(q2.data(), orig.data(), q2.size(), 1e-6f, 1e-6f,
                               "position 0 -> identity");

  // property 3: untouched tail (rotary_dim < head_dim)
  bool tail_ok = true;
  for (int64_t t = 0; t < kTokens && tail_ok; ++t)
    for (int h = 0; h < kHeads && tail_ok; ++h)
      for (int d = kRotary; d < kDim; ++d)
        tail_ok = q[(t * kHeads + h) * kDim + d] == orig[(t * kHeads + h) * kDim + d];
  inferop::testutil::check(tail_ok, "tail beyond rotary_dim untouched");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(inferop::dtype dt, float rtol, float atol) {
  using namespace inferop;
  int64_t n = kTokens * kHeads * kDim;
  std::vector<float> q(n), want(n);
  std::vector<int64_t> pos(kTokens);
  for (auto& v : q) v = testutil::normal(0.f, 1.f);
  for (int64_t t = 0; t < kTokens; ++t) pos[t] = t * 17 + (t % 3);
  want = q;
  ref::rope_neox(want.data(), pos.data(), kTokens, kHeads, kDim, kRotary, 10000.f, 1.f);

  auto qb = testutil::pack(q, dt);
  testutil::GpuBuf d_q(qb.size()), d_pos(pos.size() * 8);
  testutil::h2d(d_q.p, qb.data(), qb.size());
  testutil::h2d(d_pos.p, pos.data(), pos.size() * 8);
  rope_neox(d_q.p, static_cast<const int64_t*>(d_pos.p), dt, kTokens, kHeads, kDim, kRotary,
            10000.f, 1.f);
  std::vector<uint8_t> hb(qb.size());
  testutil::d2h(hb.data(), d_q.p, qb.size());
  auto got = testutil::unpack(hb.data(), dt, n);
  testutil::all_close(got.data(), want.data(), n, rtol, atol, "gpu rope vs ref");
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check(inferop::dtype::fp32, 1e-4f, 1e-4f);
    gpu_cross_check(inferop::dtype::bf16, 2e-2f, 2e-2f);
  });
  return inferop::testutil::finish("test_rope");
}
