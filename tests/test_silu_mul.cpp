// Standalone test : silu_and_mul (CPU golden + GPU cross-check).
#include "test_util.h"
#include "reference/reference.h"

namespace {

constexpr int64_t kTokens = 129;
constexpr int64_t kD = 512;

void cpu_golden() {
  std::vector<float> gu(kTokens * 2 * kD), out(kTokens * kD), want(kTokens * kD);
  for (auto& v : gu) v = inferop::testutil::normal(0.f, 1.f);
  inferop::ref::silu_and_mul(want.data(), gu.data(), kTokens, kD);

  // independent spot check on a few elements
  double worst = 0;
  for (int64_t t = 0; t < kTokens; t += 17)
    for (int64_t i = 0; i < kD; i += 97) {
      float g = gu[t * 2 * kD + i], u = gu[t * 2 * kD + kD + i];
      float expect = g / (1.f + std::exp(-g)) * u;
      worst = std::max(worst, double(std::fabs(expect - want[t * kD + i])));
    }
  inferop::testutil::check(worst < 1e-5, "spot check vs formula");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(inferop::dtype dt, float rtol, float atol) {
  using namespace inferop;
  int64_t n = kTokens * kD;
  std::vector<float> gu(n * 2), want(n);
  for (auto& v : gu) v = testutil::normal(0.f, 1.f);
  ref::silu_and_mul(want.data(), gu.data(), kTokens, kD);
  auto gub = testutil::pack(gu, dt);
  testutil::GpuBuf d_out(n * 2), d_in(gub.size());
  testutil::h2d(d_in.p, gub.data(), gub.size());
  silu_and_mul(d_out.p, d_in.p, dt, kTokens, kD);
  std::vector<uint8_t> hb(n * 2);
  testutil::d2h(hb.data(), d_out.p, n * 2);
  auto got = testutil::unpack(hb.data(), dt, n);
  testutil::all_close(got.data(), want.data(), n, rtol, atol, "gpu silu_and_mul vs ref");
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check(inferop::dtype::fp32, 1e-4f, 1e-4f);
    gpu_cross_check(inferop::dtype::bf16, 2e-2f, 2e-2f);
  });
  return inferop::testutil::finish("test_silu_mul");
}
