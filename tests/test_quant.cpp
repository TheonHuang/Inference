// Standalone test : per-token-group FP8 quantization (CPU golden + GPU
// cross-check). The GPU kernel must produce byte-identical q + scales to the
// CPU reference (same deterministic algorithm).
#include "test_util.h"
#include "reference/reference.h"

namespace {

constexpr int64_t kRows = 64;
constexpr int64_t kCols = 768;  // 6 groups of 128
constexpr int kGs = 128;

void cpu_golden() {
  std::vector<float> x(kRows * kCols), dq(kRows * kCols);
  std::vector<uint8_t> q(kRows * kCols);
  std::vector<float> s(kRows * (kCols / kGs));
  for (auto& v : x) v = inferop::testutil::normal(0.f, 1.f);
  inferop::ref::quant_fp8_per_token_group(q.data(), s.data(), x.data(), kRows, kCols, kGs);
  inferop::ref::dequant_fp8_per_token_group(dq.data(), q.data(), s.data(), kRows, kCols, kGs);

  // E4M3 relative error bound: 2^-4 on every element
  int bad = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    float tol = std::fabs(x[i]) * (1.f / 16.f) + 1e-7f;
    if (std::fabs(dq[i] - x[i]) > tol) bad++;
  }
  inferop::testutil::check(bad == 0, "roundtrip error <= 2^-4");

  // zero row edge case: scale=1, all zeros
  std::vector<float> z(kCols, 0.f), zq(kCols, 0), zs(1, 0);
  inferop::ref::quant_fp8_per_token_group(zq.data(), zs.data(), z.data(), 1, kCols, kGs);
  inferop::testutil::check(zs[0] == 1.0f, "zero group -> scale 1");
  bool allz = true;
  for (int i = 0; i < kCols; ++i) allz = allz && zq[i] == 0;
  inferop::testutil::check(allz, "zero group -> all zero bytes");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(inferop::dtype dt) {
  using namespace inferop;
  int64_t n = kRows * kCols;
  std::vector<float> x(n);
  for (auto& v : x) v = testutil::normal(0.f, 1.f);
  std::vector<uint8_t> q_ref(n);
  std::vector<float> s_ref(kRows * (kCols / kGs));
  ref::quant_fp8_per_token_group(q_ref.data(), s_ref.data(), x.data(), kRows, kCols, kGs);

  auto xb = testutil::pack(x, dt);
  testutil::GpuBuf d_q(n), d_s(s_ref.size() * 4), d_x(xb.size());
  testutil::h2d(d_x.p, xb.data(), xb.size());
  quant_fp8_per_token_group(d_q.p, d_s.p, d_x.p, dt, kRows, kCols, kGs);
  std::vector<uint8_t> q_got(n);
  std::vector<float> s_got(s_ref.size());
  testutil::d2h(q_got.data(), d_q.p, n);
  testutil::d2h(s_got.data(), d_s.p, s_ref.size() * 4);
  testutil::bytes_equal(q_got.data(), q_ref.data(), n, "gpu quant bytes == ref");
  testutil::all_close(s_got.data(), s_ref.data(), s_ref.size(), 0, 0, "gpu scales == ref");

  // dequant round trip on GPU vs ref dequant
  std::vector<float> dq_ref(n);
  ref::dequant_fp8_per_token_group(dq_ref.data(), q_ref.data(), s_ref.data(), kRows, kCols, kGs);
  testutil::GpuBuf d_dq(n * 2);
  dequant_fp8_per_token_group(d_dq.p, dt, d_q.p, d_s.p, kRows, kCols, kGs);
  std::vector<uint8_t> hb(n * 2);
  testutil::d2h(hb.data(), d_dq.p, n * 2);
  auto dq_got = testutil::unpack(hb.data(), dt, n);
  // note: fp16 overflows at 65504; with unit-normal data + scale it is safe here
  testutil::all_close(dq_got.data(), dq_ref.data(), n, 2e-3f, 2e-3f, "gpu dequant vs ref");
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check(inferop::dtype::fp32);
    gpu_cross_check(inferop::dtype::bf16);
  });
  return inferop::testutil::finish("test_quant");
}
