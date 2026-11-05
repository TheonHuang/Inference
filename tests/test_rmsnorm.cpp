// Standalone test : rmsnorm / fused_add_rmsnorm (CPU golden + GPU cross-check).
#include "test_util.h"
#include "reference/reference.h"

namespace {

constexpr int64_t kRows = 33;
constexpr int64_t kCols = 1024;
constexpr float kEps = 1e-6f;

void cpu_golden() {
  std::vector<float> x(kRows * kCols), w(kCols), out(kRows * kCols), want(kRows * kCols);
  for (auto& v : x) v = inferop::testutil::normal(0.f, 1.f);
  for (auto& v : w) v = inferop::testutil::uniform(0.5f, 1.5f);
  inferop::ref::rmsnorm(want.data(), x.data(), w.data(), kRows, kCols, kEps);

  // independent property: without weight, output RMS == 1
  std::vector<float> ones(kCols, 1.f), o2(kRows * kCols);
  inferop::ref::rmsnorm(o2.data(), x.data(), ones.data(), kRows, kCols, kEps);
  double ss = 0;
  for (int64_t r = 0; r < kRows; ++r)
    for (int64_t c = 0; c < kCols; ++c) ss += double(o2[r * kCols + c]) * o2[r * kCols + c];
  double rms = std::sqrt(ss / double(kRows * kCols));
  inferop::testutil::check(std::fabs(rms - 1.0) < 1e-3, "unit-weight output RMS == 1");

  // wres carries the input residual and receives res+x in place
  std::vector<float> xr(kRows * kCols), wout(kRows * kCols), wres(kRows * kCols);
  for (auto& v : wres) v = inferop::testutil::normal(0.f, 1.f);
  for (auto& v : xr) v = inferop::testutil::normal(0.f, 1.f);
  std::vector<float> res_copy = wres;
  inferop::ref::fused_add_rmsnorm(wres.data(), wout.data(), xr.data(), w.data(), kRows, kCols,
                                  kEps);
  // residual must equal res + x
  std::vector<float> want_res(kRows * kCols);
  for (size_t i = 0; i < want_res.size(); ++i) want_res[i] = res_copy[i] + xr[i];
  inferop::testutil::all_close(wres.data(), want_res.data(), kRows * kCols, 1e-5f, 1e-5f,
                               "fused residual add");
  inferop::ref::rmsnorm(want.data(), wres.data(), w.data(), kRows, kCols, kEps);
  inferop::testutil::all_close(wout.data(), want.data(), kRows * kCols, 1e-4f, 1e-4f,
                               "fused normed output");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(inferop::dtype dt, float rtol, float atol) {
  using namespace inferop;
  int64_t n = kRows * kCols;
  std::vector<float> x(n), w(kCols), want(n);
  for (auto& v : x) v = testutil::normal(0.f, 1.f);
  for (auto& v : w) v = testutil::uniform(0.5f, 1.5f);
  ref::rmsnorm(want.data(), x.data(), w.data(), kRows, kCols, kEps);

  auto xb = testutil::pack(x, dt), wb = testutil::pack(w, dt);
  testutil::GpuBuf d_out(n * 2), d_in(xb.size()), d_w(wb.size());
  testutil::h2d(d_in.p, xb.data(), xb.size());
  testutil::h2d(d_w.p, wb.data(), wb.size());
  rmsnorm(d_out.p, d_in.p, d_w.p, dt, kRows, kCols, kEps);
  std::vector<uint8_t> hb(n * 2);
  testutil::d2h(hb.data(), d_out.p, n * 2);
  auto got = testutil::unpack(hb.data(), dt, n);
  testutil::all_close(got.data(), want.data(), n, rtol, atol, "gpu rmsnorm vs ref");
}

void gpu_cross_check_fused(inferop::dtype dt, float rtol, float atol) {
  using namespace inferop;
  int64_t n = kRows * kCols;
  std::vector<float> res(n), xr(n), w(kCols), want_res(n), want_out(n);
  for (auto& v : res) v = testutil::normal(0.f, 1.f);
  for (auto& v : xr) v = testutil::normal(0.f, 1.f);
  for (auto& v : w) v = testutil::uniform(0.5f, 1.5f);
  std::vector<float> res_in = res;
  ref::fused_add_rmsnorm(res_in.data(), want_out.data(), xr.data(), w.data(), kRows, kCols, kEps);

  auto resb = testutil::pack(res, dt), xrb = testutil::pack(xr, dt), wb = testutil::pack(w, dt);
  testutil::GpuBuf d_res(resb.size()), d_out(n * 2), d_x(xrb.size()), d_w(wb.size());
  testutil::h2d(d_res.p, resb.data(), resb.size());
  testutil::h2d(d_x.p, xrb.data(), xrb.size());
  testutil::h2d(d_w.p, wb.data(), wb.size());
  fused_add_rmsnorm(d_res.p, d_out.p, d_x.p, d_w.p, dt, kRows, kCols, kEps);
  std::vector<uint8_t> hb(n * 2);
  testutil::d2h(hb.data(), d_res.p, n * 2);
  auto got_res = testutil::unpack(hb.data(), dt, n);
  testutil::all_close(got_res.data(), res_in.data(), n, rtol, atol, "gpu fused residual");
  testutil::d2h(hb.data(), d_out.p, n * 2);
  auto got_out = testutil::unpack(hb.data(), dt, n);
  testutil::all_close(got_out.data(), want_out.data(), n, rtol, atol, "gpu fused normed out");
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check(inferop::dtype::fp32, 1e-5f, 1e-5f);
    gpu_cross_check(inferop::dtype::fp16, 3e-3f, 3e-3f);
    gpu_cross_check(inferop::dtype::bf16, 2e-2f, 2e-2f);
    gpu_cross_check_fused(inferop::dtype::bf16, 3e-2f, 3e-2f);
  });
  return inferop::testutil::finish("test_rmsnorm");
}
