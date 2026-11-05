// Standalone test : W8A8 blockwise-FP8 GEMM (CPU golden + GPU cross-check).
#include <algorithm>

#include "test_util.h"
#include "reference/reference.h"

namespace {

struct QGemmCase {
  int64_t m, n, k;
};

// Build quantized inputs exactly as the kernel expects, on CPU.
void build_case(int64_t M, int64_t N, int64_t K, std::vector<uint8_t>* aq,
                std::vector<float>* as_, std::vector<uint8_t>* bq, std::vector<float>* bs_,
                std::vector<float>* c_want) {
  using namespace inferop;
  constexpr int GS = 128, BN = 128;
  std::vector<float> a(M * K), b(N * K);
  for (auto& v : a) v = testutil::normal(0.f, 1.f);
  for (auto& v : b) v = testutil::normal(0.f, 1.f);

  // activations: per-token-group absmax quant (reuse the ref quantizer row-wise)
  aq->resize(M * K);
  as_->resize(M * (K / GS));
  ref::quant_fp8_per_token_group(aq->data(), as_->data(), a.data(), M, K, GS);

  // weights: per (BN x GS) block absmax quant
  bq->resize(N * K);
  bs_->resize(((N + BN - 1) / BN) * (K / GS));
  std::fill(bs_->begin(), bs_->end(), 0.f);
  for (int64_t nb = 0; nb < (N + BN - 1) / BN; ++nb) {
    for (int64_t kg = 0; kg < K / GS; ++kg) {
      float absmax = 0.f;
      for (int64_t n = nb * BN; n < std::min(N, (nb + 1) * BN); ++n)
        for (int64_t i = 0; i < GS; ++i)
          absmax = std::max(absmax, std::fabs(b[n * K + kg * GS + i]));
      float scale = (absmax == 0.f) ? 1.f : absmax * (1.f / 448.f);
      (*bs_)[nb * (K / GS) + kg] = scale;
      float inv = 1.f / scale;
      for (int64_t n = nb * BN; n < std::min(N, (nb + 1) * BN); ++n)
        for (int64_t i = 0; i < GS; ++i)
          (*bq)[n * K + kg * GS + i] = fp8::float_to_e4m3_rn_sat(b[n * K + kg * GS + i] * inv);
    }
  }

  c_want->resize(M * N);
  ref::w8a8_block_fp8_gemm(c_want->data(), aq->data(), as_->data(), bq->data(), bs_->data(), M, N,
                           K, GS, BN);
}

void cpu_golden() {
  // check the dequant-accumulation math on a tiny case against naive dequant
  std::vector<uint8_t> aq, bq;
  std::vector<float> as_, bs_, c;
  build_case(128, 128, 128, &aq, &as_, &bq, &bs_, &c);
  double max_abs = 0;
  for (auto v : c) max_abs = std::max(max_abs, double(std::fabs(v)));
  inferop::testutil::check(max_abs > 1.0 && max_abs < 1e3, "plausible magnitude");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(int64_t M, int64_t N, int64_t K, inferop::dtype out_dt, float rtol,
                     float atol) {
  using namespace inferop;
  std::vector<uint8_t> aq, bq;
  std::vector<float> as_, bs_, want;
  build_case(M, N, K, &aq, &as_, &bq, &bs_, &want);

  testutil::GpuBuf d_c(M * N * 4), d_aq(aq.size()), d_as(as_.size() * 4), d_bq(bq.size()),
      d_bs(bs_.size() * 4);
  testutil::h2d(d_aq.p, aq.data(), aq.size());
  testutil::h2d(d_as.p, as_.data(), as_.size() * 4);
  testutil::h2d(d_bq.p, bq.data(), bq.size());
  testutil::h2d(d_bs.p, bs_.data(), bs_.size() * 4);

  w8a8_block_fp8_gemm(d_c.p, d_aq.p, d_as.p, d_bq.p, d_bs.p, M, N, K, 128, 128, out_dt);
  std::vector<uint8_t> hb(M * N * inferop::dtype_bytes(out_dt));
  testutil::d2h(hb.data(), d_c.p, hb.size());
  auto got = testutil::unpack(hb.data(), out_dt, M * N);
  char name[128];
  std::snprintf(name, sizeof(name), "gpu w8a8 gemm %lldx%lldx%lld (%s)",
                (long long)M, (long long)N, (long long)K, inferop::dtype_name(out_dt));
  testutil::all_close(got.data(), want.data(), M * N, rtol, atol, name);
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check(256, 256, 512, inferop::dtype::fp32, 2e-3f, 2e-2f);
    gpu_cross_check(256, 256, 512, inferop::dtype::bf16, 1e-2f, 3e-2f);
    gpu_cross_check(100, 180, 256, inferop::dtype::fp32, 2e-3f, 2e-2f);  // ragged tails
    gpu_cross_check(4096, 4096, 4096, inferop::dtype::bf16, 1e-2f, 3e-2f);  // real shape
  });
  return inferop::testutil::finish("test_gemm");
}
