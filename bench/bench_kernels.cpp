// inferop : standalone GPU benchmark. Measures wall time (CUDA events),
// bandwidth and compute throughput for each op and compares against the
// roofline tables in docs/PERF.md. CPU-only builds print a notice.
#include <cstdio>
#include <vector>

#include "reference/reference.h"
#include "test_util.h"

#ifdef INFEROP_HAVE_CUDA
#include <cuda_runtime.h>

namespace {

struct Sample {
  const char* name;
  float ms;
};

std::vector<Sample> g_samples;

void report(const char* name, float ms, double bytes = -1, double flops = -1) {
  printf("%-42s %10.4f ms", name, ms);
  if (bytes > 0) printf("  %9.1f GB/s", bytes / double(ms) / 1e6);
  if (flops > 0) printf("  %9.1f TFLOPS", flops / double(ms) / 1e9);
  printf("\n");
}

template <typename F>
float time_kernel(F&& fn, int iters = 20) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  fn();  // warmup
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) fn();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}

void bench_rmsnorm(inferop::dtype dt) {
  using namespace inferop;
  const int64_t rows = 4096, cols = 5120;
  int64_t n = rows * cols;
  std::vector<float> x(n, 0.5f), w(cols, 1.f);
  auto xb = testutil::pack(x, dt), wb = testutil::pack(w, dt);
  testutil::GpuBuf d_out(n * 2), d_in(xb.size()), d_w(wb.size());
  testutil::h2d(d_in.p, xb.data(), xb.size());
  testutil::h2d(d_w.p, wb.data(), wb.size());
  float ms = time_kernel([&] { rmsnorm(d_out.p, d_in.p, d_w.p, dt, rows, cols); });
  double bytes = double(n) * 2 * dtype_bytes(dt);  // read + write
  char nm[64];
  snprintf(nm, sizeof(nm), "rmsnorm %s 4096x5120", dtype_name(dt));
  report(nm, ms, bytes);
}

void bench_gemm() {
  using namespace inferop;
  const int64_t M = 4096, N = 4096, K = 4096;
  std::vector<uint8_t> aq(M * K, 0x38), bq(N * K, 0x38);
  std::vector<float> as_(M * (K / 128), 0.01f), bs_((N / 128) * (K / 128), 0.01f);
  testutil::GpuBuf d_c(M * N * 2), d_aq(aq.size()), d_as(as_.size() * 4), d_bq(bq.size()),
      d_bs(bs_.size() * 4);
  testutil::h2d(d_aq.p, aq.data(), aq.size());
  testutil::h2d(d_as.p, as_.data(), as_.size() * 4);
  testutil::h2d(d_bq.p, bq.data(), bq.size());
  testutil::h2d(d_bs.p, bs_.data(), bs_.size() * 4);
  float ms = time_kernel([&] {
    w8a8_block_fp8_gemm(d_c.p, d_aq.p, d_as.p, d_bq.p, d_bs.p, M, N, K, 128, 128, dtype::bf16);
  });
  double flops = 2.0 * double(M) * double(N) * double(K);
  double bytes = double(M * K + N * K) + double(M * N) * 2;
  report("w8a8 block-fp8 gemm 4096^3 (bf16 out)", ms, bytes, flops);
}

void bench_quant() {
  using namespace inferop;
  const int64_t rows = 4096, cols = 7168;
  int64_t n = rows * cols;
  std::vector<float> x(n, 0.5f);
  auto xb = testutil::pack(x, dtype::bf16);
  testutil::GpuBuf d_q(n), d_s(rows * (cols / 128) * 4), d_x(xb.size());
  testutil::h2d(d_x.p, xb.data(), xb.size());
  float ms = time_kernel([&] {
    quant_fp8_per_token_group(d_q.p, d_s.p, d_x.p, dtype::bf16, rows, cols, 128);
  });
  report("quant fp8 per-token-group 4096x7168", ms, double(n) * 2 + double(n) + double(n / 128) * 4);
}

void bench_paged_attn(inferop::dtype dt) {
  using namespace inferop;
  const int batch = 8, heads = 32, kvh = 8, hd = 128, bs = 16;
  int len = 4096;
  int blocks_per_seq = len / bs;
  int total_blocks = batch * blocks_per_seq;
  std::vector<float> q(batch * heads * hd, 0.1f), k(size_t(total_blocks) * bs * kvh * hd, 0.1f);
  std::vector<int32_t> tables(batch * blocks_per_seq), lens(batch, len);
  for (size_t i = 0; i < tables.size(); ++i) tables[i] = int32_t(i);
  auto qb = testutil::pack(q, dt), kb = testutil::pack(k, dt);
  testutil::GpuBuf d_out(qb.size()), d_q(qb.size()), d_k(kb.size()), d_v(kb.size()),
      d_bt(tables.size() * 4), d_len(lens.size() * 4);
  testutil::h2d(d_q.p, qb.data(), qb.size());
  testutil::h2d(d_k.p, kb.data(), kb.size());
  testutil::h2d(d_v.p, kb.data(), kb.size());
  testutil::h2d(d_bt.p, tables.data(), tables.size() * 4);
  testutil::h2d(d_len.p, lens.data(), lens.size() * 4);
  float scale = 1.f / std::sqrt(float(hd));
  float ms = time_kernel([&] {
    paged_attn_decode(d_out.p, d_q.p, d_k.p, d_v.p, static_cast<const int32_t*>(d_bt.p),
                      static_cast<const int32_t*>(d_len.p), dt, batch, heads, kvh, hd, bs,
                      blocks_per_seq, scale);
  });
  double bytes = double(batch) * len * kvh * hd * 2 * dtype_bytes(dt) * 2;  // K+V, GQA
  char nm[80];
  snprintf(nm, sizeof(nm), "paged attn decode bs=%d len=%d %s", batch, len, dtype_name(dt));
  report(nm, ms, bytes);
}

}  // namespace

int main() {
  if (!inferop::has_cuda()) {
    std::printf("no CUDA device visible; benchmark needs a GPU\n");
    return 0;
  }
  std::printf("device: %s (sm_%d)\n\n", inferop::device_name(), inferop::device_sm_version());
  bench_rmsnorm(inferop::dtype::bf16);
  bench_quant();
  bench_gemm();
  bench_paged_attn(inferop::dtype::bf16);
  return 0;
}

#else

int main() {
  std::printf("inferop built without CUDA; rebuild with INFEROP_BUILD_CUDA=ON and an nvcc "
              "toolchain to run GPU benchmarks\n");
  return 0;
}

#endif
