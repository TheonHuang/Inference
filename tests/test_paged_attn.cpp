// Standalone test : paged attention decode (CPU golden + GPU cross-check).
#include <algorithm>

#include "test_util.h"
#include "reference/reference.h"

namespace {

struct AttnCase {
  int batch, heads, kv_heads, head_dim, block_size;
  std::vector<int> lens;
};

// Build a paged KV cache: sequential disjoint blocks per sequence, with the
// first block of every sequence shared (prefix sharing sanity).
void build_case(const AttnCase& c, std::vector<float>* q, std::vector<float>* k,
                std::vector<float>* v, std::vector<int32_t>* tables, int* max_blocks) {
  using namespace inferop;
  int shared_blocks = 1;
  int per_seq_blocks = 0;
  for (int len : c.lens) per_seq_blocks = std::max(per_seq_blocks, (len + c.block_size - 1) / c.block_size);
  int blocks_needed = shared_blocks + c.batch * per_seq_blocks;
  *max_blocks = per_seq_blocks;

  k->resize(size_t(blocks_needed) * c.block_size * c.kv_heads * c.head_dim);
  v->resize(k->size());
  for (auto& x : *k) x = testutil::normal(0.f, 1.f);
  for (auto& x : *v) x = testutil::normal(0.f, 1.f);
  q->resize(size_t(c.batch) * c.heads * c.head_dim);
  for (auto& x : *q) x = testutil::normal(0.f, 1.f);

  tables->assign(size_t(c.batch) * per_seq_blocks, 0);
  for (int b = 0; b < c.batch; ++b) {
    (*tables)[b * per_seq_blocks] = 0;  // shared first block
    for (int j = 1; j < per_seq_blocks; ++j)
      (*tables)[b * per_seq_blocks + j] = 1 + b * per_seq_blocks + (j - 1);
  }
}

void cpu_golden() {
  AttnCase c{3, 32, 8, 128, 16, {1, 17, 129}};
  std::vector<float> q, k, v, out(size_t(c.batch) * c.heads * c.head_dim);
  std::vector<int32_t> tables;
  int max_blocks = 0;
  build_case(c, &q, &k, &v, &tables, &max_blocks);
  float scale = 1.f / std::sqrt(float(c.head_dim));
  inferop::ref::paged_attn_decode(out.data(), q.data(), k.data(), v.data(), tables.data(),
                                  c.lens.data(), c.batch, c.heads, c.kv_heads, c.head_dim,
                                  c.block_size, max_blocks, scale);

  // property: convex combination of V rows => output within data range
  float lo = 1e30f, hi = -1e30f;
  for (auto x : v) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  bool in_range = true;
  for (float x : out) in_range = in_range && x >= lo - 1e-3 && x <= hi + 1e-3;
  inferop::testutil::check(in_range, "outputs inside V data range");

  // property: length-1 attention returns that V row exactly
  int kvh0 = 0;  // head 0 maps to kv head 0
  int blk0 = tables[0];
  bool first_matches = true;
  for (int d = 0; d < c.head_dim; ++d) {
    float want = v[(size_t(blk0) * c.block_size + 0) * c.kv_heads * c.head_dim +
                   size_t(kvh0) * c.head_dim + d];
    first_matches = first_matches && std::fabs(out[d] - want) < 1e-4;
  }
  inferop::testutil::check(first_matches, "len==1 returns the single V row");
}

#ifdef INFEROP_HAVE_CUDA
void gpu_cross_check(const AttnCase& c, inferop::dtype dt, float rtol, float atol) {
  using namespace inferop;
  std::vector<float> q, k, v, want(size_t(c.batch) * c.heads * c.head_dim);
  std::vector<int32_t> tables;
  int max_blocks = 0;
  build_case(c, &q, &k, &v, &tables, &max_blocks);
  float scale = 1.f / std::sqrt(float(c.head_dim));
  ref::paged_attn_decode(want.data(), q.data(), k.data(), v.data(), tables.data(), c.lens.data(),
                         c.batch, c.heads, c.kv_heads, c.head_dim, c.block_size, max_blocks,
                         scale);

  auto qb = testutil::pack(q, dt), kb = testutil::pack(k, dt), vb = testutil::pack(v, dt);
  testutil::GpuBuf d_out(qb.size()), d_q(qb.size()), d_k(kb.size()), d_v(vb.size()),
      d_bt(tables.size() * 4), d_len(c.batch * 4);
  testutil::h2d(d_q.p, qb.data(), qb.size());
  testutil::h2d(d_k.p, kb.data(), kb.size());
  testutil::h2d(d_v.p, vb.data(), vb.size());
  testutil::h2d(d_bt.p, tables.data(), tables.size() * 4);
  testutil::h2d(d_len.p, c.lens.data(), c.batch * 4);
  paged_attn_decode(d_out.p, d_q.p, d_k.p, d_v.p, static_cast<const int32_t*>(d_bt.p),
                    static_cast<const int32_t*>(d_len.p), dt, c.batch, c.heads, c.kv_heads,
                    c.head_dim, c.block_size, max_blocks, scale);
  std::vector<uint8_t> hb(qb.size());
  testutil::d2h(hb.data(), d_out.p, qb.size());
  auto got = testutil::unpack(hb.data(), dt, want.size());
  char name[96];
  std::snprintf(name, sizeof(name), "gpu paged attn hd=%d %s", c.head_dim,
                inferop::dtype_name(dt));
  testutil::all_close(got.data(), want.data(), want.size(), rtol, atol, name);
}
#endif

}  // namespace

int main() {
  RUN_TEST(cpu_golden);
  GPU_SECTION({
    gpu_cross_check({3, 32, 8, 128, 16, {1, 17, 129}}, inferop::dtype::fp32, 2e-3f, 2e-3f);
    gpu_cross_check({3, 32, 8, 128, 16, {1, 17, 129}}, inferop::dtype::bf16, 2e-2f, 2e-2f);
    gpu_cross_check({2, 16, 4, 64, 16, {5, 64}}, inferop::dtype::bf16, 2e-2f, 2e-2f);
    gpu_cross_check({2, 8, 8, 256, 16, {33, 200}}, inferop::dtype::bf16, 2e-2f, 2e-2f);
    gpu_cross_check({1, 32, 8, 128, 16, {2048}}, inferop::dtype::bf16, 2e-2f, 2e-2f);
  });
  return inferop::testutil::finish("test_paged_attn");
}
