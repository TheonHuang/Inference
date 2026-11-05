// Standalone test : E4M3 (FP8) encode/decode vs an independent oracle that
// enumerates the full 256-value grid from the OCP definition. CPU-only.
#include "test_util.h"

namespace {

// Independent oracle: decode every byte per the spec (double exact math).
std::vector<double> enumerate_grid() {
  std::vector<double> g(256);
  for (int b = 0; b < 256; ++b) {
    double sign = (b >> 7) ? -1.0 : 1.0;
    int E = (b >> 3) & 0xF, m = b & 7;
    double v;
    if (E == 15 && m == 7) {
      v = std::nan("");
    } else if (E == 0) {
      v = m == 0 ? 0.0 : double(m) * std::pow(2.0, -9);
    } else {
      v = (1.0 + double(m) / 8.0) * std::pow(2.0, double(E) - 7.0);
    }
    g[b] = sign * v;
  }
  return g;
}

// Nearest-grid-point encoding with RNE ties -> even encoding, saturating.
// Monotone positive grid => linear scan suffices.
uint8_t oracle_encode(float f) {
  if (std::isnan(f)) return 0x7F;
  static std::vector<double> grid = enumerate_grid();
  bool neg = std::signbit(f);
  double a = std::fabs(double(f));
  if (a == 0.0) return neg ? 0x80 : 0x00;
  if (a >= 448.0) return neg ? 0xFE : 0x7E;  // saturate: nothing above 448
  int best = 0;
  double bestd = 1e300;
  bool tie_even = false;
  for (int b = 0; b < 128; ++b) {  // positive encodings: 0x00..0x7E (skip 0x7F NaN)
    if (b == 0x7F) continue;
    double d = std::fabs(a - grid[b]);
    if (d < bestd) {
      bestd = d;
      best = b;
      tie_even = false;
    } else if (d == bestd) {  // tie: prefer even encoding
      if ((b & 1) == 0 && (best & 1) == 1) best = b;
    }
  }
  return uint8_t(neg ? (best | 0x80) : best);
}

void test_decode_matches_grid() {
  static std::vector<double> grid = enumerate_grid();
  int bad = 0;
  for (int b = 0; b < 256; ++b) {
    float got = inferop::fp8::e4m3_to_float(uint8_t(b));
    double want = grid[b];
    if (std::isnan(want)) {
      if (!std::isnan(got)) bad++;
    } else {
      double diff = std::fabs(double(got) - want);
      double tol = std::fabs(want) * 1e-6 + 1e-45;  // exact dyadics, allow fp32 rounding
      if (diff > (want == 0.0 ? 0.0 : tol) && !(want == 0.0 && got == 0.0f)) bad++;
    }
  }
  inferop::testutil::check(bad == 0, "decode matches spec grid for all 256 bytes");
  if (bad) std::printf("  %d decode mismatches\n", bad);
}

void test_encode_matches_oracle() {
  std::vector<float> samples;
  // random
  for (int i = 0; i < 20000; ++i) samples.push_back(inferop::testutil::uniform(-600.f, 600.f));
  // edge magnitudes
  for (float v : {0.0f, -0.0f, 1.0f, -1.0f, 448.0f, 449.0f, 464.0f, 479.99998f, 480.0f, 512.0f,
                  1e6f, 5.9604645e-8f, 2.9802322e-8f, 1.5258789e-05f, 0.000015258789f,
                  0.0146484375f, 0.015624999f, 0.015625f, 0.0001f, 1e-9f, 1e38f}) {
    samples.push_back(v);
    samples.push_back(-v);
  }
  // exact grid points and exact midpoints (tie cases)
  static std::vector<double> grid = enumerate_grid();
  for (int b = 1; b < 0x7F; ++b) {
    if (b == 0x7E) {
      samples.push_back(float((grid[b] + 512.0) / 2));  // beyond-max tie region
      continue;
    }
    samples.push_back(float(grid[b]));
    samples.push_back(float((grid[b] + grid[b + 1]) / 2));
  }
  int bad = 0;
  uint8_t bad_got = 0, bad_want = 0;
  for (float f : samples) {
    uint8_t got = inferop::fp8::float_to_e4m3_rn_sat(f);
    uint8_t want = oracle_encode(f);
    if (got != want && !(std::isnan(f))) {
      if (!bad) {
        bad_got = got;
        bad_want = want;
      }
      bad++;
    }
  }
  inferop::testutil::check(bad == 0, "encode matches oracle (random + edges + ties)");
  if (bad) std::printf("  %d mismatches, first: got 0x%02x want 0x%02x\n", bad, bad_got, bad_want);
}

void test_roundtrip_error_bound() {
  // E4M3 has 3 mantissa bits -> relative error <= 2^-4 on the quantized value.
  int bad = 0;
  for (int i = 0; i < 20000; ++i) {
    float x = inferop::testutil::uniform(0.01f, 400.f);
    uint8_t q = inferop::fp8::float_to_e4m3_rn_sat(x);
    float y = inferop::fp8::e4m3_to_float(q);
    if (std::fabs(y - x) > std::fabs(x) * (1.0f / 16.0f)) bad++;
  }
  inferop::testutil::check(bad == 0, "roundtrip relative error <= 2^-4");
}

}  // namespace

int main() {
  RUN_TEST(test_decode_matches_grid);
  RUN_TEST(test_encode_matches_oracle);
  RUN_TEST(test_roundtrip_error_bound);
  return inferop::testutil::finish("test_fp8");
}
