// inferop : E4M3 (FP8) conversions shared by CUDA kernels and the CPU golden
// reference. Pure integer implementations => bit-identical on host and device,
// no reliance on hardware cvt instructions, works on every SM architecture
// (sm_70 .. sm_120+).
//
// Semantics follow the OCP FP8 spec with SATFINITE rounding:
//   * max finite value 448 (S.1111.110), NaN = S.1111.111
//   * denormals: m * 2^-9, m in [1, 7]
//   * round-to-nearest-even, saturate on overflow
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define INFEROP_HD __host__ __device__
#else
#define INFEROP_HD
#endif

namespace inferop {
namespace fp8 {

INFEROP_HD constexpr float kE4M3Max = 448.0f;

namespace detail {
INFEROP_HD inline float u32_to_f32(uint32_t w) {
  float f;
  memcpy(&f, &w, sizeof(f));
  return f;
}
INFEROP_HD inline uint32_t f32_to_u32(float f) {
  uint32_t w;
  memcpy(&w, &f, sizeof(w));
  return w;
}
}  // namespace detail

// Decode one E4M3 byte to float (all 256 inputs handled).
INFEROP_HD inline float e4m3_to_float(uint8_t b) {
  uint32_t s = uint32_t(b >> 7) << 31;
  uint32_t E = uint32_t(b >> 3) & 0xFu;
  uint32_t m = uint32_t(b) & 0x7u;
  if (E == 15u && m == 7u) return detail::u32_to_f32(0x7FC00000u | s);  // NaN
  if (E == 0u) {
    if (m == 0u) return detail::u32_to_f32(s);  // +-0
    // denormal: m * 2^-9, normalize m into [8, 16) so it maps onto the
    // fp32 "implicit-bit + 3 fraction bits" grid.
    uint32_t mm = m;
    int e32 = 121;  // see scripts/check_fp8.py for the derivation
    while (mm < 8u) {
      mm <<= 1;
      --e32;
    }
    return detail::u32_to_f32(s | (uint32_t(e32) << 23) | ((mm & 7u) << 20));
  }
  // normal: (1 + m/8) * 2^(E-7) == (8+m) * 2^(E-10)
  return detail::u32_to_f32(s | ((E + 120u) << 23) | (m << 20));
}

// Encode float to E4M3 with round-to-nearest-even, saturating (SATFINITE).
INFEROP_HD inline uint8_t float_to_e4m3_rn_sat(float f) {
  uint32_t x = detail::f32_to_u32(f);
  uint32_t sign = (x >> 24) & 0x80u;
  uint32_t a = x & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return uint8_t(sign | 0x7Fu);   // NaN
  if (a == 0x7F800000u) return uint8_t(sign | 0x7Eu);  // Inf -> 448 (satfinite)
  int e = int((a >> 23) & 0xFFu) - 127;
  if (e >= 9) return uint8_t(sign | 0x7Eu);  // |f| >= 512 -> saturate
  if (e >= -6) {                             // E4M3 normal range
    uint32_t man = (a & 0x7FFFFFu) | 0x800000u;
    uint32_t keep = man >> 20;  // in [8, 16)
    uint32_t rem = man & 0xFFFFFu;
    uint32_t r = (rem > 0x80000u || (rem == 0x80000u && (keep & 1u))) ? 1u : 0u;
    uint32_t m = keep - 8u + r;  // [0, 8]
    uint32_t E = uint32_t(e + 7);
    if (m == 8u) {
      m = 0;
      E += 1u;
    }
    if (E >= 16u || (E == 15u && m == 7u)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (E << 3) | m);
  }
  if (e >= -10) {  // E4M3 denormal range: md = RNE(m / 2^(14-e)), md in [0, 8]
    uint32_t man = (a & 0x7FFFFFu) | 0x800000u;  // fp32 normal (e >= -10)
    uint32_t sh = uint32_t(14 - e);              // 21..24
    uint32_t keep = man >> sh;                   // in [0, 7]
    uint32_t rem = man & ((1u << sh) - 1u);
    uint32_t half = 1u << (sh - 1);
    uint32_t r = (rem > half || (rem == half && (keep & 1u))) ? 1u : 0u;
    uint32_t md = keep + r;
    if (md == 8u) return uint8_t(sign | 0x08u);  // promote to min normal 2^-6
    return uint8_t(sign | md);
  }
  return uint8_t(sign);  // -> +-0 (also covers fp32 subnormals)
}

}  // namespace fp8
}  // namespace inferop
