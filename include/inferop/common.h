// inferop : common types and host-only numeric helpers.
// This header must stay CUDA-free so pure-C++ targets (reference impl, CPU
// tests) can be built without the CUDA toolkit installed.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace inferop {

enum class dtype : int32_t { fp32 = 0, fp16 = 1, bf16 = 2, fp8_e4m3 = 3 };

inline const char* dtype_name(dtype d) {
  switch (d) {
    case dtype::fp32: return "fp32";
    case dtype::fp16: return "fp16";
    case dtype::bf16: return "bf16";
    case dtype::fp8_e4m3: return "fp8_e4m3";
  }
  return "unknown";
}

inline int dtype_bytes(dtype d) {
  switch (d) {
    case dtype::fp32: return 4;
    case dtype::fp16: return 2;
    case dtype::bf16: return 2;
    case dtype::fp8_e4m3: return 1;
  }
  return 0;
}

namespace bitcast {
inline float u32_to_f32(uint32_t w) {
  float f;
  std::memcpy(&f, &w, sizeof(f));
  return f;
}
inline uint32_t f32_to_u32(float f) {
  uint32_t w;
  std::memcpy(&w, &f, sizeof(w));
  return w;
}
}  // namespace bitcast

// ---------------------------------------------------------------------------
// Host-side, header-only conversions between raw 16-bit storage and float.
// Bit-exact equivalents of the CUDA __half/__nv_bfloat16 conversions so the
// CPU golden reference and the GPU kernels agree on the wire format.
// ---------------------------------------------------------------------------
namespace host_convert {

inline float bf16_to_float(uint16_t v) { return bitcast::u32_to_f32(uint32_t(v) << 16); }

inline uint16_t float_to_bf16_rn(float f) {
  uint32_t x = bitcast::f32_to_u32(f);
  uint32_t rounded = x + 0x7FFFu + ((x >> 16) & 1u);  // round-to-nearest-even
  return uint16_t(rounded >> 16);
}

inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t(h) >> 15) & 1u;
  uint32_t exp = (uint32_t(h) >> 10) & 0x1Fu;
  uint32_t man = uint32_t(h) & 0x3FFu;
  float value;
  if (exp == 0) {
    if (man == 0) return sign ? -0.0f : 0.0f;
    value = std::ldexp(float(man), -24);  // denormal: man * 2^-24
  } else if (exp == 31) {
    if (man != 0) return std::nanf("");
    value = std::numeric_limits<float>::infinity();
  } else {
    value = std::ldexp(float(1024u + man), int(exp) - 25);  // (1024+man) * 2^(exp-25)
  }
  return sign ? -value : value;
}

inline uint16_t float_to_half_rn(float f) {
  if (std::isnan(f)) return 0x7E00;
  bool neg = std::signbit(f);
  float a = neg ? -f : f;
  if (std::isinf(a) || a > 65504.0f) return neg ? uint16_t(0xFC00) : uint16_t(0x7C00);
  if (a < 5.9604645e-8f) return neg ? uint16_t(0x8000) : uint16_t(0x0000);  // < half of min denorm
  int e = 0;
  float frac = std::frexp(a, &e);  // a = frac * 2^e, frac in [0.5, 1)
  if (e < -13) {                   // fp16 denormal: m = RNE(a * 2^24), m in (0, 1024]
    uint32_t m = uint32_t(std::nearbyint(std::ldexp(a, 24)));
    if (m >= 1024u) return neg ? uint16_t(0x8400) : uint16_t(0x0400);  // promoted to min normal
    return uint16_t((neg ? 0x8000u : 0u) | m);
  }
  uint32_t m = uint32_t(std::nearbyint(std::ldexp(frac, 11)) - 1024);  // fraction bits, RNE
  uint32_t exp = uint32_t(e - 1 + 15);
  if (m == 1024u) {  // rounding carried into exponent
    m = 0;
    exp += 1;
  }
  if (exp >= 31u) return neg ? uint16_t(0xFC00) : uint16_t(0x7C00);
  return uint16_t((neg ? 0x8000u : 0u) | (exp << 10) | m);
}

}  // namespace host_convert

}  // namespace inferop
