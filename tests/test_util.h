// inferop : tiny standalone test framework (no external deps) + data packing
// helpers. Every test binary can run in two modes:
//   * CPU golden checks        (always, no GPU required)
//   * GPU kernel cross-checks  (built when INFEROP_HAVE_CUDA is defined,
//                               auto-skipped at runtime without a device)
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "inferop/common.h"
#include "inferop/fp8.h"

#ifdef INFEROP_HAVE_CUDA
#include <cuda_runtime.h>
#include "inferop/ops.h"
#endif

namespace inferop {
namespace testutil {

struct Stats {
  int passed = 0;
  int failed = 0;
};
inline Stats& stats() {
  static Stats s;
  return s;
}

inline void check(bool ok, const char* what) {
  if (ok) {
    stats().passed++;
  } else {
    stats().failed++;
    std::printf("  [FAIL] %s\n", what);
  }
}

inline bool all_close(const float* got, const float* want, size_t n, float rtol, float atol,
                      const char* what) {
  size_t bad = 0;
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double diff = std::fabs(double(got[i]) - double(want[i]));
    double tol = double(atol) + double(rtol) * std::fabs(double(want[i]));
    if (diff > tol) {
      bad++;
      worst = std::max(worst, diff - tol);
    }
  }
  if (bad) {
    std::printf("  [FAIL] %s : %zu/%zu mismatch, worst excess %.4g\n", what, bad, n, worst);
    stats().failed++;
    return false;
  }
  stats().passed++;
  return true;
}

inline bool bytes_equal(const void* a, const void* b, size_t bytes, const char* what) {
  if (std::memcmp(a, b, bytes) == 0) {
    stats().passed++;
    return true;
  }
  std::printf("  [FAIL] %s : byte mismatch\n", what);
  stats().failed++;
  return false;
}

inline int finish(const char* name) {
  std::printf("%s : %d passed, %d failed\n", name, stats().passed, stats().failed);
  return stats().failed == 0 ? 0 : 1;
}

inline std::mt19937& rng() {
  static std::mt19937 g(20260101u);
  return g;
}
inline float uniform(float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  return d(rng());
}
inline float normal(float mu, float sigma) {
  std::normal_distribution<float> d(mu, sigma);
  return d(rng());
}

// float32 buffer -> raw dtype bytes (fp32/fp16/bf16), using the same host
// conversions as the CPU reference.
inline std::vector<uint8_t> pack(const std::vector<float>& v, dtype dt) {
  std::vector<uint8_t> out(v.size() * size_t(dtype_bytes(dt)));
  switch (dt) {
    case dtype::fp32:
      std::memcpy(out.data(), v.data(), v.size() * 4);
      break;
    case dtype::fp16:
      for (size_t i = 0; i < v.size(); ++i) {
        uint16_t h = host_convert::float_to_half_rn(v[i]);
        std::memcpy(&out[i * 2], &h, 2);
      }
      break;
    case dtype::bf16:
      for (size_t i = 0; i < v.size(); ++i) {
        uint16_t h = host_convert::float_to_bf16_rn(v[i]);
        std::memcpy(&out[i * 2], &h, 2);
      }
      break;
    case dtype::fp8_e4m3:
      for (size_t i = 0; i < v.size(); ++i) out[i] = fp8::float_to_e4m3_rn_sat(v[i]);
      break;
  }
  return out;
}

// raw dtype bytes -> float32 buffer.
inline std::vector<float> unpack(const uint8_t* raw, dtype dt, size_t n) {
  std::vector<float> out(n);
  switch (dt) {
    case dtype::fp32:
      std::memcpy(out.data(), raw, n * 4);
      break;
    case dtype::fp16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, &raw[i * 2], 2);
        out[i] = host_convert::half_to_float(h);
      }
      break;
    case dtype::bf16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, &raw[i * 2], 2);
        out[i] = host_convert::bf16_to_float(h);
      }
      break;
    case dtype::fp8_e4m3:
      for (size_t i = 0; i < n; ++i) out[i] = fp8::e4m3_to_float(raw[i]);
      break;
  }
  return out;
}

#ifdef INFEROP_HAVE_CUDA
struct GpuBuf {
  void* p = nullptr;
  explicit GpuBuf(size_t bytes) { cudaMalloc(&p, bytes); }
  ~GpuBuf() {
    if (p) cudaFree(p);
  }
  GpuBuf(const GpuBuf&) = delete;
  GpuBuf& operator=(const GpuBuf&) = delete;
};
inline void h2d(void* dst, const void* src, size_t bytes) {
  cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
inline void d2h(void* dst, const void* src, size_t bytes) {
  cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
#endif

}  // namespace testutil
}  // namespace inferop

#define RUN_TEST(fn)                            \
  do {                                          \
    std::printf("== %s ==\n", #fn);             \
    fn();                                       \
  } while (0)

#ifdef INFEROP_HAVE_CUDA
#define GPU_SECTION(stmt)                     \
  do {                                        \
    if (inferop::has_cuda()) {                \
      std::printf("-- gpu cross-check --\n"); \
      stmt;                                   \
    } else {                                  \
      std::printf("-- gpu cross-check skipped (no device) --\n"); \
    }                                         \
  } while (0)
#else
#define GPU_SECTION(stmt) \
  do {                    \
    (void)0;              \
  } while (0)
#endif
