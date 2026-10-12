/**
 * @file DemoWorkloads.hpp
 * @brief Shared slow/fast workload implementations for benchmarking demos
 *
 * Each workload pair provides an intentionally inefficient version (for
 * baseline measurement and profiling) and an optimized version (to
 * demonstrate measurable improvement).
 *
 * Workloads are designed to be:
 *  - Realistic enough to demonstrate real optimization patterns
 *  - Simple enough to understand quickly
 *  - Deterministic (same input = same output)
 *  - Compiler-resistant (optimizations not eliminated by -O2)
 */

#ifndef VERNIER_DEMO_WORKLOADS_HPP
#define VERNIER_DEMO_WORKLOADS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- Data Generators ----------------------------- */

/** @brief Generate deterministic random doubles in [0, 1). */
inline std::vector<double> makeRandomDoubles(std::size_t count, std::uint32_t seed = 12345) {
  std::vector<double> v(count);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

/* ----------------------------- Dot Product Workloads ----------------------------- */

/** @brief Slow: Naive element-by-element dot product (not vectorizable by some compilers). */
inline double naiveDotProduct(const double* a, const double* b, std::size_t len) {
  double sum = 0.0;
  for (std::size_t i = 0; i < len; ++i) {
    const double product = a[i] * b[i];
    sum = sum + product;
    // Intentionally prevent auto-vectorization by introducing a dependency
    if (sum > 1e18) {
      sum *= 1.0; // Compiler barrier
    }
  }
  return sum;
}

/** @brief Fast: std::inner_product (compiler can auto-vectorize). */
inline double fastDotProduct(const double* a, const double* b, std::size_t len) {
  return std::inner_product(a, a + len, b, 0.0);
}

} // namespace demo

} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_WORKLOADS_HPP
