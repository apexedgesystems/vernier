/**
 * @file 08_RaplProfiler_Demo.cpp
 * @brief Demo 08: Intel RAPL energy measurement
 *
 * Demonstrates using RAPL (Running Average Power Limit) to measure
 * energy consumption. Shows that optimized code not only runs faster
 * but also consumes less total energy.
 *
 * Slow: Naive dot product with dependency chain preventing vectorization
 * Fast: std::inner_product (compiler can auto-vectorize)
 *
 * Usage:
 *   @code{.sh}
 *   # Run with RAPL profiling
 *   ./BenchDemo_08_RaplProfiler --profile rapl --csv energy.csv
 *
 *   # Compare energy per operation
 *   bench summary energy.csv
 *   @endcode
 *
 * @note Requires Intel CPU (Haswell+) with MSR module loaded.
 *       May require root/sudo for MSR access.
 *
 * @see docs/08_RAPL_PROFILER.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

static constexpr std::size_t VECTOR_SIZE = 100000;

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: Naive dot product with data dependency.
 *
 * The dependency chain `if (sum > 1e18)` prevents the compiler from
 * auto-vectorizing the loop. Each iteration depends on the previous
 * sum, serializing the computation.
 *
 * Under RAPL profiling, this test consumes more Joules per operation
 * because the CPU spends more time (and draws power) on each element.
 */
PERF_THROUGHPUT(RaplProfiler, NaiveDotProduct) {
  UB_PERF_GUARD(perf);

  auto a = demo::makeRandomDoubles(VECTOR_SIZE, 42);
  auto b = demo::makeRandomDoubles(VECTOR_SIZE, 99);

  perf.warmup([&] {
    volatile double result = demo::naiveDotProduct(a.data(), b.data(), a.size());
    (void)result;
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + demo::naiveDotProduct(a.data(), b.data(), a.size()); },
      "naive_dot_product");

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

/**
 * @test Fast: std::inner_product (vectorizable).
 *
 * std::inner_product with doubles and 0.0 init allows the compiler
 * to auto-vectorize using SIMD instructions (SSE/AVX). Multiple
 * multiply-adds execute per cycle.
 *
 * Under RAPL profiling, this test consumes fewer total Joules because
 * the faster execution means less total power draw over time.
 * The energy-per-operation metric directly shows the efficiency gain.
 */
PERF_THROUGHPUT(RaplProfiler, VectorizedDotProduct) {
  UB_PERF_GUARD(perf);

  auto a = demo::makeRandomDoubles(VECTOR_SIZE, 42);
  auto b = demo::makeRandomDoubles(VECTOR_SIZE, 99);

  perf.warmup([&] {
    volatile double result = demo::fastDotProduct(a.data(), b.data(), a.size());
    (void)result;
  });

  volatile double sink = 0.0;
  auto result =
      perf.throughputLoop([&] { sink = sink + demo::fastDotProduct(a.data(), b.data(), a.size()); },
                          "vectorized_dot_product");

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

PERF_MAIN()
