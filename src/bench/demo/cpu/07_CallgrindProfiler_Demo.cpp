/**
 * @file 07_CallgrindProfiler_Demo.cpp
 * @brief Demo 07: Valgrind callgrind for deterministic instruction counting
 *
 * Demonstrates using callgrind for precise A/B comparison without
 * measurement noise. Unlike sampling profilers (perf, gperf), callgrind
 * simulates every instruction -- producing identical results on every run.
 *
 * Slow: Linear search O(n) through sorted array
 * Fast: Binary search O(log n) via std::lower_bound
 *
 * Usage:
 *   @code{.sh}
 *   # Run without profiler (normal speed)
 *   ./BenchDemo_07_CallgrindProfiler --csv baseline.csv
 *
 *   # Run under callgrind (20-50x slower but deterministic)
 *   ./BenchDemo_07_CallgrindProfiler --profile callgrind --gtest_filter="*LinearSearch*"
 *
 *   # Analyze callgrind output
 *   callgrind_annotate callgrind.out.*
 *   @endcode
 *
 * @see docs/07_CALLGRIND_PROFILER.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

static constexpr std::size_t ARRAY_SIZE = 10000;
static constexpr std::size_t NUM_SEARCHES = 100; // Searches per measurement

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: Linear search through sorted array.
 *
 * Scans from the beginning until finding the target. For a random
 * target in [0,1), the expected scan length is n/2. Under callgrind,
 * this produces a predictable instruction count proportional to n.
 *
 * callgrind_annotate will show the for-loop as the dominant cost center.
 */
PERF_LATENCY(CallgrindProfiler, LinearSearch) {
  UB_PERF_GUARD(perf);

  auto sorted = demo::makeSorted(demo::makeRandomDoubles(ARRAY_SIZE));
  auto targets = demo::makeRandomDoubles(NUM_SEARCHES, 99);

  perf.warmup([&] {
    volatile std::size_t result = 0;
    for (const auto& t : targets) {
      result = result + demo::linearSearch(sorted.data(), sorted.size(), t);
    }
    (void)result;
  });

  volatile std::size_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        std::size_t acc = 0;
        for (const auto& t : targets) {
          acc += demo::linearSearch(sorted.data(), sorted.size(), t);
        }
        sink = sink + acc;
      },
      "linear_search");

  EXPECT_GT(result.callsPerSecond, 1.0);

  (void)sink;
}

/**
 * @test Fast: Binary search through sorted array.
 *
 * Uses std::lower_bound for O(log n) search. Under callgrind, the
 * instruction count is dramatically lower -- roughly log2(10000) = 13
 * comparisons per search vs 5000 for linear.
 *
 * When comparing callgrind output, the instruction ratio directly
 * reflects the algorithmic improvement (no noise).
 */
PERF_LATENCY(CallgrindProfiler, BinarySearch) {
  UB_PERF_GUARD(perf);

  auto sorted = demo::makeSorted(demo::makeRandomDoubles(ARRAY_SIZE));
  auto targets = demo::makeRandomDoubles(NUM_SEARCHES, 99);

  perf.warmup([&] {
    volatile std::size_t result = 0;
    for (const auto& t : targets) {
      result = result + demo::binarySearch(sorted.data(), sorted.size(), t);
    }
    (void)result;
  });

  volatile std::size_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        std::size_t acc = 0;
        for (const auto& t : targets) {
          acc += demo::binarySearch(sorted.data(), sorted.size(), t);
        }
        sink = sink + acc;
      },
      "binary_search");

  EXPECT_GT(result.callsPerSecond, 100.0);

  (void)sink;
}

PERF_MAIN()
