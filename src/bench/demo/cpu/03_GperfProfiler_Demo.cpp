/**
 * @file 03_GperfProfiler_Demo.cpp
 * @brief Demo 03: gperftools profiler for function-level hotspot identification
 *
 * Demonstrates using gperftools CPU profiler to find which function
 * dominates execution time, then replacing the slow algorithm.
 *
 * Slow: bubbleSort O(n^2) -- profiler shows 95%+ time in this function
 * Fast: std::sort O(n log n) -- time distributed across introsort internals
 *
 * Usage:
 *   @code{.sh}
 *   # Baseline
 *   ./BenchDemo_03_GperfProfiler --csv baseline.csv
 *
 *   # Profile slow path (generates .prof file)
 *   ./BenchDemo_03_GperfProfiler --profile gperf --gtest_filter="*BubbleSortHotspot*"
 *
 *   # Analyze profile
 *   google-pprof --text ./BenchDemo_03_GperfProfiler *.prof
 *   google-pprof --pdf ./BenchDemo_03_GperfProfiler *.prof > hotspot.pdf
 *
 *   # Compare baseline vs optimized
 *   bench summary baseline.csv
 *   @endcode
 *
 * @see docs/03_GPERF_PROFILER.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

// 500 elements: large enough for bubble sort to be slow, small enough to finish
static constexpr std::size_t SORT_SIZE = 500;

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: Bubble sort dominates execution time.
 *
 * When profiled with gperftools, the output will show ~95% of CPU time
 * spent in demo::bubbleSort(). This is the classic "one function is the
 * bottleneck" pattern that function-level profilers excel at finding.
 */
PERF_THROUGHPUT(GperfProfiler, BubbleSortHotspot) {
  UB_PERF_GUARD(perf);

  auto original = demo::makeRandomDoubles(SORT_SIZE);

  perf.warmup([&] {
    auto copy = original;
    demo::bubbleSort(copy.data(), copy.size());
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        auto copy = original;
        demo::bubbleSort(copy.data(), copy.size());
        sink = sink + copy[0];
      },
      "bubble_sort");

  EXPECT_GT(result.callsPerSecond, 1.0);

  (void)sink;
}

/**
 * @test Fast: std::sort replaces the hotspot.
 *
 * After identifying bubbleSort as the hotspot, we replace it with
 * std::sort. The gperftools profile now shows time distributed across
 * the introsort implementation -- no single dominant hotspot.
 *
 * Expected improvement: 10-50x for 500 elements (O(n^2) vs O(n log n)).
 */
PERF_THROUGHPUT(GperfProfiler, StdSortOptimized) {
  UB_PERF_GUARD(perf);

  auto original = demo::makeRandomDoubles(SORT_SIZE);

  perf.warmup([&] {
    auto copy = original;
    demo::fastSort(copy.data(), copy.size());
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        auto copy = original;
        demo::fastSort(copy.data(), copy.size());
        sink = sink + copy[0];
      },
      "std_sort");

  EXPECT_GT(result.callsPerSecond, 100.0);

  (void)sink;
}

PERF_MAIN()
