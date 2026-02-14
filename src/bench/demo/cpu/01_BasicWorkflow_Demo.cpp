/**
 * @file 01_BasicWorkflow_Demo.cpp
 * @brief Demo 01: Basic benchmarking workflow
 *
 * Teaches the fundamental measure-export-analyze workflow:
 *  1. Measure with throughputLoop() and MemoryProfile
 *  2. Export results to CSV with --csv flag
 *  3. Analyze with bench summary and bench compare
 *
 * Usage:
 *   @code{.sh}
 *   # Run baseline
 *   ./BenchDemo_01_BasicWorkflow --csv baseline.csv
 *
 *   # Quick summary
 *   bench summary baseline.csv
 *
 *   # A/B comparison (run again after changes)
 *   ./BenchDemo_01_BasicWorkflow --csv candidate.csv
 *   bench compare baseline.csv candidate.csv
 *   @endcode
 *
 * @see docs/01_BASIC_WORKFLOW.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

static constexpr std::size_t DATA_SIZE = 100000;

/* ----------------------------- Tests ----------------------------- */

/** @test Basic throughput measurement with warmup and memory profiling. */
PERF_THROUGHPUT(BasicWorkflow, SimpleThroughput) {
  UB_PERF_GUARD(perf);

  auto data = demo::makeRandomDoubles(DATA_SIZE);

  perf.warmup([&] {
    volatile double sum = std::accumulate(data.begin(), data.end(), 0.0);
    (void)sum;
  });

  ub::MemoryProfile memProfile{
      .bytesRead = DATA_SIZE * sizeof(double), .bytesWritten = 0, .bytesAllocated = 0};

  volatile double sink = 0.0;
  auto result =
      perf.throughputLoop([&] { sink = sink + std::accumulate(data.begin(), data.end(), 0.0); },
                          "accumulate", memProfile);

  EXPECT_GT(result.callsPerSecond, 100.0) << "Accumulate should exceed 100 calls/sec";
  EXPECT_LT(result.stats.cv, 0.30) << "Measurements should be reasonably stable";
}

/** @test A/B comparison: std::accumulate vs manual loop. */
PERF_THROUGHPUT(BasicWorkflow, AccumulateVsManualLoop) {
  UB_PERF_GUARD(perf);

  auto data = demo::makeRandomDoubles(DATA_SIZE);

  perf.warmup([&] {
    volatile double sum = std::accumulate(data.begin(), data.end(), 0.0);
    (void)sum;
  });

  // Approach A: std::accumulate
  volatile double sinkA = 0.0;
  auto resultA = perf.throughputLoop(
      [&] { sinkA = sinkA + std::accumulate(data.begin(), data.end(), 0.0); }, "std_accumulate");

  // Approach B: Manual loop with pointer arithmetic
  volatile double sinkB = 0.0;
  auto resultB = perf.throughputLoop(
      [&] {
        double sum = 0.0;
        const double* ptr = data.data();
        const double* end = ptr + data.size();
        while (ptr != end) {
          sum += *ptr++;
        }
        sinkB = sinkB + sum;
      },
      "manual_loop");

  // Both should produce valid measurements
  EXPECT_GT(resultA.callsPerSecond, 100.0);
  EXPECT_GT(resultB.callsPerSecond, 100.0);

  (void)sinkA;
  (void)sinkB;
}

/** @test Quick mode demonstration -- fast iteration with relaxed thresholds. */
PERF_THROUGHPUT(BasicWorkflow, QuickModeIteration) {
  UB_PERF_GUARD(perf);

  auto data = demo::makeRandomDoubles(DATA_SIZE / 10);

  perf.warmup([&] {
    volatile double sum = std::accumulate(data.begin(), data.end(), 0.0);
    (void)sum;
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + std::accumulate(data.begin(), data.end(), 0.0); }, "quick_accumulate");

  // In quick mode, expect relaxed but still valid results
  EXPECT_GT(result.callsPerSecond, 100.0);

  (void)sink;
}

PERF_MAIN()
