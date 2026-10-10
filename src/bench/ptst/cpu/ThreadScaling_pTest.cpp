/**
 * @file ThreadScaling_pTest.cpp
 * @brief Multi-threaded performance scaling validation
 *
 * This test suite validates that the framework correctly measures multi-threaded
 * performance and calculates speedup/efficiency metrics. Tests use minimal thread
 * counts (1-2) optimized for CI environments.
 *
 * Features tested:
 *  - Single-threaded baseline performance
 *  - Multi-threaded performance scaling
 *  - Speedup calculation and validation
 *  - Efficiency metric calculation
 *
 * Expected behavior:
 *  - Multi-threaded tests show speedup vs single-threaded
 *  - Efficiency should be >50% for embarrassingly parallel workloads
 *  - Measurements stable across thread counts
 *
 * Usage:
 *   @code{.sh}
 *   # Run all thread scaling tests
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="ThreadScaling.*"
 *
 *   # Control thread count
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="ThreadScaling.*" --threads 2
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~15 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <30% for typical workloads
 *
 * @see PerfCase
 * @see PerfCase::contentionRun
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <vector>
#include <thread>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

} // anonymous namespace

/**
 * @brief Single-threaded baseline performance
 *
 * Establishes baseline performance for single-threaded execution.
 * This provides the reference for calculating speedup with multiple threads.
 *
 * @test SingleThreadBaseline
 *
 * Validates:
 *  - Single-threaded performance measurement
 *  - Baseline throughput is reasonable
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Throughput >100K calls/sec
 *  - CV <30%
 */
PERF_TEST(ThreadScaling, SingleThreadBaseline) {
  ub::PerfCase perf{"ThreadScaling.SingleThreadBaseline", config()};

  const std::size_t DATA_SIZE = 8192;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "single_thread");

  EXPECT_GT(result.callsPerSecond, 100000.0) << "Single-threaded throughput unexpectedly low";

  EXPECT_LT(result.stats.cv, 0.30) << "High variance in single-threaded baseline";

  (void)sink;
}

/**
 * @brief Multi-threaded efficiency validation
 *
 * Measures performance with 2 threads and validates speedup and efficiency
 * metrics. Only tests minimal thread counts for CI speed.
 *
 * @test EfficiencyValidation
 *
 * Validates:
 *  - Every call computes the right sum, with no shared write on the way
 *  - Multi-threaded performance exceeds single-threaded
 *  - Speedup calculation is reasonable
 *  - Efficiency >50% for embarrassingly parallel work
 *
 * Expected performance:
 *  - Speedup: >1.2x (accounting for overhead)
 *  - Efficiency: >50%
 */
PERF_TEST(ThreadScaling, EfficiencyValidation) {
  if (config().threads < 2) {
    GTEST_SKIP() << "Test requires --threads 2 or higher";
  }

  ub::PerfCase perf{"ThreadScaling.EfficiencyValidation", config()};

  const std::size_t DATA_SIZE = 8192;

  // One vector, read by every worker, and its sum taken once. Each call sums
  // the vector again and writes to shared memory only when its sum is wrong,
  // so the workers' calls stay independent: no shared result variable and no
  // shared cache line written on the measured path.
  const auto data = test::makeTestData(DATA_SIZE);
  const std::uint64_t EXPECTED_SUM = test::sumBytes(data.data(), data.size());
  std::atomic<std::uint64_t> mismatches{0};
  const auto sumAndCheck = [&] {
    if (test::sumBytes(data.data(), data.size()) != EXPECTED_SUM) {
      mismatches.fetch_add(1, std::memory_order_relaxed);
    }
  };

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  // The same calls, first on the test's own thread, then on every worker
  auto baseline = perf.throughputLoop(sumAndCheck, "baseline");
  auto multithread = perf.contentionRun(sumAndCheck, "multithread");

  EXPECT_EQ(mismatches.load(), 0u) << "Calls computed a wrong sum";

  // Calculate speedup (ratio of throughputs)
  const double speedup = multithread.callsPerSecond / baseline.callsPerSecond;
  const double efficiency = speedup / 2.0; // 2 threads

  // Validate speedup (should be >1.2x accounting for overhead)
  EXPECT_GT(speedup, 1.2) << "Multi-threaded speedup too low: " << speedup << "x";

  // Validate efficiency (should be >50%)
  EXPECT_GT(efficiency, 0.50) << "Threading efficiency too low: " << (efficiency * 100.0) << "%";

  // Validate measurements are stable
  EXPECT_LT(multithread.stats.cv, 0.30) << "High variance in multi-threaded measurements";
}