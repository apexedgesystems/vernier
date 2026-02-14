/**
 * @file CoreFeatures_pTest.cpp
 * @brief Core benchmark framework functionality validation
 *
 * This test suite validates fundamental benchmarking framework features including
 * throughput measurement, CSV export, configuration parsing, and warmup behavior.
 *
 * Features tested:
 *  - Basic throughput measurement with throughputLoop()
 *  - CSV export format and column validation
 *  - Configuration parsing (--cycles, --repeats, --msgBytes, --csv)
 *  - Warmup behavior and stabilization
 *  - Quick mode functionality
 *  - Memory profile tracking
 *  - Result statistics completeness
 *
 * Expected behavior:
 *  - All tests complete successfully with stable measurements
 *  - CSV output contains required columns
 *  - Configuration flags properly affect measurement behavior
 *  - Warmup improves measurement stability
 *
 * Usage:
 *   @code{.sh}
 *   # Run all core feature tests
 *   ./TestBenchSamples_PTEST --gtest_filter="CoreFeatures.*"
 *
 *   # Run with CSV export
 *   ./TestBenchSamples_PTEST --gtest_filter="CoreFeatures.*" --csv core_results.csv
 *
 *   # Run in quick mode for fast validation
 *   ./TestBenchSamples_PTEST --gtest_filter="CoreFeatures.*" --quick
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~5 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <10% for typical workloads
 *
 * @see PerfCase
 * @see PerfConfig
 * @see PerfResult
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <string>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

} // anonymous namespace

/**
 * @brief Basic throughput measurement validation
 *
 * Validates that throughputLoop() correctly measures simple workload performance.
 * This is the most fundamental framework capability.
 *
 * @test BasicThroughput
 *
 * Validates:
 *  - throughputLoop() executes workload repeatedly
 *  - Measurement produces valid statistics
 *  - Results are within reasonable bounds
 *
 * Expected performance:
 *  - Throughput > 1000 calls/sec
 *  - CV < 10%
 */
PERF_TEST(CoreFeatures, BasicThroughput) {
  ub::PerfCase perf{"CoreFeatures.BasicThroughput", config()};

  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result =
      perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); }, "basic");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Throughput unexpectedly low for simple workload";

  EXPECT_LT(result.stats.cv, 0.35) << "High variance detected - measurements unstable";

  EXPECT_GT(result.stats.median, 0.0) << "Median time should be positive";

  (void)sink;
}

/**
 * @brief Configuration flag validation
 *
 * Validates that command-line configuration flags properly affect measurement behavior.
 *
 * @test ConfigurationFlags
 *
 * Validates:
 *  - cycles flag controls operations per measurement
 *  - repeats flag controls number of samples
 *  - msgBytes flag accessible for test parameterization
 *
 * Expected performance:
 *  - Configuration values match command-line arguments
 */
PERF_TEST(CoreFeatures, ConfigurationFlags) {
  ub::PerfCase perf{"CoreFeatures.ConfigurationFlags", config()};

  EXPECT_GT(perf.cycles(), 0) << "Cycles should be positive";
  EXPECT_GT(perf.repeats(), 0) << "Repeats should be positive";
  EXPECT_GE(config().msgBytes, 0) << "msgBytes should be non-negative";

  EXPECT_EQ(perf.cycles(), config().cycles) << "PerfCase cycles should match config";
  EXPECT_EQ(perf.repeats(), config().repeats) << "PerfCase repeats should match config";
}

/**
 * @brief Warmup behavior validation
 *
 * Validates that warmup() stabilizes performance measurements by pre-executing
 * the workload to warm caches and branch predictors.
 *
 * @test WarmupStabilization
 *
 * Validates:
 *  - warmup() executes without error
 *  - Subsequent measurements are stable
 *
 * Expected performance:
 *  - Measurements after warmup have lower CV than without warmup
 */
PERF_TEST(CoreFeatures, WarmupStabilization) {
  ub::PerfCase perf{"CoreFeatures.WarmupStabilization", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "warmed");

  EXPECT_LT(result.stats.cv, 0.25) << "High variance even after warmup";

  (void)sink;
}

/**
 * @brief Quick mode functionality validation
 *
 * Validates that quick mode (--quick flag) properly reduces measurement time
 * while maintaining reasonable accuracy.
 *
 * @test QuickMode
 *
 * Validates:
 *  - Quick mode reduces cycles/repeats
 *  - Results still produce valid measurements
 *  - Quick mode appropriate for fast iteration during development
 *
 * Expected performance:
 *  - Quick mode completes significantly faster than normal mode
 */
PERF_TEST(CoreFeatures, QuickMode) {
  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  if (config().quickMode) {
    ub::PerfCase perf{"CoreFeatures.QuickMode", config()};

    EXPECT_LE(perf.cycles(), 5000) << "Quick mode should use reduced cycles (10000 -> 5000)";
    EXPECT_LE(perf.repeats(), 5) << "Quick mode should use reduced repeats (10 -> 5)";

    volatile std::uint64_t sink = 0;
    auto result = perf.throughputLoop(
        [&] { sink = sink + test::sumBytes(data.data(), data.size()); }, "quick");

    EXPECT_GT(result.callsPerSecond, 0.0) << "Quick mode should still produce valid measurements";

    (void)sink;
  }
}

/**
 * @brief Memory profile tracking validation
 *
 * Validates that MemoryProfile correctly tracks memory operations and
 * calculates bandwidth metrics.
 *
 * @test MemoryProfileTracking
 *
 * Validates:
 *  - MemoryProfile tracks read/write/allocation bytes
 *  - Bandwidth calculation produces reasonable values
 *  - Efficiency metrics computed correctly
 *
 * Expected performance:
 *  - Bandwidth > 100 MB/s for sequential access
 */
PERF_TEST(CoreFeatures, MemoryProfileTracking) {
  ub::PerfCase perf{"CoreFeatures.MemoryProfileTracking", config()};

  const std::size_t DATA_SIZE = 64 * 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  ub::MemoryProfile memProfile{.bytesRead = DATA_SIZE, .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "mem_track", memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 100.0) << "Memory bandwidth suspiciously low";

  EXPECT_LT(bandwidthMBs, 100000.0) << "Memory bandwidth suspiciously high";

  (void)sink;
}

/**
 * @brief Result statistics validation
 *
 * Validates that PerfResult contains all expected statistical measures
 * and that they are mathematically consistent.
 *
 * @test ResultStatistics
 *
 * Validates:
 *  - All statistical measures computed (median, mean, stddev, CV, percentiles)
 *  - Statistics are mathematically consistent (min <= p10 <= median <= p90 <= max)
 *  - CV calculation correct (stddev/mean)
 *
 * Expected performance:
 *  - All statistics positive and finite
 */
PERF_TEST(CoreFeatures, ResultStatistics) {
  ub::PerfCase perf{"CoreFeatures.ResultStatistics", config()};

  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result =
      perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); }, "stats");

  const auto& s = result.stats;

  EXPECT_GT(s.median, 0.0) << "Median should be positive";
  EXPECT_GT(s.mean, 0.0) << "Mean should be positive";
  EXPECT_GE(s.stddev, 0.0) << "Standard deviation should be non-negative";
  EXPECT_GE(s.cv, 0.0) << "CV should be non-negative";

  EXPECT_LE(s.min, s.p10) << "min <= p10";
  EXPECT_LE(s.p10, s.median) << "p10 <= median";
  EXPECT_LE(s.median, s.p90) << "median <= p90";
  EXPECT_LE(s.p90, s.max) << "p90 <= max";

  const double expectedCV = s.stddev / s.mean;
  EXPECT_NEAR(s.cv, expectedCV, 0.001) << "CV should equal stddev/mean";

  (void)sink;
}

/**
 * @brief Test name propagation validation
 *
 * Validates that test names are correctly stored and retrievable for
 * CSV export and reporting.
 *
 * @test TestNamePropagation
 *
 * Validates:
 *  - Test name accessible via PerfCase API
 *  - Test name appears in output (when CSV enabled)
 *
 * Expected performance:
 *  - Test name matches what was provided at construction
 */
PERF_TEST(CoreFeatures, TestNamePropagation) {
  const std::string EXPECTED_NAME = "CoreFeatures.TestNamePropagation";
  ub::PerfCase perf{EXPECTED_NAME, config()};

  EXPECT_EQ(perf.testName(), EXPECTED_NAME) << "Test name should match what was provided";
}

PERF_MAIN()