/**
 * @file MeasuredAPI_pTest.cpp
 * @brief PerfCase::measured() API validation
 *
 * This test suite validates the measured() API, which provides single-shot
 * timing measurements. This API was previously untested and represents a
 * critical gap in framework coverage.
 *
 * Features tested:
 *  - measured() basic functionality
 *  - Consistency with throughputLoop()
 *  - Multiple measured() calls in sequence
 *  - Stats calculation correctness
 *  - Label propagation
 *
 * Expected behavior:
 *  - measured() produces valid statistical results
 *  - Results consistent with throughputLoop() for same workload
 *  - Multiple calls work correctly
 *  - Statistics are mathematically sound
 *
 * Usage:
 *   @code{.sh}
 *   # Run all measured API tests
 *   ./TestBenchSamples_PTEST --gtest_filter="MeasuredAPI.*"
 *
 *   # With CSV output
 *   ./TestBenchSamples_PTEST --gtest_filter="MeasuredAPI.*" --csv measured.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~3 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <30% for typical workloads
 *
 * @see PerfCase::measured
 * @see PerfCase::throughputLoop
 */

#include <gtest/gtest.h>
#include <cstdint>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

} // anonymous namespace

/**
 * @brief Basic measured() API functionality
 *
 * Validates that measured() executes correctly and produces valid
 * statistical results for a simple workload.
 *
 * @test BasicMeasurement
 *
 * Validates:
 *  - measured() executes without error
 *  - Stats structure is populated
 *  - All statistical measures are positive and finite
 *  - Throughput is calculated correctly
 *
 * Expected performance:
 *  - Throughput >100K calls/sec
 *  - CV <30%
 */
PERF_TEST(MeasuredAPI, BasicMeasurement) {
  ub::PerfCase perf{"MeasuredAPI.BasicMeasurement", config()};

  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile auto result = test::sumBytes(data.data(), data.size());
      (void)result;
    }
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink = sink + test::sumBytes(data.data(), data.size());
        }
      },
      "basic");

  // Validate stats structure populated
  EXPECT_GT(result.stats.median, 0.0) << "Median should be positive";
  EXPECT_GT(result.stats.mean, 0.0) << "Mean should be positive";
  EXPECT_GE(result.stats.stddev, 0.0) << "Stddev should be non-negative";
  EXPECT_GE(result.stats.cv, 0.0) << "CV should be non-negative";

  // Validate statistical consistency
  EXPECT_LE(result.stats.min, result.stats.median) << "min <= median";
  EXPECT_LE(result.stats.median, result.stats.max) << "median <= max";

  // Validate throughput calculation
  EXPECT_GT(result.callsPerSecond, 100.0) << "Throughput suspiciously low";

  // Validate label
  EXPECT_EQ(result.label, "basic") << "Label should match what was provided";

  (void)sink;
}

/**
 * @brief Consistency between measured() and throughputLoop()
 *
 * Validates that measured() and throughputLoop() produce consistent results
 * for the same workload. They should have similar median latency.
 *
 * @test ConsistencyWithThroughputLoop
 *
 * Validates:
 *  - Both APIs measure same workload consistently
 *  - Median latency within 50% of each other
 *  - Both produce stable measurements
 *
 * Expected performance:
 *  - Results from both APIs should be similar
 */
PERF_TEST(MeasuredAPI, ConsistencyWithThroughputLoop) {
  ub::PerfCase perf{"MeasuredAPI.ConsistencyWithThroughputLoop", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile auto result = test::sumBytes(data.data(), data.size());
      (void)result;
    }
  });

  // Measure with measured() API
  volatile std::uint64_t sink1 = 0;
  auto resultMeasured = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink1 = sink1 + test::sumBytes(data.data(), data.size());
        }
      },
      "measured");

  // Measure with throughputLoop() API
  volatile std::uint64_t sink2 = 0;
  auto resultThroughput = perf.throughputLoop(
      [&] { sink2 = sink2 + test::sumBytes(data.data(), data.size()); }, "throughput");

  // Results should be within 50% of each other (loose tolerance for variance)
  const double ratio = resultMeasured.stats.median / resultThroughput.stats.median;
  EXPECT_GT(ratio, 0.5) << "measured() median much lower than throughputLoop()";
  EXPECT_LT(ratio, 2.0) << "measured() median much higher than throughputLoop()";

  // Both should produce stable measurements
  EXPECT_LT(resultMeasured.stats.cv, 0.35) << "High variance in measured()";
  EXPECT_LT(resultThroughput.stats.cv, 0.35) << "High variance in throughputLoop()";

  (void)sink1;
  (void)sink2;
}

/**
 * @brief Multiple sequential measured() calls
 *
 * Validates that multiple measured() calls can be made in sequence
 * and each produces valid independent results.
 *
 * @test MultipleSequentialCalls
 *
 * Validates:
 *  - Multiple measured() calls work correctly
 *  - Each call produces independent valid results
 *  - No interference between calls
 *
 * Expected performance:
 *  - All calls succeed
 *  - Results are independent
 */
PERF_TEST(MeasuredAPI, MultipleSequentialCalls) {
  ub::PerfCase perf{"MeasuredAPI.MultipleSequentialCalls", config()};

  const std::size_t DATA_SIZE = 2048;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile auto result = test::sumBytes(data.data(), data.size());
      (void)result;
    }
  });

  // First measured call
  volatile std::uint64_t sink1 = 0;
  auto result1 = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink1 = sink1 + test::sumBytes(data.data(), data.size());
        }
      },
      "call1");

  // Second measured call
  volatile std::uint64_t sink2 = 0;
  auto result2 = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink2 = sink2 + test::sumBytes(data.data(), data.size());
        }
      },
      "call2");

  // Third measured call
  volatile std::uint64_t sink3 = 0;
  auto result3 = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink3 = sink3 + test::sumBytes(data.data(), data.size());
        }
      },
      "call3");

  // All calls should succeed
  EXPECT_GT(result1.callsPerSecond, 0.0) << "Call 1 failed";
  EXPECT_GT(result2.callsPerSecond, 0.0) << "Call 2 failed";
  EXPECT_GT(result3.callsPerSecond, 0.0) << "Call 3 failed";

  // Labels should be correct
  EXPECT_EQ(result1.label, "call1");
  EXPECT_EQ(result2.label, "call2");
  EXPECT_EQ(result3.label, "call3");

  // All should have reasonable CV
  EXPECT_LT(result1.stats.cv, 0.35);
  EXPECT_LT(result2.stats.cv, 0.35);
  EXPECT_LT(result3.stats.cv, 0.35);

  (void)sink1;
  (void)sink2;
  (void)sink3;
}

/**
 * @brief CV calculation correctness
 *
 * Validates that the coefficient of variation is calculated correctly
 * as stddev/mean.
 *
 * @test CVCalculation
 *
 * Validates:
 *  - CV equals stddev/mean (within floating point tolerance)
 *  - CV is mathematically consistent
 *
 * Expected performance:
 *  - CV calculation accurate to 0.1%
 */
PERF_TEST(MeasuredAPI, CVCalculation) {
  ub::PerfCase perf{"MeasuredAPI.CVCalculation", config()};

  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile auto result = test::sumBytes(data.data(), data.size());
      (void)result;
    }
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink = sink + test::sumBytes(data.data(), data.size());
        }
      },
      "cv_check");

  const double expectedCV = result.stats.stddev / result.stats.mean;

  EXPECT_NEAR(result.stats.cv, expectedCV, 0.001) << "CV should equal stddev/mean";

  (void)sink;
}

/**
 * @brief Percentile ordering validation
 *
 * Validates that percentiles follow correct mathematical ordering:
 * min <= p10 <= median <= p90 <= max
 *
 * @test PercentileOrdering
 *
 * Validates:
 *  - Percentile values correctly ordered
 *  - Statistical consistency
 *
 * Expected performance:
 *  - All ordering constraints satisfied
 */
PERF_TEST(MeasuredAPI, PercentileOrdering) {
  ub::PerfCase perf{"MeasuredAPI.PercentileOrdering", config()};

  const std::size_t DATA_SIZE = 8192;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile auto result = test::sumBytes(data.data(), data.size());
      (void)result;
    }
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.measured(
      [&] {
        for (int i = 0; i < perf.cycles(); ++i) {
          sink = sink + test::sumBytes(data.data(), data.size());
        }
      },
      "percentiles");

  const auto& s = result.stats;

  EXPECT_LE(s.min, s.p10) << "min <= p10";
  EXPECT_LE(s.p10, s.median) << "p10 <= median";
  EXPECT_LE(s.median, s.p90) << "median <= p90";
  EXPECT_LE(s.p90, s.max) << "p90 <= max";

  (void)sink;
}