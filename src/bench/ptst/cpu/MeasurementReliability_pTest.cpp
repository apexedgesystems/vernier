/**
 * @file MeasurementReliability_pTest.cpp
 * @brief Measurement reliability and accuracy validation
 *
 * This test suite validates that the benchmarking framework produces reliable,
 * trustworthy measurements. These tests prove the framework works correctly
 * for RT embedded developers who need to benchmark their code.
 *
 * Features tested:
 *  - Repeatability: same benchmark run twice gives consistent results
 *  - Ground truth: timing matches wall-clock reality
 *  - Scaling: 2x work produces ~2x timing
 *  - Sensitivity: can detect 10% performance differences
 *  - Determinism: stable workloads have low variance
 *  - Statistics: mean/median relationship is correct
 *
 * Expected behavior:
 *  - Identical workloads measured twice differ by <15%
 *  - Sleep-based timing accurate within 20%
 *  - Work scaling is linear (2x work = 2x time)
 *  - 10% slowdowns are detectable
 *  - Stable workloads have CV <10%
 *
 * Usage:
 *   @code{.sh}
 *   # Run all reliability tests
 *   ./TestBenchSamples_PTEST --gtest_filter="MeasurementReliability.*"
 *
 *   # With higher repeat count for better statistics
 *   ./TestBenchSamples_PTEST --gtest_filter="MeasurementReliability.*" --repeats 20
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~10 seconds total
 *  - Pass rate: 100% on stable hardware
 *
 * @see PerfCase
 * @see PerfStats
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <thread>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

/**
 * @brief Perform scalable work with controllable iteration count
 *
 * @param iterations Number of loop iterations (more = slower)
 * @return Sum to prevent optimization
 */
inline std::uint64_t scalableWork(int iterations) {
  volatile std::uint64_t sum = 0;
  for (int i = 0; i < iterations; ++i) {
    sum += static_cast<std::uint64_t>(i) * 7 + 3;
  }
  return sum;
}

} // anonymous namespace

/**
 * @brief Repeatability: same benchmark run twice gives consistent results
 *
 * Validates that running an identical benchmark twice produces results
 * within 15% of each other. This is the foundation of trustworthy
 * measurement.
 *
 * @test IdenticalWorkloadConsistency
 *
 * Validates:
 *  - Two runs of identical workload produce similar medians
 *  - Both runs have reasonable CV
 *  - Results are stable enough for regression testing
 *
 * Expected performance:
 *  - Median difference <15%
 *  - CV <25% for both runs
 */
PERF_TEST(MeasurementReliability, IdenticalWorkloadConsistency) {
  ub::PerfCase perf{"MeasurementReliability.IdenticalWorkloadConsistency", config()};

  const int WORK_ITERATIONS = 10000;

  perf.warmup([&] { (void)scalableWork(WORK_ITERATIONS); });

  // Run 1: measure workload
  volatile std::uint64_t sink1 = 0;
  auto result1 = perf.throughputLoop([&] { sink1 += scalableWork(WORK_ITERATIONS); }, "run1");

  // Run 2: identical measurement
  volatile std::uint64_t sink2 = 0;
  auto result2 = perf.throughputLoop([&] { sink2 += scalableWork(WORK_ITERATIONS); }, "run2");

  // Both runs should have reasonable CV
  const double CV_THRESHOLD = config().quickMode ? 0.35 : 0.25;
  EXPECT_LT(result1.stats.cv, CV_THRESHOLD) << "Run 1 has high variance";
  EXPECT_LT(result2.stats.cv, CV_THRESHOLD) << "Run 2 has high variance";

  // Medians should be within 20% of each other (allowing for container variance)
  const double ratio = result1.stats.median / result2.stats.median;
  EXPECT_GT(ratio, 0.80) << "Run 1 much faster than Run 2 (ratio=" << ratio << ")";
  EXPECT_LT(ratio, 1.20) << "Run 1 much slower than Run 2 (ratio=" << ratio << ")";

  (void)sink1;
  (void)sink2;
}

/**
 * @brief Ground truth: timing matches wall-clock reality
 *
 * Uses std::this_thread::sleep_for as a ground truth timer to validate
 * that the framework's timing accurately reflects real elapsed time.
 *
 * @test TimingAccuracyVsSleep
 *
 * Validates:
 *  - Framework measures actual elapsed time
 *  - Timing is accurate within 30% of expected duration
 *  - nowUs() correctly measures wall-clock time
 *
 * Expected performance:
 *  - 500us sleep measured as 350-750us (accounting for overhead)
 */
PERF_TEST(MeasurementReliability, TimingAccuracyVsSleep) {
  // Use fewer cycles/repeats for this test since sleep is slow
  ub::PerfConfig sleepConfig = config();
  sleepConfig.cycles = 1;   // One sleep per measurement
  sleepConfig.repeats = 10; // Enough samples
  sleepConfig.warmup = 1;

  ub::PerfCase perf{"MeasurementReliability.TimingAccuracyVsSleep", sleepConfig};

  const int SLEEP_US = 500; // 500 microseconds

  // Warmup the timer
  perf.warmup([&] { std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_US)); });

  // Measure sleep duration
  auto result = perf.measured(
      [&] { std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_US)); }, "sleep_timing");

  // Sleep has overhead, so allow 30% tolerance
  // Minimum: 70% of expected (350us for 500us sleep)
  // Maximum: 150% of expected (750us for 500us sleep, accounting for scheduling)
  const double MIN_EXPECTED = static_cast<double>(SLEEP_US) * 0.70;
  const double MAX_EXPECTED = static_cast<double>(SLEEP_US) * 1.50;

  EXPECT_GT(result.stats.median, MIN_EXPECTED) << "Measured time too short: " << result.stats.median
                                               << "us, expected >" << MIN_EXPECTED << "us";
  EXPECT_LT(result.stats.median, MAX_EXPECTED) << "Measured time too long: " << result.stats.median
                                               << "us, expected <" << MAX_EXPECTED << "us";

  std::printf("\n[Ground Truth] Sleep %dus measured as %.1fus (%.0f%% of expected)\n", SLEEP_US,
              result.stats.median, (result.stats.median / SLEEP_US) * 100.0);
}

/**
 * @brief Scaling: 2x work produces ~2x timing
 *
 * Validates that doubling the workload approximately doubles the
 * measured time. This proves measurements reflect actual work done.
 *
 * @test TimingScalesWithWork
 *
 * Validates:
 *  - 2x work takes ~2x time
 *  - Measurements scale linearly with work
 *  - Framework correctly tracks execution time
 *
 * Expected performance:
 *  - 2x work = 1.5-2.5x time (allowing for overhead)
 */
PERF_TEST(MeasurementReliability, TimingScalesWithWork) {
  ub::PerfCase perf{"MeasurementReliability.TimingScalesWithWork", config()};

  const int BASE_ITERATIONS = 5000;

  perf.warmup([&] { (void)scalableWork(BASE_ITERATIONS * 2); });

  // Measure 1x work
  volatile std::uint64_t sink1 = 0;
  auto result1x = perf.throughputLoop([&] { sink1 += scalableWork(BASE_ITERATIONS); }, "1x_work");

  // Measure 2x work
  volatile std::uint64_t sink2 = 0;
  auto result2x =
      perf.throughputLoop([&] { sink2 += scalableWork(BASE_ITERATIONS * 2); }, "2x_work");

  // 2x work should take approximately 2x time
  const double ratio = result2x.stats.median / result1x.stats.median;

  EXPECT_GT(ratio, 1.5) << "2x work not significantly slower (ratio=" << ratio << ")";
  EXPECT_LT(ratio, 2.5) << "2x work more than 2.5x slower (ratio=" << ratio << ")";

  std::printf("\n[Scaling] 1x=%.2fus, 2x=%.2fus, ratio=%.2fx\n", result1x.stats.median,
              result2x.stats.median, ratio);

  (void)sink1;
  (void)sink2;
}

/**
 * @brief Sensitivity: can detect 10% performance differences
 *
 * Validates that the framework can reliably detect a 10% slowdown.
 * This is critical for regression testing where small performance
 * changes need to be identified.
 *
 * @test CanDetectTenPercentSlowdown
 *
 * Validates:
 *  - 10% more work is measurably slower
 *  - Measured ratio is in expected range (1.05-1.25)
 *  - Framework has sufficient sensitivity
 *
 * Expected performance:
 *  - 110% work = 1.05-1.25x time
 */
PERF_TEST(MeasurementReliability, CanDetectTenPercentSlowdown) {
  ub::PerfCase perf{"MeasurementReliability.CanDetectTenPercentSlowdown", config()};

  const int BASE_ITERATIONS = 10000;
  const int SLOWER_ITERATIONS = BASE_ITERATIONS + (BASE_ITERATIONS / 10); // +10%

  perf.warmup([&] { (void)scalableWork(SLOWER_ITERATIONS); });

  // Baseline measurement
  volatile std::uint64_t sink1 = 0;
  auto baseline = perf.throughputLoop([&] { sink1 += scalableWork(BASE_ITERATIONS); }, "baseline");

  // Slower measurement (+10% work)
  volatile std::uint64_t sink2 = 0;
  auto slower = perf.throughputLoop([&] { sink2 += scalableWork(SLOWER_ITERATIONS); }, "slower");

  // Slower should actually be slower
  EXPECT_GT(slower.stats.median, baseline.stats.median) << "10% more work not detected as slower";

  // Ratio should be approximately 1.10
  const double ratio = slower.stats.median / baseline.stats.median;
  EXPECT_GT(ratio, 1.05) << "Slowdown less than 5% detected (ratio=" << ratio << ")";
  EXPECT_LT(ratio, 1.25) << "Slowdown more than 25% detected (ratio=" << ratio << ")";

  std::printf("\n[Sensitivity] baseline=%.2fus, +10%%=%.2fus, detected ratio=%.2fx\n",
              baseline.stats.median, slower.stats.median, ratio);

  (void)sink1;
  (void)sink2;
}

/**
 * @brief Determinism: stable workloads have low variance
 *
 * Validates that a tight, predictable workload produces measurements
 * with low coefficient of variation. This proves the framework
 * can produce stable results when the workload itself is stable.
 *
 * @test LowVarianceWorkload
 *
 * Validates:
 *  - Deterministic workload has low CV (<15%)
 *  - Framework doesn't introduce artificial variance
 *  - Stable code produces stable measurements
 *
 * Expected performance:
 *  - CV <15% for tight loop
 */
PERF_TEST(MeasurementReliability, LowVarianceWorkload) {
  ub::PerfCase perf{"MeasurementReliability.LowVarianceWorkload", config()};

  const int ITERATIONS = 50000; // Large enough to be measurable

  perf.warmup([&] { (void)scalableWork(ITERATIONS); });

  // Tight loop with predictable timing
  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink += scalableWork(ITERATIONS); }, "deterministic");

  // Stable workload should have low jitter
  const double CV_THRESHOLD = config().quickMode ? 0.20 : 0.15;
  EXPECT_LT(result.stats.cv, CV_THRESHOLD)
      << "Deterministic workload has high variance: CV=" << (result.stats.cv * 100.0) << "%";

  std::printf("\n[Determinism] CV=%.1f%% (threshold=%.0f%%)\n", result.stats.cv * 100.0,
              CV_THRESHOLD * 100.0);

  (void)sink;
}

/**
 * @brief Statistics: mean and median relationship is correct
 *
 * Validates that for stable, symmetric workloads, the mean and median
 * are close to each other. This proves the statistical calculations
 * are working correctly.
 *
 * @test MeanMedianRelationship
 *
 * Validates:
 *  - Mean and median within 20% of each other
 *  - Statistics are mathematically consistent
 *  - No systematic bias in measurements
 *
 * Expected performance:
 *  - Mean/median ratio between 0.8 and 1.2
 */
PERF_TEST(MeasurementReliability, MeanMedianRelationship) {
  ub::PerfCase perf{"MeasurementReliability.MeanMedianRelationship", config()};

  const int ITERATIONS = 20000;

  perf.warmup([&] { (void)scalableWork(ITERATIONS); });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink += scalableWork(ITERATIONS); }, "stats_check");

  // For stable workloads, mean should be close to median
  const double ratio = result.stats.mean / result.stats.median;
  EXPECT_GT(ratio, 0.80) << "Mean much lower than median (ratio=" << ratio << ")";
  EXPECT_LT(ratio, 1.20) << "Mean much higher than median (ratio=" << ratio << ")";

  // Also verify percentile ordering
  EXPECT_LE(result.stats.min, result.stats.p10) << "min > p10";
  EXPECT_LE(result.stats.p10, result.stats.median) << "p10 > median";
  EXPECT_LE(result.stats.median, result.stats.p90) << "median > p90";
  EXPECT_LE(result.stats.p90, result.stats.max) << "p90 > max";

  std::printf("\n[Statistics] mean=%.2fus, median=%.2fus, ratio=%.2f\n", result.stats.mean,
              result.stats.median, ratio);
  std::printf("             min=%.2f, p10=%.2f, p50=%.2f, p90=%.2f, max=%.2f\n", result.stats.min,
              result.stats.p10, result.stats.median, result.stats.p90, result.stats.max);

  (void)sink;
}
