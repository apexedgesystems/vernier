/**
 * @file BranchPrediction_pTest.cpp
 * @brief Branch prediction performance impact validation
 *
 * This test suite validates that the framework correctly measures performance
 * differences between branch-heavy and branchless code implementations, demonstrating
 * the framework's ability to capture CPU micro-architectural effects.
 *
 * Features tested:
 *  - Branchy conditional counting performance
 *  - Branchless arithmetic counting performance
 *  - Performance comparison and validation
 *
 * Expected behavior:
 *  - Branchless implementation should have equal or better performance
 *  - Performance difference should be measurable and consistent
 *  - Both implementations should produce stable measurements
 *
 * Usage:
 *   @code{.sh}
 *   # Run all branch prediction tests
 *   ./TestBenchSamples_PTEST --gtest_filter="BranchPrediction.*"
 *
 *   # Compare branchy vs branchless
 *   ./TestBenchSamples_PTEST --gtest_filter="BranchPrediction.*" --csv branch_compare.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~2 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <10% for typical workloads
 *
 * @see PerfCase
 * @see vernier::bench::test::countPositiveBranchy
 * @see vernier::bench::test::countPositiveBranchless
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
 * @brief Branch-heavy conditional counting performance
 *
 * Measures performance of counting positive values using conditional
 * branches. This represents typical code with unpredictable branches
 * that may suffer from branch misprediction penalties.
 *
 * @test BranchyCount
 *
 * Validates:
 *  - Branchy implementation executes correctly
 *  - Performance measurements are stable
 *  - Results are within reasonable performance bounds
 *
 * Expected performance:
 *  - Throughput > 1M calls/sec for typical data sizes
 */
PERF_TEST(BranchPrediction, BranchyCount) {
  ub::PerfCase perf{"BranchPrediction.BranchyCount", config()};

  const std::size_t COUNT = 8192;
  auto data = test::makeSignedData(COUNT);

  perf.warmup([&] {
    volatile auto result = test::countPositiveBranchy(data.data(), data.size());
    (void)result;
  });

  volatile std::size_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + test::countPositiveBranchy(data.data(), data.size()); }, "branchy");

  EXPECT_GT(result.callsPerSecond, 100000.0) << "Branchy counting unexpectedly slow";

  EXPECT_LT(result.stats.cv, 0.20) << "High variance in branchy measurements";

  (void)sink;
}

/**
 * @brief Branchless arithmetic counting performance
 *
 * Measures performance of counting positive values using branchless
 * arithmetic. This eliminates branch misprediction penalties and
 * should show equal or better performance than branchy version.
 *
 * @test BranchlessCount
 *
 * Validates:
 *  - Branchless implementation executes correctly
 *  - Performance is equal to or better than branchy version
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Throughput >= branchy version
 */
PERF_TEST(BranchPrediction, BranchlessCount) {
  ub::PerfCase perf{"BranchPrediction.BranchlessCount", config()};

  const std::size_t COUNT = 8192;
  auto data = test::makeSignedData(COUNT);

  perf.warmup([&] {
    volatile auto result = test::countPositiveBranchless(data.data(), data.size());
    (void)result;
  });

  volatile std::size_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + test::countPositiveBranchless(data.data(), data.size()); }, "branchless");

  EXPECT_GT(result.callsPerSecond, 100000.0) << "Branchless counting unexpectedly slow";

  EXPECT_LT(result.stats.cv, 0.20) << "High variance in branchless measurements";

  (void)sink;
}