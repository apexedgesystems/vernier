/**
 * @file 05_BranchOptimization_Demo.cpp
 * @brief Demo 05: Branch prediction impact and branchless programming
 *
 * Demonstrates how unpredictable branches hurt performance and how
 * branchless coding eliminates the penalty. Shows three scenarios:
 *
 *  1. Branchy code + random data (50% mispredict rate = worst case)
 *  2. Branchy code + sorted data (predictor can learn pattern)
 *  3. Branchless code + random data (no branch = no mispredict)
 *
 * Usage:
 *   @code{.sh}
 *   # Run all three
 *   ./BenchDemo_05_BranchOptimization --csv results.csv
 *
 *   # Compare
 *   bench summary results.csv --sort median
 *
 *   # Profile branch misses
 *   ./BenchDemo_05_BranchOptimization --profile perf --gtest_filter="*RandomData*"
 *   @endcode
 *
 * @see docs/05_BRANCH_OPTIMIZATION.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

static constexpr std::size_t DATA_SIZE = 100000;
static constexpr double THRESHOLD = 0.5; // 50% of random [0,1) values exceed this

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Worst case: branchy code with random data.
 *
 * The branch `if (val > 0.5)` is taken 50% of the time in random order.
 * The CPU branch predictor cannot learn the pattern, resulting in ~50%
 * misprediction rate. Each mispredict costs ~15-20 cycles of pipeline flush.
 *
 * perf will show high branch-misses for this test.
 */
PERF_THROUGHPUT(BranchOptimization, BranchyRandomData) {
  UB_PERF_GUARD(perf);

  auto data = demo::makeRandomDoubles(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = demo::conditionalSumBranchy(data.data(), data.size(), THRESHOLD);
    (void)result;
  });

  volatile std::int64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + demo::conditionalSumBranchy(data.data(), data.size(), THRESHOLD); },
      "branchy_random");

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

/**
 * @test Better: branchy code with sorted data.
 *
 * When data is sorted, the branch `if (val > 0.5)` is taken for the
 * entire second half and not taken for the first half. The branch
 * predictor quickly learns the pattern (one transition point).
 *
 * This demonstrates why sorted data is faster for branchy algorithms --
 * the predictor achieves near-100% accuracy.
 */
PERF_THROUGHPUT(BranchOptimization, BranchySortedData) {
  UB_PERF_GUARD(perf);

  auto data = demo::makeSorted(demo::makeRandomDoubles(DATA_SIZE));

  perf.warmup([&] {
    volatile auto result = demo::conditionalSumBranchy(data.data(), data.size(), THRESHOLD);
    (void)result;
  });

  volatile std::int64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + demo::conditionalSumBranchy(data.data(), data.size(), THRESHOLD); },
      "branchy_sorted");

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

/**
 * @test Best: branchless code with random data.
 *
 * The branchless version uses multiply-by-predicate: `val * (val > threshold)`.
 * No branch instruction means no misprediction penalty, regardless of data order.
 * Performance is consistent whether data is sorted or random.
 *
 * perf will show near-zero branch-misses for this test.
 */
PERF_THROUGHPUT(BranchOptimization, BranchlessRandomData) {
  UB_PERF_GUARD(perf);

  auto data = demo::makeRandomDoubles(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = demo::conditionalSumBranchless(data.data(), data.size(), THRESHOLD);
    (void)result;
  });

  volatile std::int64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + demo::conditionalSumBranchless(data.data(), data.size(), THRESHOLD); },
      "branchless_random");

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

PERF_MAIN()
