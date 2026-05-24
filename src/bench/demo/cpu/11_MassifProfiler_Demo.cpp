/**
 * @file 11_MassifProfiler_Demo.cpp
 * @brief Demo 11: Valgrind Massif heap profiler -- find the allocation hotspot.
 *
 * Two variants exercise the same workload at different memory-allocation
 * intensities. Massif reveals the contrast and points at the fix.
 *
 *   Slow: SmallChurn      -- many short-lived heap allocations per iteration
 *   Fast: PooledReuse     -- one buffer allocated up-front, reused across iters
 *
 * Story: a hot loop calling new[]/delete[] every iteration looks innocent at
 * the source level but dominates the heap profile. Massif's allocation-site
 * timeline pins the cost; the fix (reuse a buffer) is a one-line change.
 *
 * Usage:
 *   @code{.sh}
 *   # Wrap externally with valgrind --tool=massif (slow + fast in one run):
 *   valgrind --tool=massif --massif-out-file=/tmp/slow.massif \
 *       ./BenchDemo_11_MassifProfiler \
 *       --profile massif --cycles 1 --gtest_filter='Massif.SmallChurn'
 *   ms_print /tmp/slow.massif | head -40
 *
 *   valgrind --tool=massif --massif-out-file=/tmp/fast.massif \
 *       ./BenchDemo_11_MassifProfiler \
 *       --profile massif --cycles 1 --gtest_filter='Massif.PooledReuse'
 *   ms_print /tmp/fast.massif | head -40
 *   @endcode
 *
 * @see docs/14_MASSIF_PROFILER.md for the step-by-step optimization walkthrough.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

static constexpr std::size_t WORK_SIZE = 1'000'000; // 8 MB at sizeof(double)

/**
 * @test Slow: a fresh heap allocation per iteration.
 *
 * Massif's `ms_print` output shows a sawtooth heap profile dominated by
 * std::vector / make_unique allocations from this site. Stacked timeline
 * makes the cost obvious.
 */
PERF_THROUGHPUT(Massif, SmallChurn) {
  UB_PERF_GUARD(perf);

  perf.warmup([&] {
    auto buf = std::make_unique<double[]>(WORK_SIZE);
    (void)buf;
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        // Fresh allocation each iteration -- this is the cost massif reveals.
        auto buf = std::make_unique<double[]>(WORK_SIZE);
        for (std::size_t i = 0; i < WORK_SIZE; ++i) buf[i] = static_cast<double>(i);
        sink = sink + buf[WORK_SIZE - 1];
      },
      "small_churn");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

/**
 * @test Fast: one allocation hoisted out of the loop, reused across iterations.
 *
 * Same arithmetic; the heap profile flattens dramatically because the
 * dominant allocation site has disappeared from the inner loop.
 */
PERF_THROUGHPUT(Massif, PooledReuse) {
  UB_PERF_GUARD(perf);

  // Allocate ONCE outside the measured loop.
  auto buf = std::make_unique<double[]>(WORK_SIZE);

  perf.warmup([&] {
    for (std::size_t i = 0; i < WORK_SIZE; ++i) buf[i] = static_cast<double>(i);
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        for (std::size_t i = 0; i < WORK_SIZE; ++i) buf[i] = static_cast<double>(i);
        sink = sink + buf[WORK_SIZE - 1];
      },
      "pooled_reuse");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

PERF_MAIN()
