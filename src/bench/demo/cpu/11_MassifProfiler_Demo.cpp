/**
 * @file 11_MassifProfiler_Demo.cpp
 * @brief Demo 11: Valgrind Massif heap profiler.
 *
 * Massif samples heap usage over time. Annotating which allocations dominate
 * peak memory is the canonical use: ms_print on the output file renders a
 * stacked timeline that pinpoints the largest contributors.
 *
 * Usage:
 *   @code{.sh}
 *   # Wrap externally with valgrind --tool=massif:
 *   valgrind --tool=massif \
 *       --massif-out-file=Massif.HeapWorkload.massif/massif.out \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_11_MassifProfiler \
 *       --profile massif --cycles 1 --gtest_filter='Massif.HeapWorkload'
 *
 *   # Render report:
 *   ms_print Massif.HeapWorkload.massif/massif.out | head -60
 *   @endcode
 *
 * Run unwrapped, the backend prints the wrap invocation hint and the
 * benchmark proceeds normally (no heap profile collected).
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

static constexpr std::size_t SMALL = 100'000;
static constexpr std::size_t MEDIUM = 1'000'000;

/**
 * @test Mixed-size allocations to populate the massif timeline with two
 *       distinct peaks (small + medium buffers).
 */
PERF_THROUGHPUT(Massif, HeapWorkload) {
  UB_PERF_GUARD(perf);

  perf.warmup([&] { (void)demo::makeRandomDoubles(SMALL); });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        // Several short-lived heap allocations of mixed sizes; ms_print
        // renders these as a step pattern in the heap timeline.
        auto small = std::make_unique<double[]>(SMALL);
        auto medium = std::make_unique<double[]>(MEDIUM);
        for (std::size_t i = 0; i < SMALL; ++i) small[i] = static_cast<double>(i);
        for (std::size_t i = 0; i < MEDIUM; ++i) medium[i] = static_cast<double>(i) * 0.5;
        sink = sink + small[SMALL - 1] + medium[MEDIUM - 1];
      },
      "heap_workload");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

PERF_MAIN()
