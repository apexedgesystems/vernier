/**
 * @file 15_HeaptrackProfiler_Demo.cpp
 * @brief Demo 15: heaptrack low-overhead heap profiler -- find the alloc site.
 *
 * Where massif samples heap size over time, heaptrack records every allocation
 * with its call stack at low overhead, so its allocation-site view ranks the
 * exact lines responsible for the most allocations / bytes / temporaries.
 *
 * Two variants exercise the same work at different allocation intensities:
 *   Slow: PerIterAlloc -- a fresh std::vector (no reserve) grown each iteration
 *   Fast: PooledReserve -- one vector reserved up-front, cleared and reused
 *
 * Story: building a result vector by repeated push_back without reserve looks
 * harmless, but every iteration allocates -- and reallocates as the vector
 * grows. heaptrack's "most allocations" view pins the cost to the push_back
 * site; reserving once (and reusing the buffer) removes it.
 *
 * Usage:
 *   @code{.sh}
 *   # Slow path -- many allocations / temporaries:
 *   heaptrack -o /tmp/slow.heaptrack \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_15_HeaptrackProfiler \
 *       --profile heaptrack --cycles 50 --gtest_filter='Heaptrack.PerIterAlloc'
 *   heaptrack_print /tmp/slow.heaptrack.zst | head -40
 *
 *   # Fast path -- one allocation, reused:
 *   heaptrack -o /tmp/fast.heaptrack \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_15_HeaptrackProfiler \
 *       --profile heaptrack --cycles 50 --gtest_filter='Heaptrack.PooledReserve'
 *   heaptrack_print /tmp/fast.heaptrack.zst | head -40
 *   @endcode
 *
 * @see docs/21_HEAPTRACK_PROFILER.md for the step-by-step walkthrough.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

static constexpr std::size_t WORK_SIZE = 100'000;

/**
 * @test Slow: a fresh, unreserved vector grown by push_back each iteration.
 *
 * Every iteration allocates a new vector and reallocates it repeatedly as it
 * grows from 0 to WORK_SIZE. heaptrack's allocation-site view ranks this
 * push_back as the top allocator (calls and temporaries), pointing here.
 */
PERF_THROUGHPUT(Heaptrack, PerIterAlloc) {
  UB_PERF_GUARD(perf);

  perf.warmup([&] {
    std::vector<std::uint32_t> v;
    for (std::size_t i = 0; i < WORK_SIZE; ++i)
      v.push_back(static_cast<std::uint32_t>(i));
  });

  volatile std::uint32_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        // Fresh, unreserved vector each iteration -- repeated reallocation is
        // the cost heaptrack's allocation-site view reveals.
        std::vector<std::uint32_t> v;
        for (std::size_t i = 0; i < WORK_SIZE; ++i)
          v.push_back(static_cast<std::uint32_t>(i));
        sink = v[WORK_SIZE - 1];
      },
      "per_iter_alloc");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

/**
 * @test Fast: one vector reserved up-front, cleared and reused each iteration.
 *
 * Same arithmetic and same final contents; the only difference is that the
 * single up-front reserve removes the per-iteration allocations and growth
 * reallocations. heaptrack's allocation count for the loop drops to near zero.
 */
PERF_THROUGHPUT(Heaptrack, PooledReserve) {
  UB_PERF_GUARD(perf);

  // Allocate ONCE outside the measured loop, with the final capacity.
  std::vector<std::uint32_t> v;
  v.reserve(WORK_SIZE);

  perf.warmup([&] {
    v.clear();
    for (std::size_t i = 0; i < WORK_SIZE; ++i)
      v.push_back(static_cast<std::uint32_t>(i));
  });

  volatile std::uint32_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        // clear() keeps capacity, so push_back reuses the existing buffer --
        // no allocation happens inside the measured loop.
        v.clear();
        for (std::size_t i = 0; i < WORK_SIZE; ++i)
          v.push_back(static_cast<std::uint32_t>(i));
        sink = v[WORK_SIZE - 1];
      },
      "pooled_reserve");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

PERF_MAIN()
