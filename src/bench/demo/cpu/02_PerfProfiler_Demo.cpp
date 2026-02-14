/**
 * @file 02_PerfProfiler_Demo.cpp
 * @brief Demo 02: Linux perf profiler for cache miss detection
 *
 * Demonstrates using the perf profiler to identify cache-hostile access
 * patterns, then fixing them with sequential access.
 *
 * Slow: stride-512 array walk (skips 8 cache lines per access)
 * Fast: sequential array walk (hardware prefetcher keeps up)
 *
 * Usage:
 *   @code{.sh}
 *   # Baseline without profiling
 *   ./BenchDemo_02_PerfProfiler --csv baseline.csv
 *
 *   # Profile the slow path
 *   ./BenchDemo_02_PerfProfiler --profile perf --gtest_filter="*StridedAccess*"
 *
 *   # Profile the fast path
 *   ./BenchDemo_02_PerfProfiler --profile perf --gtest_filter="*SequentialAccess*"
 *
 *   # Compare
 *   bench summary baseline.csv
 *   @endcode
 *
 * @see docs/02_PERF_PROFILER.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

// 4 MB array: large enough to exceed L2 cache (typically 256 KB - 1 MB)
static constexpr std::size_t ARRAY_SIZE = 4 * 1024 * 1024;
static constexpr std::size_t STRIDE = 512; // Skip 8 cache lines per access

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: stride-512 access pattern causes constant cache misses.
 *
 * Every access skips 512 bytes (8 cache lines). The hardware prefetcher
 * cannot predict the pattern, resulting in high L1-dcache-load-misses.
 * This is the kind of access pattern perf will flag immediately.
 */
PERF_THROUGHPUT(PerfProfiler, StridedAccess) {
  UB_PERF_GUARD(perf);

  std::vector<std::uint8_t> data(ARRAY_SIZE);
  // Fill with non-zero data to prevent zero-page optimization
  for (std::size_t i = 0; i < ARRAY_SIZE; ++i) {
    data[i] = static_cast<std::uint8_t>(i & 0xFF);
  }

  perf.warmup([&] {
    volatile auto result = demo::stridedArrayWalk(data.data(), data.size(), STRIDE);
    (void)result;
  });

  ub::MemoryProfile memProfile{
      .bytesRead = ARRAY_SIZE / STRIDE, .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + demo::stridedArrayWalk(data.data(), data.size(), STRIDE); },
      "strided_512", memProfile);

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

/**
 * @test Fast: sequential access pattern with hardware prefetching.
 *
 * Sequential access is the ideal case for the hardware prefetcher.
 * L1-dcache-load-misses will be dramatically lower than the strided version.
 * perf will show near-zero cache miss rate.
 */
PERF_THROUGHPUT(PerfProfiler, SequentialAccess) {
  UB_PERF_GUARD(perf);

  std::vector<std::uint8_t> data(ARRAY_SIZE);
  for (std::size_t i = 0; i < ARRAY_SIZE; ++i) {
    data[i] = static_cast<std::uint8_t>(i & 0xFF);
  }

  perf.warmup([&] {
    volatile auto result = demo::sequentialArrayWalk(data.data(), data.size());
    (void)result;
  });

  ub::MemoryProfile memProfile{.bytesRead = ARRAY_SIZE, .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + demo::sequentialArrayWalk(data.data(), data.size()); }, "sequential",
      memProfile);

  EXPECT_GT(result.callsPerSecond, 10.0);

  // Sequential should be faster due to prefetching
  EXPECT_LT(result.stats.cv, 0.30);

  (void)sink;
}

PERF_MAIN()
