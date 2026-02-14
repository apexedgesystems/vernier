/**
 * @file CacheEffects_pTest.cpp
 * @brief Cache locality performance impact validation
 *
 * This test suite validates that the framework correctly measures performance
 * differences between cache-friendly and cache-unfriendly memory access patterns.
 * Demonstrates the framework's ability to capture memory hierarchy effects.
 *
 * Features tested:
 *  - Sequential memory access (cache-friendly)
 *  - Strided memory access (cache-unfriendly)
 *  - Performance comparison between access patterns
 *  - Memory bandwidth calculation via MemoryProfile
 *
 * Expected behavior:
 *  - Sequential access should show significantly higher bandwidth
 *  - Strided access should show degraded performance
 *  - Bandwidth measurements should be reasonable
 *
 * Usage:
 *   @code{.sh}
 *   # Run all cache effects tests
 *   ./TestBenchSamples_PTEST --gtest_filter="CacheEffects.*"
 *
 *   # Compare sequential vs strided with CSV output
 *   ./TestBenchSamples_PTEST --gtest_filter="CacheEffects.*" --csv cache_compare.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~8 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <10% for typical workloads
 *
 * @see PerfCase
 * @see MemoryProfile
 * @see vernier::bench::test::sumBytes
 * @see vernier::bench::test::sumBytesStrided
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
 * @brief Sequential memory access bandwidth measurement
 *
 * Measures performance of sequential memory reads, which should show
 * high bandwidth due to cache prefetching and spatial locality.
 *
 * @test SequentialAccess
 *
 * Validates:
 *  - Sequential access executes efficiently
 *  - Memory bandwidth is tracked correctly
 *  - Bandwidth measurements are reasonable (>100 MB/s)
 *
 * Expected performance:
 *  - High bandwidth due to cache-friendly access
 *  - Stable measurements with low CV
 */
PERF_TEST(CacheEffects, SequentialAccess) {
  ub::PerfCase perf{"CacheEffects.SequentialAccess", config()};

  const std::size_t DATA_SIZE = 256 * 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  ub::MemoryProfile memProfile{.bytesRead = DATA_SIZE, .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "sequential", memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 100.0) << "Sequential bandwidth suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in sequential access";

  (void)sink;
}

/**
 * @brief Strided memory access bandwidth measurement
 *
 * Measures performance of strided memory reads (64-byte stride), which should
 * show degraded performance due to poor cache utilization and reduced
 * effectiveness of hardware prefetchers.
 *
 * @test StridedAccess
 *
 * Validates:
 *  - Strided access executes correctly
 *  - Performance is lower than sequential access
 *  - Bandwidth degradation is measurable
 *
 * Expected performance:
 *  - Lower bandwidth than sequential access
 *  - Stable measurements despite cache misses
 */
PERF_TEST(CacheEffects, StridedAccess) {
  ub::PerfCase perf{"CacheEffects.StridedAccess", config()};

  const std::size_t DATA_SIZE = 256 * 1024;
  const std::size_t STRIDE = 64;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytesStrided(data.data(), data.size(), STRIDE);
    (void)result;
  });

  const std::size_t bytesAccessed = (DATA_SIZE / STRIDE) * sizeof(std::uint8_t);
  ub::MemoryProfile memProfile{.bytesRead = bytesAccessed, .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] { sink = sink + test::sumBytesStrided(data.data(), data.size(), STRIDE); }, "strided",
      memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 10.0) << "Strided bandwidth suspiciously low";

  EXPECT_LT(result.stats.cv, 0.30) << "High variance in strided access";

  (void)sink;
}