/**
 * @file AllocationOverhead_pTest.cpp
 * @brief Memory allocation performance overhead validation
 *
 * This test suite validates that the framework correctly measures performance
 * differences between code that allocates memory per operation versus code that
 * reuses pre-allocated buffers.
 *
 * Features tested:
 *  - Per-call allocation overhead measurement
 *  - Buffer reuse performance measurement
 *  - Performance comparison validation
 *  - Allocation tracking in MemoryProfile
 *
 * Expected behavior:
 *  - Buffer reuse should significantly outperform per-call allocation
 *  - Allocation overhead should be measurable
 *  - Both patterns should produce stable measurements
 *
 * Usage:
 *   @code{.sh}
 *   # Run all allocation overhead tests
 *   ./TestBenchSamples_PTEST --gtest_filter="AllocationOverhead.*"
 *
 *   # Compare allocation patterns
 *   ./TestBenchSamples_PTEST --gtest_filter="AllocationOverhead.*" --csv alloc_compare.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~2 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <10% for typical workloads
 *
 * @see PerfCase
 * @see MemoryProfile
 * @see vernier::bench::test::allocateAndFill
 * @see vernier::bench::test::reuseAndFill
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

} // anonymous namespace

/**
 * @brief Per-call allocation overhead measurement
 *
 * Measures performance when allocating a new buffer on each iteration,
 * simulating code that doesn't reuse memory. This should show significant
 * overhead from allocation and deallocation.
 *
 * @test AllocateEachCall
 *
 * Validates:
 *  - Allocation overhead is measurable
 *  - Performance is lower than buffer reuse
 *  - Measurements are stable despite allocation variance
 *
 * Expected performance:
 *  - Slower than buffer reuse pattern
 *  - Throughput still reasonable (>1000 calls/sec)
 */
PERF_TEST(AllocationOverhead, AllocateEachCall) {
  ub::PerfCase perf{"AllocationOverhead.AllocateEachCall", config()};

  const std::size_t BUFFER_SIZE = 4096;

  perf.warmup([&] { test::allocateAndFill(BUFFER_SIZE); });

  ub::MemoryProfile memProfile{
      .bytesRead = 0, .bytesWritten = BUFFER_SIZE, .bytesAllocated = BUFFER_SIZE};

  auto result =
      perf.throughputLoop([&] { test::allocateAndFill(BUFFER_SIZE); }, "allocate", memProfile);

  EXPECT_GT(result.callsPerSecond, 1000.0) << "Allocation throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in allocation pattern";
}

/**
 * @brief Buffer reuse performance measurement
 *
 * Measures performance when reusing a pre-allocated buffer across iterations,
 * simulating optimized code that minimizes allocation overhead. This should
 * show significantly better performance than per-call allocation.
 *
 * @test ReuseBuffer
 *
 * Validates:
 *  - Buffer reuse eliminates allocation overhead
 *  - Performance significantly better than allocate-per-call
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Much faster than per-call allocation
 *  - High throughput (>10000 calls/sec)
 */
PERF_TEST(AllocationOverhead, ReuseBuffer) {
  ub::PerfCase perf{"AllocationOverhead.ReuseBuffer", config()};

  const std::size_t BUFFER_SIZE = 4096;
  std::vector<std::uint8_t> buffer;

  perf.warmup([&] { test::reuseAndFill(buffer, BUFFER_SIZE); });

  ub::MemoryProfile memProfile{.bytesRead = 0, .bytesWritten = BUFFER_SIZE, .bytesAllocated = 0};

  auto result =
      perf.throughputLoop([&] { test::reuseAndFill(buffer, BUFFER_SIZE); }, "reuse", memProfile);

  EXPECT_GT(result.callsPerSecond, 10000.0) << "Buffer reuse throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in buffer reuse pattern";
}