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
 *  - AllocateEachCall allocates a 4 KiB buffer, fills it and frees it on
 *    every call; ReuseBuffer fills a caller-owned buffer of the same size
 *    that grew once, during warmup
 *  - The cases do not differ by the allocate/free pair alone. In an
 *    optimized build the reuse path may also call vector::resize and take
 *    its fill length from the vector at run time, while the allocating path
 *    fills a length known at compile time. The difference between the two
 *    results indicates allocation cost; it does not isolate it
 *  - Reference observation, not a guarantee: in an optimized x86-64 build
 *    with a general-purpose system allocator, both cases take tens of
 *    nanoseconds per call and AllocateEachCall is roughly 1.5x to 2x slower.
 *    The ratio depends on the compiler, the allocator and the host
 *  - An unexpected ordering proves nothing by itself; timing noise alone can
 *    produce one. Treat it as a reason to inspect the generated code or an
 *    allocation count (for example from a heap profiler): an optimizing
 *    compiler may remove an allocation whose contents are never observed,
 *    which is why the helpers keep the buffer observable
 *  - Both patterns should produce stable measurements
 *
 * Usage:
 *   @code{.sh}
 *   # Run all allocation overhead tests
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="AllocationOverhead.*"
 *
 *   # Compare allocation patterns
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="AllocationOverhead.*" --csv alloc_compare.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: well under a second with default settings
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
 * simulating code that doesn't reuse memory. Each call allocates one buffer,
 * fills it and frees it; the helper keeps the buffer observable so that an
 * optimizing compiler does not discard that work.
 *
 * @test AllocateEachCall
 *
 * Validates:
 *  - Allocation overhead is measurable
 *  - Measurements are stable despite allocation variance
 *
 * Expected performance:
 *  - Reference observation: roughly 1.5x to 2x slower than ReuseBuffer;
 *    compiler, allocator and host dependent (see the file comment)
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
 * simulating optimized code that minimizes allocation overhead. The buffer
 * grows once, during warmup; the measured calls resize it to the size it
 * already has and fill it.
 *
 * @test ReuseBuffer
 *
 * Validates:
 *  - Buffer reuse eliminates allocation overhead
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Reference observation: faster than AllocateEachCall; the difference
 *    indicates allocation cost but is not an isolated measurement of the
 *    allocate/free pair (see the file comment)
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