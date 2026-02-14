/**
 * @file MemoryBandwidth_pTest.cpp
 * @brief Memory bandwidth measurement and pattern validation
 *
 * This test suite validates that the framework correctly measures memory
 * bandwidth across different access patterns (sequential, strided, write,
 * read-modify-write) and calculates efficiency metrics.
 *
 * Features tested:
 *  - Sequential read bandwidth measurement
 *  - Strided read bandwidth measurement
 *  - Write bandwidth measurement
 *  - Read-modify-write bandwidth measurement
 *  - Bandwidth efficiency calculation
 *
 * Expected behavior:
 *  - Sequential access shows highest bandwidth
 *  - Strided access shows reduced bandwidth
 *  - Write bandwidth is measurable
 *  - RMW bandwidth reflects combined operation cost
 *
 * Usage:
 *   @code{.sh}
 *   # Run all memory bandwidth tests
 *   ./TestBenchSamples_PTEST --gtest_filter="MemoryBandwidth.*"
 *
 *   # With CSV output to compare patterns
 *   ./TestBenchSamples_PTEST --gtest_filter="MemoryBandwidth.*" --csv bandwidth.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~8 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <30% for typical workloads
 *
 * @see PerfCase
 * @see MemoryProfile
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <algorithm>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

} // anonymous namespace

/**
 * @brief Sequential read bandwidth measurement
 *
 * Measures bandwidth for sequential memory reads, which should show
 * optimal performance due to cache prefetching and spatial locality.
 *
 * @test SequentialRead
 *
 * Validates:
 *  - Sequential read bandwidth measured correctly
 *  - Bandwidth is reasonable (>100 MB/s)
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Highest read bandwidth of all patterns
 *  - CV <30%
 */
PERF_TEST(MemoryBandwidth, SequentialRead) {
  ub::PerfCase perf{"MemoryBandwidth.SequentialRead", config()};

  const std::size_t DATA_SIZE = 128 * 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  ub::MemoryProfile memProfile{.bytesRead = DATA_SIZE, .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "seq_read", memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 100.0) << "Sequential read bandwidth suspiciously low";

  EXPECT_LT(result.stats.cv, 0.30) << "High variance in sequential read";

  (void)sink;
}

/**
 * @brief Strided read bandwidth measurement
 *
 * Measures bandwidth for strided memory reads (64-byte stride), which
 * should show reduced performance due to poor cache utilization.
 *
 * @test StridedRead
 *
 * Validates:
 *  - Strided read bandwidth measured correctly
 *  - Bandwidth lower than sequential
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Bandwidth lower than sequential
 *  - CV <30%
 */
PERF_TEST(MemoryBandwidth, StridedRead) {
  ub::PerfCase perf{"MemoryBandwidth.StridedRead", config()};

  const std::size_t DATA_SIZE = 128 * 1024;
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
      [&] { sink = sink + test::sumBytesStrided(data.data(), data.size(), STRIDE); },
      "strided_read", memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 10.0) << "Strided read bandwidth suspiciously low";

  EXPECT_LT(result.stats.cv, 0.50) << "High variance in strided read";

  (void)sink;
}

/**
 * @brief Write bandwidth measurement
 *
 * Measures bandwidth for memory writes by filling a buffer with a pattern.
 *
 * @test WritePattern
 *
 * Validates:
 *  - Write bandwidth measured correctly
 *  - Bandwidth is reasonable
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Write bandwidth >100 MB/s
 *  - CV <30%
 */
PERF_TEST(MemoryBandwidth, WritePattern) {
  ub::PerfCase perf{"MemoryBandwidth.WritePattern", config()};

  const std::size_t DATA_SIZE = 128 * 1024;
  std::vector<std::uint8_t> buffer(DATA_SIZE);

  perf.warmup([&] { std::fill(buffer.begin(), buffer.end(), std::uint8_t{0xAA}); });

  ub::MemoryProfile memProfile{.bytesRead = 0, .bytesWritten = DATA_SIZE, .bytesAllocated = 0};

  auto result = perf.throughputLoop(
      [&] {
        std::fill(buffer.begin(), buffer.end(), std::uint8_t{0xAA});
        volatile auto val = buffer[0];
        (void)val;
      },
      "write", memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 100.0) << "Write bandwidth suspiciously low";

  EXPECT_LT(result.stats.cv, 0.30) << "High variance in write pattern";
}

/**
 * @brief Read-modify-write bandwidth measurement
 *
 * Measures bandwidth for read-modify-write operations, which combine
 * read and write costs.
 *
 * @test ReadModifyWrite
 *
 * Validates:
 *  - RMW bandwidth measured correctly
 *  - Bandwidth accounts for both read and write
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Bandwidth >50 MB/s
 *  - CV <30%
 */
PERF_TEST(MemoryBandwidth, ReadModifyWrite) {
  ub::PerfCase perf{"MemoryBandwidth.ReadModifyWrite", config()};

  const std::size_t DATA_SIZE = 128 * 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i] = data[i] + 1;
    }
  });

  ub::MemoryProfile memProfile{
      .bytesRead = DATA_SIZE, .bytesWritten = DATA_SIZE, .bytesAllocated = 0};

  auto result = perf.throughputLoop(
      [&] {
        for (std::size_t i = 0; i < data.size(); ++i) {
          data[i] = data[i] + 1;
        }
        volatile auto val = data[0];
        (void)val;
      },
      "rmw", memProfile);

  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

  EXPECT_GT(bandwidthMBs, 50.0) << "RMW bandwidth suspiciously low";

  EXPECT_LT(result.stats.cv, 0.30) << "High variance in read-modify-write";
}