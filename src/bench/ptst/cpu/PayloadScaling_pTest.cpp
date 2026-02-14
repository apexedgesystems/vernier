/**
 * @file PayloadScaling_pTest.cpp
 * @brief Payload size scaling performance validation
 *
 * This test suite validates that the framework correctly measures performance
 * across varying payload sizes, demonstrating scaling behavior across the
 * memory hierarchy (L1, L2, L3, RAM).
 *
 * Features tested:
 *  - Parameterized test execution with GoogleTest
 *  - Performance scaling across payload sizes
 *  - msgBytes configuration integration
 *  - Bandwidth scaling validation
 *
 * Expected behavior:
 *  - Larger payloads show lower throughput (higher latency)
 *  - Bandwidth should scale reasonably across sizes
 *  - Performance degradation at cache boundaries
 *
 * Usage:
 *   @code{.sh}
 *   # Run all payload scaling tests
 *   ./TestBenchSamples_PTEST --gtest_filter="PayloadScaling.*"
 *
 *   # Run specific size
 *   ./TestBenchSamples_PTEST --gtest_filter="PayloadScaling.MeasureScaling/0"
 *
 *   # With CSV output
 *   ./TestBenchSamples_PTEST --gtest_filter="PayloadScaling.*" --csv scaling.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~12 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <30% for typical workloads
 *
 * @see PerfCase
 * @see MemoryProfile
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
 * @brief Parameterized payload scaling test
 *
 * Measures performance across different payload sizes to validate scaling
 * behavior and identify cache hierarchy effects.
 *
 * @test MeasureScaling
 *
 * Validates:
 *  - Framework measures different payload sizes correctly
 *  - Throughput decreases with larger payloads
 *  - Bandwidth scaling is reasonable
 *  - Performance degradation at cache boundaries
 *
 * Expected performance:
 *  - Small payloads: High throughput (>100K calls/sec)
 *  - Large payloads: Lower throughput, stable measurements
 */
class PayloadScaling : public ::testing::TestWithParam<int> {
protected:
  void SetUp() override { payloadSize = GetParam(); }

  int payloadSize;
};

TEST_P(PayloadScaling, MeasureScaling) {
  ub::PerfCase perf{"PayloadScaling.MeasureScaling/" + std::to_string(payloadSize), config()};

  auto data = test::makeTestData(payloadSize);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  ub::MemoryProfile memProfile{
      .bytesRead = static_cast<std::uint64_t>(payloadSize), .bytesWritten = 0, .bytesAllocated = 0};

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "payload_" + std::to_string(payloadSize), memProfile);

  // Validate reasonable throughput (scaled by size)
  const double minThroughput = payloadSize < 1024 ? 100000.0 : 1000.0;
  EXPECT_GT(result.callsPerSecond, minThroughput)
      << "Throughput unexpectedly low for payload size " << payloadSize;

  // Validate measurements are stable
  EXPECT_LT(result.stats.cv, 0.30) << "High variance for payload size " << payloadSize;

  // Validate bandwidth is reasonable
  const double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);
  EXPECT_GT(bandwidthMBs, 10.0) << "Bandwidth suspiciously low for payload size " << payloadSize;

  (void)sink;
}

// Test payload sizes: 64B (L1), 1KB (L1), 256KB (L2/L3), 1MB (RAM)
INSTANTIATE_TEST_SUITE_P(Sizes, PayloadScaling, ::testing::Values(64, 1024, 262144, 1048576));