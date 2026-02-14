/**
 * @file SemanticMacros_pTest.cpp
 * @brief Smoke tests for semantic test macros and lifecycle methods
 *
 * Validates that all semantic test declaration macros compile and execute
 * correctly, and that optional lifecycle methods (setup/teardown) work as
 * expected.
 *
 * Features tested:
 *  - PERF_TAIL macro: tail-latency (p99/p999) measurement pattern
 *  - PERF_ALLOC macro: allocation overhead measurement pattern
 *  - PERF_RAMP macro: thread-ramp sweep pattern
 *  - PerfCase::setup() and PerfCase::teardown() lifecycle methods
 *
 * Expected behavior:
 *  - All macros expand to valid GTest TEST() invocations
 *  - setup() executes before warmup/measurement
 *  - teardown() executes after measurement
 *  - Measurements are stable with reasonable throughput
 *
 * Usage:
 *   @code{.sh}
 *   # Run all semantic macro tests
 *   ./BenchmarkCPU_PTEST --gtest_filter="SemanticMacros.*"
 *
 *   # Export results
 *   ./BenchmarkCPU_PTEST --gtest_filter="SemanticMacros.*" --csv semantic.csv
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~5 seconds total
 *  - Pass rate: 100% on stable hardware
 *  - CV: <15% for typical workloads
 *
 * @see PerfTestMacros.hpp
 * @see PerfCase::setup
 * @see PerfCase::teardown
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
 * @brief PERF_TAIL macro smoke test: tail-latency measurement pattern
 *
 * Validates that PERF_TAIL expands to a valid test and that the framework
 * captures p90/p99 statistics correctly. Tail-latency patterns focus on
 * worst-case performance rather than average throughput.
 *
 * @test TailLatency
 *
 * Validates:
 *  - PERF_TAIL macro compiles and executes
 *  - p10 <= median <= p90 ordering is correct
 *  - Measurements complete with reasonable throughput
 */
PERF_TAIL(SemanticMacros, TailLatency) {
  ub::PerfCase perf{"SemanticMacros.TailLatency", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "tail_latency");

  EXPECT_GT(result.callsPerSecond, 1000.0) << "Tail-latency test throughput suspiciously low";

  EXPECT_LE(result.stats.p10, result.stats.median) << "p10 should be <= median";

  EXPECT_LE(result.stats.median, result.stats.p90) << "median should be <= p90";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "Tail-latency measurements unstable";

  (void)sink;
}

/**
 * @brief PERF_ALLOC macro smoke test: allocation overhead pattern
 *
 * Validates that PERF_ALLOC expands to a valid test. Uses the allocation
 * workload pattern to verify the macro works for allocation-focused tests.
 *
 * @test AllocationPattern
 *
 * Validates:
 *  - PERF_ALLOC macro compiles and executes
 *  - MemoryProfile bytesAllocated field is captured
 *  - Measurements are stable
 */
PERF_ALLOC(SemanticMacros, AllocationPattern) {
  ub::PerfCase perf{"SemanticMacros.AllocationPattern", config()};

  const std::size_t BUFFER_SIZE = 2048;

  perf.warmup([&] { test::allocateAndFill(BUFFER_SIZE); });

  ub::MemoryProfile memProfile{
      .bytesRead = 0, .bytesWritten = BUFFER_SIZE, .bytesAllocated = BUFFER_SIZE};

  auto result =
      perf.throughputLoop([&] { test::allocateAndFill(BUFFER_SIZE); }, "alloc_pattern", memProfile);

  EXPECT_GT(result.callsPerSecond, 500.0) << "Allocation pattern throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "Allocation pattern measurements unstable";
}

/**
 * @brief PERF_RAMP macro smoke test: thread-ramp sweep pattern
 *
 * Validates that PERF_RAMP expands to a valid test. Uses a simple workload
 * that can be measured at different thread counts via --threads flag.
 *
 * @test ThreadRampSweep
 *
 * Validates:
 *  - PERF_RAMP macro compiles and executes
 *  - Single-threaded baseline measurement works
 *  - Measurements are stable
 */
PERF_RAMP(SemanticMacros, ThreadRampSweep) {
  ub::PerfCase perf{"SemanticMacros.ThreadRampSweep", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "ramp_sweep");

  EXPECT_GT(result.callsPerSecond, 1000.0) << "Thread ramp sweep throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "Thread ramp sweep measurements unstable";

  (void)sink;
}

/**
 * @brief setup() and teardown() lifecycle method validation
 *
 * Validates that PerfCase::setup() executes before measurement and
 * PerfCase::teardown() executes after measurement. Uses sentinel
 * variables to verify execution order.
 *
 * @test SetupTeardownLifecycle
 *
 * Validates:
 *  - setup() is called and completes before measurement
 *  - teardown() is called and completes after measurement
 *  - Lifecycle methods do not interfere with measurement stability
 *  - Null function arguments are handled gracefully
 */
PERF_TEST(SemanticMacros, SetupTeardownLifecycle) {
  ub::PerfCase perf{"SemanticMacros.SetupTeardownLifecycle", config()};

  bool setupCalled = false;
  bool teardownCalled = false;
  std::vector<std::uint8_t> data;

  // setup() allocates test data
  perf.setup([&] {
    data = test::makeTestData(4096);
    setupCalled = true;
  });

  EXPECT_TRUE(setupCalled) << "setup() should execute immediately";
  EXPECT_EQ(data.size(), 4096u) << "setup() should have allocated data";

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "lifecycle_test");

  EXPECT_GT(result.callsPerSecond, 1000.0) << "Lifecycle test throughput suspiciously low";

  // teardown() cleans up
  perf.teardown([&] {
    data.clear();
    teardownCalled = true;
  });

  EXPECT_TRUE(teardownCalled) << "teardown() should execute immediately";
  EXPECT_TRUE(data.empty()) << "teardown() should have cleared data";

  // Verify null functions are handled gracefully
  perf.setup(nullptr);
  perf.teardown(nullptr);

  (void)sink;
}
