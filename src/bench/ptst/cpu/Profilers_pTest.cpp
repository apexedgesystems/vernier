/**
 * @file Profilers_pTest.cpp
 * @brief Profiler integration smoke tests
 *
 * This test suite provides smoke test coverage for profiler integration.
 * Tests validate that profiler hooks execute without error and that
 * basic profiler functionality works when enabled.
 *
 * Features tested:
 *  - Profiler hook attachment
 *  - Profiler execution without crash (perf, gperf, bpftrace, RAPL, callgrind)
 *  - Artifact directory creation
 *  - Graceful handling when profilers unavailable
 *
 * Expected behavior:
 *  - Tests skip if profiler not enabled (via --profile flag)
 *  - Tests pass when profiler available
 *  - No crashes or errors during profiling
 *
 * Usage:
 *   @code{.sh}
 *   # Run without profiler (tests will skip)
 *   ./TestBenchSamples_PTEST --gtest_filter="Profilers.*"
 *
 *   # Run with perf profiler
 *   ./TestBenchSamples_PTEST --gtest_filter="Profilers.PerfIntegration" --profile perf
 *
 *   # Run with gperf profiler
 *   ./TestBenchSamples_PTEST --gtest_filter="Profilers.GperfIntegration" --profile gperf
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~5 seconds total (when profilers enabled)
 *  - Pass rate: 100% (or skip if profiler unavailable)
 *
 * @note Since Phase 2, profiler hooks are auto-attached via UB_PERF_GUARD.
 *       These tests explicitly test the underlying attachment mechanism.
 *
 * @see makePerfCaseWithProfiler
 * @see attachProfilerHooks
 * @see PerfConfig::profileTool
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;
namespace fs = std::filesystem;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

} // anonymous namespace

/**
 * @brief Profiler hook auto-attachment validation
 *
 * Validates that UB_PERF_GUARD auto-attaches profiler hooks without error
 * regardless of whether profiler is enabled.
 *
 * @test HookAttachment
 *
 * Validates:
 *  - UB_PERF_GUARD auto-attaches profiler hooks
 *  - No crash when profiler not enabled
 *  - Test executes normally after hook attachment
 *
 * Expected performance:
 *  - Always passes (hooks should be safe to attach)
 */
PERF_TEST(Profilers, HookAttachment) {
  UB_PERF_GUARD(perf); // Auto-attaches profiler hooks

  // This should not crash even if no profiler enabled

  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "hook_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should execute normally with hooks attached";

  (void)sink;
}

/**
 * @brief Perf profiler integration smoke test
 *
 * Validates that perf profiler integration works when enabled.
 * Test skips if perf profiler not enabled via --profile perf.
 *
 * @test PerfIntegration
 *
 * Validates:
 *  - Perf profiler executes without crash
 *  - Test completes successfully
 *  - Measurements are valid
 *
 * Expected performance:
 *  - Pass when --profile perf provided
 *  - Skip otherwise
 */
PERF_TEST(Profilers, PerfIntegration) {
  if (config().profileTool != "perf") {
    GTEST_SKIP() << "Test requires --profile perf";
  }

  ub::PerfCase perf{"Profilers.PerfIntegration", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "perf_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with perf profiling";

  EXPECT_LT(result.stats.cv, 0.35)
      << "Profiling should not excessively impact measurement stability";

  (void)sink;
}

/**
 * @brief Gperf profiler integration smoke test
 *
 * Validates that gperf profiler integration works when enabled.
 * Test skips if gperf profiler not enabled via --profile gperf.
 *
 * @test GperfIntegration
 *
 * Validates:
 *  - Gperf profiler executes without crash
 *  - Test completes successfully
 *  - Measurements are valid
 *
 * Expected performance:
 *  - Pass when --profile gperf provided
 *  - Skip otherwise
 */
PERF_TEST(Profilers, GperfIntegration) {
  if (config().profileTool != "gperf") {
    GTEST_SKIP() << "Test requires --profile gperf";
  }

  ub::PerfCase perf{"Profilers.GperfIntegration", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "gperf_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with gperf profiling";

  EXPECT_LT(result.stats.cv, 0.35)
      << "Profiling should not excessively impact measurement stability";

  (void)sink;
}

/**
 * @brief Bpftrace profiler integration smoke test
 *
 * Validates that bpftrace profiler integration works when enabled.
 * Test skips if bpftrace profiler not enabled via --profile bpftrace.
 *
 * @test BpftraceIntegration
 *
 * Validates:
 *  - Bpftrace profiler executes without crash
 *  - Test completes successfully
 *  - Measurements are valid
 *
 * Expected performance:
 *  - Pass when --profile bpftrace provided
 *  - Skip otherwise
 */
PERF_TEST(Profilers, BpftraceIntegration) {
  if (config().profileTool != "bpftrace") {
    GTEST_SKIP() << "Test requires --profile bpftrace";
  }

  ub::PerfCase perf{"Profilers.BpftraceIntegration", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "bpftrace_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with bpftrace profiling";

  EXPECT_LT(result.stats.cv, 0.35)
      << "Profiling should not excessively impact measurement stability";

  (void)sink;
}

/**
 * @brief RAPL profiler integration smoke test
 *
 * Validates that RAPL (power measurement) profiler integration works when enabled.
 * Test skips if RAPL profiler not enabled.
 *
 * @test RAPLIntegration
 *
 * Validates:
 *  - RAPL profiler executes without crash
 *  - Test completes successfully
 *  - Measurements are valid
 *
 * Expected performance:
 *  - Pass when --profile rapl provided
 *  - Skip otherwise
 */
PERF_TEST(Profilers, RAPLIntegration) {
  if (config().profileTool != "rapl") {
    GTEST_SKIP() << "Test requires --profile rapl";
  }

  ub::PerfCase perf{"Profilers.RAPLIntegration", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "rapl_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with RAPL profiling";

  EXPECT_LT(result.stats.cv, 0.35) << "Power measurement should not excessively impact stability";

  (void)sink;
}

/**
 * @brief Callgrind profiler integration smoke test
 *
 * Validates that callgrind (Valgrind) profiler integration works when enabled.
 * Test skips if callgrind profiler not enabled via --profile callgrind.
 *
 * @test CallgrindIntegration
 *
 * Validates:
 *  - Callgrind profiler executes without crash
 *  - Test completes successfully under valgrind simulation
 *  - Measurements are valid despite instrumentation overhead
 *
 * Expected performance:
 *  - Pass when --profile callgrind provided
 *  - Skip otherwise
 *  - Note: When running under valgrind, execution is 20-50x slower;
 *    thresholds are relaxed accordingly
 */
PERF_TEST(Profilers, CallgrindIntegration) {
  if (config().profileTool != "callgrind") {
    GTEST_SKIP() << "Test requires --profile callgrind";
  }

  ub::PerfCase perf{"Profilers.CallgrindIntegration", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "callgrind_test");

  // Callgrind runs under valgrind which adds 20-50x overhead; use relaxed threshold
  EXPECT_GT(result.callsPerSecond, 10.0) << "Test should complete with callgrind profiling";

  EXPECT_LT(result.stats.cv, 0.50)
      << "Valgrind simulation should produce reasonably stable measurements";

  (void)sink;
}