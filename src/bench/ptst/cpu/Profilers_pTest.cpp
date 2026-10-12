/**
 * @file Profilers_pTest.cpp
 * @brief Profiler integration samples: a backend attached by the profiler guard
 *
 * HookAttachment shows that UB_PERF_GUARD attaches the profiler hooks whether
 * or not --profile names a profiler. PerfIntegration, GperfIntegration,
 * BpftraceIntegration and CallgrindIntegration run only under their own
 * --profile and build their case with the same guard, which attaches that
 * backend to the measurement. When the backend's readiness check refuses the
 * request, the guard attaches a no-op instead and the run prints why; the
 * guard decides once per run.
 *
 * The CSV row of a profiled case names the backend and its capture folder
 * (profileTool, profileDir); a refused request names no folder. There perf
 * writes stat.txt, gperftools cpu.prof and bpftrace a report per script, and
 * valgrind writes callgrind's profile when the process exits. What a backend
 * wrote is read after the run, from that folder. RAPLIntegration builds a
 * PerfCase without the hooks, so --profile attaches no profiler to it.
 *
 * Usage:
 *   @code{.sh}
 *   # Without --profile only HookAttachment measures; the other cases skip
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Profilers.*"
 *
 *   # One backend, its capture folders under ./profiles
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Profilers.GperfIntegration" --profile gperf --artifact-root profiles
 *
 *   # callgrind records a process only when valgrind runs it
 *   valgrind --tool=callgrind --instr-atstart=no \
 *       --callgrind-out-file=profiles/Profilers.CallgrindIntegration.callgrind/callgrind.out \
 *       ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Profilers.CallgrindIntegration" --profile callgrind \
 *       --artifact-root profiles
 *   @endcode
 *
 * perf needs access to this process's counters and bpftrace needs root or
 * BENCH_SUDO; `--profile-check --profile <name>` says what this machine allows.
 *
 * @see makePerfCaseWithProfiler
 * @see attachProfilerHooks
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
 * @brief perf attached by the guard
 *
 * Under --profile perf, perf stat counts this process during the measured
 * repeats and writes its report, stat.txt, into the case's capture folder.
 *
 * @test PerfIntegration
 */
PERF_TEST(Profilers, PerfIntegration) {
  if (config().profileTool != "perf") {
    GTEST_SKIP() << "Test requires --profile perf";
  }

  UB_PERF_GUARD(perf);

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
 * @brief gperftools attached by the guard
 *
 * Under --profile gperf, ProfilerStart() before the measured repeats and
 * ProfilerStop() after them write cpu.prof into the case's capture folder.
 *
 * @test GperfIntegration
 */
PERF_TEST(Profilers, GperfIntegration) {
  if (config().profileTool != "gperf") {
    GTEST_SKIP() << "Test requires --profile gperf";
  }

  UB_PERF_GUARD(perf);

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
 * @brief bpftrace attached by the guard
 *
 * Under --profile bpftrace, one tracer per selected script traces this
 * process during the measured repeats and writes its report into the case's
 * capture folder.
 *
 * @test BpftraceIntegration
 */
PERF_TEST(Profilers, BpftraceIntegration) {
  if (config().profileTool != "bpftrace") {
    GTEST_SKIP() << "Test requires --profile bpftrace";
  }

  UB_PERF_GUARD(perf);

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
 * @brief callgrind attached by the guard
 *
 * Under --profile callgrind, in a process valgrind's callgrind runs from a wrap
 * started with --instr-atstart=no, the backend switches instrumentation on
 * for the measured repeats and off after them; valgrind writes the profile
 * when the process exits. Execution is 20-50x slower under valgrind; the
 * thresholds below are relaxed accordingly.
 *
 * @test CallgrindIntegration
 */
PERF_TEST(Profilers, CallgrindIntegration) {
  if (config().profileTool != "callgrind") {
    GTEST_SKIP() << "Test requires --profile callgrind";
  }

  UB_PERF_GUARD(perf);

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