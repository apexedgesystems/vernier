/**
 * @file Contention_pTest.cpp
 * @brief Multi-threading synchronization primitive performance validation
 *
 * This test suite measures one shared counter incremented by every worker,
 * once behind a std::mutex and once as a std::atomic, through the
 * contentionRun() API for measuring multi-threaded performance.
 *
 * Features tested:
 *  - Mutex contention measurement
 *  - Atomic contention measurement
 *  - contentionRun() API validation
 *
 * Expected behavior:
 *  - No increment is lost: each case checks its counter against the calls
 *    its worker made
 *  - Contention overhead should be measurable
 *  - Both patterns should produce stable measurements
 *  - The two rows of a --csv run compare the primitives; neither case
 *    asserts which one is faster
 *
 * Usage:
 *   @code{.sh}
 *   # Run all contention tests
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Contention.*"
 *
 *   # Compare synchronization primitives
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Contention.*" --csv contention_compare.csv
 *
 *   # Control thread count: a core per worker plus one for the test's own
 *   # thread, since workers pinned to fewer cores take turns
 *   taskset -c 1-3 ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Contention.*" --threads 2
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~12 seconds total (optimized for CI)
 *  - Pass rate: 100% on stable hardware
 *  - CV: <10% for typical workloads
 *
 * @see PerfCase
 * @see PerfCase::contentionRun
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <algorithm>

#include "src/bench/inc/Perf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

/**
 * @brief Checks a contention case's counter against the calls its worker made.
 *
 * contentionRun() calls the worker repeats() x threads() x cycles() times.
 * With --target-time its calibration calls the worker too, before the
 * measured phase, so the call count is then at least that; cycles() read
 * after the run is the calibrated count. Every call increments the counter
 * @p iters times, so the counter is exactly calls x iters.
 */
void expectExactCount(const ub::PerfCase& perf, std::uint64_t calls, std::uint64_t counter,
                      int iters) {
  const std::uint64_t MEASURED_CALLS =
      static_cast<std::uint64_t>(perf.repeats()) * perf.threads() * perf.cycles();
  std::printf("[%s] calls %llu counter %llu (%d increments per call)\n", perf.testName().c_str(),
              static_cast<unsigned long long>(calls), static_cast<unsigned long long>(counter),
              iters);

  EXPECT_GT(calls, 0u) << "The worker never ran";
  if (perf.config().targetTimeUs > 0) {
    EXPECT_GE(calls, MEASURED_CALLS) << "Fewer worker calls than the measured phase makes";
  } else {
    EXPECT_EQ(calls, MEASURED_CALLS) << "Worker calls differ from repeats x threads x cycles";
  }
  EXPECT_EQ(counter, calls * static_cast<std::uint64_t>(iters)) << "Increments were lost or added";
}

} // anonymous namespace

/**
 * @brief Mutex contention measurement
 *
 * Measures performance of incrementing a shared counter protected by std::mutex
 * under multi-threaded contention: every increment takes and releases the lock.
 * This represents typical lock-based synchronization and should show
 * measurable overhead.
 *
 * @test MutexContention
 *
 * Validates:
 *  - The lock protects the counter: no increment is lost
 *  - Every measured call of the worker ran
 *  - contentionRun() API functions properly
 *
 * Expected performance:
 *  - Moderate throughput under contention
 *  - Stable measurements
 */
PERF_TEST(Contention, MutexContention) {
  ub::PerfCase perf{"Contention.MutexContention", config()};

  std::mutex mtx;
  std::uint64_t counter = 0; // guarded by mtx
  std::atomic<std::uint64_t> calls{0};

  const int ITERS = std::min(1000, config().cycles);

  auto result = perf.contentionRun(
      [&] {
        for (int i = 0; i < ITERS; ++i) {
          std::lock_guard<std::mutex> lock(mtx);
          ++counter;
        }
        calls.fetch_add(1, std::memory_order_relaxed);
      },
      "mutex");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Mutex contention throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in mutex contention";

  expectExactCount(perf, calls.load(), counter, ITERS);
}

/**
 * @brief Atomic contention measurement
 *
 * Measures performance of incrementing a shared counter using std::atomic
 * under multi-threaded contention. Compare its row with MutexContention's
 * to see what the lock costs on the machine at hand.
 *
 * @test AtomicContention
 *
 * Validates:
 *  - Atomic operations work correctly under contention: no increment is lost
 *  - Every measured call of the worker ran
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Stable measurements
 */
PERF_TEST(Contention, AtomicContention) {
  ub::PerfCase perf{"Contention.AtomicContention", config()};

  std::atomic<std::uint64_t> counter{0};
  std::atomic<std::uint64_t> calls{0};

  const int ITERS = std::min(1000, config().cycles);

  auto result = perf.contentionRun(
      [&] {
        for (int i = 0; i < ITERS; ++i) {
          counter.fetch_add(1, std::memory_order_relaxed);
        }
        calls.fetch_add(1, std::memory_order_relaxed);
      },
      "atomic");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Atomic contention throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in atomic contention";

  expectExactCount(perf, calls.load(), counter.load(), ITERS);
}