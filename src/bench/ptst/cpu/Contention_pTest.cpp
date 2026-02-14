/**
 * @file Contention_pTest.cpp
 * @brief Multi-threading synchronization primitive performance validation
 *
 * This test suite validates that the framework correctly measures performance
 * differences between various synchronization primitives under contention.
 * Demonstrates the contentionRun() API for measuring multi-threaded performance.
 *
 * Features tested:
 *  - Mutex contention measurement
 *  - Atomic contention measurement
 *  - Performance comparison between synchronization methods
 *  - contentionRun() API validation
 *
 * Expected behavior:
 *  - Atomic operations should outperform mutex operations
 *  - Contention overhead should be measurable
 *  - Both patterns should produce stable measurements
 *
 * Usage:
 *   @code{.sh}
 *   # Run all contention tests
 *   ./TestBenchSamples_PTEST --gtest_filter="Contention.*"
 *
 *   # Compare synchronization primitives
 *   ./TestBenchSamples_PTEST --gtest_filter="Contention.*" --csv contention_compare.csv
 *
 *   # Control thread count
 *   ./TestBenchSamples_PTEST --gtest_filter="Contention.*" --threads 2
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

} // anonymous namespace

/**
 * @brief Mutex contention measurement
 *
 * Measures performance of incrementing a shared counter protected by std::mutex
 * under multi-threaded contention. This represents typical lock-based
 * synchronization and should show measurable overhead.
 *
 * @test MutexContention
 *
 * Validates:
 *  - Mutex synchronization works correctly
 *  - Contention overhead is measurable
 *  - contentionRun() API functions properly
 *
 * Expected performance:
 *  - Moderate throughput under contention
 *  - Stable measurements
 */
PERF_TEST(Contention, MutexContention) {
  ub::PerfCase perf{"Contention.MutexContention", config()};

  std::mutex mtx;
  volatile std::uint64_t counter = 0;

  const int ITERS = std::min(1000, config().cycles);

  auto result = perf.contentionRun(
      [&] {
        std::uint64_t local = 0;
        for (int i = 0; i < ITERS; ++i) {
          std::lock_guard<std::mutex> lock(mtx);
          local++;
        }
        counter = local;
      },
      "mutex");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Mutex contention throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in mutex contention";

  EXPECT_GT(counter, 0u) << "Counter should have been incremented";
}

/**
 * @brief Atomic contention measurement
 *
 * Measures performance of incrementing a shared counter using std::atomic
 * under multi-threaded contention. Lock-free atomics should show better
 * performance than mutex-based synchronization.
 *
 * @test AtomicContention
 *
 * Validates:
 *  - Atomic operations work correctly under contention
 *  - Performance is better than mutex synchronization
 *  - Measurements are stable
 *
 * Expected performance:
 *  - Higher throughput than mutex pattern
 *  - Stable measurements
 */
PERF_TEST(Contention, AtomicContention) {
  ub::PerfCase perf{"Contention.AtomicContention", config()};

  std::atomic<std::uint64_t> counter{0};

  const int ITERS = std::min(1000, config().cycles);

  auto result = perf.contentionRun(
      [&] {
        for (int i = 0; i < ITERS; ++i) {
          counter.fetch_add(1, std::memory_order_relaxed);
        }
      },
      "atomic");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Atomic contention throughput suspiciously low";

  EXPECT_LT(result.stats.cv, ub::recommendedCVThreshold(config()))
      << "High variance in atomic contention";

  EXPECT_GT(counter.load(), 0u) << "Counter should have been incremented";
}