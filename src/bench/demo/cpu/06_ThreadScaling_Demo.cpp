/**
 * @file 06_ThreadScaling_Demo.cpp
 * @brief Demo 06: Lock contention vs lock-free optimization
 *
 * Demonstrates the framework's contentionRun() API for multi-threaded
 * benchmarking. Shows how mutex contention destroys throughput and how
 * atomic operations restore scalability.
 *
 * Slow: Mutex-protected counter (threads serialize on lock)
 * Fast: Atomic counter with relaxed ordering (true parallelism)
 *
 * Usage:
 *   @code{.sh}
 *   # Run with 4 threads
 *   ./BenchDemo_06_ThreadScaling --threads 4 --csv results.csv
 *
 *   # Compare scaling: 2 vs 4 vs 8 threads
 *   ./BenchDemo_06_ThreadScaling --threads 2 --csv t2.csv
 *   ./BenchDemo_06_ThreadScaling --threads 4 --csv t4.csv
 *   ./BenchDemo_06_ThreadScaling --threads 8 --csv t8.csv
 *   @endcode
 *
 * @see docs/06_THREAD_SCALING.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <mutex>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

static constexpr int OPS_PER_WORKER = 10000;

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: Mutex-protected counter under contention.
 *
 * Uses contentionRun() to spawn N worker threads, each incrementing
 * a shared counter through a mutex. As thread count increases, threads
 * spend more time waiting for the lock than doing useful work.
 *
 * This is the classic "lock contention" anti-pattern.
 */
PERF_CONTENTION(ThreadScaling, MutexContention) {
  UB_PERF_GUARD(perf);

  std::mutex mtx;
  std::uint64_t counter = 0;

  perf.warmup([&] {
    counter = 0;
    demo::incrementMutex(mtx, counter, OPS_PER_WORKER);
  });

  auto result = perf.contentionRun([&] { demo::incrementMutex(mtx, counter, OPS_PER_WORKER); },
                                   "mutex_increment");

  EXPECT_GT(result.callsPerSecond, 1.0);
}

/**
 * @test Fast: Atomic counter under contention (lock-free).
 *
 * Uses contentionRun() with atomic operations. Each thread can
 * increment without waiting for a lock. Relaxed memory ordering
 * allows maximum hardware parallelism.
 *
 * Expected improvement: 5-20x depending on thread count and
 * hardware CAS implementation.
 */
PERF_CONTENTION(ThreadScaling, AtomicLockFree) {
  UB_PERF_GUARD(perf);

  std::atomic<std::uint64_t> counter{0};

  perf.warmup([&] {
    counter.store(0, std::memory_order_relaxed);
    demo::incrementAtomic(counter, OPS_PER_WORKER);
  });

  auto result = perf.contentionRun([&] { demo::incrementAtomic(counter, OPS_PER_WORKER); },
                                   "atomic_increment");

  EXPECT_GT(result.callsPerSecond, 1.0);
}

/**
 * @test Single-threaded baseline for comparison.
 *
 * Runs the atomic version with 1 thread to establish the
 * uncontended baseline. Compare with multi-threaded results
 * to calculate scaling efficiency.
 */
PERF_THROUGHPUT(ThreadScaling, SingleThreadBaseline) {
  UB_PERF_GUARD(perf);

  std::atomic<std::uint64_t> counter{0};

  perf.warmup([&] {
    counter.store(0, std::memory_order_relaxed);
    demo::incrementAtomic(counter, OPS_PER_WORKER);
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        counter.store(0, std::memory_order_relaxed);
        demo::incrementAtomic(counter, OPS_PER_WORKER);
        sink = counter.load(std::memory_order_relaxed);
      },
      "single_thread_baseline");

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

PERF_MAIN()
