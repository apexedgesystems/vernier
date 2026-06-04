/**
 * @file 14_HelgrindProfiler_Demo.cpp
 * @brief Demo 14: Valgrind Helgrind / DRD thread-error detector -- find the race.
 *
 * Helgrind is a correctness tool for threaded code, run alongside benchmarks
 * to prove a parallel optimization is actually safe. It detects data races,
 * lock-ordering violations, and misuse of the POSIX threads API, and pins each
 * one to the two conflicting source lines.
 *
 * Two tests:
 *   Slow/Buggy: RacyCounter   -- threads increment a shared unguarded long
 *   Fast/Safe:  AtomicCounter -- same workload via std::atomic, race-free
 *
 * Story: the unsynchronized increment looks fine and even "works" at low
 * thread counts, but Helgrind flags the read/write race with the two stacks
 * that collide. The fix (std::atomic) is reported clean -- and is also faster
 * than a mutex would be.
 *
 * Usage:
 *   @code{.sh}
 *   # Buggy run (expect: "Possible data race ... during write of size 8"):
 *   valgrind --tool=helgrind \
 *       --log-file=Helgrind.RacyCounter.helgrind/helgrind.log \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_14_HelgrindProfiler \
 *       --profile helgrind --cycles 5 --gtest_filter='Helgrind.RacyCounter'
 *
 *   # Clean run (expect: no race reported):
 *   valgrind --tool=helgrind \
 *       --log-file=Helgrind.AtomicCounter.helgrind/helgrind.log \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_14_HelgrindProfiler \
 *       --profile helgrind --cycles 5 --gtest_filter='Helgrind.AtomicCounter'
 *
 *   # DRD is the alternate detector; vernier selects it via --profile-args drd:
 *   valgrind --tool=drd \
 *       --log-file=Helgrind.RacyCounter.helgrind/drd.log \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_14_HelgrindProfiler \
 *       --profile helgrind --profile-args drd --cycles 5 \
 *       --gtest_filter='Helgrind.RacyCounter'
 *   @endcode
 *
 * Helgrind/DRD slow execution ~20-100x; use --cycles 5 (or fewer) so the
 * threads still actually overlap but the run stays usable.
 *
 * @see docs/20_HELGRIND_PROFILER.md for the step-by-step walkthrough.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "src/bench/inc/Perf.hpp"

static constexpr int NUM_THREADS = 4;
static constexpr int LOOPS_PER_THREAD = 2000;

/**
 * @test Slow/Buggy: several threads increment a shared unsynchronized long.
 *
 * `sharedCounter += 1` is a read-modify-write with no synchronization, so two
 * threads can read the same value and lose an update. Helgrind reports
 * "Possible data race during write of size 8" with the two colliding stacks,
 * both pointing at the increment line below.
 *
 * Do not copy; this is here so helgrind has a genuine race to find. The test
 * still asserts callsPerSecond > 1 so the benchmark completes -- the race
 * corrupts the count but does not crash.
 */
PERF_THROUGHPUT(Helgrind, RacyCounter) {
  UB_PERF_GUARD(perf);

  long sharedCounter = 0;

  auto worker = [&] {
    for (int i = 0; i < LOOPS_PER_THREAD; ++i) {
      // Intentional data race: unsynchronized read-modify-write on a value
      // shared across threads. Helgrind/DRD flags this exact line.
      sharedCounter += 1;
    }
  };

  perf.warmup([&] {
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t)
      threads.emplace_back(worker);
    for (auto& th : threads)
      th.join();
  });

  volatile long sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t)
          threads.emplace_back(worker);
        for (auto& th : threads)
          th.join();
        sink = sharedCounter;
      },
      "racy_counter");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

/**
 * @test Fast/Safe: same workload, std::atomic counter eliminates the race.
 *
 * The read-modify-write is now a single atomic operation; there is no window
 * for two threads to clobber each other. Helgrind reports no data race for
 * this test, and the atomic is cheaper than the mutex alternative.
 */
PERF_THROUGHPUT(Helgrind, AtomicCounter) {
  UB_PERF_GUARD(perf);

  std::atomic<long> sharedCounter{0};

  auto worker = [&] {
    for (int i = 0; i < LOOPS_PER_THREAD; ++i) {
      sharedCounter.fetch_add(1, std::memory_order_relaxed);
    }
  };

  perf.warmup([&] {
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t)
      threads.emplace_back(worker);
    for (auto& th : threads)
      th.join();
  });

  volatile long sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t)
          threads.emplace_back(worker);
        for (auto& th : threads)
          th.join();
        sink = sharedCounter.load(std::memory_order_relaxed);
      },
      "atomic_counter");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

PERF_MAIN()
