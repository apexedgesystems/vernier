/**
 * @file 13_OffCpuProfiler_Demo.cpp
 * @brief Demo 13: Off-CPU profiling -- see where threads block, then unblock them.
 *
 * On-CPU profilers (gperf, perf, callgrind) only see threads while they
 * are running. Off-CPU profiling shows where they *stop* running -- the
 * blocked side of the thread life cycle.
 *
 * Two variants in the spirit of the existing demo pattern:
 *
 *   Slow: MutexCounter -- two threads contend for a mutex around an int
 *   Fast: AtomicCounter -- same workload using std::atomic, no blocking
 *
 * Story: a profiler shows mutex contention as `pthread_mutex_lock` /
 * `futex_wait` time. The off-CPU profile pins the source line; the fix
 * (atomic counter) eliminates the blocked time entirely.
 *
 * Requires sudo + tracefs access to actually collect stacks. The backend
 * gracefully degrades and prints the hint when neither is available;
 * benchmarks still run, just without off-CPU collection.
 *
 * Usage:
 *   @code{.sh}
 *   sudo ./BenchDemo_13_OffCpuProfiler --profile offcpu --quick \
 *       --gtest_filter='OffCpu.MutexCounter'
 *   cat OffCpu.MutexCounter.offcpu/offcpu.txt
 *
 *   sudo ./BenchDemo_13_OffCpuProfiler --profile offcpu --quick \
 *       --gtest_filter='OffCpu.AtomicCounter'
 *   cat OffCpu.AtomicCounter.offcpu/offcpu.txt
 *   @endcode
 *
 * @see docs/16_OFFCPU_PROFILER.md for the walkthrough.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "src/bench/inc/Perf.hpp"

namespace ub = vernier::bench;

static constexpr int LOOPS_PER_THREAD = 5000;

/**
 * @test Slow: two threads contend for a mutex around a counter increment.
 *
 * Off-CPU profile attributes most blocked time to pthread_mutex_lock /
 * futex_wait paths in the std::lock_guard at this source line.
 */
PERF_THROUGHPUT(OffCpu, MutexCounter) {
  UB_PERF_GUARD(perf);

  std::mutex m;
  volatile long sharedCounter = 0;

  auto worker = [&] {
    for (int i = 0; i < LOOPS_PER_THREAD; ++i) {
      std::lock_guard<std::mutex> g(m);
      sharedCounter += 1;
    }
  };

  perf.warmup([&] {
    std::thread t1(worker), t2(worker);
    t1.join();
    t2.join();
  });

  auto result = perf.throughputLoop(
      [&] {
        std::thread t1(worker), t2(worker);
        t1.join();
        t2.join();
      },
      "mutex_counter");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sharedCounter;
}

/**
 * @test Fast: same workload, std::atomic counter eliminates the blocking.
 *
 * Off-CPU profile should show no lock-related stacks for this case --
 * the threads now spin productively on the cache line instead of going
 * off-CPU into the kernel scheduler.
 */
PERF_THROUGHPUT(OffCpu, AtomicCounter) {
  UB_PERF_GUARD(perf);

  std::atomic<long> sharedCounter{0};

  auto worker = [&] {
    for (int i = 0; i < LOOPS_PER_THREAD; ++i) {
      sharedCounter.fetch_add(1, std::memory_order_relaxed);
    }
  };

  perf.warmup([&] {
    std::thread t1(worker), t2(worker);
    t1.join();
    t2.join();
  });

  auto result = perf.throughputLoop(
      [&] {
        std::thread t1(worker), t2(worker);
        t1.join();
        t2.join();
      },
      "atomic_counter");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sharedCounter;
}

PERF_MAIN()
