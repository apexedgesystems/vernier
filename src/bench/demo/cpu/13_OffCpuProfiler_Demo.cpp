/**
 * @file 13_OffCpuProfiler_Demo.cpp
 * @brief Demo 13: Off-CPU profiling via bpftrace.
 *
 * On-CPU profilers (gperf, perf, callgrind) show where threads burn CPU
 * cycles. Off-CPU profiling shows where they *stop* burning cycles --
 * sleeps, mutex contention, I/O waits, scheduler delays. The two views
 * together explain a thread's full life cycle.
 *
 * Usage (requires sudo to attach kprobes):
 *   @code{.sh}
 *   sudo ./build/native-linux-debug/bin/ptests/BenchDemo_13_OffCpuProfiler \
 *       --profile offcpu --quick --gtest_filter='OffCpu.SleepHeavy'
 *
 *   # Read the stack dump (ranked by total off-CPU nanoseconds per stack):
 *   cat OffCpu.SleepHeavy.offcpu/offcpu.txt
 *   @endcode
 *
 * Run without sudo: the backend prints a "need root" hint and the
 * benchmark proceeds normally without off-CPU collection.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "src/bench/inc/Perf.hpp"

namespace ub = vernier::bench;

/**
 * @test Sleep-dominated workload: each iteration sleeps a millisecond
 *       between two trivial pieces of work. Off-CPU profile shows the
 *       std::this_thread::sleep_for stack as the top consumer.
 */
PERF_THROUGHPUT(OffCpu, SleepHeavy) {
  UB_PERF_GUARD(perf);

  perf.warmup([&] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });

  volatile int sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        sink += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        sink += 2;
      },
      "sleep_1ms_per_iter");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

/**
 * @test Mutex-contended workload: two threads contend for a single mutex
 *       in a tight loop. Off-CPU profile attributes most blocked time to
 *       pthread_mutex_lock / futex_wait paths.
 */
PERF_THROUGHPUT(OffCpu, MutexContention) {
  UB_PERF_GUARD(perf);

  std::mutex m;
  volatile int sharedCounter = 0;

  auto contend = [&] {
    for (int i = 0; i < 1000; ++i) {
      std::lock_guard<std::mutex> g(m);
      sharedCounter += 1;
    }
  };

  perf.warmup([&] {
    std::thread t1(contend), t2(contend);
    t1.join();
    t2.join();
  });

  auto result = perf.throughputLoop(
      [&] {
        std::thread t1(contend), t2(contend);
        t1.join();
        t2.join();
      },
      "mutex_contention");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sharedCounter;
}

PERF_MAIN()
