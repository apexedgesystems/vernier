/**
 * @file 12_MemcheckProfiler_Demo.cpp
 * @brief Demo 12: Valgrind Memcheck for memory error / leak detection.
 *
 * Memcheck is most useful as a *correctness gate* run alongside benchmarks
 * after an optimization pass -- it catches leaks, use-after-free, and reads
 * of uninitialized memory introduced by the pass. Clean code reports
 * "definitely lost: 0 bytes" and the benchmark proceeds normally.
 *
 * Two tests:
 *  - CleanWorkload      heap workload with proper RAII; memcheck reports 0 leaks
 *  - WithDeliberateLeak intentionally leaks one buffer per iteration
 *
 * Usage:
 *   @code{.sh}
 *   # Clean run (expect: definitely lost: 0 bytes):
 *   valgrind --tool=memcheck --leak-check=full \
 *       --log-file=Memcheck.CleanWorkload.memcheck/log.txt \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_12_MemcheckProfiler \
 *       --profile memcheck --cycles 1 --gtest_filter='Memcheck.CleanWorkload'
 *
 *   # Leaky run (expect: nonzero "definitely lost" with stack trace):
 *   valgrind --tool=memcheck --leak-check=full \
 *       --log-file=Memcheck.WithDeliberateLeak.memcheck/log.txt \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_12_MemcheckProfiler \
 *       --profile memcheck --cycles 1 --gtest_filter='Memcheck.WithDeliberateLeak'
 *   @endcode
 *
 * Memcheck slows execution ~20x; use --cycles 1 (or 2) for usable runtimes.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

static constexpr std::size_t WORK = 100'000;

/** @test Clean workload: RAII unique_ptr; memcheck reports zero leaks. */
PERF_THROUGHPUT(Memcheck, CleanWorkload) {
  UB_PERF_GUARD(perf);

  perf.warmup([&] {
    auto buf = std::make_unique<double[]>(WORK);
    (void)buf;
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        auto buf = std::make_unique<double[]>(WORK);
        for (std::size_t i = 0; i < WORK; ++i) buf[i] = static_cast<double>(i);
        sink = sink + buf[WORK - 1];
      },
      "clean_workload");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

/**
 * @test Deliberate leak: a raw new[] without a delete[]. Memcheck flags
 *       this as "definitely lost" with the source-line backtrace.
 *
 * Do not copy; this is here so memcheck has something to find. The test
 * still asserts a callsPerSecond > 1 so the benchmark completes; leaks do
 * not crash the program, they just accumulate.
 */
PERF_THROUGHPUT(Memcheck, WithDeliberateLeak) {
  UB_PERF_GUARD(perf);

  // Tiny iteration count -- memcheck is slow and we don't want to actually
  // OOM the run, just produce one leaky cycle.
  perf.warmup([&] { /* nothing */ });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        // Intentional: allocate but never free. Memcheck will report
        // "definitely lost: N bytes in M blocks" where M scales with calls.
        double* leaked = new double[WORK];
        for (std::size_t i = 0; i < WORK; ++i) leaked[i] = static_cast<double>(i);
        sink = sink + leaked[WORK - 1];
        // leaked NOT deleted -- on purpose.
      },
      "with_deliberate_leak");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

PERF_MAIN()
