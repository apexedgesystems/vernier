/**
 * @file 13_OffCpuProfiler_Demo.cpp
 * @brief Demo 13: where threads sleep, and for how long, which a CPU sampler
 *        cannot see
 *
 * Threads join the shared example's words and add the joined length to one
 * total, two ways (13_OffCpuProfiler_Totals.hpp), one CSV row each:
 *  1. CoarseLock: one lock, held for the whole call, so the joins take turns
 *     and a thread that finds the lock taken sleeps until it is free
 *  2. NoSharing: each thread adds to a total of its own and hands it over
 *     once, when the thread ends
 * contentionRun() starts --threads threads that make the call at the same
 * time. With --profile offcpu, each case's capture holds the stacks at which
 * its threads went to sleep and, for each thread, how long it was off the CPU
 * after going to sleep.
 *
 * Usage (a core for each thread, and one for the thread that starts them):
 *   @code{.sh}
 *   BENCH_SUDO=1 taskset -c 0-3 ./BenchDemo_13_OffCpuProfiler --profile offcpu \
 *       --threads 3 --cycles 1000 --repeats 5
 *   cat OffCpu.CoarseLock.offcpu/offcpu.txt OffCpu.NoSharing.offcpu/offcpu.txt
 *   @endcode
 *
 * That both versions reach the same total is checked apart from the demo, by
 * utst/13_OffCpuProfiler_uTest.cpp.
 *
 * @see docs/16_OFFCPU_PROFILER.md for the walkthrough.
 */

#include <gtest/gtest.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/cpu/13_OffCpuProfiler_Totals.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

using vernier::bench::demo::offcpu_demo::addToThreadTotal;
using vernier::bench::demo::offcpu_demo::addUnderCoarseLock;
using vernier::bench::demo::offcpu_demo::PART_COUNT;
using vernier::bench::demo::offcpu_demo::PART_SEED;
using vernier::bench::demo::offcpu_demo::SharedTotal;

/* ----------------------------- Tests ----------------------------- */

/** @test Threads add joined lengths to one total, holding its lock for the whole call. */
PERF_CONTENTION(OffCpu, CoarseLock) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  perf.warmup([&] { addUnderCoarseLock(total, PARTS); });
  perf.contentionRun([&] { addUnderCoarseLock(total, PARTS); }, "coarse_lock");
}

/** @test Threads add joined lengths to totals of their own, sharing nothing while they run. */
PERF_CONTENTION(OffCpu, NoSharing) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);

  perf.warmup([&] { addToThreadTotal(PARTS); });
  perf.contentionRun([&] { addToThreadTotal(PARTS); }, "no_sharing");
}

PERF_MAIN()
