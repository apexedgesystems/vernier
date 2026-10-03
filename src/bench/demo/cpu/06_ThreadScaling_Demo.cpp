/**
 * @file 06_ThreadScaling_Demo.cpp
 * @brief Demo 06: what a call costs while other threads make the same call
 *
 * Threads join the shared example's words and add the joined length to one
 * total, two ways (06_ThreadScaling_Totals.hpp), one CSV row each:
 *  1. CoarseLock: one lock, held for the whole call, so the joins take turns
 *  2. NoSharing: each thread adds to a total of its own and hands it over
 *     once, when the thread ends
 * contentionRun() starts --threads threads that make the call at the same
 * time; each row records how many.
 *
 * Usage (a core for each thread, and one for the thread that starts them):
 *   @code{.sh}
 *   # One thread
 *   taskset -c 2,3 ./BenchDemo_06_ThreadScaling --threads 1 --repeats 10 --csv one.csv
 *
 *   # Three threads
 *   taskset -c 0-3 ./BenchDemo_06_ThreadScaling --threads 3 --repeats 10 --csv three.csv
 *   @endcode
 *
 * That both versions reach the same total is checked apart from the demo, by
 * utst/06_ThreadScaling_uTest.cpp.
 *
 * @see docs/06_THREAD_SCALING.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/cpu/06_ThreadScaling_Totals.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

using vernier::bench::demo::thread_scaling_demo::addToThreadTotal;
using vernier::bench::demo::thread_scaling_demo::addUnderCoarseLock;
using vernier::bench::demo::thread_scaling_demo::PART_COUNT;
using vernier::bench::demo::thread_scaling_demo::PART_SEED;
using vernier::bench::demo::thread_scaling_demo::SharedTotal;

/* ----------------------------- Tests ----------------------------- */

/** @test Threads add joined lengths to one total, holding its lock for the whole call. */
PERF_CONTENTION(ThreadScaling, CoarseLock) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  perf.warmup([&] { addUnderCoarseLock(total, PARTS); });
  perf.contentionRun([&] { addUnderCoarseLock(total, PARTS); }, "coarse_lock");
}

/** @test Threads add joined lengths to totals of their own, sharing nothing while they run. */
PERF_CONTENTION(ThreadScaling, NoSharing) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);

  perf.warmup([&] { addToThreadTotal(PARTS); });
  perf.contentionRun([&] { addToThreadTotal(PARTS); }, "no_sharing");
}

PERF_MAIN()
