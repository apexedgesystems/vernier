/**
 * @file 13_OffCpuProfiler_uTest.cpp
 * @brief The check behind walkthrough 16: both of demo 13's versions reach the
 *        same total.
 *
 * Not part of demo 13, whose source shows only what it teaches. This program
 * runs each version through contentionRun(), as the demo does, with a fixed
 * number of threads, calls and repeats, and requires every call's joined
 * length in the total. Where the threads wait, and for how long, is measured
 * on the rig; this checks what the machine's load cannot change. ctest runs it
 * under the demo label.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L demo
 *   ./build/bin/tests/TestDemoOffCpu      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/13_OffCpuProfiler_Totals.hpp"

#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"

#include <cstddef>

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace demo = vernier::bench::demo;

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::demo::offcpu_demo::addToThreadTotal;
using vernier::bench::demo::offcpu_demo::addUnderCoarseLock;
using vernier::bench::demo::offcpu_demo::finishedThreadsTotal;
using vernier::bench::demo::offcpu_demo::PART_COUNT;
using vernier::bench::demo::offcpu_demo::PART_SEED;
using vernier::bench::demo::offcpu_demo::SharedTotal;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The run each version gets: threads, calls per thread and repeats.
constexpr int THREADS = 4;
constexpr int CALLS_PER_THREAD = 25;
constexpr int REPEATS = 3;

/// A configuration under which contentionRun() makes exactly
/// THREADS x CALLS_PER_THREAD x REPEATS calls.
PerfConfig fixedRun() {
  PerfConfig cfg;
  cfg.threads = THREADS;
  cfg.cycles = CALLS_PER_THREAD;
  cfg.repeats = REPEATS;
  return cfg;
}

/// Every call's joined length, added up.
std::size_t everyCallsLength(const std::vector<std::string>& parts) {
  return static_cast<std::size_t>(THREADS) * CALLS_PER_THREAD * REPEATS * demo::joinedSize(parts);
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test The coarse-lock version's shared total holds every call's joined length */
TEST(OffCpuTotalsTest, CoarseLockCountsEveryCall) {
  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  PerfCase perf{"OffCpuTotalsTest.CoarseLock", fixedRun()};
  perf.contentionRun([&] { addUnderCoarseLock(total, PARTS); }, "coarse_lock");

  EXPECT_EQ(total.value, everyCallsLength(PARTS));
}

/** @test Every no-sharing thread has handed its total over when contentionRun returns */
TEST(OffCpuTotalsTest, NoSharingCountsEveryCallOnceItsThreadsEnd) {
  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  finishedThreadsTotal = 0;

  PerfCase perf{"OffCpuTotalsTest.NoSharing", fixedRun()};
  perf.contentionRun([&] { addToThreadTotal(PARTS); }, "no_sharing");

  EXPECT_EQ(finishedThreadsTotal.load(), everyCallsLength(PARTS));
}
