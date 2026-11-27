/**
 * @file PerfMeasuredTime_uTest.cpp
 * @brief The per-call times the measured loops record, against operations of
 * known duration.
 *
 * Notes:
 *  - These cases run in the serial Timing lane (TIMING_PATTERNS in this
 *    directory's CMakeLists.txt).
 */

#include "src/bench/inc/PerfHarness.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::PerfResult;

namespace {

/// Spin on the steady clock for at least @p duration.
void spinFor(std::chrono::nanoseconds duration) {
  const auto UNTIL = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < UNTIL) {
  }
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test Records a sub-microsecond repeat at the clock's resolution, not in whole microseconds */
TEST(PerfCaseTimingTest, SubMicrosecondRepeatIsNotQuantized) {
  PerfConfig cfg;
  cfg.cycles = 1;
  cfg.repeats = 51;
  PerfCase perf{"Timing.SubMicrosecondRepeat", cfg};

  const PerfResult RESULT = perf.measured([] { spinFor(std::chrono::nanoseconds(300)); });

  // Each repeat's window holds its whole 300 ns spin, so no repeat reads
  // less than 0.3 us. Cut to whole microseconds, every repeat shorter than a
  // microsecond reads 0, and most of these are. A median of a whole number of
  // microseconds is a correct reading too, so the median is not constrained.
  EXPECT_GE(RESULT.stats.min, 0.3) << "median " << RESULT.stats.median;
}

/**
 * @test Records a sleep's per-call time within 0.9 to 1.5 times the sleep
 *
 * The 2 ms sleep is the operation measured and the reference the harness is
 * checked against; the case waits for its 11 sleeps (about 25 ms) and nothing
 * else. A sleep lasts at least its duration, so the lower edge holds under any
 * load, and a per-call value cut in half is under 0.75 times the sleep
 * whenever the true median meets the upper edge.
 */
TEST(PerfCaseTimingTest, MeasuredPerCallTimeMatchesASleep) {
  constexpr double SLEEP_US = 2000.0;
  PerfConfig cfg;
  cfg.cycles = 1;
  cfg.repeats = 11;
  PerfCase perf{"Timing.MeasuredSleep", cfg};

  const PerfResult RESULT =
      perf.measured([] { std::this_thread::sleep_for(std::chrono::milliseconds(2)); });

  EXPECT_GE(RESULT.stats.median, 0.9 * SLEEP_US);
  EXPECT_LE(RESULT.stats.median, 1.5 * SLEEP_US);
}

/**
 * @test Divides a contention repeat by the calls of every worker it started
 *
 * Four workers each make 20 calls a repeat of a 2 ms sleep, so a repeat spans
 * about 20 sleeps and holds 80 calls: a quarter of the sleep a call. The timed
 * region also holds the workers' start and join, which 20 calls a repeat keep
 * small beside the sleeps. The case waits for its 5 repeats (about 0.2 s) and
 * nothing else. A sleep lasts at least its duration, so a repeat lasts at
 * least 20 sleeps: the lower edge holds under any load. A divisor without the
 * workers reads that repeat over 20 calls, at least a whole sleep a call under
 * any load, and the upper edge is just below it; the divisor that counts every
 * worker reaches it only if a repeat takes four times its sleeps.
 */
TEST(PerfCaseTimingTest, ContentionPerCallTimeCountsEveryWorker) {
  constexpr int THREADS = 4;
  constexpr double SLEEP_US = 2000.0;
  PerfConfig cfg;
  cfg.cycles = 20;
  cfg.repeats = 5;
  cfg.threads = THREADS;
  PerfCase perf{"Timing.ContentionSleep", cfg};

  const PerfResult RESULT =
      perf.contentionRun([] { std::this_thread::sleep_for(std::chrono::milliseconds(2)); });

  const double PER_CALL_US = SLEEP_US / THREADS;
  EXPECT_GE(RESULT.stats.median, 0.9 * PER_CALL_US);
  EXPECT_LT(RESULT.stats.median, SLEEP_US);
}
