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
#include <cmath>

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
  // less than 0.3 us. Cut to whole microseconds, most read 0 and the rest 1.
  EXPECT_GE(RESULT.stats.min, 0.3) << "median " << RESULT.stats.median;
  EXPECT_NE(RESULT.stats.median, std::floor(RESULT.stats.median))
      << "the median is a whole number of microseconds";
}
