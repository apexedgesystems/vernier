/**
 * @file PerfTargetTime_uTest.cpp
 * @brief --target-time building blocks: the duration parser and the
 * calibration math (pure functions; the timed path is exercised live).
 */
#include <gtest/gtest.h>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"

#include <chrono>
#include <cstdint>
#include <thread>

namespace {

using vernier::bench::calibratedCycles;
using vernier::bench::parseDurationUs;

TEST(ParseDurationUs, SuffixesAndBareMilliseconds) {
  EXPECT_EQ(parseDurationUs("500us"), 500);
  EXPECT_EQ(parseDurationUs("250ms"), 250000);
  EXPECT_EQ(parseDurationUs("2s"), 2000000);
  EXPECT_EQ(parseDurationUs("100"), 100000); // bare number = milliseconds
  EXPECT_EQ(parseDurationUs("0.5s"), 500000);
  EXPECT_EQ(parseDurationUs("1.5ms"), 1500);
}

TEST(ParseDurationUs, RejectsGarbage) {
  EXPECT_EQ(parseDurationUs(""), -1);
  EXPECT_EQ(parseDurationUs("abc"), -1);
  EXPECT_EQ(parseDurationUs("10xs"), -1);
  EXPECT_EQ(parseDurationUs("-5ms"), -1);
  EXPECT_EQ(parseDurationUs("0"), -1);
  EXPECT_EQ(parseDurationUs("ms"), -1);
}

TEST(CalibratedCycles, TargetOverEstimate) {
  EXPECT_EQ(calibratedCycles(100000, 10.0), 10000); // 100ms of 10us calls
  EXPECT_EQ(calibratedCycles(1000, 1.0), 1000);
}

TEST(CalibratedCycles, ClampsBothEnds) {
  EXPECT_EQ(calibratedCycles(10, 1000.0), 1);               // floor
  EXPECT_EQ(calibratedCycles(60000, 200000.0), 1);          // 60ms target, 200ms call
  EXPECT_EQ(calibratedCycles(5000, 1000.0), 5);             // below ten is not rounded up
  EXPECT_EQ(calibratedCycles(2000000000, 0.001), 50000000); // ceiling
  // Sub-precision estimates hit the safety floor rather than exploding.
  EXPECT_EQ(calibratedCycles(1000, 0.0), 1000000);
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test Sizes a tens-of-nanoseconds operation so one repeat spans about the requested time */
TEST(PerfCaseTargetTimeTest, FastOperationRoundNearTarget) {
  constexpr int TARGET_US = 40000;
  constexpr int REPEATS = 5;
  vernier::bench::PerfConfig cfg;
  cfg.targetTimeUs = TARGET_US;
  cfg.repeats = REPEATS;

  // Sixteen xorshift steps through a volatile: tens of nanoseconds per call,
  // far below the microsecond clock one call would be timed with.
  volatile std::uint64_t state = 88172645463325252ULL;
  const auto op = [&] {
    for (int i = 0; i < 16; ++i) {
      std::uint64_t x = state;
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      state = x;
    }
  };

  // Bring the core out of idle first, as a benchmark's warmup phase does;
  // otherwise the sample measures the frequency ramp, not the operation.
  const auto spinUntil = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
  while (std::chrono::steady_clock::now() < spinUntil) {
    op();
  }

  vernier::bench::PerfCase perf{"TargetTime.FastOperation", cfg};
  const auto begin = std::chrono::steady_clock::now();
  const vernier::bench::PerfResult result = perf.throughputLoop(op);
  const double wallUs =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();

  // One repeat = cycles x per-call median. A factor of four either way
  // absorbs scheduler noise; a miscalibration is off by an order of magnitude.
  const double roundUs = result.stats.median * static_cast<double>(perf.cycles());
  EXPECT_GT(roundUs, TARGET_US / 4.0) << "cycles=" << perf.cycles();
  EXPECT_LT(roundUs, TARGET_US * 4.0) << "cycles=" << perf.cycles();

  // Nominal run is calibration plus five 40 ms repeats (0.2 s).
  EXPECT_LT(wallUs, 1e6) << "cycles=" << perf.cycles();
}

/** @test Runs one cycle per repeat when a single call already exceeds the requested time */
TEST(PerfCaseTargetTimeTest, SlowOperationRunsOneCycle) {
  constexpr int REPEATS = 2;
  vernier::bench::PerfConfig cfg;
  cfg.targetTimeUs = 5000;
  cfg.repeats = REPEATS;

  int calls = 0;
  const auto op = [&] {
    ++calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  };

  vernier::bench::PerfCase perf{"TargetTime.SlowOperation", cfg};
  (void)perf.throughputLoop(op);

  EXPECT_EQ(perf.cycles(), 1);
  // One calibration call, then one call per repeat.
  EXPECT_EQ(calls, 1 + REPEATS);
}
