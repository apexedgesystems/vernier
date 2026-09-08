/**
 * @file PerfTargetTime_uTest.cpp
 * @brief --target-time building blocks: the duration parser and the
 * calibration math (pure functions; the timed path is exercised live).
 */
#include <gtest/gtest.h>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"

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
  EXPECT_EQ(calibratedCycles(10, 1000.0), 10);              // floor
  EXPECT_EQ(calibratedCycles(2000000000, 0.001), 50000000); // ceiling
  // Sub-precision estimates hit the safety floor rather than exploding.
  EXPECT_EQ(calibratedCycles(1000, 0.0), 1000000);
}

} // namespace
