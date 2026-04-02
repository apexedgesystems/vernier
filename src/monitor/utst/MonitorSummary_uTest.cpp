/**
 * @file MonitorSummary_uTest.cpp
 * @brief Unit tests for vernier::monitor::MonitorSummary.
 *
 * Notes:
 *  - Tests are platform-agnostic: assert invariants, not exact values.
 */

#include "src/monitor/inc/MonitorSummary.hpp"

#include <gtest/gtest.h>

#include <cstring>

#include <string>

using vernier::monitor::MonitorSummary;
using vernier::monitor::MonitorTag;
using vernier::monitor::Sample;
using vernier::monitor::SampleKind;

/* ----------------------------- Default Construction ----------------------------- */

/** @test Empty summary has zero entries */
TEST(MonitorSummaryDefaultTest, EmptySummary) {
  const MonitorSummary SUMMARY;
  EXPECT_EQ(SUMMARY.size(), 0u);
}

/* ----------------------------- MonitorSummary Method Tests ----------------------------- */

/** @test Recording scope samples accumulates stats */
TEST(MonitorSummaryTest, ScopeAccumulation) {
  MonitorSummary summary;

  Sample s;
  s.tag = MonitorTag("test", 1);
  std::strncpy(s.scope, "work", sizeof(s.scope));
  s.kind = SampleKind::SCOPE;

  s.durationNs = 1000000; // 1ms
  summary.record(s);

  s.durationNs = 3000000; // 3ms
  summary.record(s);

  s.durationNs = 2000000; // 2ms
  summary.record(s);

  EXPECT_EQ(summary.size(), 1u);

  const auto& ENTRIES = summary.entries();
  const auto IT = ENTRIES.find("test/1::work");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 3u);
  EXPECT_DOUBLE_EQ(IT->second.median(), 2.0); // 2ms median
  EXPECT_DOUBLE_EQ(IT->second.minVal, 1.0);
  EXPECT_DOUBLE_EQ(IT->second.maxVal, 3.0);
}

/** @test Breaches are counted separately */
TEST(MonitorSummaryTest, BreachCounting) {
  MonitorSummary summary;

  Sample normal;
  normal.tag = MonitorTag("x", 1);
  std::strncpy(normal.scope, "s", sizeof(normal.scope));
  normal.kind = SampleKind::SCOPE;
  normal.durationNs = 1000000;
  summary.record(normal);

  Sample breach;
  breach.tag = MonitorTag("x", 1);
  std::strncpy(breach.scope, "s", sizeof(breach.scope));
  breach.kind = SampleKind::THRESHOLD_BREACH;
  breach.durationNs = 5000000;
  breach.value = 2000; // threshold in us
  summary.record(breach);

  const auto& ENTRIES = summary.entries();
  const auto IT = ENTRIES.find("x/1::s");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 2u);
  EXPECT_EQ(IT->second.breaches, 1u);
}

/** @test Counter samples accumulate sum */
TEST(MonitorSummaryTest, CounterSum) {
  MonitorSummary summary;

  Sample s;
  s.tag = MonitorTag("cnt", 2);
  std::strncpy(s.scope, "frames", sizeof(s.scope));
  s.kind = SampleKind::COUNTER;

  s.value = 1.0;
  summary.record(s);
  s.value = 5.0;
  summary.record(s);

  const auto& ENTRIES = summary.entries();
  const auto IT = ENTRIES.find("cnt/2::frames");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 2u);
  EXPECT_DOUBLE_EQ(IT->second.sum, 6.0);
}

/** @test Gauge records point-in-time values */
TEST(MonitorSummaryTest, GaugeValues) {
  MonitorSummary summary;

  Sample s;
  s.tag = MonitorTag("g", 3);
  std::strncpy(s.scope, "depth", sizeof(s.scope));
  s.kind = SampleKind::GAUGE;

  s.value = 10.0;
  summary.record(s);
  s.value = 50.0;
  summary.record(s);
  s.value = 30.0;
  summary.record(s);

  const auto& ENTRIES = summary.entries();
  const auto IT = ENTRIES.find("g/3::depth");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_DOUBLE_EQ(IT->second.median(), 30.0);
  EXPECT_DOUBLE_EQ(IT->second.maxVal, 50.0);
  EXPECT_DOUBLE_EQ(IT->second.minVal, 10.0);
}

/** @test Multiple tags create separate entries */
TEST(MonitorSummaryTest, SeparateTags) {
  MonitorSummary summary;

  Sample s1;
  s1.tag = MonitorTag("decoder", 1);
  std::strncpy(s1.scope, "work", sizeof(s1.scope));
  s1.kind = SampleKind::SCOPE;
  s1.durationNs = 1000000;
  summary.record(s1);

  Sample s2;
  s2.tag = MonitorTag("encoder", 2);
  std::strncpy(s2.scope, "work", sizeof(s2.scope));
  s2.kind = SampleKind::SCOPE;
  s2.durationNs = 2000000;
  summary.record(s2);

  EXPECT_EQ(summary.size(), 2u);
  EXPECT_NE(summary.entries().find("decoder/1::work"), summary.entries().end());
  EXPECT_NE(summary.entries().find("encoder/2::work"), summary.entries().end());
}
