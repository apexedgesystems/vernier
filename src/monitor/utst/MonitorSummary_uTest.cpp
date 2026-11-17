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
using vernier::monitor::ScopeStats;

/* ----------------------------- Default Construction ----------------------------- */

/** @test Empty summary has zero entries */
TEST(MonitorSummaryDefaultTest, EmptySummary) {
  const MonitorSummary SUMMARY;
  EXPECT_EQ(SUMMARY.size(), 0u);
}

/* ----------------------------- ScopeStats Method Tests ----------------------------- */

/** @test Takes an even count's median as the mean of the two middle values */
TEST(ScopeStatsTest, EvenCountMedianAveragesTheMiddlePair) {
  ScopeStats stats;
  for (const double V : {4.0, 1.0, 3.0, 2.0}) {
    stats.record(V, false);
  }
  EXPECT_DOUBLE_EQ(stats.median(), 2.5);
}

/** @test Takes P99 as the nearest rank: 198 of 1..200, 99 of 1..100, a lone value itself */
TEST(ScopeStatsTest, P99IsTheNearestRank) {
  const auto P99_OF_ONE_TO = [](int n) {
    ScopeStats stats;
    for (int v = n; v >= 1; --v) {
      stats.record(static_cast<double>(v), false);
    }
    return stats.p99();
  };
  EXPECT_DOUBLE_EQ(P99_OF_ONE_TO(200), 198.0);
  EXPECT_DOUBLE_EQ(P99_OF_ONE_TO(100), 99.0);

  // p99() copies and sorts this one value. Inlined here, g++ 13 and 14 at
  // -O3 report the memmove libstdc++ uses for more than one element as past
  // the end (-Warray-bounds, and with that off -Wstringop-overflow and
  // -Wstringop-overread); one element never takes that path. The three are
  // off for these three lines only.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#endif
  ScopeStats lone;
  lone.record(7.5, false);
  EXPECT_DOUBLE_EQ(lone.p99(), 7.5);
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif
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

/** @test Gauge records point-in-time values, its range starting at the first reading */
TEST(MonitorSummaryTest, GaugeValues) {
  MonitorSummary summary;

  Sample s;
  s.tag = MonitorTag("g", 3);
  std::strncpy(s.scope, "depth", sizeof(s.scope));
  s.kind = SampleKind::GAUGE;

  // All negative: a maximum that started at 0 would stay 0, above every reading.
  s.value = -5.0;
  summary.record(s);
  s.value = -3.0;
  summary.record(s);
  s.value = -9.0;
  summary.record(s);

  const auto& ENTRIES = summary.entries();
  const auto IT = ENTRIES.find("g/3::depth");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_DOUBLE_EQ(IT->second.median(), -5.0);
  EXPECT_DOUBLE_EQ(IT->second.maxVal, -3.0);
  EXPECT_DOUBLE_EQ(IT->second.minVal, -9.0);
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
