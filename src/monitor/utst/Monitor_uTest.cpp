/**
 * @file Monitor_uTest.cpp
 * @brief Unit tests for vernier::monitor::Monitor.
 *
 * Notes:
 *  - Tests are platform-agnostic: assert invariants, not exact values.
 *  - Platform-specific features gracefully skip when unavailable.
 */

#include "src/monitor/inc/Monitor.hpp"

#include <gtest/gtest.h>

#include <cstdio>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

using vernier::monitor::Monitor;
using vernier::monitor::MonitorConfig;
using vernier::monitor::MonitorTag;
using vernier::monitor::ScopeGuard;
using vernier::monitor::SINK_FILE;
using vernier::monitor::SINK_NONE;

/* ----------------------------- Monitor Method Tests ----------------------------- */

/** @test Monitor starts and stops cleanly */
TEST(MonitorTest, StartStop) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);

  mon.start();
  EXPECT_TRUE(mon.isRunning());
  mon.stop();
  EXPECT_FALSE(mon.isRunning());
}

/** @test Double start is idempotent */
TEST(MonitorTest, DoubleStart) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);

  mon.start();
  mon.start(); // Should not crash
  mon.stop();
}

/** @test Double stop is idempotent */
TEST(MonitorTest, DoubleStop) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);

  mon.start();
  mon.stop();
  mon.stop(); // Should not crash
}

/** @test Scope recording works end-to-end */
TEST(MonitorTest, ScopeRecording) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("test", 1);
  {
    ScopeGuard guard(mon, "work", TAG);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Allow I/O thread to drain
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  EXPECT_EQ(mon.summary().size(), 1u);
  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("test/1::work");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 1u);
  EXPECT_GT(IT->second.median(), 0.0); // Duration in ms
}

/** @test Counter increment recording */
TEST(MonitorTest, CounterIncrement) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("counter", 2);
  mon.increment("frames", TAG, 1.0);
  mon.increment("frames", TAG, 1.0);
  mon.increment("frames", TAG, 1.0);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("counter/2::frames");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 3u);
  EXPECT_DOUBLE_EQ(IT->second.sum, 3.0);
}

/** @test Gauge recording */
TEST(MonitorTest, GaugeRecording) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("gauge", 3);
  mon.gauge("depth", TAG, 42.0);
  mon.gauge("depth", TAG, 100.0);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("gauge/3::depth");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 2u);
  EXPECT_DOUBLE_EQ(IT->second.maxVal, 100.0);
}

/** @test Threshold breach detection */
TEST(MonitorTest, ThresholdBreach) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.setThreshold("work", 4, 1000); // 1ms threshold
  mon.start();

  const MonitorTag TAG("slow", 4);

  // Record a scope that exceeds 1ms
  {
    ScopeGuard guard(mon, "work", TAG);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("slow/4::work");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_GE(IT->second.breaches, 1u);
}

/** @test Disabled monitor skips all recording */
TEST(MonitorTest, DisabledSkips) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  cfg.enabled = false;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("skip", 5);
  mon.increment("counter", TAG, 1.0);
  {
    ScopeGuard guard(mon, "work", TAG);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  EXPECT_EQ(mon.summary().size(), 0u);
}

/** @test Runtime enable/disable toggle */
TEST(MonitorTest, RuntimeToggle) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("toggle", 6);

  // Record one sample while enabled
  mon.increment("counter", TAG, 1.0);

  // Disable and record another
  mon.setEnabled(false);
  mon.increment("counter", TAG, 1.0);

  // Re-enable and record another
  mon.setEnabled(true);
  mon.increment("counter", TAG, 1.0);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("toggle/6::counter");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 2u); // Skipped the disabled one
}

/** @test File sink writes output */
TEST(MonitorTest, FileSinkOutput) {
  const auto TMP_PATH = std::filesystem::temp_directory_path() / "vernier_mon_test.log";
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.sinks = SINK_FILE;
  cfg.filePath = TMP_PATH.string();
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("file", 7);
  mon.increment("event", TAG, 1.0);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  // Verify file was written
  std::ifstream f(TMP_PATH);
  ASSERT_TRUE(f.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(f, line));
  EXPECT_TRUE(line.find("COUNTER") != std::string::npos);
  EXPECT_TRUE(line.find("file/7") != std::string::npos);
  EXPECT_TRUE(line.find("event") != std::string::npos);

  std::filesystem::remove(TMP_PATH);
}

/** @test VERNIER_MONITOR_SCOPE macro compiles and works */
TEST(MonitorTest, ScopeMacro) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("macro", 8);
  {
    VERNIER_MONITOR_SCOPE(mon, "stage", TAG);
    // Simulate work
    volatile int x = 0;
    for (int i = 0; i < 100; ++i)
      x += i;
    (void)x;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  mon.stop();

  EXPECT_GE(mon.summary().size(), 1u);
}
