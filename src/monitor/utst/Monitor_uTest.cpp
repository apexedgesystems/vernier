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

#include <unistd.h>

#include <cstddef>
#include <cstdint>
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
using vernier::monitor::SINK_CONSOLE;
using vernier::monitor::SINK_FILE;
using vernier::monitor::SINK_NONE;

namespace {

/* ----------------------------- Test Helpers ----------------------------- */

/// Number of newline-terminated records in a sink file (0 if it does not exist).
std::size_t countLines(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    return 0;
  }
  std::size_t lines = 0;
  std::string line;
  while (std::getline(f, line)) {
    ++lines;
  }
  return lines;
}

/// Unique sink path per test, so a shuffled or repeated run never shares a file.
std::filesystem::path sinkPath(const char* stem) {
  return std::filesystem::temp_directory_path() /
         ("vernier_mon_" + std::string(stem) + "_" + std::to_string(::getpid()) + ".log");
}

/**
 * @brief Redirects the process's stderr to a temp file until text() is read.
 *
 * Works at the file-descriptor level, so it sees what the console sink and the
 * summary write from the monitor's I/O thread as well as from this one.
 */
class StderrCapture {
public:
  StderrCapture() : file_(std::tmpfile()) {
    if (file_ == nullptr) {
      return;
    }
    std::fflush(stderr);
    saved_ = ::dup(STDERR_FILENO);
    ::dup2(::fileno(file_), STDERR_FILENO);
  }

  ~StderrCapture() {
    restore();
    if (file_ != nullptr) {
      std::fclose(file_);
    }
  }

  StderrCapture(const StderrCapture&) = delete;
  StderrCapture& operator=(const StderrCapture&) = delete;

  /// Stop capturing and return everything written so far.
  std::string text() {
    restore();
    std::string out;
    if (file_ == nullptr) {
      return out;
    }
    std::rewind(file_);
    char buf[256];
    std::size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), file_)) > 0) {
      out.append(buf, got);
    }
    return out;
  }

private:
  void restore() {
    if (saved_ >= 0) {
      std::fflush(stderr);
      ::dup2(saved_, STDERR_FILENO);
      ::close(saved_);
      saved_ = -1;
    }
  }

  std::FILE* file_;
  int saved_ = -1;
};

} // namespace

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

/**
 * @test Samples still queued when stop() is called reach both the summary and
 *       the configured sink.
 *
 * The producer runs to completion and calls stop() immediately, leaving the
 * queue non-empty: stop()'s join is the synchronization point, so no sleep is
 * needed and none is used. Repeated rounds because a single round could find
 * the queue already empty.
 */
TEST(MonitorTest, StopDrainsSamplesQueuedBeforeStop) {
  const auto TMP_PATH = sinkPath("pending");
  constexpr int ROUNDS = 20;
  constexpr unsigned long SAMPLES_PER_ROUND = 64;
  const MonitorTag TAG("pending", 9);

  for (int round = 0; round < ROUNDS; ++round) {
    std::filesystem::remove(TMP_PATH);

    MonitorConfig cfg;
    cfg.sinks = SINK_FILE;
    cfg.filePath = TMP_PATH.string();
    cfg.queueCapacity = 4096; // >> SAMPLES_PER_ROUND: overflow cannot explain a loss
    Monitor mon(cfg);
    mon.start();

    for (unsigned long i = 0; i < SAMPLES_PER_ROUND; ++i) {
      mon.increment("queued", TAG, 1.0);
    }
    mon.stop();

    ASSERT_EQ(mon.queue().droppedCount(), 0u) << "round " << round;

    const auto& ENTRIES = mon.summary().entries();
    const auto IT = ENTRIES.find("pending/9::queued");
    ASSERT_NE(IT, ENTRIES.end()) << "round " << round;
    EXPECT_EQ(IT->second.count, SAMPLES_PER_ROUND) << "summary, round " << round;
    EXPECT_EQ(countLines(TMP_PATH), SAMPLES_PER_ROUND) << "file sink, round " << round;
  }

  std::filesystem::remove(TMP_PATH);
}

/* ----------------------------- Lifecycle Contract Tests ----------------------------- */

/** @test Construction alone starts no worker, opens no file and reports nothing */
TEST(MonitorTest, ConstructionStartsNothing) {
  const auto TMP_PATH = sinkPath("constructed");
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.sinks = static_cast<std::uint8_t>(SINK_CONSOLE | SINK_FILE);
  cfg.filePath = TMP_PATH.string();

  const MonitorTag TAG("unstarted", 14);
  std::string captured;
  {
    StderrCapture capture;
    {
      Monitor mon(cfg);
      EXPECT_FALSE(mon.isRunning());
      mon.increment("events", TAG, 1.0); // queued, but nothing drains it
      EXPECT_EQ(mon.summary().size(), 0u);
    } // destructor: stop() finds nothing to stop
    captured = capture.text();
  }

  EXPECT_TRUE(captured.empty()) << "stderr: " << captured;
  EXPECT_FALSE(std::filesystem::exists(TMP_PATH)) << "output file opened without start()";

  std::filesystem::remove(TMP_PATH);
}

/** @test A disabled start() creates no worker, opens no file and prints nothing */
TEST(MonitorTest, DisabledStartIsInert) {
  const auto TMP_PATH = sinkPath("disabled_start");
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.enabled = false;
  cfg.sinks = static_cast<std::uint8_t>(SINK_CONSOLE | SINK_FILE);
  cfg.filePath = TMP_PATH.string();
  Monitor mon(cfg);

  std::string captured;
  {
    StderrCapture capture;
    mon.start();
    EXPECT_FALSE(mon.isRunning());
    mon.stop();
    captured = capture.text();
  }

  EXPECT_TRUE(captured.empty()) << "stderr: " << captured;
  EXPECT_FALSE(std::filesystem::exists(TMP_PATH)) << "output file opened while disabled";
  EXPECT_EQ(mon.summary().size(), 0u);

  std::filesystem::remove(TMP_PATH);
}

/** @test setEnabled(true) followed by start() activates a monitor that started disabled */
TEST(MonitorTest, EnableThenStartActivatesAfterDisabledStart) {
  MonitorConfig cfg;
  cfg.enabled = false;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);

  mon.start();
  ASSERT_FALSE(mon.isRunning());

  mon.setEnabled(true);
  mon.start();
  EXPECT_TRUE(mon.isRunning());

  const MonitorTag TAG("late", 10);
  mon.increment("events", TAG, 1.0);
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("late/10::events");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 1u);
}

/** @test A file-only monitor collects and writes without printing to the console */
TEST(MonitorTest, ConsoleOffSuppressesAutomaticSummary) {
  const auto TMP_PATH = sinkPath("file_only");
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.sinks = SINK_FILE;
  cfg.filePath = TMP_PATH.string();
  Monitor mon(cfg);

  const MonitorTag TAG("fileonly", 11);
  std::string captured;
  {
    StderrCapture capture;
    mon.start();
    mon.increment("events", TAG, 1.0);
    mon.stop();
    captured = capture.text();
  }

  EXPECT_TRUE(captured.empty()) << "stderr: " << captured;

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("fileonly/11::events");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 1u);
  EXPECT_EQ(countLines(TMP_PATH), 1u);

  std::filesystem::remove(TMP_PATH);
}

/** @test SINK_NONE still collects in memory and writes nowhere */
TEST(MonitorTest, SinkNoneCollectsInMemoryAndPrintsNothing) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);

  const MonitorTag TAG("silent", 12);
  std::string captured;
  {
    StderrCapture capture;
    mon.start();
    mon.increment("events", TAG, 1.0);
    mon.increment("events", TAG, 1.0);
    mon.increment("events", TAG, 1.0);
    mon.stop();
    captured = capture.text();
  }

  EXPECT_TRUE(captured.empty()) << "stderr: " << captured;

  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("silent/12::events");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 3u);
}

/**
 * @test Disabling a running monitor rejects later samples and keeps earlier ones,
 *       including in the summary printed at stop.
 */
TEST(MonitorTest, DisableMidRunKeepsEarlierHistory) {
  const auto TMP_PATH = sinkPath("disable_midrun");
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.sinks = static_cast<std::uint8_t>(SINK_CONSOLE | SINK_FILE);
  cfg.filePath = TMP_PATH.string();
  Monitor mon(cfg);

  const MonitorTag TAG("history", 13);
  std::string captured;
  {
    StderrCapture capture;
    mon.start();
    mon.increment("kept", TAG, 1.0);
    mon.increment("kept", TAG, 1.0);
    mon.increment("kept", TAG, 1.0);

    mon.setEnabled(false);
    mon.increment("rejected", TAG, 1.0);
    mon.increment("rejected", TAG, 1.0);

    mon.stop();
    captured = capture.text();
  }

  EXPECT_NE(captured.find("vernier::monitor summary"), std::string::npos) << "stderr: " << captured;
  EXPECT_NE(captured.find("history/13"), std::string::npos) << "stderr: " << captured;

  const auto& ENTRIES = mon.summary().entries();
  const auto KEPT = ENTRIES.find("history/13::kept");
  ASSERT_NE(KEPT, ENTRIES.end());
  EXPECT_EQ(KEPT->second.count, 3u);
  EXPECT_EQ(ENTRIES.find("history/13::rejected"), ENTRIES.end());
  EXPECT_EQ(mon.queue().droppedCount(), 0u);
  EXPECT_EQ(countLines(TMP_PATH), 3u);

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
