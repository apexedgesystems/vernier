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

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

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

/// A sink file's records, each split at its tabs (none if the file does not exist).
std::vector<std::vector<std::string>> readRecords(const std::filesystem::path& path) {
  std::vector<std::vector<std::string>> records;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (;;) {
      const std::size_t TAB = line.find('\t', start);
      fields.push_back(line.substr(start, TAB == std::string::npos ? TAB : TAB - start));
      if (TAB == std::string::npos) {
        break;
      }
      start = TAB + 1;
    }
    records.push_back(fields);
  }
  return records;
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

/** @test Records a scope's duration from its two timestamps, and its start as its timestamp */
TEST(MonitorTest, ScopeRecording) {
  const auto TMP_PATH = sinkPath("scope");
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.sinks = SINK_FILE;
  cfg.filePath = TMP_PATH.string();
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("test", 1);
  mon.recordScope("x", TAG, 1000, 2501000); // 2.5 ms from 1000 ns
  mon.stop();

  EXPECT_EQ(mon.summary().size(), 1u);
  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("test/1::x");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 1u);
  EXPECT_DOUBLE_EQ(IT->second.median(), 2.5); // ms

  const auto RECORDS = readRecords(TMP_PATH);
  ASSERT_EQ(RECORDS.size(), 1u);
  ASSERT_GE(RECORDS[0].size(), 5u);
  EXPECT_EQ(RECORDS[0][0], "1000");    // the timestamp: the scope's start
  EXPECT_EQ(RECORDS[0][4], "2500000"); // the duration in nanoseconds

  std::filesystem::remove(TMP_PATH);
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

/** @test Flags only a scope longer than the threshold of its own scope and tag id */
TEST(MonitorTest, ThresholdBreach) {
  const auto TMP_PATH = sinkPath("breach");
  std::filesystem::remove(TMP_PATH);

  MonitorConfig cfg;
  cfg.sinks = SINK_FILE;
  cfg.filePath = TMP_PATH.string();
  Monitor mon(cfg);
  mon.setThreshold("work", 4, 1000); // 1 ms
  mon.start();

  // Under, at and over the threshold (only longer breaches), then 5 ms on the
  // same scope with another tag id and on another scope with the same id.
  const MonitorTag TAG4("slow", 4);
  const MonitorTag TAG5("slow", 5);
  mon.recordScope("work", TAG4, 1000, 1000 + 999000);
  mon.recordScope("work", TAG4, 1000, 1000 + 1000000);
  mon.recordScope("work", TAG4, 1000, 1000 + 1001000);
  mon.recordScope("work", TAG5, 1000, 1000 + 5000000);
  mon.recordScope("other", TAG4, 1000, 1000 + 5000000);
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto WORK4 = ENTRIES.find("slow/4::work");
  const auto WORK5 = ENTRIES.find("slow/5::work");
  const auto OTHER4 = ENTRIES.find("slow/4::other");
  ASSERT_NE(WORK4, ENTRIES.end());
  ASSERT_NE(WORK5, ENTRIES.end());
  ASSERT_NE(OTHER4, ENTRIES.end());
  EXPECT_EQ(WORK4->second.count, 3u);
  EXPECT_EQ(WORK4->second.breaches, 1u);
  EXPECT_EQ(WORK5->second.breaches, 0u);
  EXPECT_EQ(OTHER4->second.breaches, 0u);

  // The breach's record carries the threshold, in microseconds, as its value.
  const auto RECORDS = readRecords(TMP_PATH);
  ASSERT_EQ(RECORDS.size(), 5u);
  std::size_t breaches = 0;
  for (const auto& record : RECORDS) {
    if (record.size() == 7 && record[6] == "THRESHOLD_BREACH") {
      ++breaches;
      EXPECT_EQ(record[2], "slow/4");
      EXPECT_EQ(record[3], "work");
      EXPECT_EQ(record[4], "1001000");
      EXPECT_EQ(record[5], "1000.000000");
    }
  }
  EXPECT_EQ(breaches, 1u);

  std::filesystem::remove(TMP_PATH);
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
 * Each round pushes 64 samples and calls stop() at once. Nothing holds the I/O
 * thread back, so a round tests stop() with samples still queued only if the
 * I/O thread, which start() has just created, has not drained them by the
 * time it sees the stop. The pushes take microseconds, so that is the usual
 * case, and a drain that loses a queued sample then fails the round. A round
 * whose samples were drained first passes however stop() treats queued
 * samples, so the round is repeated: a loss goes unseen only if every round
 * drains early. stop() joins the I/O thread, so the checks need no sleep.
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

/** @test A threshold set before a restart flags after it; one set while stopped, from the next */
TEST(MonitorTest, ThresholdSetBeforeARestartStillFlagsAfterIt) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  const MonitorTag TAG("restart", 17);
  mon.setThreshold("work", 17, 1000);
  mon.start();
  mon.stop();

  mon.start(); // the same thresholds
  mon.recordScope("work", TAG, 0, 2000000);
  mon.recordScope("late", TAG, 0, 2000000);
  mon.stop();

  mon.setThreshold("late", 17, 1000);
  mon.start(); // one threshold more
  mon.recordScope("late", TAG, 0, 2000000);
  mon.stop();

  const auto& ENTRIES = mon.summary().entries();
  const auto WORK = ENTRIES.find("restart/17::work");
  ASSERT_NE(WORK, ENTRIES.end());
  EXPECT_EQ(WORK->second.count, 1u);
  EXPECT_EQ(WORK->second.breaches, 1u);
  const auto LATE = ENTRIES.find("restart/17::late");
  ASSERT_NE(LATE, ENTRIES.end());
  EXPECT_EQ(LATE->second.count, 2u);
  EXPECT_EQ(LATE->second.breaches, 1u) << "only the sample after its start() is a breach";
}

/**
 * @test Keeps every threshold table a live producer may be reading across restarts
 *
 * A producer records 2 ms scopes on "work", the last of nine 1000 us thresholds,
 * while the main thread starts and stops the monitor, changing that threshold
 * before each start() so that each start() publishes a new table while the
 * producer may still be scanning the one before. The producer records through
 * each start() and pauses for each stop(), whose contract is that producers
 * have finished. Built with ASan, a table freed under the producer is a
 * heap-use-after-free; with TSan, a publication without ordering is a data race.
 * In any build, each sample recorded after the first round, when a table is
 * published, is a breach, and every sample is either summarized or dropped.
 */
TEST(MonitorTest, ThresholdTableSurvivesRestartWithLiveProducer) {
  constexpr int ROUNDS = 100;
  constexpr int PAUSE = 0;
  constexpr int RECORD = 1;
  constexpr int QUIT = 2;
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  cfg.queueCapacity = 64;
  Monitor mon(cfg);
  for (int i = 0; i < 8; ++i) {
    mon.setThreshold(("stage" + std::to_string(i)).c_str(), 16, 1000);
  }
  mon.setThreshold("work", 16, 1000);

  // The main thread sets the phase; the producer acknowledges it once it acts
  // on it, so a stop() never starts while a sample is being recorded.
  std::atomic<int> phase{PAUSE};
  std::atomic<int> acked{PAUSE};
  std::atomic<int> round{0};
  std::uint64_t calls = 0; // the producer's until it is joined
  const MonitorTag FIRST("first", 16);
  const MonitorTag LATER("later", 16);
  std::thread producer([&] {
    for (;;) {
      const int NOW = phase.load(std::memory_order_acquire);
      if (NOW == QUIT) {
        break;
      }
      if (NOW == RECORD) {
        const MonitorTag& tag = round.load(std::memory_order_relaxed) == 0 ? FIRST : LATER;
        mon.recordScope("work", tag, 0, 2000000);
        ++calls;
      }
      acked.store(NOW, std::memory_order_release);
    }
  });
  const auto setPhase = [&](int next) {
    phase.store(next, std::memory_order_release);
    while (acked.load(std::memory_order_acquire) != next) {
      std::this_thread::yield();
    }
  };

  for (int r = 0; r < ROUNDS; ++r) {
    round.store(r, std::memory_order_relaxed);
    mon.setThreshold("work", 16, 1000 + static_cast<std::uint64_t>(r % 2));
    setPhase(RECORD);
    mon.start();
    setPhase(PAUSE);
    mon.stop();
  }
  phase.store(QUIT, std::memory_order_release);
  producer.join();

  const auto& ENTRIES = mon.summary().entries();
  const auto EARLY = ENTRIES.find("first/16::work");
  const auto LATE = ENTRIES.find("later/16::work");
  ASSERT_NE(EARLY, ENTRIES.end());
  ASSERT_NE(LATE, ENTRIES.end());
  EXPECT_EQ(LATE->second.breaches, LATE->second.count) << "a sample found no threshold";
  EXPECT_EQ(EARLY->second.count + LATE->second.count + mon.queue().droppedCount(), calls);
}

/** @test VERNIER_MONITOR_SCOPE times the rest of its scope as one sample */
TEST(MonitorTest, ScopeMacro) {
  MonitorConfig cfg;
  cfg.sinks = SINK_NONE;
  Monitor mon(cfg);
  mon.start();

  const MonitorTag TAG("macro", 8);
  {
    VERNIER_MONITOR_SCOPE(mon, "stage", TAG);
    // The work the guard times: a sleep lasts at least its 2 ms.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  mon.stop();

  EXPECT_EQ(mon.summary().size(), 1u);
  const auto& ENTRIES = mon.summary().entries();
  const auto IT = ENTRIES.find("macro/8::stage");
  ASSERT_NE(IT, ENTRIES.end());
  EXPECT_EQ(IT->second.count, 1u);
  EXPECT_GE(IT->second.median(), 2.0); // ms
}
