/**
 * @file ProfilerPerfLifecycle_uTest.cpp
 * @brief Unit tests for the perf backend's run: its start, its stop and the
 * output it leaves.
 *
 * A fake perf (fixtures/readiness/fake_perf.sh) stands in for the tool, and
 * FAKE_PERF_MODE selects how the run's perf behaves. Each test builds the
 * profiler from a plan that names the fake, runs one measured window and
 * reads what the run recorded as failed and what perf left in the folder.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using vernier::bench::parsePerfMode;
using vernier::bench::PerfConfig;
using vernier::bench::PerfPlan;
using vernier::bench::PerfStatProfiler;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessStage;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

std::string readText(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

/** @brief One measured window under a fake perf in a private folder. */
class PerfLifecycleTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    perf_ = dir_.install("fake_perf.sh", "perf");
    ProfilerRegistry::instance().resetFailures();
  }

  void TearDown() override { ProfilerRegistry::instance().resetFailures(); }

  /** @brief What a window left: the failures recorded and the artifact folder. */
  struct Window {
    std::vector<ProfileFailure> failures;
    std::string folder;
  };

  /**
   * @brief Run one window with the fake in @p mode and @p args; when
   * @p afterExit, the stop waits for the fake to report its own exit first.
   */
  Window window(const std::string& mode, const std::string& args = "", bool afterExit = false) {
    auto plan = std::make_shared<PerfPlan>();
    plan->perf = perf_;
    plan->mode = parsePerfMode(args);
    PerfConfig cfg;
    cfg.profileTool = "perf";
    cfg.profileArgs = args;
    cfg.artifactRoot = dir_.path();
    ScopedEnv fakeMode("FAKE_PERF_MODE", mode);
    ScopedEnv log("FAKE_LOG", dir_.logPath());
    StderrCapture quiet;
    Window out;
    {
      PerfStatProfiler profiler(cfg, "Perf.Life", plan);
      out.folder = profiler.artifactDir();
      profiler.beforeMeasure();
      if (afterExit) {
        // The fake logs its exit itself: no timing is assumed.
        for (int i = 0; i < 500 && dir_.logLines("perf exited").empty(); ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        EXPECT_FALSE(dir_.logLines("perf exited").empty()) << dir_.log();
      }
      profiler.afterMeasure(vernier::bench::Stats{});
    }
    out.failures = ProfilerRegistry::instance().failures();
    return out;
  }

  FakeToolDir dir_;
  std::string perf_;
};

} // namespace

/** @test A perf stopped by SIGINT leaves its counts, and nothing fails. */
TEST_F(PerfLifecycleTest, CleanStopKeepsCounts) {
  const Window W = window("ok");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(readText(W.folder + "/stat.txt").find("Performance counter stats"), std::string::npos);
}

/** @test The stop returns once perf has written, not after a fixed wait. */
TEST_F(PerfLifecycleTest, StopWaitsForPerfToWrite) {
  const Window W = window("slow-stop");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(readText(W.folder + "/stat.txt").find("Performance counter stats"), std::string::npos)
      << "the hook returned before perf wrote its counts";
}

/** @test A perf that ends before the measured phase fails the capture with its words. */
TEST_F(PerfLifecycleTest, EarlyExitFails) {
  const Window W = window("exit-early");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].backend, "perf");
  EXPECT_EQ(W.failures[0].test, "Perf.Life");
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "unusable: perf ended (exit status 1) before the measured phase: perf: Error: failed "
            "to open counters: No such process");
}

/** @test A perf that ends during the measured phase fails its completion. */
TEST_F(PerfLifecycleTest, EndDuringTheWindowFails) {
  const Window W = window("exit-soon", "", /*afterExit=*/true);
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: perf ended (exit status 3) during the measured phase, so its "
            "output covers part of it at most: perf: Error: the target process exited");
}

/** @test A perf that ignores SIGINT is stopped by SIGTERM, and its output does not count. */
TEST_F(PerfLifecycleTest, SigtermStopFails) {
  const Window W = window("ignore-int");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: perf did not finish writing within 5 s of SIGINT and was "
            "stopped by SIGTERM, so its output may be incomplete");
}

/** @test A perf that ignores SIGINT and SIGTERM is killed, and its output does not count. */
TEST_F(PerfLifecycleTest, HangIsKilledAndFails) {
  const Window W = window("hang");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_NE(W.failures[0].result.report.message.find("was stopped by SIGKILL"), std::string::npos)
      << W.failures[0].result.report.message;
}

/** @test stat.txt holding an error message instead of counts fails (a zero count would not). */
TEST_F(PerfLifecycleTest, ErrorTextIsNotACount) {
  const Window W = window("error-text");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(
      W.failures[0].result.report.message,
      "completion: unusable: " + W.folder +
          "/stat.txt holds no counts: perf: Error: the fake perf could not read its counters");
}

/** @test Record mode's data file, confirmed by perf, passes; without it the capture fails. */
TEST_F(PerfLifecycleTest, RecordNeedsItsConfirmedData) {
  const Window GOOD = window("ok", "record -g");
  EXPECT_TRUE(GOOD.failures.empty()) << GOOD.failures.front().result.report.message;
  std::error_code ec;
  EXPECT_GT(std::filesystem::file_size(GOOD.folder + "/perf.data", ec), 0U);
  ProfilerRegistry::instance().resetFailures();
  std::filesystem::remove(GOOD.folder + "/perf.data", ec);
  const Window BAD = window("error-text", "record -g");
  ASSERT_EQ(BAD.failures.size(), 1U);
  EXPECT_EQ(BAD.failures[0].result.report.message,
            "completion: missing: " + BAD.folder +
                "/perf.data was not written: perf: Error: the fake perf could not read its "
                "counters");
}
