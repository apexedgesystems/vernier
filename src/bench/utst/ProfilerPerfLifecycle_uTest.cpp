/**
 * @file ProfilerPerfLifecycle_uTest.cpp
 * @brief Unit tests for the perf backend's run: its start, its stop and the
 * output it leaves.
 *
 * A fake perf (fixtures/readiness/fake_perf.sh) stands in for the tool, and
 * FAKE_PERF_MODE selects how the run's perf behaves. Each test builds the
 * profiler from a plan that names the fake, runs one measured window and
 * reads what the run recorded as failed and what perf left in the folder; the
 * check's tests ask checkPerfRequest about the same fake.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <csignal>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using vernier::bench::checkPerfRequest;
using vernier::bench::parsePerfMode;
using vernier::bench::PerfConfig;
using vernier::bench::PerfPlan;
using vernier::bench::PerfStatProfiler;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

/** @brief True once process @p pid has ended: a zombie or gone from /proc. */
bool ended(pid_t pid) {
  std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (!std::getline(stat, line)) {
    return true;
  }
  const std::size_t CLOSE = line.rfind(')');
  return CLOSE != std::string::npos && line.compare(CLOSE + 1, 3, " Z ") == 0;
}

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

  /** @brief What a window left: the failures recorded, the folder, how long the start took. */
  struct Window {
    std::vector<ProfileFailure> failures;
    std::string folder;
    std::chrono::steady_clock::duration startTook{};
  };

  /**
   * @brief Run one window with the fake in @p mode and @p args; when
   * @p afterExit, the stop waits for the fake to report its own exit first.
   * @p answersPing is what the check found: the launch then waits for perf's
   * answer on its control fifo, else the fixed grace.
   */
  Window window(const std::string& mode, const std::string& args = "", bool afterExit = false,
                bool answersPing = true) {
    auto plan = std::make_shared<PerfPlan>();
    plan->perf = perf_;
    plan->mode = parsePerfMode(args);
    plan->answersPing = answersPing;
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
      const auto START = std::chrono::steady_clock::now();
      profiler.beforeMeasure();
      out.startTook = std::chrono::steady_clock::now() - START;
      if (afterExit) {
        // The fake logs its exit just before it exits: wait for the line,
        // then for the process to have ended (a zombie until the stop reaps
        // it), so that no timing is assumed.
        for (int i = 0; i < 500 && dir_.logLines("perf exited").empty(); ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const std::vector<std::string> EXITED = dir_.logLines("perf exited");
        EXPECT_FALSE(EXITED.empty()) << dir_.log();
        if (!EXITED.empty()) {
          const pid_t FAKE = std::stoi(EXITED.front().substr(EXITED.front().find('=') + 1));
          for (int i = 0; i < 500 && !ended(FAKE); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
          }
          EXPECT_TRUE(ended(FAKE)) << "perf " << FAKE << " logged its exit but still runs";
        }
      }
      profiler.afterMeasure(vernier::bench::Stats{});
    }
    out.failures = ProfilerRegistry::instance().failures();
    return out;
  }

  /** @brief The check's decision about the fake in @p mode, for @p args. */
  ReadinessResult check(const std::string& mode, const std::string& args = "") const {
    ReadinessRequest request;
    request.backend = "perf";
    request.profileArgs = args;
    request.scope = ReadinessScope::PREFLIGHT;
    return checkPerfRequest(request, dir_.context({{"FAKE_PERF_MODE", mode}}, 0));
  }

  /** @brief True when no process of a "pid=" in the fake's log is still alive. */
  bool fakesGone() const {
    for (const std::string& line : dir_.logLines("perf ")) {
      const std::size_t AT = line.rfind(" pid=");
      if (AT != std::string::npos &&
          ::kill(static_cast<pid_t>(std::atol(line.c_str() + AT + 5)), 0) == 0) {
        return false;
      }
    }
    return true;
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

/**
 * @test A perf that ends after SIGINT otherwise than a perf that has written
 * its report ends fails the capture, whatever it wrote: here a count, then
 * "Error: final read failed", and exit status 42.
 */
TEST_F(PerfLifecycleTest, OtherEndAfterSigintFails) {
  const Window W = window("error-exit");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: perf ended (exit status 42) after SIGINT, not as a perf that "
            "has written its report ends (by SIGINT, or exit status 130 or 0), so its output does "
            "not count: perf: 0.101234567 seconds time elapsed | Error: final read failed");
}

/** @test The control: perf ending by SIGINT itself, as it does after its report, passes. */
TEST_F(PerfLifecycleTest, EndBySigintKeepsCounts) {
  const Window W = window("raise-int");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(readText(W.folder + "/stat.txt").find("66,055"), std::string::npos);
}

/** @test stat's heading alone, with perf's usual exit status 130, is no count. */
TEST_F(PerfLifecycleTest, HeadingAloneIsNoCount) {
  const Window W = window("heading-only");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  const std::string MESSAGE = W.failures[0].result.report.message;
  EXPECT_EQ(MESSAGE.rfind("completion: unusable: " + W.folder +
                              "/stat.txt holds no counts: perf: Performance counter stats for "
                              "process id '",
                          0),
            0U)
      << MESSAGE;
}

/**
 * @test Events perf marks <not supported> or <not counted> are named when
 * none is counted, apart from missing data; one counted event among them
 * passes, as on a CPU without one of the counters.
 */
TEST_F(PerfLifecycleTest, UnavailableEventsAreNamed) {
  const Window NONE = window("none-counted");
  ASSERT_EQ(NONE.failures.size(), 1U);
  EXPECT_EQ(NONE.failures[0].result.report.message,
            "completion: unsupported: " + NONE.folder +
                "/stat.txt holds no count: perf counted none of its events here (cpu-cycles:u "
                "<not supported>, task-clock <not counted>)");
  ProfilerRegistry::instance().resetFailures();
  const Window SOME = window("some-unsupported");
  EXPECT_TRUE(SOME.failures.empty()) << SOME.failures.front().result.report.message;
}

/** @test An error perf prints after its counts fails them, even with exit status 130. */
TEST_F(PerfLifecycleTest, ErrorAfterTheCountsFails) {
  const Window W = window("error-after");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: " + W.folder +
                "/stat.txt holds an error from perf after its counts, which therefore do not "
                "count: perf: 0.101234567 seconds time elapsed | Error: final read failed");
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

/* ----------------------------- Start Handshake ----------------------------- */

/**
 * @test The measured phase starts once perf answers on its control fifo: with
 * a perf that answers 2 s after it starts, beforeMeasure() returns after at
 * least 2 s (a lower bound, which load cannot pass falsely), and nothing fails.
 */
TEST_F(PerfLifecycleTest, MeasuredPhaseWaitsForTheAck) {
  const Window W = window("slow-ack");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_GE(W.startTook, std::chrono::milliseconds(2000));
  EXPECT_EQ(dir_.logLines("perf " + perf_ + " stat -e " + vernier::bench::PERF_STAT_EVENTS +
                          " -p " + std::to_string(::getpid()) + " --control fifo:")
                .size(),
            1U)
      << dir_.log();
}

/**
 * @test A perf that never answers on its control fifo fails the request at
 * the collection stage once the wait's bound has passed, and is stopped.
 */
TEST_F(PerfLifecycleTest, NoAckFailsTheRequest) {
  const Window W = window("no-ack");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "unusable: perf did not answer on its --control fifo within 5 s, so it was not known "
            "to be counting; it was stopped, and this case is not profiled");
  EXPECT_GE(W.startTook, std::chrono::milliseconds(PerfStatProfiler::PERF_ACK_WAIT_MS));
  EXPECT_TRUE(fakesGone()) << dir_.log();
}

/**
 * @test A perf the check found not answering is started without --control,
 * after the fixed grace, and its capture is kept as before.
 */
TEST_F(PerfLifecycleTest, NoControlKeepsTheFixedWait) {
  const Window W = window("ok", "", /*afterExit=*/false, /*answersPing=*/false);
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_GE(W.startTook, std::chrono::milliseconds(PerfStatProfiler::PERF_START_GRACE_MS));
  const auto LAUNCH = dir_.logLines("perf " + perf_ + " stat -e ");
  ASSERT_EQ(LAUNCH.size(), 1U) << dir_.log();
  EXPECT_EQ(LAUNCH[0].find("--control"), std::string::npos) << LAUNCH[0];
  EXPECT_NE(readText(W.folder + "/stat.txt").find("Performance counter stats"), std::string::npos);
}

/**
 * @test Without the handshake (a perf the check found not answering, and
 * perf mem) a perf that ends at its start fails the capture before the
 * measured phase, at the collection stage, with its words: the start's grace
 * sees it gone (the fake ends within milliseconds; the grace is 200 ms).
 */
TEST_F(PerfLifecycleTest, EarlyExitWithoutTheHandshakeFails) {
  const std::string SAID = "unusable: perf ended (exit status 1) before the measured phase: perf: "
                           "Error: failed to open counters: No such process";
  const Window SILENT = window("exit-early", "", /*afterExit=*/false, /*answersPing=*/false);
  ASSERT_EQ(SILENT.failures.size(), 1U);
  EXPECT_EQ(SILENT.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(SILENT.failures[0].result.report.message, SAID);
  ProfilerRegistry::instance().resetFailures();
  const Window MEM = window("exit-early", "mem");
  ASSERT_EQ(MEM.failures.size(), 1U);
  EXPECT_EQ(MEM.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(MEM.failures[0].result.report.message, SAID);
  EXPECT_EQ(dir_.log().find("--control"), std::string::npos) << dir_.log();
}

/** @test perf mem keeps the fixed grace, also with a perf that answers. */
TEST_F(PerfLifecycleTest, MemKeepsTheFixedStart) {
  const Window W = window("ok", "mem");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  const auto LAUNCH = dir_.logLines("perf " + perf_ + " mem record ");
  ASSERT_EQ(LAUNCH.size(), 1U) << dir_.log();
  EXPECT_EQ(LAUNCH[0].find("--control"), std::string::npos) << LAUNCH[0];
}

/**
 * @test The check's access probe finds whether perf answers a ping: one that
 * does is ready and planned for the wait; one that does not answer, or does
 * not take --control (the probe then runs again without it), is a caveat
 * naming the fixed start, and its plan keeps the fixed grace.
 */
TEST_F(PerfLifecycleTest, CheckLearnsWhetherPerfAnswers) {
  const ReadinessResult ANSWERS = check("ok");
  EXPECT_EQ(ANSWERS.cause, ReadinessCause::READY) << ANSWERS.report.message;
  const auto ANSWERS_PLAN = std::dynamic_pointer_cast<const PerfPlan>(ANSWERS.plan);
  ASSERT_NE(ANSWERS_PLAN, nullptr);
  EXPECT_TRUE(ANSWERS_PLAN->answersPing);

  const ReadinessResult SILENT = check("no-ack");
  EXPECT_EQ(SILENT.cause, ReadinessCause::CAVEAT);
  EXPECT_EQ(SILENT.report.message,
            "perf stat counts this process, but " + perf_ +
                " did not answer a ping on its --control fifo, so the measured phase starts "
                "after a fixed 200 ms instead of once perf is counting");
  const auto SILENT_PLAN = std::dynamic_pointer_cast<const PerfPlan>(SILENT.plan);
  ASSERT_NE(SILENT_PLAN, nullptr);
  EXPECT_FALSE(SILENT_PLAN->answersPing);

  const std::string PROBE = "perf " + perf_ + " stat -x, -e " + vernier::bench::PERF_STAT_EVENTS +
                            " -p " + std::to_string(::getpid()) + " --timeout 100";
  const std::size_t PROBES_BEFORE = dir_.logLines(PROBE).size();
  const ReadinessResult OLD = check("no-control");
  EXPECT_EQ(OLD.cause, ReadinessCause::CAVEAT);
  EXPECT_EQ(OLD.report.message,
            "perf stat counts this process, but " + perf_ +
                " does not take --control, so the measured phase starts after a fixed 200 ms "
                "instead of once perf is counting");
  const auto OLD_PLAN = std::dynamic_pointer_cast<const PerfPlan>(OLD.plan);
  ASSERT_NE(OLD_PLAN, nullptr);
  EXPECT_FALSE(OLD_PLAN->answersPing);
  const auto PROBES = dir_.logLines(PROBE);
  ASSERT_EQ(PROBES.size(), PROBES_BEFORE + 2) << dir_.log();
  EXPECT_NE(PROBES[PROBES_BEFORE].find(" --control fifo:"), std::string::npos) << PROBES.back();
  EXPECT_EQ(PROBES.back().find("--control"), std::string::npos) << PROBES.back();

  EXPECT_EQ(check("no-ack", "record -g").report.message,
            "unverified: perf stat counts this process; perf record itself is not probed before "
            "the run; " +
                perf_ +
                " did not answer a ping on its --control fifo, so the measured phase starts after "
                "a fixed 200 ms");
  EXPECT_EQ(check("ok", "mem").report.message,
            "unverified: perf stat counts this process; perf mem itself is not probed before the "
            "run, and the measured phase starts after a fixed 200 ms");
}
