/**
 * @file ProfilerGperfLifecycle_uTest.cpp
 * @brief Unit tests for the gperf backend's run: the capture its measured
 * window starts and stops, and the profile it leaves.
 *
 * The capture is gperftools' own, in this test process, so these tests need a
 * build with gperftools and skip without it. A fake analyzer
 * (fixtures/readiness/fake_pprof.sh) stands in for google-pprof where an
 * analysis is asked for, and logs each of its runs.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#if UB_HAS_GPERF_CPU
#include <gperftools/profiler.h>
#endif
#if UB_HAS_GPERF_HEAP
#include <gperftools/heap-profiler.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using vernier::bench::GperfPlan;
using vernier::bench::GperfProfiler;
using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::Stats;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

/** @brief The file of the CPU profile gperftools runs now; "" when none runs. */
std::string runningCpuProfile() {
#if UB_HAS_GPERF_CPU
  ProfilerState state{};
  ProfilerGetCurrentState(&state);
  return state.enabled != 0 ? std::string{state.profile_name} : std::string{};
#else
  return {};
#endif
}

/** @brief True while gperftools runs a heap profile in this process. */
bool heapProfileRuns() {
#if UB_HAS_GPERF_HEAP
  return IsHeapProfilerRunning() != 0;
#else
  return false;
#endif
}

/** @brief The size of @p path, or -1 when there is no such file. */
long long fileSize(const std::string& path) {
  std::error_code ec;
  const auto SIZE = std::filesystem::file_size(path, ec);
  return ec ? -1 : static_cast<long long>(SIZE);
}

/** @brief Work for a measured window: enough to be sampled now and then. */
void work() {
  volatile double sink = 0;
  for (int i = 0; i < 200000; ++i) {
    sink = sink + i * 0.5;
  }
}

/** @brief gperf requests into a private folder, with a fake analyzer first on PATH. */
class GperfLifecycleTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (UB_HAS_GPERF_CPU == 0) {
      GTEST_SKIP() << "gperftools CPU profiling is not compiled in";
    }
    ASSERT_TRUE(dir_.ok());
    pprof_ = dir_.install("fake_pprof.sh", "google-pprof");
    ProfilerRegistry::instance().resetFailures();
    ProfilerRegistry::instance().resetReadiness();
  }

  void TearDown() override {
    ProfilerRegistry::instance().resetFailures();
    ProfilerRegistry::instance().resetReadiness();
  }

  /** @brief A gperf request into this test's folder, no watchdog; with --profile-analyze. */
  PerfConfig config(bool analyze = true) const {
    PerfConfig cfg;
    cfg.profileTool = "gperf";
    cfg.profileAnalyze = analyze;
    cfg.artifactRoot = dir_.path();
    cfg.cycles = 1;
    cfg.repeats = 2;
    cfg.profileTestTimeoutSecs = 0;
    return cfg;
  }

  /** @brief A checked plan for CPU and @p heap; with the fake analyzer when @p analyze. */
  std::shared_ptr<const GperfPlan> plan(bool analyze, bool heap = false) const {
    auto out = std::make_shared<GperfPlan>();
    out->modes.cpu = !heap;
    out->modes.heap = heap;
    out->analyze = analyze;
    out->analyzer = pprof_;
    out->analysisReady = analyze;
    return out;
  }

  /** @brief The CPU profile of case @p test. */
  std::string cpuProf(const std::string& test) const {
    return dir_.path() + "/" + test + ".gperf/cpu.prof";
  }

  /**
   * @brief The failures the run recorded, each as "<test>: <message>"; the
   * message begins with its stage when that is not collection.
   */
  static std::vector<std::string> failures() {
    std::vector<std::string> out;
    for (const ProfileFailure& F : ProfilerRegistry::instance().failures()) {
      out.push_back(F.test + ": " + F.result.report.message);
    }
    return out;
  }

  /** @brief True when the fake analyzer was run on case @p test's profile. */
  bool analyzed(const std::string& test) const {
    return dir_.log().find(test + ".gperf/cpu.prof") != std::string::npos;
  }

  FakeToolDir dir_;
  std::string pprof_;
};

} // namespace

/**
 * @test A case whose measured callback throws: the profiler its hooks own
 * stops the capture when the case ends, analyzes nothing and records no
 * failure; the next case of the process starts, writes and analyzes its own
 * profile.
 */
TEST_F(GperfLifecycleTest, ThrowingCaseStopsItsCapture) {
  // The registry's decision finds the fake analyzer first; git, for the run's
  // metadata, stays reachable behind it.
  const char* INHERITED = std::getenv("PATH");
  ScopedEnv path("PATH", dir_.path() + ":" + (INHERITED != nullptr ? INHERITED : "/usr/bin:/bin"));
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  {
    PerfCase pc{"Gperf.Throws", config()};
    vernier::bench::attachProfilerHooks(pc, config());
    EXPECT_THROW(pc.measured([] {
      work();
      throw std::runtime_error("the workload failed");
    }),
                 std::runtime_error);
  }
  EXPECT_GT(fileSize(cpuProf("Gperf.Throws")), 0)
      << "the capture was not stopped when its case ended (an unstopped one writes nothing)";
  EXPECT_FALSE(analyzed("Gperf.Throws")) << "an unfinished measurement was analyzed\n"
                                         << dir_.log();

  {
    PerfCase next{"Gperf.Next", config()};
    vernier::bench::attachProfilerHooks(next, config());
    next.measured(work);
  }
  EXPECT_GT(fileSize(cpuProf("Gperf.Next")), 0) << "the next case's capture did not start";
  EXPECT_TRUE(analyzed("Gperf.Next")) << dir_.log();
  const auto FAILED = ProfilerRegistry::instance().failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front().result.report.message;
}

/**
 * @test A start gperftools refuses (another CPU profile runs in the process)
 * fails that case at collection, and its stop leaves the other capture
 * running: only the profiler that started a capture stops it.
 */
TEST_F(GperfLifecycleTest, RefusedStartFailsAndLeavesTheOtherCapture) {
  StderrCapture quiet;
  GperfProfiler first(config(false), "Gperf.First", plan(false));
  GperfProfiler second(config(false), "Gperf.Second", plan(false));
  first.beforeMeasure();
  ASSERT_EQ(runningCpuProfile(), cpuProf("Gperf.First"));
  second.beforeMeasure();
  work();
  second.afterMeasure(Stats{});
  EXPECT_EQ(runningCpuProfile(), cpuProf("Gperf.First"))
      << "the case whose start was refused stopped the other case's capture";
  first.afterMeasure(Stats{});
  EXPECT_EQ(runningCpuProfile(), "");
  EXPECT_GT(fileSize(cpuProf("Gperf.First")), 0);
  EXPECT_EQ(fileSize(cpuProf("Gperf.Second")), -1);
  const std::vector<std::string> FAILED = failures();
  ASSERT_EQ(FAILED.size(), 1U) << "the refused start is the run's one failure";
  EXPECT_EQ(FAILED[0], "Gperf.Second: unusable: gperftools did not start a CPU profile "
                       "into " +
                           cpuProf("Gperf.Second") +
                           " (ProfilerStart returned 0): another one already runs in this "
                           "process, or the file cannot be written; this case is not profiled");
}

/**
 * @test A capture whose file is gone when it stops fails that case at
 * completion, with --profile-analyze and without it, and nothing is
 * analyzed: the raw output is checked whether or not an analysis follows.
 */
TEST_F(GperfLifecycleTest, CaptureWithoutItsFileFails) {
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  for (const bool ANALYZE : {false, true}) {
    const std::string TEST = ANALYZE ? "Gperf.GoneAnalyzed" : "Gperf.Gone";
    {
      GperfProfiler profiler(config(ANALYZE), TEST, plan(ANALYZE));
      profiler.beforeMeasure();
      work();
      // gperftools writes on into the file it opened; the name it had is gone.
      ASSERT_TRUE(std::filesystem::remove(cpuProf(TEST)));
      profiler.afterMeasure(Stats{});
    }
    EXPECT_EQ(
        failures(),
        std::vector<std::string>{TEST + ": completion: missing: " + cpuProf(TEST) +
                                 " was not written: gperftools stopped the CPU profile and left no "
                                 "capture there"})
        << (ANALYZE ? "with" : "without") << " --profile-analyze";
    EXPECT_FALSE(analyzed(TEST)) << dir_.log();
    ProfilerRegistry::instance().resetFailures();
  }
}

/** @test An empty file where the capture should be fails at completion too, unanalyzed. */
TEST_F(GperfLifecycleTest, EmptyCaptureFails) {
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  {
    GperfProfiler profiler(config(), "Gperf.Empty", plan(true));
    profiler.beforeMeasure();
    work();
    ASSERT_TRUE(std::filesystem::remove(cpuProf("Gperf.Empty")));
    std::ofstream(cpuProf("Gperf.Empty")).flush();
    profiler.afterMeasure(Stats{});
  }
  EXPECT_EQ(failures(), std::vector<std::string>{
                            "Gperf.Empty: completion: unusable: " + cpuProf("Gperf.Empty") +
                            " is empty: gperftools stopped the CPU profile and left no capture "
                            "there"});
  EXPECT_FALSE(analyzed("Gperf.Empty")) << dir_.log();
}

/** @test The control: a capture gperftools starts and stops is this run's file, then analyzed. */
TEST_F(GperfLifecycleTest, CaptureWritesItsFileAndIsAnalyzed) {
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  {
    GperfProfiler profiler(config(), "Gperf.Kept", plan(true));
    profiler.beforeMeasure();
    EXPECT_EQ(runningCpuProfile(), cpuProf("Gperf.Kept"));
    work();
    profiler.afterMeasure(Stats{});
  }
  EXPECT_EQ(runningCpuProfile(), "");
  EXPECT_GT(fileSize(cpuProf("Gperf.Kept")), 0);
  EXPECT_TRUE(analyzed("Gperf.Kept")) << dir_.log();
  EXPECT_TRUE(failures().empty()) << failures().front();
}

/**
 * @test In a build with heap profiling: a case's heap capture removes the
 * previous run's dumps and writes its own; a second case's start, while the
 * first's runs, fails at collection and leaves the first's running.
 */
TEST_F(GperfLifecycleTest, HeapCaptureIsTheCasesOwn) {
  if (UB_HAS_GPERF_HEAP == 0) {
    GTEST_SKIP() << "heap profiling is not compiled in (VERNIER_LINK_TCMALLOC=OFF)";
  }
  StderrCapture quiet;
  const std::string FIRST_DIR = dir_.path() + "/Gperf.HeapFirst.gperf";
  std::filesystem::create_directories(FIRST_DIR);
  std::ofstream(FIRST_DIR + "/heap.0007.heap") << "a previous run's dump\n";
  {
    GperfProfiler first(config(false), "Gperf.HeapFirst", plan(false, true));
    GperfProfiler second(config(false), "Gperf.HeapSecond", plan(false, true));
    first.beforeMeasure();
    EXPECT_FALSE(std::filesystem::exists(FIRST_DIR + "/heap.0007.heap"))
        << "the previous run's dump is still there";
    ASSERT_TRUE(heapProfileRuns());
    second.beforeMeasure();
    second.afterMeasure(Stats{});
    EXPECT_TRUE(heapProfileRuns()) << "the refused case stopped the other case's heap capture";
    first.afterMeasure(Stats{});
    EXPECT_FALSE(heapProfileRuns());
  }
  bool dumped = false;
  for (const auto& ENTRY : std::filesystem::directory_iterator(FIRST_DIR)) {
    dumped = dumped || (ENTRY.path().extension() == ".heap" && fileSize(ENTRY.path().string()) > 0);
  }
  EXPECT_TRUE(dumped) << "the first case wrote no heap dump";
  const std::vector<std::string> FAILED = failures();
  ASSERT_EQ(FAILED.size(), 1U);
  EXPECT_EQ(FAILED[0], "Gperf.HeapSecond: unusable: a gperftools heap profile already "
                       "runs in this process, so this case's could not start; its heap is not "
                       "profiled");
}
