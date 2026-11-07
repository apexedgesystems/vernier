/**
 * @file ProfilerGperfLifecycle_uTest.cpp
 * @brief Unit tests for the gperf backend's run: the capture its measured
 * window starts and stops, and the profile it leaves.
 *
 * The capture is gperftools' own, in this test process, so these tests need a
 * build with gperftools and skip without it. A fake analyzer
 * (fixtures/readiness/fake_pprof.sh) stands in for google-pprof where an
 * analysis is asked for, and logs each of its runs. In a build with heap
 * profiling this program replaces the global allocation functions, through
 * which the tests of both captures at once watch the profile reader (Both
 * Captures, below); with the thread or the address sanitizer it replaces
 * none, and those tests skip, naming it.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#if UB_HAS_GPERF_CPU
#include <gperftools/profiler.h>
#endif
#if UB_HAS_GPERF_HEAP
#include <gperftools/heap-profiler.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
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

/** @brief What @p path holds; "" when it cannot be read. */
std::string contents(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/** @brief The heap dumps (*.heap) in @p dir, sorted. */
std::vector<std::string> heapDumpsIn(const std::string& dir) {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& ENTRY : std::filesystem::directory_iterator(dir, ec)) {
    if (ENTRY.path().extension() == ".heap") {
      out.push_back(ENTRY.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

/**
 * @brief Makes a folder read-only (mode 0555) while it lives, and gives it
 * its mode back after.
 */
class ReadOnlyFolder {
public:
  explicit ReadOnlyFolder(std::string dir) : dir_(std::move(dir)) {
    namespace fs = std::filesystem;
    std::error_code ec;
    mode_ = fs::status(dir_, ec).permissions();
    if (!ec) {
      fs::permissions(dir_,
                      fs::perms::owner_read | fs::perms::owner_exec | fs::perms::group_read |
                          fs::perms::group_exec | fs::perms::others_read | fs::perms::others_exec,
                      fs::perm_options::replace, ec);
    }
    ok_ = !ec;
  }
  ~ReadOnlyFolder() {
    std::error_code ec;
    std::filesystem::permissions(dir_, mode_, std::filesystem::perm_options::replace, ec);
  }
  ReadOnlyFolder(const ReadOnlyFolder&) = delete;
  ReadOnlyFolder& operator=(const ReadOnlyFolder&) = delete;

  /** @brief True when the folder was made read-only. */
  [[nodiscard]] bool ok() const { return ok_; }

private:
  std::string dir_;
  std::filesystem::perms mode_{std::filesystem::perms::unknown};
  bool ok_ = false;
};

/** @brief Work for a measured window: enough to be sampled now and then. */
void work() {
  volatile double sink = 0;
  for (int i = 0; i < 200000; ++i) {
    sink = sink + i * 0.5;
  }
}

/** @brief A measured window whose workload fails part-way. */
void workThenThrow() {
  work();
  throw std::runtime_error("the workload failed");
}

/**
 * @brief The registry's decision finds the fake analyzer first on PATH, and
 * git, for the run's metadata, stays reachable behind it; the fake logs to
 * the folder's log.
 */
class FakeAnalyzerFirst {
public:
  explicit FakeAnalyzerFirst(const FakeToolDir& dir)
      : path_("PATH", dir.path() + ":" + inheritedPath()), log_("FAKE_LOG", dir.logPath()) {}

private:
  static std::string inheritedPath() {
    const char* inherited = std::getenv("PATH");
    return inherited != nullptr ? inherited : "/usr/bin:/bin";
  }

  ScopedEnv path_;
  ScopedEnv log_;
};

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
    // A capture a failed test left on would fail the tests after it.
#if UB_HAS_GPERF_CPU
    if (!runningCpuProfile().empty()) {
      ProfilerStop();
    }
#endif
#if UB_HAS_GPERF_HEAP
    if (heapProfileRuns()) {
      HeapProfilerStop();
    }
#endif
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

  /** @brief A checked plan for both captures, without analysis. */
  std::shared_ptr<const GperfPlan> bothPlan() const {
    auto out = std::make_shared<GperfPlan>(*plan(false));
    out->modes.cpu = true;
    out->modes.heap = true;
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
 * has stopped the capture by the time the case ends, analyzes nothing and
 * records no failure; the next case of the process starts, writes and
 * analyzes its own profile.
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
 * @test A case whose measured callback throws, kept by the caller that
 * caught the exception: its capture is stopped at the throw, while the case
 * lives, unanalyzed, and nothing is recorded as failed.
 */
TEST_F(GperfLifecycleTest, ThrowingCaseKeptStopsItsCaptureAtOnce) {
  FakeAnalyzerFirst analyzer(dir_);
  StderrCapture quiet;
  PerfCase pc{"Gperf.ThrowsKept", config()};
  vernier::bench::attachProfilerHooks(pc, config());
  EXPECT_THROW(pc.measured(workThenThrow), std::runtime_error);
  EXPECT_EQ(runningCpuProfile(), "") << "the capture is still on while its case is kept";
  EXPECT_GT(fileSize(cpuProf("Gperf.ThrowsKept")), 0)
      << "the capture was not stopped (an unstopped gperftools profile has written nothing)";
  EXPECT_FALSE(analyzed("Gperf.ThrowsKept")) << "an unfinished measurement was analyzed\n"
                                             << dir_.log();
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
}

/**
 * @test While a case whose callback threw is kept, another case of the
 * process starts, writes and analyzes its own capture.
 */
TEST_F(GperfLifecycleTest, AnotherCaseCapturesWhileTheThrowingOneIsKept) {
  FakeAnalyzerFirst analyzer(dir_);
  StderrCapture quiet;
  PerfCase thrown{"Gperf.ThrowsFirst", config()};
  vernier::bench::attachProfilerHooks(thrown, config());
  EXPECT_THROW(thrown.measured(workThenThrow), std::runtime_error);

  PerfCase other{"Gperf.Other", config()};
  vernier::bench::attachProfilerHooks(other, config());
  other.measured(work);
  EXPECT_EQ(runningCpuProfile(), "");
  EXPECT_GT(fileSize(cpuProf("Gperf.Other")), 0) << "the other case's capture did not start";
  EXPECT_TRUE(analyzed("Gperf.Other")) << dir_.log();
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
}

/**
 * @test A case whose callback threw measures again: that measurement starts,
 * writes and analyzes its own capture, and nothing is on once the case ends.
 */
TEST_F(GperfLifecycleTest, ThrowingCaseMeasuresAgainAndCaptures) {
  FakeAnalyzerFirst analyzer(dir_);
  StderrCapture quiet;
  {
    PerfCase pc{"Gperf.Retried", config()};
    vernier::bench::attachProfilerHooks(pc, config());
    EXPECT_THROW(pc.measured(workThenThrow), std::runtime_error);
    EXPECT_FALSE(analyzed("Gperf.Retried")) << dir_.log();
    pc.measured(work);
    EXPECT_EQ(runningCpuProfile(), "") << "a capture is still on after the measurement ended";
    EXPECT_TRUE(analyzed("Gperf.Retried")) << "the measurement again was not captured\n"
                                           << dir_.log();
  }
  EXPECT_EQ(runningCpuProfile(), "") << "a capture outlived its case";
  EXPECT_GT(fileSize(cpuProf("Gperf.Retried")), 0);
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
}

/**
 * @test A window that ended through neither afterMeasure() nor
 * abortMeasure(), as hooks attached without the abort hook leave it, leaves
 * its capture on; the profiler's next start stops that capture first, as its
 * own, and starts this window's. Nothing is refused, the profiler stops what
 * it started, and nothing is on once it is done.
 */
TEST_F(GperfLifecycleTest, StartOverItsOwnCaptureStopsThatFirst) {
  StderrCapture quiet;
  {
    GperfProfiler profiler(config(false), "Gperf.Again", plan(false));
    profiler.beforeMeasure();
    work();
    profiler.beforeMeasure();
    EXPECT_EQ(runningCpuProfile(), cpuProf("Gperf.Again"));
    work();
    profiler.afterMeasure(Stats{});
    EXPECT_EQ(runningCpuProfile(), "") << "the profiler lost the capture it started";
  }
  EXPECT_EQ(runningCpuProfile(), "") << "a capture outlived its profiler";
  EXPECT_GT(fileSize(cpuProf("Gperf.Again")), 0);
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
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
 * @test The writable-reuse control: a previous run's CPU profile in a folder
 * this user can write is removed first, and this capture's file takes its
 * place and is analyzed.
 */
TEST_F(GperfLifecycleTest, PreviousCaptureIsReplaced) {
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  ASSERT_TRUE(std::filesystem::create_directories(dir_.path() + "/Gperf.Reused.gperf"));
  std::ofstream(cpuProf("Gperf.Reused")) << "a previous run's profile\n";
  {
    GperfProfiler profiler(config(), "Gperf.Reused", plan(true));
    profiler.beforeMeasure();
    EXPECT_EQ(runningCpuProfile(), cpuProf("Gperf.Reused"));
    work();
    profiler.afterMeasure(Stats{});
  }
  const std::string NOW = contents(cpuProf("Gperf.Reused"));
  EXPECT_FALSE(NOW.empty());
  EXPECT_NE(NOW, "a previous run's profile\n") << "the previous run's profile is still there";
  EXPECT_TRUE(analyzed("Gperf.Reused")) << dir_.log();
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
}

/**
 * @test A previous run's CPU profile that cannot be removed (its folder made
 * read-only) fails the case at collection before a capture starts, naming
 * the file and why; the previous profile is left as it was, and nothing is
 * captured or analyzed, so it is never taken for this run's.
 */
TEST_F(GperfLifecycleTest, PreviousCaptureThatCannotBeRemovedFailsTheStart) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "root removes files from a read-only folder";
  }
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  const std::string CASE_DIR = dir_.path() + "/Gperf.Stale.gperf";
  ASSERT_TRUE(std::filesystem::create_directories(CASE_DIR));
  std::ofstream(cpuProf("Gperf.Stale")) << "a previous run's profile\n";
  {
    const ReadOnlyFolder LOCKED(CASE_DIR);
    ASSERT_TRUE(LOCKED.ok());
    GperfProfiler profiler(config(), "Gperf.Stale", plan(true));
    profiler.beforeMeasure();
    EXPECT_EQ(runningCpuProfile(), "") << "a capture started over the previous run's profile";
    work();
    profiler.afterMeasure(Stats{});
  }
  EXPECT_EQ(contents(cpuProf("Gperf.Stale")), "a previous run's profile\n");
  EXPECT_FALSE(analyzed("Gperf.Stale")) << dir_.log();
  EXPECT_EQ(failures(), std::vector<std::string>{
                            "Gperf.Stale: unusable: the previous run's CPU profile could not be "
                            "cleared (" +
                            cpuProf("Gperf.Stale") +
                            ": Permission denied), so this case's could not be told from it; this "
                            "case is not profiled"});
}

/**
 * @test In a build with heap profiling, three heap captures of one case:
 * its first heap capture writes a dump (the fresh control); its second, in
 * the same folder, removes that dump and writes its own (the writable-reuse
 * control); with the folder then made read-only, its third fails at
 * collection before the heap profile starts, naming the dump it could not
 * remove, which is left as it was and never taken for this attempt's.
 */
TEST_F(GperfLifecycleTest, PreviousHeapDumpThatCannotBeRemovedFailsTheStart) {
  if (UB_HAS_GPERF_HEAP == 0) {
    GTEST_SKIP() << "heap profiling is not compiled in (VERNIER_LINK_TCMALLOC=OFF)";
  }
  if (::geteuid() == 0) {
    GTEST_SKIP() << "root removes files from a read-only folder";
  }
  StderrCapture quiet;
  const std::string CASE_DIR = dir_.path() + "/Gperf.HeapStale.gperf";
  GperfProfiler profiler(config(false), "Gperf.HeapStale", plan(false, true));
  profiler.beforeMeasure();
  profiler.afterMeasure(Stats{});
  const std::vector<std::string> FIRST = heapDumpsIn(CASE_DIR);
  ASSERT_EQ(FIRST.size(), 1U) << "the first capture wrote no dump";
  profiler.beforeMeasure();
  profiler.afterMeasure(Stats{});
  const std::vector<std::string> SECOND = heapDumpsIn(CASE_DIR);
  ASSERT_EQ(SECOND.size(), 1U) << "the second capture did not replace the first's dump";
  ASSERT_GT(fileSize(SECOND[0]), 0);
  {
    const std::vector<std::string> FAILED = failures();
    ASSERT_TRUE(FAILED.empty()) << FAILED.front();
  }
  {
    const ReadOnlyFolder LOCKED(CASE_DIR);
    ASSERT_TRUE(LOCKED.ok());
    profiler.beforeMeasure();
    EXPECT_FALSE(heapProfileRuns()) << "a heap capture started over the previous run's dump";
    profiler.afterMeasure(Stats{});
  }
  EXPECT_EQ(heapDumpsIn(CASE_DIR), SECOND) << "the previous run's dump was not left as it was";
  EXPECT_EQ(failures(), std::vector<std::string>{
                            "Gperf.HeapStale: unusable: the previous run's heap dumps could not be "
                            "cleared (" +
                            SECOND[0] +
                            ": Permission denied), so this case's could not be told from them; "
                            "its heap is not profiled"});
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

/* ------------------------------ Both Captures ------------------------------ */

// A capture of both kinds ends both before its profile is read, so the
// reader's work is in neither, and a failure of it leaves neither on. The
// tests below watch the reader through this program's allocation functions:
// on the thread that arms the watch, each allocation of at least
// READER_BUFFER_BYTES is counted, with whether a capture ran when it was
// made, and one can be made to fail. While a capture completes, only the
// reader allocates that much: the tests put a profile of PROFILE_BYTES in
// place of the capture, and the reader takes the whole file in. The functions
// are replaced only in a build with heap profiling, the one build that
// captures both; tcmalloc's malloc and free serve them there, and its heap
// profiler records what they allocate as it records any allocation. A build
// with the thread or the address sanitizer replaces none, and the tests skip,
// naming it (GPERF_READER_WATCH_SANITIZER, from this directory's
// CMakeLists.txt): the thread sanitizer's runtime defines these functions
// itself, and clang's cannot link beside a replacement; the address
// sanitizer's serves malloc and free itself, in place of tcmalloc.

namespace {

/// The smallest allocation the watch counts. While a capture completes, only
/// the reader's buffer for the profile below is this large.
constexpr std::size_t READER_BUFFER_BYTES = std::size_t{1} << 20;

/// The size of the profile the tests put in place of a capture.
constexpr std::size_t PROFILE_BYTES = 4 * READER_BUFFER_BYTES;

/// The sanitizer this build is configured with (-DSANITIZER) when it is the
/// address or the thread sanitizer, "asan" or "tsan"; empty otherwise. Such a
/// build does not replace the allocation functions, so it cannot watch the
/// reader.
#ifdef GPERF_READER_WATCH_SANITIZER
constexpr std::string_view WATCH_SANITIZER = GPERF_READER_WATCH_SANITIZER;
#else
constexpr std::string_view WATCH_SANITIZER = "";
#endif

/** @brief What the allocation functions watch, on the thread that armed them. */
struct ReaderWatch {
  bool armed = false;     ///< Count this thread's allocations of READER_BUFFER_BYTES or more.
  bool failOnce = false;  ///< Fail the next one with std::bad_alloc.
  int seen = 0;           ///< The allocations counted.
  int whileCapturing = 0; ///< Of those, the ones made while a CPU or heap capture ran.
};

thread_local ReaderWatch readerWatch;

/** @brief Arms the watch on this thread while it lives; its counts stay readable after. */
class WatchTheReader {
public:
  explicit WatchTheReader(bool failOnce) { readerWatch = ReaderWatch{true, failOnce, 0, 0}; }
  ~WatchTheReader() {
    readerWatch.armed = false;
    readerWatch.failOnce = false;
  }
  WatchTheReader(const WatchTheReader&) = delete;
  WatchTheReader& operator=(const WatchTheReader&) = delete;
};

/**
 * @brief Write at @p path a complete profile of PROFILE_BYTES: a header at
 * gperftools' default period, one record whose stack fills the file, and the
 * trailer.
 */
void writeLargeProfile(const std::string& path) {
  constexpr std::size_t SLOTS = PROFILE_BYTES / sizeof(std::uintptr_t);
  std::vector<std::uintptr_t> slots = {0, 3, 0, 10000, 0, 1, SLOTS - 10};
  slots.resize(SLOTS - 3, 0x1000);
  slots.insert(slots.end(), {0, 1, 0});
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(slots.data()),
             static_cast<std::streamsize>(slots.size() * sizeof(std::uintptr_t)));
}

/** @brief True when the case folder @p dir holds a heap dump that is not empty. */
bool heapDumped(const std::string& dir) {
  const std::vector<std::string> DUMPS = heapDumpsIn(dir);
  return std::any_of(DUMPS.begin(), DUMPS.end(),
                     [](const std::string& dump) { return fileSize(dump) > 0; });
}

/**
 * @brief Why the tests of both captures cannot run in this build: it has no
 * heap profiling, or it has the thread or the address sanitizer and so does
 * not watch the reader; empty when they can.
 */
std::string bothCapturesUntestable() {
  if (UB_HAS_GPERF_HEAP == 0) {
    return "heap profiling is not compiled in (VERNIER_LINK_TCMALLOC=OFF)";
  }
  if (WATCH_SANITIZER.empty()) {
    return "";
  }
  const std::string WHY =
      "this program is built with a sanitizer (SANITIZER=" + std::string(WATCH_SANITIZER) +
      "), so it does not replace the allocation functions these tests watch the profile "
      "reader through";
  if (WATCH_SANITIZER == "tsan") {
    return WHY + ": the thread sanitizer's runtime defines them itself, and clang's cannot "
                 "link beside a replacement";
  }
  return WHY + ": the address sanitizer's runtime serves malloc and free itself, in place of "
               "tcmalloc";
}

} // namespace

#if UB_HAS_GPERF_HEAP && !defined(GPERF_READER_WATCH_SANITIZER)

namespace {

/** @brief True while gperftools runs a CPU profile in this process; allocates nothing. */
bool cpuProfileRuns() noexcept {
#if UB_HAS_GPERF_CPU
  ProfilerState state{};
  ProfilerGetCurrentState(&state);
  return state.enabled != 0;
#else
  return false;
#endif
}

/** @brief Count an allocation of @p size for an armed watch, and fail it if asked to. */
void watchAllocation(std::size_t size) {
  ReaderWatch& watch = readerWatch;
  if (!watch.armed || size < READER_BUFFER_BYTES) {
    return;
  }
  ++watch.seen;
  if (cpuProfileRuns() || heapProfileRuns()) {
    ++watch.whileCapturing;
  }
  if (watch.failOnce) {
    watch.failOnce = false;
    throw std::bad_alloc();
  }
}

} // namespace

void* operator new(std::size_t size) {
  watchAllocation(size);
  for (;;) {
    if (void* block = std::malloc(size == 0 ? 1 : size)) {
      return block;
    }
    const std::new_handler HANDLER = std::get_new_handler();
    if (HANDLER == nullptr) {
      throw std::bad_alloc();
    }
    HANDLER();
  }
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

// Every delete is out of line: inlined into a caller, free() would meet a
// pointer the compiler knows came from operator new, and it warns about the
// pair.
[[gnu::noinline]] void operator delete(void* block) noexcept { std::free(block); }

[[gnu::noinline]] void operator delete[](void* block) noexcept { std::free(block); }

[[gnu::noinline]] void operator delete(void* block, std::size_t /*size*/) noexcept {
  std::free(block);
}

[[gnu::noinline]] void operator delete[](void* block, std::size_t /*size*/) noexcept {
  std::free(block);
}

[[gnu::noinline]] void operator delete(void* block, const std::nothrow_t& /*tag*/) noexcept {
  std::free(block);
}

[[gnu::noinline]] void operator delete[](void* block, const std::nothrow_t& /*tag*/) noexcept {
  std::free(block);
}

#endif

/**
 * @test In a build with heap profiling and neither the thread nor the address
 * sanitizer, a capture of both kinds: both have ended by the time the reader
 * takes the profile in, so neither records its buffer; the heap profile's
 * final dump is written, and nothing fails.
 */
TEST_F(GperfLifecycleTest, CombinedCaptureEndsBeforeItsProfileIsRead) {
  if (const std::string WHY = bothCapturesUntestable(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  StderrCapture quiet;
  const std::string STAGED = dir_.path() + "/staged.prof";
  writeLargeProfile(STAGED);
  {
    GperfProfiler profiler(config(false), "Gperf.Both", bothPlan());
    profiler.beforeMeasure();
    ASSERT_EQ(runningCpuProfile(), cpuProf("Gperf.Both"));
    ASSERT_TRUE(heapProfileRuns());
    work();
    // gperftools writes on into the file it opened; the name now holds the
    // large profile.
    std::error_code ec;
    std::filesystem::rename(STAGED, cpuProf("Gperf.Both"), ec);
    ASSERT_FALSE(ec) << ec.message();
    const WatchTheReader WATCH(/*failOnce=*/false);
    profiler.afterMeasure(Stats{});
  }
  EXPECT_GT(readerWatch.seen, 0) << "the reader's buffer was not seen, so nothing was watched";
  EXPECT_EQ(readerWatch.whileCapturing, 0) << "the profile was read while a capture ran";
  EXPECT_EQ(runningCpuProfile(), "");
  EXPECT_FALSE(heapProfileRuns());
  EXPECT_TRUE(heapDumped(dir_.path() + "/Gperf.Both.gperf")) << "no final heap dump";
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
}

/**
 * @test In a build with heap profiling and neither the thread nor the address
 * sanitizer, a capture of both kinds whose reader fails: the exception
 * reaches the caller as it was thrown, and while the profiler is kept neither
 * capture is on. The heap profile's final dump is written, the raw profile is
 * kept, nothing else is recorded as failed, and another case's captures of
 * both kinds start and end.
 */
TEST_F(GperfLifecycleTest, CombinedCaptureWhoseReadFailsLeavesNoCaptureOn) {
  if (const std::string WHY = bothCapturesUntestable(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  StderrCapture quiet;
  const std::string STAGED = dir_.path() + "/staged.prof";
  writeLargeProfile(STAGED);
  GperfProfiler profiler(config(false), "Gperf.BothFails", bothPlan());
  profiler.beforeMeasure();
  ASSERT_TRUE(heapProfileRuns());
  work();
  std::error_code ec;
  std::filesystem::rename(STAGED, cpuProf("Gperf.BothFails"), ec);
  ASSERT_FALSE(ec) << ec.message();
  {
    const WatchTheReader WATCH(/*failOnce=*/true);
    EXPECT_THROW(profiler.afterMeasure(Stats{}), std::bad_alloc);
  }
  EXPECT_EQ(readerWatch.seen, 1) << "the failure was not the reader's";
  EXPECT_EQ(readerWatch.whileCapturing, 0) << "the profile was read while a capture ran";
  EXPECT_EQ(runningCpuProfile(), "") << "the CPU capture is on while its profiler is kept";
  EXPECT_FALSE(heapProfileRuns()) << "the heap capture is on while its profiler is kept";
  EXPECT_TRUE(heapDumped(dir_.path() + "/Gperf.BothFails.gperf")) << "no final heap dump";
  EXPECT_EQ(fileSize(cpuProf("Gperf.BothFails")), static_cast<long long>(PROFILE_BYTES))
      << "the raw profile was not kept";
  {
    const std::vector<std::string> FAILED = failures();
    EXPECT_TRUE(FAILED.empty()) << FAILED.front();
  }
  {
    GperfProfiler next(config(false), "Gperf.BothNext", bothPlan());
    next.beforeMeasure();
    EXPECT_EQ(runningCpuProfile(), cpuProf("Gperf.BothNext"));
    work();
    next.afterMeasure(Stats{});
  }
  EXPECT_TRUE(heapDumped(dir_.path() + "/Gperf.BothNext.gperf"))
      << "another case's heap capture did not start";
  EXPECT_EQ(runningCpuProfile(), "");
  EXPECT_FALSE(heapProfileRuns());
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
}

/* ------------------------------ Sampling Rate ------------------------------ */

namespace {

using vernier::bench::GperfRateSource;

/**
 * @brief The sampling period the header of the CPU profile at @p path records,
 * in microseconds (its fourth pointer-sized slot); -1 when it has no header.
 * Read here, apart from the backend's own reader.
 */
long long recordedPeriodUs(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::uintptr_t slots[5] = {};
  if (!in.read(reinterpret_cast<char*>(slots), sizeof slots) || slots[0] != 0 || slots[1] < 3 ||
      slots[2] != 0) {
    return -1;
  }
  return static_cast<long long>(slots[3]);
}

/**
 * @brief The period gperftools samples this process with: 1000000 / its
 * CPUPROFILE_FREQUENCY, rounded down, or 10000 without it; none when the
 * value is one a gperf request refuses (the tests then skip).
 */
std::optional<int> thisProcessPeriodUs() {
  const char* value = std::getenv(vernier::bench::SAMPLING_RATE_VARIABLE);
  if (value == nullptr) {
    return 10000;
  }
  const std::optional<int> HZ = vernier::bench::parseProfileFrequency(value);
  return HZ ? std::optional<int>{1000000 / *HZ} : std::nullopt;
}

/** @brief @p plan, expecting @p periodUs from @p source at @p hz. */
std::shared_ptr<const GperfPlan> expecting(const std::shared_ptr<const GperfPlan>& plan,
                                           int periodUs, GperfRateSource source, int hz) {
  auto out = std::make_shared<GperfPlan>(*plan);
  out->expectedPeriodUs = periodUs;
  out->samplingSource = source;
  out->samplingHz = hz;
  return out;
}

/** @brief Why this process cannot be profiled at the period it started with. */
std::string refusedVariable() {
  return std::string{"this process started with CPUPROFILE_FREQUENCY='"} +
         std::getenv(vernier::bench::SAMPLING_RATE_VARIABLE) + "', which a gperf request refuses";
}

} // namespace

/**
 * @test A capture through the hooks a benchmark attaches records the period
 * the decision expects, taken from this process's own environment, and
 * nothing fails; the rate line is printed only for a rate from the
 * environment, so at the default the run prints what it printed before.
 */
TEST_F(GperfLifecycleTest, CaptureRecordsTheDecisionsPeriod) {
  const std::optional<int> PERIOD = thisProcessPeriodUs();
  if (!PERIOD) {
    GTEST_SKIP() << refusedVariable();
  }
  StderrCapture capture;
  {
    PerfCase pc{"Gperf.Decided", config(false)};
    vernier::bench::attachProfilerHooks(pc, config(false));
    pc.measured(work);
  }
  const std::string ERR = capture.text();
  EXPECT_EQ(recordedPeriodUs(cpuProf("Gperf.Decided")), *PERIOD);
  const std::vector<std::string> FAILED = failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front();
  const bool INHERITED = std::getenv(vernier::bench::SAMPLING_RATE_VARIABLE) != nullptr;
  EXPECT_EQ(ERR.find("[gperf] Gperf.Decided: cpu.prof records a period of ") != std::string::npos,
            INHERITED)
      << ERR;
}

/**
 * @test A profile whose period is not the one the plan expects fails its
 * case at completion, naming both periods and where the expected one came
 * from; the profile is kept as gperftools wrote it, and, being a complete
 * profile, it is still analyzed.
 */
TEST_F(GperfLifecycleTest, CaptureAtAnotherPeriodFailsAndIsKept) {
  const std::optional<int> PERIOD = thisProcessPeriodUs();
  if (!PERIOD) {
    GTEST_SKIP() << refusedVariable();
  }
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  const int OTHER = *PERIOD == 1000 ? 4000 : 1000;
  {
    GperfProfiler profiler(
        config(), "Gperf.OtherRate",
        expecting(plan(true), OTHER, GperfRateSource::REQUESTED, 1000000 / OTHER));
    profiler.beforeMeasure();
    work();
    profiler.afterMeasure(Stats{});
  }
  const std::string PROF = cpuProf("Gperf.OtherRate");
  EXPECT_EQ(recordedPeriodUs(PROF), *PERIOD) << "the profile was not kept as written";
  const std::vector<ProfileFailure> FAILED = ProfilerRegistry::instance().failures();
  ASSERT_EQ(FAILED.size(), 1U);
  EXPECT_EQ(FAILED[0].test, "Gperf.OtherRate");
  EXPECT_EQ(FAILED[0].result.stage, vernier::bench::ReadinessStage::COMPLETION);
  EXPECT_EQ(FAILED[0].result.report.message,
            "completion: unusable: " + PROF + " records a period of " + std::to_string(*PERIOD) +
                " us (" + std::to_string(1000000 / *PERIOD) + " Hz) where " +
                std::to_string(OTHER) + " us was expected for " + std::to_string(1000000 / OTHER) +
                " Hz requested: gperftools started with another rate; raw profile kept");
  EXPECT_EQ(FAILED[0].result.report.hint,
            "Set CPUPROFILE_FREQUENCY before the benchmark starts (bench run "
            "--profile-frequency does), not from inside it.");
  EXPECT_TRUE(analyzed("Gperf.OtherRate")) << "a complete profile was not analyzed\n" << dir_.log();
}

/**
 * @test A profile at the period expected prints the rate line for a rate
 * asked for or taken from the environment, with the samples it holds, and
 * nothing at gperftools' default.
 */
TEST_F(GperfLifecycleTest, RateLineOnlyForARequestedOrInheritedRate) {
  const std::optional<int> PERIOD = thisProcessPeriodUs();
  if (!PERIOD) {
    GTEST_SKIP() << refusedVariable();
  }
  const int HZ = 1000000 / *PERIOD;
  const std::string AT =
      "[gperf] Gperf.Line: cpu.prof records a period of " + std::to_string(*PERIOD) + " us (";
  const std::vector<std::pair<GperfRateSource, std::string>> SOURCES = {
      {GperfRateSource::REQUESTED, std::to_string(HZ) + " Hz requested"},
      {GperfRateSource::INHERITED,
       "CPUPROFILE_FREQUENCY=" + std::to_string(HZ) + " from the environment"},
      {GperfRateSource::DEFAULT, ""}};
  for (const auto& [SOURCE, TEXT] : SOURCES) {
    StderrCapture capture;
    {
      GperfProfiler profiler(
          config(false), "Gperf.Line",
          expecting(plan(false), *PERIOD, SOURCE, SOURCE == GperfRateSource::DEFAULT ? 0 : HZ));
      profiler.beforeMeasure();
      work();
      profiler.afterMeasure(Stats{});
    }
    const std::string ERR = capture.text();
    if (TEXT.empty()) {
      EXPECT_EQ(ERR.find("[gperf] Gperf.Line"), std::string::npos) << "a line at the default\n"
                                                                   << ERR;
    } else {
      const std::regex LINE(
          std::regex_replace(AT + TEXT + "), ", std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\$&)") +
          "[0-9]+ samples?\n");
      EXPECT_TRUE(std::regex_search(ERR, LINE)) << ERR;
    }
    const std::vector<std::string> FAILED = failures();
    EXPECT_TRUE(FAILED.empty()) << FAILED.front();
  }
}

/**
 * @test A file at the capture's path that is not a complete profile fails the
 * case at completion, saying what is wrong with it, and is neither analyzed
 * nor replaced.
 */
TEST_F(GperfLifecycleTest, CaptureThatIsNotAProfileFails) {
  ScopedEnv log("FAKE_LOG", dir_.logPath());
  StderrCapture quiet;
  {
    GperfProfiler profiler(config(), "Gperf.Foreign", plan(true));
    profiler.beforeMeasure();
    work();
    // gperftools writes on into the file it opened; the name now holds another.
    ASSERT_TRUE(std::filesystem::remove(cpuProf("Gperf.Foreign")));
    std::ofstream(cpuProf("Gperf.Foreign")) << "not a profile\n";
    profiler.afterMeasure(Stats{});
  }
  EXPECT_EQ(failures(), std::vector<std::string>{
                            "Gperf.Foreign: completion: unusable: " + cpuProf("Gperf.Foreign") +
                            " is not a complete gperftools CPU profile: its 14 bytes hold no "
                            "header (5 slots of " +
                            std::to_string(sizeof(std::uintptr_t)) + " bytes); raw profile kept"});
  EXPECT_EQ(contents(cpuProf("Gperf.Foreign")), "not a profile\n");
  EXPECT_FALSE(analyzed("Gperf.Foreign")) << dir_.log();
}
