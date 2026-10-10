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

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::ProfilerRegistry;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

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

  /** @brief A CPU request with --profile-analyze, into this test's folder, no watchdog. */
  PerfConfig config() const {
    PerfConfig cfg;
    cfg.profileTool = "gperf";
    cfg.profileAnalyze = true;
    cfg.artifactRoot = dir_.path();
    cfg.cycles = 1;
    cfg.repeats = 2;
    cfg.profileTestTimeoutSecs = 0;
    return cfg;
  }

  /** @brief The CPU profile of case @p test. */
  std::string cpuProf(const std::string& test) const {
    return dir_.path() + "/" + test + ".gperf/cpu.prof";
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
