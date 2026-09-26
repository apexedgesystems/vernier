/**
 * @file ProfilerTcmalloc_uTest.cpp
 * @brief Unit tests for the tcmalloc opt-in and the backends that depend on it.
 *
 * Notes:
 *  - tcmalloc replaces the allocator for the whole process, so these tests
 *    compare what the build asked for (VERNIER_HAS_TCMALLOC) with what the
 *    loader mapped, and check the two backends that must tell the user.
 *  - Each expectation holds in both configurations; which branch runs
 *    depends on how the tree was configured.
 *  - A heap request without heap support is a readiness error that the
 *    registry prints once per process, so it is asserted in a child process
 *    that starts with that state clear.
 */

#include "src/bench/inc/ProfilerGperf.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerHeaptrack.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using vernier::bench::EnvReport;
using vernier::bench::HeaptrackProfiler;
using vernier::bench::PerfConfig;
using vernier::bench::Profiler;
using vernier::bench::ProfilerRegistry;
using vernier::bench::test::StderrCapture;

namespace {

bool tcmallocMapped() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find("libtcmalloc") != std::string::npos) {
      return true;
    }
  }
  return false;
}

#if defined(VERNIER_HAS_TCMALLOC)
constexpr bool TCMALLOC_REQUESTED = true;
#else
constexpr bool TCMALLOC_REQUESTED = false;
#endif

/** @brief True when the gperf backend is compiled in. */
constexpr bool GPERF_COMPILED_IN = UB_HAS_GPERF_CPU != 0 || UB_HAS_GPERF_HEAP != 0;

/** @brief Create one heap-mode profiler as a run does; true when the explanation was printed. */
bool heapRequestExplains(const PerfConfig& cfg) {
  StderrCapture capture;
  const std::unique_ptr<Profiler> profiler = Profiler::make(cfg, "Tcmalloc.GperfHeap");
  return capture.text().find("VERNIER_LINK_TCMALLOC") != std::string::npos;
}

/**
 * @brief Child-process body: report which of two heap-mode requests explained.
 *
 * Removes the child's artifact root itself, because a death-test child never
 * reaches TearDown().
 */
[[noreturn]] void reportHeapExplanations(const PerfConfig& cfg, const std::filesystem::path& root) {
  const bool first = heapRequestExplains(cfg);
  const bool second = heapRequestExplains(cfg);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::fprintf(stderr, "heap-explanation first=%d second=%d\n", first ? 1 : 0, second ? 1 : 0);
  std::exit(0);
}

} // namespace

/** @brief Gives each test a private artifact root and removes it afterwards. */
class TcmallocOptInTest : public ::testing::Test {
protected:
  PerfConfig cfg_{};
  std::filesystem::path root_;

  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("vernier_tcmalloc_utest_" + std::to_string(::getpid()));
    std::filesystem::create_directories(root_);
    cfg_.artifactRoot = root_.string();
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }
};

/* ----------------------------- API Tests ----------------------------- */

/** @test Maps tcmalloc into the process exactly when the build opted in */
TEST_F(TcmallocOptInTest, MappedOnlyWhenRequested) {
  EXPECT_EQ(tcmallocMapped(), TCMALLOC_REQUESTED);
}

/** @test Warns that C++ allocations will be missing exactly when tcmalloc is mapped */
TEST_F(TcmallocOptInTest, HeaptrackWarnsOnlyWhenTcmallocMapped) {
  HeaptrackProfiler profiler(cfg_, "Tcmalloc.Heaptrack");

  StderrCapture capture;
  profiler.beforeMeasure();
  const std::string err = capture.text();

  const bool warned = err.find("libtcmalloc") != std::string::npos;
  EXPECT_EQ(warned, tcmallocMapped()) << err;
}

/** @test Doctor reports heaptrack as a warning naming tcmalloc exactly when tcmalloc is mapped */
TEST_F(TcmallocOptInTest, HeaptrackDoctorWarnsOnlyWhenTcmallocMapped) {
  // The check bench doctor prints: it runs in this process, as it does in a
  // benchmark started with --profile-check.
  const EnvReport report = ProfilerRegistry::instance().runCheck("heaptrack");
  if (report.status == EnvReport::Status::Error) {
    GTEST_SKIP() << "heaptrack cannot run here, so the allocator is not checked: "
                 << report.message;
  }

  const bool warned = report.status == EnvReport::Status::Warning;
  EXPECT_EQ(warned, tcmallocMapped()) << report.message;
  if (warned) {
    EXPECT_NE(report.message.find("libtcmalloc"), std::string::npos) << report.message;
    EXPECT_NE(report.hint.find("VERNIER_LINK_TCMALLOC"), std::string::npos) << report.hint;
  }
}

/** @brief Same fixture, named so GoogleTest schedules the death test first. */
using TcmallocOptInDeathTest = TcmallocOptInTest;

/** @test Reports an unavailable heap mode on the first request in a process and never again */
TEST_F(TcmallocOptInDeathTest, GperfHeapModeExplainsOncePerProcess) {
  cfg_.profileTool = "gperf";
  cfg_.profileArgs = "heap";

  if (!GPERF_COMPILED_IN) {
    GTEST_SKIP() << "gperftools headers not present at build time";
  }

  // "threadsafe" re-executes the test binary for the child, so the
  // once-per-process state starts clear whatever ran before in this process;
  // the default style forks and would inherit it.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  const std::string expected =
      std::string("heap-explanation first=") + (UB_HAS_GPERF_HEAP == 0 ? "1" : "0") + " second=0";
  EXPECT_EXIT(reportHeapExplanations(cfg_, root_), ::testing::ExitedWithCode(0), expected);

  // Heap support is never compiled in without the allocator it needs.
  EXPECT_TRUE(UB_HAS_GPERF_HEAP == 0 || TCMALLOC_REQUESTED);
}
