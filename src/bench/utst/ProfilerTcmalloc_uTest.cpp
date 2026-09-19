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
 */

#include "src/bench/inc/ProfilerGperf.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerHeaptrack.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using vernier::bench::HeaptrackProfiler;
using vernier::bench::makeGperfProfiler;
using vernier::bench::PerfConfig;
using vernier::bench::Profiler;
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

/** @test Says how to enable heap mode exactly when heap profiling is not compiled in */
TEST_F(TcmallocOptInTest, GperfHeapModeExplainsWhenUnavailable) {
  cfg_.profileTool = "gperf";
  cfg_.profileArgs = "heap";

  // The explanation prints once per process; this is the only test in the
  // binary that requests heap mode.

  StderrCapture capture;
  const std::unique_ptr<Profiler> profiler = makeGperfProfiler(cfg_, "Tcmalloc.GperfHeap");
  const std::string err = capture.text();

  if (profiler == nullptr) {
    GTEST_SKIP() << "gperftools headers not present at build time";
  }

  const bool explained = err.find("VERNIER_LINK_TCMALLOC") != std::string::npos;
  EXPECT_EQ(explained, UB_HAS_GPERF_HEAP == 0) << err;
  // Heap support is never compiled in without the allocator it needs.
  EXPECT_TRUE(UB_HAS_GPERF_HEAP == 0 || TCMALLOC_REQUESTED);
}
