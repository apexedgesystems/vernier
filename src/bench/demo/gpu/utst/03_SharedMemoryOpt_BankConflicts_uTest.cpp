/**
 * @file 03_SharedMemoryOpt_BankConflicts_uTest.cpp
 * @brief The check behind walkthrough 12: Nsight Compute counts the bank
 *        conflicts of demo 03's tiled transpose, and none in the padded one.
 *
 * Not part of demo 03, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_Gpu_03_SharedMemoryOpt, whose path the
 * build passes in) under ncu with the three metrics the walkthrough reads
 * and holds every launch of every kernel to what the walkthrough says: the
 * naive kernel loads nothing from shared memory; the tiled kernels execute
 * one shared-load instruction per warp; the conflicting tile's reads take
 * 32 wavefronts per instruction and the padded tile's one, and ncu's
 * bank-conflict counter reads accordingly. The process plumbing is
 * walkthrough 15's (12_MemcheckProfiler_Check.hpp); the reading of ncu's
 * CSV and the verdict are 03_SharedMemoryOpt_BankConflicts_Check.hpp's, whose
 * tests need no GPU and run in every configuration
 * (03_SharedMemoryOpt_BankConflictsReport_uTest.cpp). ctest runs this check
 * under the demo and ncu labels where the GPU demos are built.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L ncu             # root where the counters are administrators' only
 *   ./build/bin/tests/TestDemoBankConflicts   # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/gpu/utst/03_SharedMemoryOpt_BankConflicts_Check.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace bc = vernier::bench::demo::bank_conflicts_check;
namespace check = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using bc::CONFLICT;
using bc::describeLaunches;
using bc::expectWalkthroughCounts;
using bc::METRIC_CONFLICTS;
using bc::METRIC_INSTRUCTIONS;
using bc::METRIC_WAVEFRONTS;
using bc::NAIVE;
using bc::NCU_NO_PERMISSION;
using bc::ncuLineWith;
using bc::PADDED;
using bc::Readings;
using bc::readReport;
using bc::Report;

namespace {

/// The demo binary the check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_GPU_03_BINARY;

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Nsight Compute counts 32 wavefronts per shared-load instruction in
 *       the conflicting tile and one in the padded tile, and its
 *       bank-conflict counter reads accordingly.
 *
 * Runs the demo binary under ncu with the three metrics on the fewest
 * launches the harness makes (one warmup and one measured launch per kernel,
 * after the harness's own first), reads ncu's CSV, and checks every launch
 * (expectWalkthroughCounts): the naive kernel executes no shared-load
 * instruction and reports no conflict; the tiled kernels execute one
 * shared-load instruction per warp of the launch; the conflicting kernel's
 * SASS-level wavefronts are exactly TILE_DIM times its instructions and its
 * L1TEX counter at least TILE_DIM - 2 times them; the padded kernel's
 * wavefronts equal its instructions and its counter, which the report must
 * give as a whole number, stays below 2% of the conflicting kernel's
 * smallest reading. A kernel the report names no launch of fails the check.
 *
 * Skipped where ncu is not on PATH or no CUDA device is present, decided
 * before anything runs; where ncu refuses the counters to this user,
 * quoting its line; and where ncu prints n/a for one of the metrics on this
 * GPU or in this version, quoting the row, once the demo's tests have run to
 * their end under it. The demo's three tests must pass under ncu and ncu
 * must exit 0; anything else fails with the run's last lines.
 */
TEST(BankConflicts, CountedByNsightCompute) {
  if (!vernier::bench::profiler_env::isOnPath("ncu")) {
    GTEST_SKIP() << "ncu is not on PATH; this test runs the demo under Nsight Compute";
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device available; this test runs the demo's kernels under ncu";
  }

  std::error_code found;
  const std::string DEMO = fs::canonical(DEMO_BINARY, found).string();
  ASSERT_FALSE(found) << "the demo binary is missing: " << DEMO_BINARY;

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo-gpu03-XXXXXX").string();
  ASSERT_NE(::mkdtemp(dirTemplate.data()), nullptr) << "cannot create a temporary directory";
  const fs::path DIR = dirTemplate;
  const fs::path CSV = DIR / "metrics.csv";
  const fs::path OUTPUT = DIR / "program.txt";

  // ncu replays every launch, so the demo makes the fewest the harness
  // allows; its metrics go to the log file, the demo's output to another.
  const std::vector<std::string> ARGS = {"ncu",
                                         "--target-processes",
                                         "all",
                                         "--metrics",
                                         std::string(METRIC_INSTRUCTIONS) + "," +
                                             METRIC_WAVEFRONTS + "," + METRIC_CONFLICTS,
                                         "--csv",
                                         "--log-file",
                                         CSV.string(),
                                         DEMO,
                                         "--gtest_filter=SharedMemoryOpt.*",
                                         "--gtest_print_time=0",
                                         "--gpu-warmup",
                                         "1",
                                         "--cycles",
                                         "1",
                                         "--repeats",
                                         "1"};
  const check::ChildExit END = check::runLogged(ARGS, OUTPUT);
  const std::string PROGRAM = check::readText(OUTPUT);
  const std::string LOG = check::readText(CSV);

  const std::string REFUSED = ncuLineWith(LOG + PROGRAM, NCU_NO_PERMISSION);
  if (!REFUSED.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "ncu may not read the GPU's performance counters as this user (on a Jetson "
                    "they are for administrators: run this test as root). It printed:\n"
                 << REFUSED;
  }

  ASSERT_TRUE(check::testsStarted(PROGRAM))
      << "the demo did not start under ncu: ncu " << check::describe(END) << ". The run printed:\n"
      << check::lastLines(PROGRAM + LOG, 40);
  ASSERT_NE(PROGRAM.find("[  PASSED  ] 3 tests."), std::string::npos)
      << "the demo's three tests did not all pass under ncu (its output is in " << DIR << "):\n"
      << check::lastLines(PROGRAM, 40);
  ASSERT_TRUE(check::exitedWith(END, 0))
      << "ncu " << check::describe(END) << " (its log is " << CSV << "):\n"
      << check::lastLines(LOG + PROGRAM, 20);

  const Report REPORT = readReport(LOG);
  if (!REPORT.unavailableRow.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "ncu has no value for a metric the check reads, on this GPU or in this "
                    "version. Its row:\n"
                 << REPORT.unavailableRow;
  }
  const Readings& READINGS = REPORT.readings;
  std::printf("[BankConflicts.CountedByNsightCompute]  per launch: naive %s | conflicting %s | "
              "padded %s\n",
              describeLaunches(READINGS, NAIVE).c_str(),
              describeLaunches(READINGS, CONFLICT).c_str(),
              describeLaunches(READINGS, PADDED).c_str());
  expectWalkthroughCounts(READINGS);

  if (HasFailure()) {
    std::printf("ncu output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}
