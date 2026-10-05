/**
 * @file 05_NvtxAnnotation_Ranges_uTest.cpp
 * @brief The check behind walkthrough 13: under Nsight Systems, every call of
 *        demo 05's G1Phases carries its three named NVTX ranges, in order,
 *        inside the test's own range, each holding its phase's GPU work.
 *
 * Not part of demo 05, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_Gpu_05_NvtxAnnotation, whose path the build
 * passes in) under nsys with --profile nsight, reads the report's push/pop
 * trace and GPU trace through `nsys stats`, and holds them to what the
 * walkthrough shows:
 *  - one range per test, named after it, NvtxAnnotation.G1 and
 *    NvtxAnnotation.G1Phases, neither inside the other;
 *  - G1's range holds its measured calls' kernels and no range: the
 *    unannotated view;
 *  - G1Phases's range holds copy_in, kernel and copy_out once per measured
 *    call, in that order, one after another, directly inside it, and the
 *    warmup call's three lie outside every test's range;
 *  - in every call, copy_in holds its two copies to the device, kernel its one
 *    SAXPY kernel and copy_out its copy back, each from start to end.
 * Every count follows from the flags the check passes; nothing compares a
 * duration. A capture that does not end with both tests passed and a report
 * fails, unless nsys stops before the tests start, saying it cannot create
 * its temporary files: that skips, quoting nsys (judgeCapture()). The reading
 * of nsys's CSV, the assessment and the capture's judgement are the check's
 * private support (05_NvtxAnnotation_Check.hpp); the process plumbing is
 * walkthrough 15's (12_MemcheckProfiler_Check.hpp). That support's own tests
 * need neither nsys nor a device, so they are a program of their own, built
 * in every configuration (05_NvtxAnnotation_RangesReport_uTest.cpp). ctest
 * runs this one, where the GPU demos are built, under the demo and nsight
 * labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L nsight
 *   ./build/bin/tests/TestDemoNvtxRanges      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/gpu/utst/05_NvtxAnnotation_Check.hpp"

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace check = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using vernier::bench::demo::nvtx_check::assessRanges;
using vernier::bench::demo::nvtx_check::captureUnderNsys;
using vernier::bench::demo::nvtx_check::CaptureVerdict;
using vernier::bench::demo::nvtx_check::GpuOp;
using vernier::bench::demo::nvtx_check::kernelMargins;
using vernier::bench::demo::nvtx_check::MEASURED_CALLS;
using vernier::bench::demo::nvtx_check::NvtxRange;
using vernier::bench::demo::nvtx_check::readGpuOps;
using vernier::bench::demo::nvtx_check::readRanges;
using vernier::bench::demo::nvtx_check::RunDirectory;
using vernier::bench::demo::nvtx_check::WARMUP_CALLS;

namespace {

/* ----------------------------- Constants ----------------------------- */

/// The demo binary the check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_GPU_05_BINARY;

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Nsight Systems records each measured call of G1Phases as copy_in,
 *       kernel, copy_out inside the test's range, each around its own GPU
 *       work, and G1's range with no range inside
 *
 * Runs the demo's two tests under nsys, reads the report with nsys stats and
 * checks it with assessRanges(). Skipped where nsys is not on PATH, no CUDA
 * device is present, or the build has no NVTX headers, all decided before
 * anything runs; and where nsys stops before the demo's tests start because
 * it cannot create its temporary files, quoting nsys (judgeCapture()). The
 * demo must list its tests on its own, its two tests must pass under nsys,
 * and nsys must exit 0 and write the report; anything else fails, saying how
 * the run ended and what it printed, and a failing run keeps its files and
 * says where.
 */
TEST(NvtxRanges, RecordedByNsightSystems) {
#if !VERNIER_NVTX_USABLE
  GTEST_SKIP() << "this build has no NVTX headers (the CUDA toolkit's nvtx3), so the demo's "
                  "ranges and the Nsight backend's compile to nothing";
#else
  if (!vernier::bench::profiler_env::isOnPath("nsys")) {
    GTEST_SKIP() << "nsys is not on PATH; this test runs the demo under Nsight Systems";
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device available; this test runs the demo's kernels under nsys";
  }

  std::error_code found;
  const std::string DEMO = fs::canonical(DEMO_BINARY, found).string();
  ASSERT_FALSE(found) << "the demo binary is missing: " << DEMO_BINARY;

  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made()) << "cannot create a temporary directory";

  // The demo lists its tests on its own first: a demo that cannot start at
  // all fails here, as the demo's, before nsys is involved.
  const check::ChildExit LISTED = check::runLogged({DEMO, "--gtest_list_tests"}, DIR / "list.txt");
  const std::string LIST = check::readText(DIR / "list.txt");
  ASSERT_TRUE(check::exitedWith(LISTED, 0) && LIST.find("G1Phases") != std::string::npos)
      << "the demo does not list its tests (it " << check::describe(LISTED) << "):\n"
      << check::lastLines(LIST);

  const CaptureVerdict CAPTURE = captureUnderNsys({"nsys"}, DEMO, DIR.path());
  if (CAPTURE.action == CaptureVerdict::Action::SKIP) {
    GTEST_SKIP() << CAPTURE.message;
  }
  if (CAPTURE.action == CaptureVerdict::Action::FAIL) {
    FAIL() << CAPTURE.message;
  }
  const fs::path REPORT = DIR / "capture.nsys-rep";

  const check::ChildExit STATS = check::runLogged(
      {"nsys", "stats", "--force-export=true", "--report", "nvtx_pushpop_trace", "--report",
       "cuda_gpu_trace", "--format", "csv", "--output", (DIR / "trace").string(), REPORT.string()},
      DIR / "stats.txt");
  ASSERT_TRUE(check::exitedWith(STATS, 0))
      << "nsys stats " << check::describe(STATS) << ":\n"
      << check::lastLines(check::readText(DIR / "stats.txt"), 20);

  std::string error;
  const std::vector<NvtxRange> RANGES =
      readRanges(check::readText(DIR / "trace_nvtx_pushpop_trace.csv"), error);
  const std::vector<GpuOp> OPS =
      error.empty() ? readGpuOps(check::readText(DIR / "trace_cuda_gpu_trace.csv"), error)
                    : std::vector<GpuOp>{};
  ASSERT_TRUE(error.empty()) << error << "\nnsys stats printed:\n"
                             << check::lastLines(check::readText(DIR / "stats.txt"), 20);

  const auto [BEFORE, AFTER] = kernelMargins(RANGES, OPS);
  std::printf("[NvtxRanges.RecordedByNsightSystems]  %zu ranges, %zu GPU operations; kernel "
              "ranges open %.3f us before their kernels at least and close %.3f us after\n",
              RANGES.size(), OPS.size(), static_cast<double>(BEFORE) / 1e3,
              static_cast<double>(AFTER) / 1e3);

  const std::vector<std::string> PROBLEMS = assessRanges(RANGES, OPS, MEASURED_CALLS, WARMUP_CALLS);
  for (const std::string& problem : PROBLEMS) {
    ADD_FAILURE() << problem;
  }
#endif
}
