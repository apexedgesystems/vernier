/**
 * @file 04_ComputeSanitizerProfiler_Report_uTest.cpp
 * @brief How demo 04's check reads compute-sanitizer's report, held to report
 *        text from real runs, in every build.
 *
 * The check (04_ComputeSanitizerProfiler_uTest.cpp, built where the GPU demos
 * are) runs the demo under the tool and decides on what
 * 04_ComputeSanitizerProfiler_Check.hpp reads from each report. These tests
 * hold that reading to report text from real runs. They need neither the
 * tool, nor CUDA, nor a device, so every build has them, and ctest runs them
 * under the check's labels, demo and compute-sanitizer.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L compute-sanitizer
 *   ./build/bin/tests/TestDemoComputeSanitizerReport      # these alone, by hand
 *   @endcode
 */

#include "src/bench/demo/gpu/utst/04_ComputeSanitizerProfiler_Check.hpp"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace check = vernier::bench::demo::sanitizer_check;

/* ----------------------------- Report Reading Tests ----------------------------- */

// What decides the checks' pass and failures, on report text taken from
// real runs on a Jetson AGX Thor (compute-sanitizer 2025.3.1) and in the
// dev-cuda image (2025.4.0).

namespace {

/// One invalid read as the tool printed it on the Thor, with @p allocation as
/// its allocation line.
std::string readSnippet(const std::string& allocation) {
  return "========= COMPUTE-SANITIZER\n"
         "========= Invalid __global__ read of size 4 bytes\n"
         "=========     at vernier::bench::demo::sanitizer_demo::<unnamed>::saxpyUnguarded(float, "
         "const float *, float *)+0x100 in 04_ComputeSanitizerProfiler_Unguarded.cu:33\n"
         "=========     by thread (255,0,0) in block (4095,0,0)\n"
         "=========     Access to 0xd67bffffc is out of bounds\n"
         "=========     " +
         allocation +
         "\n"
         "=========     Saved host backtrace up to driver entry point at kernel launch time\n"
         "=========         Host Frame: main [0xabeb] in "
         "BenchDemo_Gpu_04_ComputeSanitizerProfiler\n"
         "========= \n"
         "========= Program hit cudaErrorLaunchFailure (error 719) due to \"unspecified launch "
         "failure\" on CUDA API call to cudaDeviceSynchronize.\n"
         "=========     Saved host backtrace up to driver entry point at error\n"
         "========= \n"
         "========= Target application returned an error\n"
         "========= ERROR SUMMARY: 4 errors\n";
}

constexpr const char* THOR_ALLOCATION =
    "and is 1 bytes after the nearest allocation at 0xd67800000 of size 4,194,300 bytes";
constexpr const char* IMAGE_ALLOCATION =
    "and is 1 bytes after the nearest allocation at 0x771b9a200000 of size 4194300 bytes";

} // namespace

/** @test A report is read into its one access, with every line the checks look at */
TEST(SanitizerReportTest, InvalidAccessReadsItsLines) {
  const std::vector<check::InvalidAccess> ACCESSES =
      check::invalidAccesses(readSnippet(THOR_ALLOCATION));

  ASSERT_EQ(ACCESSES.size(), 1u);
  EXPECT_EQ(ACCESSES[0].kind, "Invalid __global__ read of size 4 bytes");
  EXPECT_TRUE(check::frameAt(ACCESSES[0].frame, "saxpyUnguarded",
                             "04_ComputeSanitizerProfiler_Unguarded.cu:33"));
  EXPECT_EQ(ACCESSES[0].thread, "by thread (255,0,0) in block (4095,0,0)");
  EXPECT_EQ(ACCESSES[0].address, "Access to 0xd67bffffc is out of bounds");
  EXPECT_EQ(check::bytesAfterAllocation(ACCESSES[0]), 1);
  EXPECT_EQ(check::allocationSize(ACCESSES[0]), 4194300);
  EXPECT_EQ(check::errorSummary(readSnippet(THOR_ALLOCATION)), 4);
}

/** @test The allocation's size reads the same with and without thousands separators */
TEST(SanitizerReportTest, AllocationSizeReadsBothVersions) {
  const std::vector<check::InvalidAccess> THOR =
      check::invalidAccesses(readSnippet(THOR_ALLOCATION));
  const std::vector<check::InvalidAccess> IMAGE =
      check::invalidAccesses(readSnippet(IMAGE_ALLOCATION));

  ASSERT_EQ(THOR.size(), 1u);
  ASSERT_EQ(IMAGE.size(), 1u);
  EXPECT_EQ(check::allocationSize(THOR[0]), 4194300);
  EXPECT_EQ(check::allocationSize(IMAGE[0]), 4194300);
  EXPECT_EQ(check::bytesAfterAllocation(IMAGE[0]), 1);
}

/** @test The API failures after the access are not accesses; a clean report has none */
TEST(SanitizerReportTest, ApiFailuresAndCleanReportsAreNoAccesses) {
  EXPECT_EQ(check::invalidAccesses(readSnippet(THOR_ALLOCATION)).size(), 1u);
  const std::string CLEAN = "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n";
  EXPECT_TRUE(check::invalidAccesses(CLEAN).empty());
  EXPECT_EQ(check::errorSummary(CLEAN), 0);
  EXPECT_EQ(check::errorSummary(""), -1);
}

/** @test The first summary is the total; the print-limit line after it is not */
TEST(SanitizerReportTest, FirstSummaryIsTheTotal) {
  EXPECT_EQ(check::errorSummary("========= ERROR SUMMARY: 259 errors\n"
                                "========= ERROR SUMMARY: 159 errors were not printed. Use "
                                "--print-limit option to adjust the number of printed errors\n"),
            259);
}

/** @test A frame needs the kernel and the location; a frame without a location is at none */
TEST(SanitizerReportTest, FrameAtNeedsTheKernelAndTheLocation) {
  const std::string WITH_LINE =
      "at vernier::bench::demo::sanitizer_demo::<unnamed>::saxpyUnguarded(float, const float *, "
      "float *)+0x100 in 04_ComputeSanitizerProfiler_Unguarded.cu:33";
  const std::string WITH_DIRECTORY =
      "at saxpyUnguarded(float, const float *, float *)+0x100 in "
      "/src/bench/demo/gpu/04_ComputeSanitizerProfiler_Unguarded.cu:33";
  const std::string WITHOUT_LINE =
      "at vernier::bench::demo::sanitizer_demo::<unnamed>::saxpyUnguarded(float, const float *, "
      "float *)+0x100";

  EXPECT_TRUE(
      check::frameAt(WITH_LINE, "saxpyUnguarded", "04_ComputeSanitizerProfiler_Unguarded.cu:33"));
  EXPECT_TRUE(check::frameAt(WITH_DIRECTORY, "saxpyUnguarded",
                             "04_ComputeSanitizerProfiler_Unguarded.cu:33"));
  EXPECT_FALSE(
      check::frameAt(WITH_LINE, "saxpyUnguarded", "04_ComputeSanitizerProfiler_Unguarded.cu:32"));
  EXPECT_FALSE(
      check::frameAt(WITH_LINE, "saxpyKernel", "04_ComputeSanitizerProfiler_Unguarded.cu:33"));
  EXPECT_FALSE(
      check::frameAt(WITH_LINE, "saxpyUnguarded", "x04_ComputeSanitizerProfiler_Unguarded.cu:33"));
  EXPECT_FALSE(check::frameAt(WITHOUT_LINE, "saxpyUnguarded",
                              "04_ComputeSanitizerProfiler_Unguarded.cu:33"));
}

/** @test The unguarded statement's line is found once, or not at all */
TEST(SanitizerReportTest, LineOfFindsTheStatementOnce) {
  EXPECT_EQ(check::lineOf("a\n  y[I] = a * x[I] + y[I];\nb\n", check::UNGUARDED_STATEMENT), 2u);
  EXPECT_EQ(check::lineOf("y[I] = a * x[I] + y[I];\ny[I] = a * x[I] + y[I];\n",
                          check::UNGUARDED_STATEMENT),
            0u);
  EXPECT_EQ(check::lineOf("a\nb\n", check::UNGUARDED_STATEMENT), 0u);
}
