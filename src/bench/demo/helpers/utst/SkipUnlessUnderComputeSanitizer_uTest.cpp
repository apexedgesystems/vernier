/**
 * @file SkipUnlessUnderComputeSanitizer_uTest.cpp
 * @brief Unit tests for the demo helper that skips a case unless it runs
 *        under Compute Sanitizer.
 *
 * Notes:
 *  - The helper decides from the process it runs in, so the tests run this
 *    binary as a child on a probe case that uses the helper, once plainly and
 *    once under compute-sanitizer, and read what GoogleTest printed for the
 *    probe. This binary calls no CUDA, so the tool is told not to require it
 *    (`--require-cuda-init no`); what it exports to the process is the same.
 *  - The tool run skips in a build with the address sanitizer, whose runtime
 *    aborts the program before its tests start once the tool's libraries are
 *    injected ("AddressSanitizer: alloc-dealloc-mismatch"), and where
 *    compute-sanitizer is not on PATH, both decided before anything runs;
 *    any other way of not reaching the probe fails.
 *  - Tests are platform-agnostic and independent of execution order.
 */

#include "src/bench/demo/helpers/SkipUnlessUnderComputeSanitizer.hpp"

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/gpu/utst/04_ComputeSanitizerProfiler_Check.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <string>
#include <system_error>

#include <gtest/gtest.h>

namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using vernier::bench::demo::reasonToSkipUnlessUnderComputeSanitizer;
using vernier::bench::demo::SKIP_UNLESS_UNDER_COMPUTE_SANITIZER_REASON;
using vernier::bench::demo::sanitizer_check::BUILT_WITH_ASAN;
using vernier::bench::profiler_env::isOnPath;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The probe's name, as the child runs select it and GoogleTest prints it.
constexpr const char* PROBE_TEST = "SkipUnlessUnderComputeSanitizerProbe.RunsOnlyUnderTheTool";

/// What the probe prints once the helper has let it run.
constexpr const char* PROBE_RAN = "probe: the helper let this case run";

/// Path of this binary, which the child runs execute.
std::string selfPath() {
  std::array<char, 4096> buf{};
  const ssize_t LEN = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  return LEN > 0 ? std::string(buf.data(), static_cast<std::size_t>(LEN)) : std::string{};
}

/// A new temporary directory for one test's runs; empty when it cannot be
/// made.
fs::path scratchDir() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo-helpers-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

} // namespace

/* ----------------------------- Probe ----------------------------- */

/** @test The probe the child runs select: skips unless under the tool, then reports that it ran */
TEST(SkipUnlessUnderComputeSanitizerProbe, RunsOnlyUnderTheTool) {
  DEMO_SKIP_UNLESS_UNDER_COMPUTE_SANITIZER();
  EXPECT_TRUE(reasonToSkipUnlessUnderComputeSanitizer().empty());
  std::puts(PROBE_RAN);
}

/* ----------------------------- API Tests ----------------------------- */

/** @test The reason tells the reader both ways to run the case */
TEST(SkipUnlessUnderComputeSanitizerTest, ReasonSaysHowToRunTheCase) {
  const std::string reason = SKIP_UNLESS_UNDER_COMPUTE_SANITIZER_REASON;

  EXPECT_NE(reason.find("compute-sanitizer --tool=memcheck"), std::string::npos) << reason;
  EXPECT_NE(reason.find("bench run --profile compute-sanitizer"), std::string::npos) << reason;
}

/** @test In a plain run the probe reports SKIPPED with the reason, and does not run */
TEST(SkipUnlessUnderComputeSanitizerTest, PlainRunSkipsTheProbe) {
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "could not create a scratch directory";

  const vg::ChildExit END = vg::runLogged(
      {selfPath(), std::string("--gtest_filter=") + PROBE_TEST, "--gtest_print_time=0"},
      DIR / "plain.txt");
  const std::string OUTPUT = vg::readText(DIR / "plain.txt");
  std::error_code ec;
  fs::remove_all(DIR, ec);

  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << OUTPUT;
  EXPECT_NE(OUTPUT.find(std::string("[  SKIPPED ] ") + PROBE_TEST), std::string::npos) << OUTPUT;
  EXPECT_NE(OUTPUT.find(SKIP_UNLESS_UNDER_COMPUTE_SANITIZER_REASON), std::string::npos) << OUTPUT;
  EXPECT_EQ(OUTPUT.find(PROBE_RAN), std::string::npos) << OUTPUT;
}

/** @test Under compute-sanitizer the probe runs and passes */
TEST(SkipUnlessUnderComputeSanitizerTest, ToolRunRunsTheProbe) {
  if constexpr (BUILT_WITH_ASAN) {
    GTEST_SKIP() << "this binary is built with the address sanitizer, whose runtime aborts the "
                    "program before its tests start once compute-sanitizer's libraries are "
                    "injected (`AddressSanitizer: alloc-dealloc-mismatch`); run this test in a "
                    "build without it";
  }
  if (!isOnPath("compute-sanitizer")) {
    GTEST_SKIP() << "compute-sanitizer is not on PATH; this test runs the probe under it";
  }
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "could not create a scratch directory";

  const vg::ChildExit END =
      vg::runLogged({"compute-sanitizer", "--require-cuda-init", "no", selfPath(),
                     std::string("--gtest_filter=") + PROBE_TEST, "--gtest_print_time=0"},
                    DIR / "tool.txt");
  const std::string OUTPUT = vg::readText(DIR / "tool.txt");
  std::error_code ec;
  fs::remove_all(DIR, ec);

  // Any way of not reaching the probe fails, with what the run printed.
  ASSERT_TRUE(vg::testsStarted(OUTPUT))
      << "the probe did not start under compute-sanitizer (" << vg::describe(END)
      << "). The run printed:\n"
      << (OUTPUT.empty() ? std::string("(nothing)\n") : vg::lastLines(OUTPUT, 40));
  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << OUTPUT;
  EXPECT_NE(OUTPUT.find(PROBE_RAN), std::string::npos) << OUTPUT;
  EXPECT_TRUE(vg::testPassed(OUTPUT, PROBE_TEST) && vg::oneTestPassed(OUTPUT)) << OUTPUT;
  EXPECT_EQ(OUTPUT.find("[  SKIPPED ]"), std::string::npos) << OUTPUT;
}
