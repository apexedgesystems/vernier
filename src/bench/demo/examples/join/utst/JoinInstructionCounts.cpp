/**
 * @file JoinInstructionCounts.cpp
 * @brief The check behind walkthrough 07: under callgrind, joinV0 executes
 * several times joinV1's instructions per call, and a second run of the same
 * work counts the same total
 *
 * Not part of demo 07, whose source shows only the two measured versions.
 * This program runs itself under callgrind as a counting worker, six times,
 * and reads each run's program total, so the count per call is exact and
 * nothing depends on callgrind's call graph. ctest runs it as
 * CallgrindDemoInstructionCountsTest (labels callgrind and demo); walkthrough
 * 07's step 4 runs it by hand.
 *
 * Usage:
 *   @code{.sh}
 *   ./JoinInstructionCounts      # skips where valgrind is not installed
 *   @endcode
 *
 * The build compiles its sanitizer setting in (JOIN_COUNTS_SANITIZER, from
 * -DSANITIZER=asan|tsan|ubsan): a sanitizer's instrumentation would be counted
 * with the join code, so a sanitizer build skips the check, saying what was
 * seen when one was counted.
 */

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/demo/examples/join/utst/CallgrindRuns.hpp"

namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::callgrind_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The input demo 07 measures: 1,000 words from a fixed seed, joined by ','.
constexpr std::size_t PART_COUNT = 1000;
constexpr unsigned PART_SEED = 42;
constexpr char SEPARATOR = ',';

/// Turns Worker into a counting run: "<V0|V1> <calls>", for example "V0 10".
constexpr const char* COUNT_CALLS_VARIABLE = "VERNIER_JOIN_COUNT_CALLS";

/// Calls in the two counting runs of each version. Everything else a counting
/// run does is the same in both, so the difference between the two
/// whole-program counts is exactly the extra calls. Both numbers have two
/// digits, so both runs see an environment of the same size.
constexpr int COUNT_CALLS_LOW = 10;
constexpr int COUNT_CALLS_HIGH = 20;

/// V0 must execute more than this many times V1's instructions per call. The
/// smallest ratio measured while choosing it came from an unoptimized x86 build
/// (6.9x); optimized builds measure several times more, on x86 and on the Arm
/// reference rig. Instruction counts carry no noise, so the margin only has to
/// cover build and platform differences.
constexpr double MIN_INSTRUCTION_RATIO = 5.0;

/// The sanitizer this program is built with, as the build set it: "asan",
/// "tsan", "ubsan", or empty. Decided at compile time, so the check skips
/// before it starts anything.
#ifdef JOIN_COUNTS_SANITIZER
constexpr std::string_view BUILD_SANITIZER = JOIN_COUNTS_SANITIZER;
#else
constexpr std::string_view BUILD_SANITIZER = "";
#endif

/// Why a build with @p sanitizer cannot be counted: what was seen when such a
/// build of this program ran under callgrind. Used only in a sanitizer build.
[[maybe_unused]] constexpr std::string_view sanitizerSkipReason(std::string_view sanitizer) {
  if (sanitizer == "asan") {
    return "this program is built with the address sanitizer (SANITIZER=asan): its counting "
           "worker ran outside valgrind and left no profile (clang), or valgrind stopped "
           "reading it (GCC), so there is nothing to count";
  }
  if (sanitizer == "tsan") {
    return "this program is built with the thread sanitizer (SANITIZER=tsan): its runtime "
           "made two identical counting runs count different totals, so the second-run "
           "check cannot hold";
  }
  if (sanitizer == "ubsan") {
    return "this program is built with the undefined-behaviour sanitizer (SANITIZER=ubsan): "
           "its checks are counted with the join code, and V0 came to 4.9 times V1's "
           "instructions per call, under the 5x this check asserts for the code built "
           "without them";
  }
  return "this program is built with a sanitizer this check does not know";
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test The counting worker: calls one version a given number of times and
 * does nothing else, so a callgrind count of the run holds nothing that
 * depends on time.
 *
 * UnderCallgrind runs it under callgrind with VERNIER_JOIN_COUNT_CALLS set
 * ("V0 10"); without the variable it skips.
 */
TEST(JoinInstructionCounts, Worker) {
  const char* job = std::getenv(COUNT_CALLS_VARIABLE);
  if (job == nullptr) {
    GTEST_SKIP() << "UnderCallgrind runs this worker under callgrind";
  }
  std::string version;
  int calls = 0;
  std::istringstream(job) >> version >> calls;
  ASSERT_TRUE((version == "V0" || version == "V1") && calls > 0)
      << COUNT_CALLS_VARIABLE << " is '" << job << "', not '<V0|V1> <calls>'";

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  const auto JOIN = (version == "V0") ? &demo::joinV0 : &demo::joinV1;
  // Written, never read: the volatile store keeps the calls from being removed.
  [[maybe_unused]] volatile std::size_t sink = 0;
  for (int call = 0; call < calls; ++call) {
    sink = JOIN(PARTS, SEPARATOR).size();
  }
}

/**
 * @test V0 executes more than MIN_INSTRUCTION_RATIO times V1's instructions per
 * call, and a second callgrind run of the same work counts the same number.
 *
 * Runs this program's Worker under callgrind six times: each version with 10
 * and with 20 calls, and with 10 calls again. A version's instructions per
 * call are the difference between its two program totals divided by the
 * difference in calls, so nothing depends on callgrind's call graph, which
 * on the Arm rig credited joinV0 with calls it never received.
 *
 * Skipped, saying why, in a build with a sanitizer (sanitizerSkipReason()),
 * where valgrind is not installed, or where valgrind stops reading this
 * program's debug information before the program runs (debugInfoGiveUp()).
 * A counting run that does not reach the worker for any other reason fails,
 * with what the run printed.
 */
TEST(JoinInstructionCounts, UnderCallgrind) {
  if constexpr (!BUILD_SANITIZER.empty()) {
    GTEST_SKIP() << sanitizerSkipReason(BUILD_SANITIZER);
  }
  if (std::system("command -v valgrind >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "valgrind is not installed; this test counts instructions under callgrind";
  }

  std::array<char, 4096> self{};
  const ssize_t SELF_LENGTH = ::readlink("/proc/self/exe", self.data(), self.size() - 1);
  ASSERT_GT(SELF_LENGTH, 0) << "cannot find this binary's path";
  const std::string SELF(self.data(), static_cast<std::size_t>(SELF_LENGTH));

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-join-counts-XXXXXX").string();
  ASSERT_NE(::mkdtemp(dirTemplate.data()), nullptr) << "cannot create a temporary directory";
  const fs::path DIR = dirTemplate;

  struct Run {
    const char* version;
    int calls;
    std::uint64_t total;
  };
  std::array<Run, 6> runs{{{"V0", COUNT_CALLS_LOW, 0},
                           {"V0", COUNT_CALLS_HIGH, 0},
                           {"V0", COUNT_CALLS_LOW, 0},
                           {"V1", COUNT_CALLS_LOW, 0},
                           {"V1", COUNT_CALLS_HIGH, 0},
                           {"V1", COUNT_CALLS_LOW, 0}}};
  for (std::size_t i = 0; i < runs.size(); ++i) {
    const fs::path PROFILE = DIR / ("run" + std::to_string(i) + ".callgrind.out");
    const fs::path LOG = DIR / ("run" + std::to_string(i) + ".log");
    // On the Arm reference rig, identical runs count a few instructions apart
    // without fallback-llsc, valgrind's alternative handling of load- and
    // store-exclusive instruction pairs (MIPS and ARM64 only), and the same
    // total with it. On x86 the hint changes nothing.
    const check::ChildExit END =
        check::runLogged({"valgrind", "--tool=callgrind", "--sim-hints=fallback-llsc",
                          "--callgrind-out-file=" + check::escapePercent(PROFILE.string()), SELF,
                          "--gtest_filter=JoinInstructionCounts.Worker", "--gtest_print_time=0"},
                         std::string(COUNT_CALLS_VARIABLE) + "=" + runs[i].version + " " +
                             std::to_string(runs[i].calls),
                         LOG);
    const std::string LOG_TEXT = check::readText(LOG);

    // Only valgrind's own reason skips: its debug-information reader gave up
    // before the program ran. A crash, a signal, an early exit, a failed launch
    // or no output at all is a failure of the run, reported with what the run
    // printed (the log is short when the tests never started).
    if (!check::testsStarted(LOG_TEXT)) {
      const std::string GAVE_UP = check::debugInfoGiveUp(END, LOG_TEXT);
      std::error_code ec;
      fs::remove_all(DIR, ec);
      if (!GAVE_UP.empty()) {
        GTEST_SKIP() << "valgrind gave up reading this program's debug information before "
                        "it ran. It printed:\n"
                     << GAVE_UP;
      }
      FAIL() << "the counting worker did not start under callgrind: valgrind "
             << check::describe(END) << ". The run printed:\n"
             << (LOG_TEXT.empty() ? std::string("(nothing)\n") : check::lastLines(LOG_TEXT, 40));
    }
    ASSERT_TRUE(check::exitedCleanly(END))
        << "the counting run under callgrind " << check::describe(END) << " (log " << LOG << "):\n"
        << check::lastLines(LOG_TEXT);
    ASSERT_TRUE(check::oneTestPassed(LOG_TEXT))
        << "the counting worker did not run to its end under callgrind (log " << LOG << "):\n"
        << check::lastLines(LOG_TEXT);
    runs[i].total = check::programTotal(PROFILE);
    ASSERT_GT(runs[i].total, 0u) << "no program total in " << PROFILE;
  }

  ASSERT_GT(runs[1].total, runs[0].total);
  ASSERT_GT(runs[4].total, runs[3].total);
  const double EXTRA_CALLS = COUNT_CALLS_HIGH - COUNT_CALLS_LOW;
  const double V0_PER_CALL = static_cast<double>(runs[1].total - runs[0].total) / EXTRA_CALLS;
  const double V1_PER_CALL = static_cast<double>(runs[4].total - runs[3].total) / EXTRA_CALLS;
  const bool IDENTICAL = runs[2].total == runs[0].total && runs[5].total == runs[3].total;
  std::printf("[JoinInstructionCounts]  V0 %.1f instr/call  V1 %.1f instr/call  "
              "%.1fx  second run: %s\n",
              V0_PER_CALL, V1_PER_CALL, V0_PER_CALL / V1_PER_CALL,
              IDENTICAL ? "identical" : "different");

  EXPECT_GT(V0_PER_CALL, MIN_INSTRUCTION_RATIO * V1_PER_CALL)
      << "V0 " << V0_PER_CALL << " instructions per call is not " << MIN_INSTRUCTION_RATIO
      << "x V1's " << V1_PER_CALL << ": the demo has stopped demonstrating";
  EXPECT_EQ(runs[2].total, runs[0].total)
      << "callgrind counted a different total for the same V0 run the second time";
  EXPECT_EQ(runs[5].total, runs[3].total)
      << "callgrind counted a different total for the same V1 run the second time";

  if (!HasFailure()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
  } else {
    std::printf("callgrind output kept in %s\n", DIR.c_str());
  }
}
