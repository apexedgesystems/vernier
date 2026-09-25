/**
 * @file 07_CallgrindProfiler_Demo.cpp
 * @brief Demo 07: exact instruction counts with callgrind
 *
 * Measures the two versions of the shared join example and checks what
 * callgrind counts for them:
 *  1. JoinV0 and JoinV1 each measure one version, so each writes its own row
 *  2. Profiled under `bench run --profile callgrind`, the same two tests show
 *     which functions and source lines execute the instructions
 *  3. InstructionCounts runs CountCalls under callgrind and fails when V0
 *     stops executing several times V1's instructions per call, or when a
 *     second run counts a different number
 *
 * Usage:
 *   @code{.sh}
 *   # Time both versions
 *   ./BenchDemo_07_CallgrindProfiler --gtest_filter='CallgrindProfiler.JoinV*' \
 *     --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Count their instructions under callgrind, then read the counts
 *   bench run ./BenchDemo_07_CallgrindProfiler --profile callgrind --cycles 10 \
 *     --repeats 1 -- --gtest_filter='CallgrindProfiler.JoinV*'
 *   callgrind_annotate bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out
 *
 *   # Instructions per call of each version
 *   ./BenchDemo_07_CallgrindProfiler --gtest_filter=CallgrindProfiler.InstructionCounts
 *   @endcode
 *
 * @see docs/07_CALLGRIND_PROFILER.md for the step-by-step walkthrough
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
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/cpu/07_CallgrindProfiler_Check.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::callgrind_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, the size demo 01 measures. V0's cost grows with its square.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// Turns CountCalls into a counting run: "<V0|V1> <calls>", for example "V0 10".
static constexpr const char* COUNT_CALLS_VARIABLE = "VERNIER_DEMO_COUNT_CALLS";

/// Calls in InstructionCounts' two counting runs of each version. Everything
/// else a counting run does is the same in both, so the difference between the
/// two whole-program counts is exactly the extra calls. Both numbers have two
/// digits, so both runs see an environment of the same size.
static constexpr int COUNT_CALLS_LOW = 10;
static constexpr int COUNT_CALLS_HIGH = 20;

/// V0 must execute more than this many times V1's instructions per call. The
/// smallest ratio measured while choosing it came from an unoptimized x86 build
/// (6.9x); optimized builds measure several times more, on x86 and on the Arm
/// reference rig. Instruction counts carry no noise, so the margin only has to
/// cover build and platform differences.
static constexpr double MIN_INSTRUCTION_RATIO = 5.0;

/* ----------------------------- File Helpers ----------------------------- */

namespace {

/// Bytes a joined string holds: every part plus one separator each.
std::size_t joinedSize(const std::vector<std::string>& parts) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }
  return total;
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(CallgrindProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(CallgrindProfiler, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/**
 * @test Calls one version a given number of times and measures nothing, so a
 * callgrind count of the run holds nothing that depends on time.
 *
 * InstructionCounts runs it under callgrind with VERNIER_DEMO_COUNT_CALLS set
 * ("V0 10"); without the variable it skips.
 */
PERF_TEST(CallgrindProfiler, CountCalls) {
  const char* job = std::getenv(COUNT_CALLS_VARIABLE);
  if (job == nullptr) {
    GTEST_SKIP() << "InstructionCounts runs this test under callgrind";
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
 * Runs CountCalls under callgrind six times: each version with 10 and with 20
 * calls, and with 10 calls again. A version's instructions per call are the
 * difference between its two program totals divided by the difference in
 * calls, so nothing depends on callgrind's call graph, which on Arm can credit
 * a function with calls it never received. Writes no CSV row.
 *
 * Skipped where valgrind is not installed, or where it gives up reading this
 * binary's debug information before the program runs, as a valgrind older
 * than the compiler does, and says so. A counting run that does not reach its
 * test for any other reason fails, with what the run printed.
 */
PERF_TEST(CallgrindProfiler, InstructionCounts) {
  if (std::system("command -v valgrind >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "valgrind is not installed; this test counts instructions under callgrind";
  }

  std::array<char, 4096> self{};
  const ssize_t SELF_LENGTH = ::readlink("/proc/self/exe", self.data(), self.size() - 1);
  ASSERT_GT(SELF_LENGTH, 0) << "cannot find this binary's path";
  const std::string SELF(self.data(), static_cast<std::size_t>(SELF_LENGTH));

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo07-XXXXXX").string();
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
                          "--gtest_filter=CallgrindProfiler.CountCalls", "--gtest_print_time=0"},
                         std::string(COUNT_CALLS_VARIABLE) + "=" + runs[i].version + " " +
                             std::to_string(runs[i].calls),
                         LOG);
    const std::string LOG_TEXT = check::readText(LOG);

    // Only valgrind's own reason skips: its debug-information reader gave up
    // before the program ran. A crash, a signal, an early exit, a failed launch
    // or no output at all is a failure of the run, reported with what the run
    // printed (the log is short when the tests never started).
    if (!check::testsStarted(LOG_TEXT)) {
      const std::string GAVE_UP = check::debugInfoGiveUp(LOG_TEXT);
      std::error_code ec;
      fs::remove_all(DIR, ec);
      if (!GAVE_UP.empty()) {
        GTEST_SKIP() << "valgrind could not read this binary's debug information: " << GAVE_UP;
      }
      FAIL() << "CountCalls did not start under callgrind: valgrind " << check::describe(END)
             << ". The run printed:\n"
             << (LOG_TEXT.empty() ? std::string("(nothing)\n") : check::lastLines(LOG_TEXT, 40));
    }
    ASSERT_TRUE(check::exitedCleanly(END))
        << "the counting run under callgrind " << check::describe(END) << " (log " << LOG << "):\n"
        << check::lastLines(LOG_TEXT);
    ASSERT_TRUE(check::oneTestPassed(LOG_TEXT))
        << "CountCalls did not run to its end under callgrind (log " << LOG << "):\n"
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
  std::printf("[CallgrindProfiler.InstructionCounts]  V0 %.1f instr/call  V1 %.1f instr/call  "
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

PERF_MAIN()
