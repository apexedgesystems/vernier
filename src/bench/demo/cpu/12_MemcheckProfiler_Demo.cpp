/**
 * @file 12_MemcheckProfiler_Demo.cpp
 * @brief Demo 12: memcheck -- a memory error a timer cannot see
 *
 * Measures the two versions of the shared join example and carries a third,
 * deliberately wrong join for memcheck to find:
 *  1. JoinV0 and JoinV1 each measure one version, one CSV row each
 *  2. JoinOffByOne calls the wrong join, which returns the right string and
 *     writes one byte past its buffer; it runs only under valgrind and skips
 *     itself anywhere else
 *  3. FindsTheOffByOne runs JoinOffByOne under memcheck as a child and fails
 *     unless memcheck reports the write, then runs JoinV1 the same way and
 *     fails unless memcheck reports nothing; the process plumbing and the log
 *     reading it uses are in 12_MemcheckProfiler_Check.hpp
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_12_MemcheckProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Find the bug: memcheck's log lands in
 *   # bench-out/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log
 *   bench run ./BenchDemo_12_MemcheckProfiler --profile memcheck -- \
 *     --gtest_filter=Memcheck.JoinOffByOne
 *
 *   # Read it
 *   cat bench-out/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log
 *   @endcode
 *
 * @see docs/15_MEMCHECK_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/cpu/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/cpu/12_MemcheckProfiler_OffByOne.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;
namespace wrong = vernier::bench::demo::memcheck_demo;
namespace check = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, as in demo 01, so the walkthroughs measure the same call.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// Calls JoinOffByOne makes. Each writes one byte past its buffer, so memcheck
/// counts this many of the same error and reports the context once.
static constexpr long OFF_BY_ONE_CALLS = 3;

/// The exit status FindsTheOffByOne asks valgrind for when memcheck reported
/// an error, so a run that found something is told apart from one that did
/// not by its status alone. Not 1, which is also what a failed test exits with.
static constexpr int MEMCHECK_ERROR_EXIT = 99;

/// The name memcheck's report must carry in its stacks.
static constexpr const char* OFF_BY_ONE_FUNCTION = "joinOffByOne";

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: two temporaries and a full copy per part. */
PERF_THROUGHPUT(Memcheck, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: one allocation per call. */
PERF_THROUGHPUT(Memcheck, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/**
 * @test The wrong join returns the right string OFF_BY_ONE_CALLS times, and
 *       writes past its buffer every time. Runs only under valgrind.
 *
 * Measures nothing: under valgrind a timing means nothing, and a fixed number
 * of calls is what memcheck's error count divides by. Anywhere else the case
 * skips itself and says how to run it.
 */
PERF_TEST(Memcheck, JoinOffByOne) {
  DEMO_SKIP_UNLESS_UNDER_VALGRIND();

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  const std::string EXPECTED = demo::joinV1(PARTS, SEPARATOR);
  for (long call = 0; call < OFF_BY_ONE_CALLS; ++call) {
    EXPECT_EQ(wrong::joinOffByOne(PARTS, SEPARATOR), EXPECTED) << "call " << call;
  }
}

/**
 * @test memcheck reports JoinOffByOne's write, and nothing for JoinV1.
 *
 * Runs this binary under memcheck twice, as `bench run --profile memcheck`
 * wraps it plus valgrind's error list: on JoinOffByOne, whose report must
 * count one invalid write of one byte per call (in one context, or in one per
 * call where the compiler unrolled the loop), place every one right after a
 * block the size of the joined string and name joinOffByOne in both of its
 * stacks, with valgrind exiting with the error status it was given; then on
 * JoinV1 with one call per phase, whose summary must count no error and whose
 * valgrind must exit 0. One report is suppressed in both runs, a start-up
 * memory probe of gperftools' profiler library that is not the program's (the
 * check header says which). Writes no CSV row.
 *
 * Skipped in a build with a sanitizer (which valgrind does not run as an
 * ordinary binary), without valgrind, under --profile (it runs memcheck
 * itself), where valgrind gives up reading this binary's debug information
 * before the program runs, as a valgrind older than the compiler does, and
 * where valgrind runs the binary but cannot read its symbols, so that its
 * report names no function of it; that skip comes last, once everything but
 * the names has passed. The last two quote valgrind's own lines. A run that
 * does not reach its test for any other reason fails, with what the run
 * printed, and so does a report that does not name joinOffByOne from a binary
 * whose symbols valgrind read.
 */
PERF_TEST(Memcheck, FindsTheOffByOne) {
  if constexpr (demo::BUILT_WITH_A_SANITIZER) {
    GTEST_SKIP() << demo::SANITIZER_UNDER_VALGRIND_REASON;
  }
  if (!ub::detail::getPerfConfig().profileTool.empty()) {
    GTEST_SKIP() << "runs memcheck itself; run it without --profile";
  }
  if (!ub::profiler_env::isOnPath("valgrind")) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the wrong join under memcheck";
  }

  std::array<char, 4096> self{};
  const ssize_t SELF_LENGTH = ::readlink("/proc/self/exe", self.data(), self.size() - 1);
  ASSERT_GT(SELF_LENGTH, 0) << "cannot find this binary's path";
  const std::string SELF(self.data(), static_cast<std::size_t>(SELF_LENGTH));

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo12-XXXXXX").string();
  ASSERT_NE(::mkdtemp(dirTemplate.data()), nullptr) << "cannot create a temporary directory";
  const fs::path DIR = dirTemplate;

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  const long JOINED = static_cast<long>(demo::joinedSize(PARTS));

  // The wrong join under memcheck.
  const check::MemcheckRun WRONG = check::runUnderMemcheck(SELF, "Memcheck.JoinOffByOne", {},
                                                           DIR / "off_by_one", MEMCHECK_ERROR_EXIT);
  if (!check::testsStarted(WRONG.output)) {
    const std::string GAVE_UP = check::debugInfoGiveUp(WRONG.log + WRONG.output);
    std::error_code ec;
    fs::remove_all(DIR, ec);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind gave up reading this binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    FAIL() << "JoinOffByOne did not start under memcheck: valgrind " << check::describe(WRONG.end)
           << ". The run printed:\n"
           << check::lastLines(WRONG.log + WRONG.output, 40);
  }
  ASSERT_TRUE(check::oneTestPassed(WRONG.output))
      << "JoinOffByOne did not run to its end under memcheck (its output is in " << DIR << "):\n"
      << check::lastLines(WRONG.output);

  const check::ErrorSummary WRONG_SUMMARY = check::errorSummary(WRONG.log);
  const std::vector<check::ReportedError> ERRORS = check::errorList(WRONG.log);
  const std::vector<const check::ReportedError*> WRITES =
      check::errorsOfKind(ERRORS, "Invalid write of size 1");
  const long WRITE_COUNT = check::occurrences(WRITES);
  std::printf("[Memcheck.FindsTheOffByOne]  JoinOffByOne: %ld errors from %ld contexts; invalid "
              "write of size 1: %ld times in %zu contexts, %ld bytes after a block of size %ld\n",
              WRONG_SUMMARY.errors, WRONG_SUMMARY.contexts, WRITE_COUNT, WRITES.size(),
              WRITES.empty() ? -1L : check::bytesAfterBlock(*WRITES.front()),
              WRITES.empty() ? -1L : check::blockSize(*WRITES.front()));

  ASSERT_FALSE(WRITES.empty()) << "memcheck reported no invalid write of size 1 for JoinOffByOne ("
                               << WRONG_SUMMARY.errors << " errors from " << WRONG_SUMMARY.contexts
                               << " contexts): the wrong join has stopped being wrong (log "
                               << DIR / "off_by_one" / "memcheck.log" << ")";
  EXPECT_EQ(WRITE_COUNT, OFF_BY_ONE_CALLS) << "one invalid write per call was expected";
  // Where valgrind could not read this binary's symbols, its log says so and
  // its report names no function of the binary: the names are not looked
  // for, and the test skips at its end on valgrind's lines.
  const std::string SYMBOLS_UNREADABLE = check::symbolsUnreadable(WRONG.log, SELF);
  for (const check::ReportedError* write : WRITES) {
    EXPECT_EQ(check::bytesAfterBlock(*write), 0)
        << "the write is not right after the end of a heap block:\n"
        << write->text;
    EXPECT_EQ(check::blockSize(*write), JOINED)
        << "the block memcheck describes is not the joined string's buffer:\n"
        << write->text;
    // The write's own stack and the block's allocation stack.
    if (SYMBOLS_UNREADABLE.empty()) {
      EXPECT_GE(check::framesNaming(*write, OFF_BY_ONE_FUNCTION), 2)
          << "the report does not name " << OFF_BY_ONE_FUNCTION << " in both stacks:\n"
          << write->text;
    }
  }
  EXPECT_GE(WRONG_SUMMARY.errors, OFF_BY_ONE_CALLS);
  EXPECT_TRUE(check::exitedWith(WRONG.end, MEMCHECK_ERROR_EXIT))
      << "valgrind " << check::describe(WRONG.end) << ", not the --error-exitcode "
      << MEMCHECK_ERROR_EXIT << " it was given for a run with errors";

  // The correct join under memcheck: nothing to report.
  const check::MemcheckRun CLEAN =
      check::runUnderMemcheck(SELF, "Memcheck.JoinV1", {"--cycles", "1", "--repeats", "1"},
                              DIR / "clean", MEMCHECK_ERROR_EXIT);
  ASSERT_TRUE(check::testsStarted(CLEAN.output) && check::oneTestPassed(CLEAN.output))
      << "JoinV1 did not run to its end under memcheck: valgrind " << check::describe(CLEAN.end)
      << " (its output is in " << DIR << "):\n"
      << check::lastLines(CLEAN.log + CLEAN.output, 40);
  const check::ErrorSummary CLEAN_SUMMARY = check::errorSummary(CLEAN.log);
  std::printf("[Memcheck.FindsTheOffByOne]  JoinV1: %ld errors from %ld contexts\n",
              CLEAN_SUMMARY.errors, CLEAN_SUMMARY.contexts);
  EXPECT_EQ(CLEAN_SUMMARY.errors, 0)
      << "memcheck reported errors for JoinV1 (log " << DIR / "clean" / "memcheck.log" << ")";
  EXPECT_EQ(CLEAN_SUMMARY.contexts, 0);
  EXPECT_TRUE(check::exitedWith(CLEAN.end, 0))
      << "valgrind " << check::describe(CLEAN.end) << " for a run without errors";

  if (HasFailure()) {
    std::printf("memcheck output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
  if (!SYMBOLS_UNREADABLE.empty()) {
    GTEST_SKIP() << "memcheck reported the write as expected, and nothing for JoinV1, but "
                    "valgrind could not read this binary's symbols, so its report names no "
                    "function of it and "
                 << OFF_BY_ONE_FUNCTION << " was not looked for. It printed:\n"
                 << SYMBOLS_UNREADABLE;
  }
}

PERF_MAIN()
