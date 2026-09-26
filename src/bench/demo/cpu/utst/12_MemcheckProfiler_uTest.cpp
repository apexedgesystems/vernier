/**
 * @file 12_MemcheckProfiler_uTest.cpp
 * @brief The check behind walkthrough 15: memcheck reports demo 12's wrong
 *        join, and nothing for the correct one.
 *
 * Not part of demo 12, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_12_MemcheckProfiler, whose path the build
 * passes in) under memcheck as a child and reads what each run left;
 * 12_MemcheckProfiler_Check.hpp holds the process plumbing and the log
 * reading. ctest runs it under the demo and memcheck labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L memcheck
 *   ./build/bin/tests/TestDemoMemcheck      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"

#include "src/bench/demo/cpu/12_MemcheckProfiler_Workload.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using vernier::bench::demo::memcheck_demo::OFF_BY_ONE_CALLS;
using vernier::bench::demo::memcheck_demo::PART_COUNT;
using vernier::bench::demo::memcheck_demo::PART_SEED;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The demo binary the check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_12_BINARY;

/// The exit status FindsTheOffByOne asks valgrind for when memcheck reported
/// an error, so a run that found something is told apart from one that did
/// not by its status alone. Not 1, which is also what a failed test exits with.
constexpr int MEMCHECK_ERROR_EXIT = 99;

/// The name memcheck's report must carry in its stacks.
constexpr const char* OFF_BY_ONE_FUNCTION = "joinOffByOne";

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test memcheck reports JoinOffByOne's write, and nothing for JoinV1.
 *
 * Runs the demo binary under memcheck twice, as `bench run --profile memcheck`
 * wraps it plus valgrind's error list: on JoinOffByOne, whose report must
 * count one invalid write of one byte per call (in one context, or in one per
 * call where the compiler unrolled the loop), place every one right after a
 * block the size of the joined string and name joinOffByOne in both of its
 * stacks, with valgrind exiting with the error status it was given; then on
 * JoinV1 with one call per phase, whose summary must count no error and whose
 * valgrind must exit 0. One report is suppressed in both runs, a start-up
 * memory probe of gperftools' profiler library that is not the program's (the
 * check header says which).
 *
 * Skipped in a build with a sanitizer (which valgrind does not run as an
 * ordinary binary), without valgrind, where valgrind gives up reading the demo
 * binary's debug information before the program runs, as a valgrind older than
 * the compiler does, and where valgrind runs the demo binary but cannot read
 * its symbols, so that its report names no function of it; that skip comes
 * last, once everything but the names has passed. The last two quote
 * valgrind's own lines. A run that does not reach its test for any other
 * reason fails, with what the run printed, and so does a report that does not
 * name joinOffByOne from a binary whose symbols valgrind read.
 */
TEST(Memcheck, FindsTheOffByOne) {
  if constexpr (demo::BUILT_WITH_A_SANITIZER) {
    GTEST_SKIP() << demo::SANITIZER_UNDER_VALGRIND_REASON;
  }
  if (!vernier::bench::profiler_env::isOnPath("valgrind")) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the wrong join under memcheck";
  }

  std::error_code found;
  const std::string DEMO = fs::canonical(DEMO_BINARY, found).string();
  ASSERT_FALSE(found) << "the demo binary is missing: " << DEMO_BINARY;

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo12-XXXXXX").string();
  ASSERT_NE(::mkdtemp(dirTemplate.data()), nullptr) << "cannot create a temporary directory";
  const fs::path DIR = dirTemplate;

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  const long JOINED = static_cast<long>(demo::joinedSize(PARTS));

  // The wrong join under memcheck.
  const check::MemcheckRun WRONG = check::runUnderMemcheck(DEMO, "Memcheck.JoinOffByOne", {},
                                                           DIR / "off_by_one", MEMCHECK_ERROR_EXIT);
  if (!check::testsStarted(WRONG.output)) {
    const std::string GAVE_UP = check::debugInfoGiveUp(WRONG.log + WRONG.output);
    std::error_code ec;
    fs::remove_all(DIR, ec);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind gave up reading the demo binary's debug information before the "
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
  // Where valgrind could not read the demo binary's symbols, its log says so
  // and its report names no function of the binary: the names are not looked
  // for, and the test skips at its end on valgrind's lines.
  const std::string SYMBOLS_UNREADABLE = check::symbolsUnreadable(WRONG.log, DEMO);
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
      check::runUnderMemcheck(DEMO, "Memcheck.JoinV1", {"--cycles", "1", "--repeats", "1"},
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
                    "valgrind could not read the demo binary's symbols, so its report names no "
                    "function of it and "
                 << OFF_BY_ONE_FUNCTION << " was not looked for. It printed:\n"
                 << SYMBOLS_UNREADABLE;
  }
}
