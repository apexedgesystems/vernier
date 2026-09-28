/**
 * @file 14_HelgrindProfiler_uTest.cpp
 * @brief The check behind walkthrough 20: helgrind names demo 14's racy
 *        line, and reports nothing for the locked version.
 *
 * Not part of demo 14, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_14_HelgrindProfiler, whose path the build
 * passes in, with the racy source's) under helgrind as a child and reads the
 * report each run left; 14_HelgrindProfiler_Check.hpp holds the runs and the
 * report reading. ctest runs it under the demo and helgrind labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L helgrind
 *   ./build/bin/tests/TestDemoHelgrind      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/14_HelgrindProfiler_Check.hpp"

#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::helgrind_check;
namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The demo binary the checks run, and the racy source; the build passes
/// both paths.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_14_BINARY;
constexpr const char* RACY_SOURCE = VERNIER_DEMO_14_RACY_SOURCE;

/// The demo's case that races, and the one the checks run as the control.
constexpr const char* RACY_CASE = "Helgrind.RacyTotal";
constexpr const char* LOCKED_CASE = "Helgrind.LockedTotal";

/// The racy statement, as the racy source has it: the report must name its
/// line. Found in the source, so an edit that moves it moves the expectation.
constexpr const char* RACY_STATEMENT = "total += joinV1(parts, sep).size();";

/// The function that holds it, and its file's name as a report prints it.
constexpr const char* RACY_FUNCTION = "helgrind_demo::addJoinedLength";
constexpr const char* RACY_FILE = "14_HelgrindProfiler_Racy.cpp";

/// What valgrind exits with when helgrind reported an error: not 1, which is
/// also what a failed test exits with.
constexpr int HELGRIND_ERROR_EXIT = 99;

/// True where StartGate marks its flags for helgrind: valgrind's helgrind.h
/// is available to this build, and NVALGRIND does not compile its requests
/// out. Built without that, contentionRun's gate is reported, and the
/// locked control has nothing to show.
#if defined(__has_include)
#if __has_include(<valgrind/helgrind.h>) && !defined(NVALGRIND)
constexpr bool GATE_MARKED = true;
#else
constexpr bool GATE_MARKED = false;
#endif
#else
constexpr bool GATE_MARKED = false;
#endif

/// The demo binary's canonical path, or an empty string when it is missing.
std::string demoPath() {
  std::error_code ec;
  const fs::path PATH = fs::canonical(DEMO_BINARY, ec);
  return ec ? std::string() : PATH.string();
}

/// A new temporary directory for one check's runs; empty when it cannot be
/// made.
fs::path scratchDir() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo14-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test helgrind reports RacyTotal's race, at the racy line, with no lock
 *       held.
 *
 * Runs the demo binary under helgrind as `bench run --profile helgrind` wraps
 * it, plus valgrind's error list and an exit code for errors. RacyTotal must
 * run to its end and pass (its answer is right: valgrind runs one thread at a
 * time), helgrind must report at least one race, and every race it reports
 * must be a read or a write of the total's size, with no lock held, in the
 * racy function at the racy statement's line, against an earlier access with
 * no lock held at the same line; valgrind must exit with the error status it
 * was given.
 *
 * Skipped in a build with the address or the thread sanitizer (which
 * valgrind cannot check), without valgrind, where valgrind gives up reading
 * the demo binary's debug information before the program runs, and where
 * valgrind cannot read the demo binary's symbols: valgrind says so for that
 * binary, and the race's own frame, in the binary, is unnamed. Only the
 * function and line go unchecked then, and the skip comes last, once
 * everything else has passed. The last two skips quote valgrind's lines.
 */
TEST(Helgrind, FindsTheRace) {
  if constexpr (demo::BUILT_WITH_ASAN_OR_TSAN) {
    GTEST_SKIP() << demo::SANITIZER_UNDER_VALGRIND_REASON;
  }
  if (!vernier::bench::profiler_env::isOnPath("valgrind")) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the racy case under helgrind";
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const std::size_t RACY_LINE = check::lineOf(vg::readText(RACY_SOURCE), RACY_STATEMENT);
  ASSERT_NE(RACY_LINE, 0u) << "the racy source does not hold \"" << RACY_STATEMENT
                           << "\" on exactly one line: " << RACY_SOURCE;
  const std::string RACY_LOCATION = std::string(RACY_FILE) + ":" + std::to_string(RACY_LINE);
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const check::HelgrindRun RUN =
      check::runUnderHelgrind(DEMO, RACY_CASE, {}, DIR, HELGRIND_ERROR_EXIT);
  if (!vg::testsStarted(RUN.output)) {
    const std::string GAVE_UP = vg::debugInfoGiveUp(RUN.log + RUN.output);
    std::error_code ec;
    fs::remove_all(DIR, ec);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind gave up reading the demo binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    FAIL() << "RacyTotal did not start under helgrind: valgrind " << vg::describe(RUN.end)
           << ". The run printed:\n"
           << vg::lastLines(RUN.log + RUN.output, 40);
  }
  ASSERT_TRUE(vg::testPassed(RUN.output, RACY_CASE) && vg::oneTestPassed(RUN.output))
      << "RacyTotal did not run to its end under helgrind (its output is in " << DIR << "):\n"
      << vg::lastLines(RUN.output);

  const vg::ErrorSummary SUMMARY = vg::errorSummary(RUN.log);
  const std::vector<check::RaceReport> RACES = check::raceReports(RUN.log);
  std::printf("[Helgrind.FindsTheRace]  RacyTotal: %ld errors from %ld contexts; %zu race reports "
              "read, looking for %s\n",
              SUMMARY.errors, SUMMARY.contexts, RACES.size(), RACY_LOCATION.c_str());
  ASSERT_FALSE(RACES.empty()) << "helgrind reported no race for RacyTotal: the racy addition has "
                                 "stopped racing (log "
                              << DIR / "helgrind.log" << ")";

  // Where valgrind could not read the demo binary's symbols, its log says so
  // and the race's own frame, in the binary, is unnamed: the function and
  // line are not looked for, and the test skips at its end on valgrind's
  // lines. A frame valgrind named is read as it is.
  const std::string SYMBOLS_UNREADABLE = vg::symbolsUnreadable(RUN.log, DEMO);
  bool namesUnreadable = false;
  for (const check::RaceReport& report : RACES) {
    const std::string SHOWN = report.race.kind + " of size " + std::to_string(report.race.size) +
                              ", " + report.race.frame;
    EXPECT_TRUE(report.race.kind == "read" || report.race.kind == "write") << SHOWN;
    EXPECT_EQ(report.race.size, static_cast<long>(sizeof(std::size_t)))
        << "a race on something other than the total: " << SHOWN;
    EXPECT_EQ(report.race.locksHeld, "none") << SHOWN;
    EXPECT_FALSE(report.conflict.kind.empty()) << "no earlier access is shown for " << SHOWN;
    EXPECT_EQ(report.conflict.locksHeld, "none") << SHOWN;
    if (!SYMBOLS_UNREADABLE.empty() && check::frameUnnamedIn(report.race.frame, DEMO)) {
      namesUnreadable = true;
      continue;
    }
    EXPECT_TRUE(check::frameAt(report.race.frame, RACY_FUNCTION, RACY_LOCATION))
        << "the race is not reported at " << RACY_LOCATION << " in " << RACY_FUNCTION << ": "
        << SHOWN;
    EXPECT_TRUE(check::frameAt(report.conflict.frame, RACY_FUNCTION, RACY_LOCATION))
        << "the earlier access is not at " << RACY_LOCATION << ": " << report.conflict.frame;
  }
  EXPECT_TRUE(vg::exitedWith(RUN.end, HELGRIND_ERROR_EXIT))
      << "valgrind " << vg::describe(RUN.end) << ", not the --error-exitcode "
      << HELGRIND_ERROR_EXIT << " it was given for a run with errors";

  if (HasFailure()) {
    std::printf("helgrind output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
  if (namesUnreadable) {
    GTEST_SKIP() << "helgrind reported the race as expected, but valgrind could not read the demo "
                    "binary's symbols, so its report names no function or line of it and "
                 << RACY_LOCATION << " was not looked for. It printed:\n"
                 << SYMBOLS_UNREADABLE;
  }
}

/**
 * @test helgrind reports nothing for LockedTotal on four threads.
 *
 * Runs the locked case under helgrind as FindsTheRace runs the racy one, on
 * four threads, two calls each, one repeat: it must run to its end, and
 * helgrind must count no error, contentionRun's start gate included; valgrind
 * must exit 0. Skipped where FindsTheRace skips before running anything, on
 * valgrind's give-up lines, and in a build where the gate is not marked for
 * helgrind.
 */
TEST(Helgrind, LockedTotalReportsNothing) {
  if constexpr (demo::BUILT_WITH_ASAN_OR_TSAN) {
    GTEST_SKIP() << demo::SANITIZER_UNDER_VALGRIND_REASON;
  }
  if (!vernier::bench::profiler_env::isOnPath("valgrind")) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the locked case under helgrind";
  }
  if constexpr (!GATE_MARKED) {
    GTEST_SKIP() << "valgrind's helgrind.h is not available to this build, or NVALGRIND is "
                    "defined, so contentionRun's start gate is not marked for helgrind and is "
                    "reported";
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const check::HelgrindRun RUN = check::runUnderHelgrind(
      DEMO, LOCKED_CASE, {"--threads", "4", "--cycles", "2", "--repeats", "1"}, DIR,
      HELGRIND_ERROR_EXIT);
  if (!vg::testsStarted(RUN.output)) {
    const std::string GAVE_UP = vg::debugInfoGiveUp(RUN.log + RUN.output);
    std::error_code ec;
    fs::remove_all(DIR, ec);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind gave up reading the demo binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    FAIL() << "LockedTotal did not start under helgrind: valgrind " << vg::describe(RUN.end)
           << ". The run printed:\n"
           << vg::lastLines(RUN.log + RUN.output, 40);
  }
  const vg::ErrorSummary SUMMARY = vg::errorSummary(RUN.log);
  std::printf("[Helgrind.LockedTotalReportsNothing]  LockedTotal: %ld errors from %ld contexts\n",
              SUMMARY.errors, SUMMARY.contexts);
  EXPECT_TRUE(vg::testPassed(RUN.output, LOCKED_CASE) && vg::oneTestPassed(RUN.output))
      << "LockedTotal did not run to its end under helgrind:\n"
      << vg::lastLines(RUN.output);
  EXPECT_EQ(SUMMARY.errors, 0) << "helgrind reported errors for LockedTotal (log "
                               << DIR / "helgrind.log" << ")";
  EXPECT_EQ(SUMMARY.contexts, 0);
  EXPECT_TRUE(vg::exitedWith(RUN.end, 0))
      << "valgrind " << vg::describe(RUN.end) << " for a run without errors";

  if (HasFailure()) {
    std::printf("helgrind output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/** @test Run without valgrind, RacyTotal reports SKIPPED, says how to run it, and does not run */
TEST(Helgrind, RacyTotalSkipsOutsideValgrind) {
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const vg::ChildExit END =
      vg::runLogged({DEMO, std::string("--gtest_filter=") + RACY_CASE, "--gtest_print_time=0"},
                    DIR / "plain.txt");
  const std::string OUTPUT = vg::readText(DIR / "plain.txt");
  std::error_code ec;
  fs::remove_all(DIR, ec);

  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << OUTPUT;
  EXPECT_NE(OUTPUT.find(std::string("[  SKIPPED ] ") + RACY_CASE), std::string::npos) << OUTPUT;
  EXPECT_FALSE(vg::testPassed(OUTPUT, RACY_CASE)) << OUTPUT;
  EXPECT_NE(OUTPUT.find("valgrind --tool=helgrind"), std::string::npos) << OUTPUT;
  EXPECT_NE(OUTPUT.find("bench run --profile helgrind"), std::string::npos) << OUTPUT;
}

/* ----------------------------- Report Reading Tests ----------------------------- */

// What decides FindsTheRace's pass, its failures and its skip, on report text
// taken from a real run (the laptop's valgrind 3.18.1), the function's
// argument list shortened.

namespace {

/// One race report with its conflicting access, as helgrind prints it, with
/// @p frame as the frame of both accesses and @p locks as the racing access's
/// locks.
std::string raceSnippet(const std::string& frame, const std::string& locks) {
  return "==7== Possible data race during read of size 8 at 0x1FFEFFEE08 by thread #3\n"
         "==7== Locks held: " +
         locks +
         "\n"
         "==7==    " +
         frame +
         "\n"
         "==7==    by 0x4D0AA82: start_thread (pthread_create.c:442)\n"
         "==7== \n"
         "==7== This conflicts with a previous write of size 8 by thread #2\n"
         "==7== Locks held: none\n"
         "==7==    " +
         frame +
         "\n"
         "==7==  Address 0x1ffeffee08 is on thread #1's stack\n"
         "==7== \n"
         "==7== ----------------------------------------------------------------\n";
}

/// The racy function's frame at @p line, as helgrind named it.
std::string racyFrame(int line) {
  return "at 0x1223D0: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, "
         "std::vector<...> const&, char) (14_HelgrindProfiler_Racy.cpp:" +
         std::to_string(line) + ")";
}

} // namespace

/** @test A report is read into the access that raced and the one it conflicts with */
TEST(HelgrindReportTest, RaceReportReadsBothAccesses) {
  const std::vector<check::RaceReport> REPORTS =
      check::raceReports(raceSnippet(racyFrame(21), "none"));

  ASSERT_EQ(REPORTS.size(), 1u);
  EXPECT_EQ(REPORTS[0].race.kind, "read");
  EXPECT_EQ(REPORTS[0].race.size, 8);
  EXPECT_EQ(REPORTS[0].race.locksHeld, "none");
  EXPECT_EQ(REPORTS[0].race.frame, racyFrame(21));
  EXPECT_EQ(REPORTS[0].conflict.kind, "write");
  EXPECT_EQ(REPORTS[0].conflict.size, 8);
  EXPECT_EQ(REPORTS[0].conflict.locksHeld, "none");
  EXPECT_EQ(REPORTS[0].conflict.frame, racyFrame(21));
}

/** @test A held lock is read as helgrind printed it, not as none */
TEST(HelgrindReportTest, LocksHeldAreReadAsPrinted) {
  const std::vector<check::RaceReport> REPORTS =
      check::raceReports(raceSnippet(racyFrame(21), "1, at address 0x1FFEFFEE10"));

  ASSERT_EQ(REPORTS.size(), 1u);
  EXPECT_EQ(REPORTS[0].race.locksHeld, "1, at address 0x1FFEFFEE10");
}

/** @test A frame at another line, or in another function, is not the racy line */
TEST(HelgrindReportTest, FrameAtNeedsTheFunctionAndTheLine) {
  EXPECT_TRUE(check::frameAt(racyFrame(21), RACY_FUNCTION, "14_HelgrindProfiler_Racy.cpp:21"));
  EXPECT_FALSE(check::frameAt(racyFrame(20), RACY_FUNCTION, "14_HelgrindProfiler_Racy.cpp:21"));
  EXPECT_FALSE(check::frameAt(racyFrame(21), RACY_FUNCTION, "14_HelgrindProfiler_Racy.cpp:2"));
  EXPECT_FALSE(
      check::frameAt(racyFrame(21), "helgrind_demo::addTotal", "14_HelgrindProfiler_Racy.cpp:21"));
}

/** @test A directory before the file is the same place; a longer file name is not */
TEST(HelgrindReportTest, FrameAtAcceptsTheFilesDirectory) {
  // As valgrind 3.22 printed it for a clang 21 Debug build
  const std::string WITH_DIRECTORY =
      "at 0x145699: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, "
      "std::vector<...> const&, char) (src/bench/demo/cpu/14_HelgrindProfiler_Racy.cpp:21)";
  const std::string OTHER_FILE =
      "at 0x145699: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, "
      "std::vector<...> const&, char) (src/bench/demo/cpu/x14_HelgrindProfiler_Racy.cpp:21)";

  EXPECT_TRUE(check::frameAt(WITH_DIRECTORY, RACY_FUNCTION, "14_HelgrindProfiler_Racy.cpp:21"));
  EXPECT_FALSE(check::frameAt(WITH_DIRECTORY, RACY_FUNCTION, "14_HelgrindProfiler_Racy.cpp:1"));
  EXPECT_FALSE(check::frameAt(OTHER_FILE, RACY_FUNCTION, "14_HelgrindProfiler_Racy.cpp:21"));
}

/** @test An unnamed frame in the binary is unnamed there, not in a library */
TEST(HelgrindReportTest, FrameUnnamedOnlyInTheBinary) {
  const std::string BINARY = "/b/bin/ptests/BenchDemo_14_HelgrindProfiler";

  EXPECT_TRUE(check::frameUnnamedIn("at 0x1223D0: ??? (in " + BINARY + ")", BINARY));
  EXPECT_FALSE(check::frameUnnamedIn("at 0x4A1D252: ??? (in /usr/lib/libstdc++.so.6)", BINARY));
  EXPECT_FALSE(check::frameUnnamedIn(racyFrame(21), BINARY));
}

/** @test A log without a race report reads as none */
TEST(HelgrindReportTest, NoRaceReadsAsNone) {
  EXPECT_TRUE(check::raceReports("==7== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 30 "
                                 "from 7)\n")
                  .empty());
}

/** @test The racy statement's line is found once, or not at all */
TEST(HelgrindReportTest, LineOfFindsTheStatementOnce) {
  EXPECT_EQ(check::lineOf("a\n  total += x;\nb\n", "total += x;"), 2u);
  EXPECT_EQ(check::lineOf("total += x;\ntotal += x;\n", "total += x;"), 0u);
  EXPECT_EQ(check::lineOf("a\nb\n", "total += x;"), 0u);
}
