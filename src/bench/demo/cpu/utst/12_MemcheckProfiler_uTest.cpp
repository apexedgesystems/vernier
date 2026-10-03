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

#include <csignal>
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

/// The demo's case that runs the wrong join, and the one the check runs as its
/// control.
constexpr const char* WRONG_CASE = "Memcheck.JoinOffByOne";
constexpr const char* CLEAN_CASE = "Memcheck.JoinV1";

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
 * wraps it plus valgrind's error list: on JoinOffByOne, which must run to its
 * end and whose report must count one invalid write of one byte per call (in
 * one context, or in one per call where the compiler unrolled the loop), place
 * every one right after a block the size of the joined string and name
 * joinOffByOne in both of its stacks, with valgrind exiting with the error
 * status it was given; then on JoinV1 with one call per phase, which must run
 * to its end, whose summary must count no error and whose valgrind must exit
 * 0. One report is suppressed in both runs, a start-up memory probe of
 * gperftools' profiler library that is not the program's (the check header
 * says which).
 *
 * Skipped in a build with the address or the thread sanitizer (which valgrind
 * cannot check), without valgrind, where valgrind gives up reading the demo
 * binary's debug information before the program runs, as a valgrind older than
 * the compiler does, where an assertion in valgrind's ELF debug-information
 * reader stops it before the program starts (valgrind 3.18.1 on a GCC 11.4
 * Debug build that mold linked), and where valgrind cannot read the demo
 * binary's symbols: valgrind says so for that binary, and the write's own
 * frame, in the binary, is unnamed. Only the names go unchecked then, and the
 * skip comes last, once everything else has passed; a failure wins over it.
 * The last three skips quote valgrind's own lines. A run that does not reach
 * its test for any other reason fails, with what the run printed, and so does
 * a report whose frames valgrind named but not as joinOffByOne.
 */
TEST(Memcheck, FindsTheOffByOne) {
  if constexpr (demo::BUILT_WITH_ASAN_OR_TSAN) {
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
  const check::MemcheckRun WRONG =
      check::runUnderMemcheck(DEMO, WRONG_CASE, {}, DIR / "off_by_one", MEMCHECK_ERROR_EXIT);
  if (!check::testsStarted(WRONG.output)) {
    const std::string GAVE_UP = check::debugInfoGiveUp(WRONG.log + WRONG.output);
    const std::string ASSERTED =
        check::readerAssertionBeforeStart(WRONG.end, WRONG.output, WRONG.log);
    std::error_code ec;
    fs::remove_all(DIR, ec);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind gave up reading the demo binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    if (!ASSERTED.empty()) {
      GTEST_SKIP() << "valgrind stopped before the demo binary started: an assertion failed in "
                      "its debug-information reader and valgrind "
                   << check::describe(WRONG.end) << ", so memcheck checked nothing. It printed:\n"
                   << ASSERTED;
    }
    FAIL() << "JoinOffByOne did not start under memcheck: valgrind " << check::describe(WRONG.end)
           << ". The run printed:\n"
           << check::lastLines(WRONG.log + WRONG.output, 40);
  }
  ASSERT_TRUE(check::testPassed(WRONG.output, WRONG_CASE) && check::oneTestPassed(WRONG.output))
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
  // and the write's own frame, in the binary, is unnamed: the names are not
  // looked for, and the test skips at its end on valgrind's lines. A frame
  // valgrind named is read as it is.
  const std::string SYMBOLS_UNREADABLE = check::symbolsUnreadable(WRONG.log, DEMO);
  bool namesUnreadable = false;
  for (const check::ReportedError* write : WRITES) {
    EXPECT_EQ(check::bytesAfterBlock(*write), 0)
        << "the write is not right after the end of a heap block:\n"
        << write->text;
    EXPECT_EQ(check::blockSize(*write), JOINED)
        << "the block memcheck describes is not the joined string's buffer:\n"
        << write->text;
    if (!SYMBOLS_UNREADABLE.empty() && check::ownFrameUnnamedIn(*write, DEMO)) {
      namesUnreadable = true;
      continue;
    }
    // The write's own stack and the block's allocation stack.
    EXPECT_GE(check::framesNaming(*write, OFF_BY_ONE_FUNCTION), 2)
        << "the report does not name " << OFF_BY_ONE_FUNCTION << " in both stacks:\n"
        << write->text;
  }
  EXPECT_GE(WRONG_SUMMARY.errors, OFF_BY_ONE_CALLS);
  EXPECT_TRUE(check::exitedWith(WRONG.end, MEMCHECK_ERROR_EXIT))
      << "valgrind " << check::describe(WRONG.end) << ", not the --error-exitcode "
      << MEMCHECK_ERROR_EXIT << " it was given for a run with errors";

  // The correct join under memcheck: nothing to report.
  const check::MemcheckRun CLEAN = check::runUnderMemcheck(
      DEMO, CLEAN_CASE, {"--cycles", "1", "--repeats", "1"}, DIR / "clean", MEMCHECK_ERROR_EXIT);
  ASSERT_TRUE(check::testPassed(CLEAN.output, CLEAN_CASE) && check::oneTestPassed(CLEAN.output))
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
  if (namesUnreadable) {
    GTEST_SKIP() << "memcheck reported the write as expected, and nothing for JoinV1, but "
                    "valgrind could not read the demo binary's symbols, so its report names no "
                    "function of it and "
                 << OFF_BY_ONE_FUNCTION << " was not looked for. It printed:\n"
                 << SYMBOLS_UNREADABLE;
  }
}

/* ----------------------------- Log Reading Tests ----------------------------- */

// What decides FindsTheOffByOne's skip and its failures, on log text taken
// from real runs (the laptop's valgrind 3.18.1: its mold build's warning, the
// give-up of its clang 21 Debug build, the reader's assertion on a GCC 11.4
// Debug build that mold linked, before a program started and after one had,
// and a program's own failed assertion), with a short binary path and process
// id. The other assertion lines are shaped as valgrind prints its assertions;
// no run printed them.

namespace {

/// The binary the snippets below are about.
constexpr const char* SNIPPET_BINARY = "/b/bin/ptests/BenchDemo_12_MemcheckProfiler";

/// valgrind's opening lines for the check's run of the wrong case.
constexpr const char* OPENING_LINES =
    "==7== Memcheck, a memory error detector\n"
    "==7== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.\n"
    "==7== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info\n"
    "==7== Command: /b/bin/ptests/BenchDemo_12_MemcheckProfiler "
    "--gtest_filter=Memcheck.JoinOffByOne --gtest_print_time=0\n"
    "==7== Parent PID: 6\n"
    "==7== \n";

/// The assertion valgrind 3.18.1 failed on a GCC 11.4 Debug build of the demo
/// that mold linked, before the program started.
constexpr const char* READER_ASSERTION =
    "valgrind: m_debuginfo/readelf.c:2478 (vgModuleLocal_read_elf_debug_info): Assertion "
    "'di->bss_svma + di->bss_size == svma' failed.";

/// valgrind's log as that run left it, with @p line in its assertion's place:
/// the opening lines, an empty line and @p line, nothing after it.
std::string logEndingWith(const std::string& line) {
  return std::string(OPENING_LINES) + "\n" + line + "\n";
}

/// How that run ended: valgrind killed by SIGSEGV. The program wrote nothing.
constexpr check::ChildExit KILLED_BY_SIGSEGV{check::ChildExit::How::Signaled, SIGSEGV};

/// What valgrind printed after the same assertion once a program had started
/// (it had printed GoogleTest's first line, then loaded the demo's Debug
/// binary), most frames cut; it then exited with status 1.
constexpr const char* CRASH_REPORT =
    "\n"
    "host stacktrace:\n"
    "==7==    at 0x5804284A: ??? (in /usr/libexec/valgrind/memcheck-amd64-linux)\n"
    "\n"
    "sched status:\n"
    "  running_tid=1\n"
    "\n"
    "Thread 1: status = VgTs_Runnable syscall 9 (lwpid 7)\n"
    "==7==    at 0x4026CB7: __mmap64 (mmap64.c:58)\n"
    "==7==    by 0x4902687: dlopen@@GLIBC_2.34 (dlopen.c:81)\n"
    "\n"
    "\n"
    "Note: see also the FAQ in the source distribution.\n"
    "It contains workarounds to several common problems.\n"
    "In particular, if Valgrind aborted or crashed after\n"
    "identifying problems in your program, there's a good chance\n"
    "that fixing those problems will prevent Valgrind aborting or\n"
    "crashing, especially if it happened in m_mallocfree.c.\n"
    "\n"
    "If that doesn't help, please report this bug to: www.valgrind.org\n"
    "\n"
    "In the bug report, send all the above text, the valgrind\n"
    "version, and what OS and version you are using.  Thanks.\n"
    "\n";

/// valgrind's lines when it could not read @p file's symbols.
std::string mappingWarning(const std::string& file) {
  return "--7-- WARNING: Serious error when reading debug info\n"
         "--7-- When reading debug info from " +
         file +
         ":\n"
         "--7-- Can't make sense of .rodata section mapping\n";
}

/// The error list's entry for the write, with @p frame as its own frame.
check::ReportedError writeWithOwnFrame(const std::string& frame) {
  const std::vector<check::ReportedError> ERRORS =
      check::errorList("==7== 3 errors in context 1 of 2:\n"
                       "==7== Invalid write of size 1\n"
                       "==7==    " +
                       frame +
                       "\n"
                       "==7==  Address 0x4ec48d2 is 0 bytes after a block of size 7,490 alloc'd\n"
                       "==7==    at 0x484A2F3: operator new[](unsigned long) (in "
                       "/usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)\n"
                       "==7== \n");
  return ERRORS.empty() ? check::ReportedError{} : ERRORS.front();
}

} // namespace

/** @test The skip quotes valgrind's own lines about the binary, as they are */
TEST(MemcheckLogTest, SymbolsUnreadableQuotesValgrindAboutTheBinary) {
  const std::string LINES = mappingWarning(SNIPPET_BINARY);
  const std::string LOG =
      "==7== Memcheck, a memory error detector\n" + LINES + "==7== Invalid write of size 1\n";

  EXPECT_EQ(check::symbolsUnreadable(LOG, SNIPPET_BINARY), LINES.substr(0, LINES.size() - 1));
}

/** @test The same warning about a library is not the binary's */
TEST(MemcheckLogTest, SymbolsUnreadableIgnoresALibrary) {
  const std::string LOG =
      mappingWarning("/b/lib/libbench.so.2.0.3") + mappingWarning("/b/lib/libgtest.so.1.16.0");

  EXPECT_TRUE(check::symbolsUnreadable(LOG, SNIPPET_BINARY).empty());
}

/** @test Another of valgrind's reasons about the binary is not this one */
TEST(MemcheckLogTest, SymbolsUnreadableIgnoresAnotherReason) {
  const std::string LOG = "--7-- WARNING: Serious error when reading debug info\n"
                          "--7-- When reading debug info from " +
                          std::string(SNIPPET_BINARY) +
                          ":\n"
                          "--7-- get_Form_contents: unhandled DW_FORM\n";

  EXPECT_TRUE(check::symbolsUnreadable(LOG, SNIPPET_BINARY).empty());
}

/** @test A missing log says nothing */
TEST(MemcheckLogTest, SymbolsUnreadableIsEmptyWithoutALog) {
  EXPECT_TRUE(check::symbolsUnreadable("", SNIPPET_BINARY).empty());
}

/** @test Only an unnamed frame in the binary is unnamed there */
TEST(MemcheckLogTest, OwnFrameUnnamedOnlyInTheBinary) {
  const std::string BINARY = SNIPPET_BINARY;

  EXPECT_TRUE(
      check::ownFrameUnnamedIn(writeWithOwnFrame("at 0x133EB9: ??? (in " + BINARY + ")"), BINARY));
  EXPECT_FALSE(check::ownFrameUnnamedIn(
      writeWithOwnFrame("at 0x133EB9: ??? (in /b/lib/libbench.so.2.0.3)"), BINARY));
}

/** @test A frame valgrind named is read as it is, warning or not: a wrong name fails */
TEST(MemcheckLogTest, AReadableWrongNameIsNotExcused) {
  const std::string BINARY = SNIPPET_BINARY;
  const check::ReportedError WRITE =
      writeWithOwnFrame("at 0x12ED79: vernier::bench::demo::memcheck_demo::joinOther("
                        "std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:32)");

  EXPECT_FALSE(check::symbolsUnreadable(mappingWarning(BINARY), BINARY).empty());
  EXPECT_FALSE(check::ownFrameUnnamedIn(WRITE, BINARY));
  EXPECT_EQ(check::framesNaming(WRITE, OFF_BY_ONE_FUNCTION), 0);
}

/** @test The named test's own OK line, not a longer name's, a skip or another test's */
TEST(MemcheckLogTest, TestPassedNamesTheTest) {
  EXPECT_TRUE(check::testPassed("[ RUN      ] Memcheck.JoinV1\n[       OK ] Memcheck.JoinV1\n",
                                CLEAN_CASE));
  EXPECT_TRUE(check::testPassed("[       OK ] Memcheck.JoinV1 (260 ms)\n", CLEAN_CASE));
  EXPECT_FALSE(check::testPassed("[       OK ] Memcheck.JoinV10\n", CLEAN_CASE));
  EXPECT_FALSE(check::testPassed("[  SKIPPED ] Memcheck.JoinOffByOne\n", WRONG_CASE));
  EXPECT_FALSE(check::testPassed("[       OK ] Memcheck.JoinV1\n", WRONG_CASE));
}

/** @test A run that died before its tests is not valgrind giving up */
TEST(MemcheckLogTest, DebugInfoGiveUpOnlyOnValgrindsLines) {
  EXPECT_TRUE(check::debugInfoGiveUp(
                  "==7== Process terminating with default action of signal 11 (SIGSEGV)\n")
                  .empty());
  EXPECT_FALSE(check::debugInfoGiveUp(
                   "==7== Valgrind: debuginfo reader: Possibly corrupted debuginfo file.\n"
                   "==7== Valgrind: I can't recover.  Giving up.  Sorry.\n")
                   .empty());
}

/** @test valgrind's reader assertion before the program started is quoted as printed */
TEST(MemcheckLogTest, ReaderAssertionBeforeStartQuotesValgrind) {
  EXPECT_EQ(
      check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, "", logEndingWith(READER_ASSERTION)),
      READER_ASSERTION);
}

/** @test The same assertion once the program had started is not taken, on any one sign of it */
TEST(MemcheckLogTest, ReaderAssertionAfterTheProgramStartedIsNotTaken) {
  const std::string STARTED = "[==========] Running 1 test from 1 test suite.\n";
  const std::string AFTER_START_LOG = logEndingWith(READER_ASSERTION) + CRASH_REPORT;
  constexpr check::ChildExit EXITED_1{check::ChildExit::How::Exited, 1};

  EXPECT_TRUE(check::readerAssertionBeforeStart(EXITED_1, STARTED, AFTER_START_LOG).empty());
  // The program's output, valgrind's report after the line, valgrind's exit
  // status: each on its own
  EXPECT_TRUE(
      check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, STARTED, logEndingWith(READER_ASSERTION))
          .empty());
  EXPECT_TRUE(check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, "", AFTER_START_LOG).empty());
  EXPECT_TRUE(
      check::readerAssertionBeforeStart(EXITED_1, "", logEndingWith(READER_ASSERTION)).empty());
}

/** @test The assertion is not taken after anything of the run, or without valgrind's opening */
TEST(MemcheckLogTest, ReaderAssertionNeedsNothingBeforeIt) {
  const std::string AFTER_A_REPORT =
      std::string(OPENING_LINES) +
      "==7== Invalid write of size 1\n"
      "==7==    at 0x12ED79: vernier::bench::demo::memcheck_demo::joinOffByOne("
      "std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:32)\n"
      "==7== \n"
      "\n" +
      READER_ASSERTION + "\n";

  EXPECT_TRUE(check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, "", AFTER_A_REPORT).empty());
  EXPECT_TRUE(
      check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, "", std::string(READER_ASSERTION) + "\n")
          .empty());
}

/** @test Only a killed valgrind stopped before the program started */
TEST(MemcheckLogTest, ReaderAssertionNeedsValgrindKilled) {
  const std::string LOG = logEndingWith(READER_ASSERTION);

  EXPECT_TRUE(
      check::readerAssertionBeforeStart({check::ChildExit::How::Exited, 0}, "", LOG).empty());
  EXPECT_TRUE(
      check::readerAssertionBeforeStart({check::ChildExit::How::NotStarted, 2}, "", LOG).empty());
}

/** @test Another assertion, or the reader's text not as a line of its own, is not taken */
TEST(MemcheckLogTest, OnlyTheReadersAssertionIsTaken) {
  const std::vector<std::string> OTHERS = {
      // The same function's other assertion, and another function's
      "valgrind: m_debuginfo/readelf.c:2478 (vgModuleLocal_read_elf_debug_info): Assertion "
      "'di->sbss_svma + di->sbss_size == svma' failed.",
      "valgrind: m_debuginfo/readelf.c:2478 (read_elf_symtab__normal): Assertion "
      "'di->bss_svma + di->bss_size == svma' failed.",
      // Another reader's source, valgrind's core elsewhere, the tool itself
      "valgrind: m_debuginfo/readdwarf3.c:2478 (vgModuleLocal_read_elf_debug_info): Assertion "
      "'di->bss_svma + di->bss_size == svma' failed.",
      "valgrind: m_mallocfree.c:303 (get_bszB_as_is): Assertion 'bszB_lo == bszB_hi' failed.",
      "Memcheck: mc_leakcheck.c:1106 (lc_scan_memory): Assertion 'bad_scanned_addr >= "
      "VG_ROUNDUP(start, sizeof(Addr))' failed.",
      // The reader's text behind a log prefix, without its line number, with
      // more after it
      std::string("==7== ") + READER_ASSERTION,
      "valgrind: m_debuginfo/readelf.c: (vgModuleLocal_read_elf_debug_info): Assertion "
      "'di->bss_svma + di->bss_size == svma' failed.",
      std::string(READER_ASSERTION) + " Sorry.",
  };
  for (const std::string& other : OTHERS) {
    EXPECT_TRUE(
        check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, "", logEndingWith(other)).empty())
        << other;
  }
  // The same assertion at another line of another release's source
  const std::string ELSEWHERE = "valgrind: m_debuginfo/readelf.c:2512 "
                                "(vgModuleLocal_read_elf_debug_info): Assertion "
                                "'di->bss_svma + di->bss_size == svma' failed.";
  EXPECT_EQ(check::readerAssertionBeforeStart(KILLED_BY_SIGSEGV, "", logEndingWith(ELSEWHERE)),
            ELSEWHERE);
}

/** @test The program's own failed assertion before its tests is not valgrind's */
TEST(MemcheckLogTest, TheProgramsAssertionIsNotTaken) {
  // As valgrind 3.18.1 logged a program whose assert() failed before main,
  // most frames cut
  const std::string OUTPUT = "assert_before_main: assert_before_main.cpp:5: "
                             "FailsAtStartup::FailsAtStartup(): Assertion `!\"the program's own "
                             "assertion before main\"' failed.";
  const std::string LOG =
      "==7== Memcheck, a memory error detector\n"
      "==7== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.\n"
      "==7== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info\n"
      "==7== Command: ./assert_before_main\n"
      "==7== Parent PID: 6\n"
      "==7== \n"
      "==7== \n"
      "==7== Process terminating with default action of signal 6 (SIGABRT)\n"
      "==7==    at 0x49089BC: __pthread_kill_implementation (pthread_kill.c:44)\n"
      "==7==    by 0x48ABE95: __assert_fail (assert.c:103)\n"
      "==7== \n"
      "==7== HEAP SUMMARY:\n"
      "==7==     in use at exit: 0 bytes in 0 blocks\n"
      "==7==   total heap usage: 3 allocs, 3 frees, 544 bytes allocated\n"
      "==7== \n"
      "==7== All heap blocks were freed -- no leaks are possible\n"
      "==7== \n"
      "==7== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)\n";
  constexpr check::ChildExit KILLED_BY_SIGABRT{check::ChildExit::How::Signaled, SIGABRT};

  EXPECT_TRUE(check::readerAssertionBeforeStart(KILLED_BY_SIGABRT, OUTPUT + "\n", LOG).empty());
  EXPECT_TRUE(
      check::readerAssertionBeforeStart(KILLED_BY_SIGABRT, "", logEndingWith(OUTPUT)).empty());
}

/** @test The reader's give-up and its assertion are each their own reading */
TEST(MemcheckLogTest, GiveUpAndAssertionAreTold) {
  // As valgrind 3.18.1 logged the GCC 11.4 UBSan Debug demo linked by GNU ld
  const std::string GIVE_UP_LOG =
      std::string(OPENING_LINES) +
      "==7== Valgrind: debuginfo reader: ensure_valid failed:\n"
      "==7== Valgrind:   during call to ML_(img_get_UChar)\n"
      "==7== Valgrind:   request for range [1948945, +1) exceeds\n"
      "==7== Valgrind:   valid image size of 1948944 for image:\n"
      "==7== Valgrind:   \"/b/bin/ptests/BenchDemo_12_MemcheckProfiler\"\n"
      "==7== \n"
      "==7== Valgrind: debuginfo reader: Possibly corrupted debuginfo file.\n"
      "==7== Valgrind: I can't recover.  Giving up.  Sorry.\n"
      "==7== \n";

  EXPECT_FALSE(check::debugInfoGiveUp(GIVE_UP_LOG).empty());
  EXPECT_TRUE(check::readerAssertionBeforeStart({check::ChildExit::How::Exited, 1}, "", GIVE_UP_LOG)
                  .empty());
  EXPECT_TRUE(check::debugInfoGiveUp(logEndingWith(READER_ASSERTION)).empty());
}
