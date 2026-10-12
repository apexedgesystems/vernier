/**
 * @file ValgrindReaderAssertion_uTest.cpp
 * @brief Unit tests for ValgrindReaderAssertion.hpp: valgrind's line when an
 * assertion in its debug-information reader stopped it before the program
 * started.
 *
 * Notes:
 *  - The texts are valgrind 3.18.1's, from real runs on x86-64 with a short
 *    program path and process id: callgrind on the stream it shares with the
 *    program (the reader's assertion on a GCC 11.4 Debug binary linked by
 *    mold, the reader giving up, a program that aborts before its tests) and
 *    helgrind with a log file of its own. A line of the program's or of
 *    valgrind's before the assertion, and a crash report after it, are shaped
 *    as valgrind and such a program print them; no run printed them there.
 */

#include "src/bench/utst/ValgrindReaderAssertion.hpp"

#include <gtest/gtest.h>

#include <string>

using vernier::bench::test::readerAssertionBeforeStart;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The reader's assertion, as valgrind 3.18.1 printed it.
constexpr const char* READER_ASSERTION =
    "valgrind: m_debuginfo/readelf.c:2478 (vgModuleLocal_read_elf_debug_info): Assertion "
    "'di->bss_svma + di->bss_size == svma' failed.";

/// callgrind's opening lines on the stream it shares with the program, as it
/// printed them for a check's counting run.
constexpr const char* CALLGRIND_OPENING =
    "==7== Callgrind, a call-graph generating cache profiler\n"
    "==7== Copyright (C) 2002-2017, and GNU GPL'd, by Josef Weidendorfer et al.\n"
    "==7== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info\n"
    "==7== Command: /b/bin/JoinInstructionCounts --gtest_filter=JoinInstructionCounts.Worker "
    "--gtest_print_time=0\n"
    "==7== \n"
    "==7== For interactive control, run 'callgrind_control -h'.\n";

/// helgrind's opening lines in a log file of its own.
constexpr const char* HELGRIND_LOG_OPENING =
    "==7== Helgrind, a thread error detector\n"
    "==7== Copyright (C) 2007-2017, and GNU GPL'd, by OpenWorks LLP et al.\n"
    "==7== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info\n"
    "==7== Command: /b/bin/ptests/BenchDemo_14_HelgrindProfiler --gtest_filter=Helgrind.RacyTotal "
    "--gtest_print_time=0\n"
    "==7== Parent PID: 6\n"
    "==7== \n";

/// The stream as callgrind left it when the assertion stopped it before the
/// program started: its opening lines, an empty line, the assertion.
std::string callgrindStoppedBeforeStart() {
  return std::string(CALLGRIND_OPENING) + "\n" + READER_ASSERTION + "\n";
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test On callgrind's shared stream, the assertion right after its opening is quoted as printed
 */
TEST(ValgrindReaderAssertionTest, CallgrindStreamStoppedBeforeStartIsQuoted) {
  EXPECT_EQ(readerAssertionBeforeStart(true, "", callgrindStoppedBeforeStart()), READER_ASSERTION);
}

/** @test In a log file of its own, the assertion right after valgrind's opening is quoted */
TEST(ValgrindReaderAssertionTest, LogFileStoppedBeforeStartIsQuoted) {
  const std::string LOG = std::string(HELGRIND_LOG_OPENING) + "\n" + READER_ASSERTION + "\n";

  EXPECT_EQ(readerAssertionBeforeStart(true, "", LOG), READER_ASSERTION);
}

/** @test valgrind's line is taken only when valgrind was killed by a signal */
TEST(ValgrindReaderAssertionTest, OnlyAKilledValgrindStoppedBeforeStart) {
  EXPECT_TRUE(readerAssertionBeforeStart(false, "", callgrindStoppedBeforeStart()).empty());
}

/** @test A program that wrote anything had started: in its own file or on the shared stream */
TEST(ValgrindReaderAssertionTest, AProgramThatWroteHadStarted) {
  const std::string LOG = std::string(HELGRIND_LOG_OPENING) + "\n" + READER_ASSERTION + "\n";
  const std::string PROGRAM_FIRST = std::string(CALLGRIND_OPENING) +
                                    "Running main() from gmock_main.cc\n"
                                    "\n" +
                                    READER_ASSERTION + "\n";

  EXPECT_TRUE(readerAssertionBeforeStart(true, "Running main() from gmock_main.cc\n", LOG).empty());
  EXPECT_TRUE(readerAssertionBeforeStart(true, "", PROGRAM_FIRST).empty());
}

/** @test Another of valgrind's own lines before the assertion is not the end of its opening */
TEST(ValgrindReaderAssertionTest, ValgrindsOtherLinesDoNotEndItsOpening) {
  const std::string READER_LINE_FIRST = std::string(CALLGRIND_OPENING) +
                                        "==7== Valgrind: debuginfo reader: ensure_valid failed:\n"
                                        "\n" +
                                        READER_ASSERTION + "\n";

  EXPECT_TRUE(readerAssertionBeforeStart(true, "", READER_LINE_FIRST).empty());
}

/** @test valgrind's crash report after the line says the program had run */
TEST(ValgrindReaderAssertionTest, ACrashReportAfterTheLineIsNotTaken) {
  const std::string THEN_REPORT = callgrindStoppedBeforeStart() +
                                  "\n"
                                  "host stacktrace:\n"
                                  "==7==    at 0x5804284A: ??? (in "
                                  "/usr/libexec/valgrind/callgrind-amd64-linux)\n";

  EXPECT_TRUE(readerAssertionBeforeStart(true, "", THEN_REPORT).empty());
}

/** @test A program's own abort before its tests, on the shared stream, is no assertion of
 * valgrind's */
TEST(ValgrindReaderAssertionTest, AStartupFaultIsNotTaken) {
  // As callgrind printed it for a program that aborts in its static
  // initialization under valgrind, most frames cut; valgrind was killed by
  // SIGABRT
  const std::string FAULT = std::string(CALLGRIND_OPENING) +
                            "StartupFaultUnderValgrind: aborting before the tests start, as "
                            "intended\n"
                            "==7== \n"
                            "==7== Process terminating with default action of signal 6 (SIGABRT)\n"
                            "==7==    at 0x4BC99BC: pthread_kill@@GLIBC_2.34 (pthread_kill.c:44)\n"
                            "==7== \n"
                            "==7== Events    : Ir\n"
                            "==7== Collected : 3439864\n"
                            "==7== \n"
                            "==7== I   refs:      3,439,864\n";

  EXPECT_TRUE(readerAssertionBeforeStart(true, "", FAULT).empty());
}

/** @test The reader's give-up is no assertion: the same opening, valgrind's other words */
TEST(ValgrindReaderAssertionTest, TheReadersGiveUpIsNotTaken) {
  // As callgrind printed it when its reader gave up on a library; valgrind
  // exited with status 1
  const std::string GIVE_UP = std::string(CALLGRIND_OPENING) +
                              "==7== Valgrind: debuginfo reader: Possibly corrupted debuginfo "
                              "file.\n"
                              "==7== Valgrind: I can't recover.  Giving up.  Sorry.\n"
                              "==7== \n";

  EXPECT_TRUE(readerAssertionBeforeStart(false, "", GIVE_UP).empty());
  EXPECT_TRUE(readerAssertionBeforeStart(true, "", GIVE_UP).empty());
}
