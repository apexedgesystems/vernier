/**
 * @file 09_BpftraceProfiler_uTest.cpp
 * @brief The checks behind walkthrough 09: what demo 09's writes and totals
 *        do, and what write_latency.bt counts for its per-line case.
 *
 * Not part of demo 09, whose source shows only what it teaches. The first
 * checks count and compare, so the machine's load cannot change them: how
 * many write() calls each write version makes (the kernel's own count for the
 * calling thread, syscw in /proc/thread-self/io), what the two write, and the
 * total each threaded version reaches through contentionRun(). The last one
 * runs the demo binary (BenchDemo_09_BpftraceProfiler, whose path the build
 * passes in) under bpftrace with write_latency.bt, as the walkthrough does,
 * and reads the histogram it left; 09_BpftraceProfiler_Check.hpp holds the
 * traced run and the report reading. ctest runs them under the demo label,
 * the traced one also under the bpftrace label.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L demo
 *   ./build/bin/tests/TestDemoBpftrace      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/09_BpftraceProfiler_Check.hpp"

#include "src/bench/demo/cpu/09_BpftraceProfiler_Workload.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/ProfilerBpftrace.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::bpftrace_check;
namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::demo::bpftrace_demo::addToThreadTotal;
using vernier::bench::demo::bpftrace_demo::addUnderCoarseLock;
using vernier::bench::demo::bpftrace_demo::finishedThreadsTotal;
using vernier::bench::demo::bpftrace_demo::LINE_END;
using vernier::bench::demo::bpftrace_demo::linesOf;
using vernier::bench::demo::bpftrace_demo::PART_COUNT;
using vernier::bench::demo::bpftrace_demo::PART_SEED;
using vernier::bench::demo::bpftrace_demo::SharedTotal;
using vernier::bench::demo::bpftrace_demo::writeBatched;
using vernier::bench::demo::bpftrace_demo::writeEachLine;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The demo binary the traced check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_09_BINARY;

/// The demo's case the traced check runs.
constexpr const char* TRACED_CASE = "BpftraceProfiler.WritePerLine";

/// The traced run: calls per repeat and repeats, fixed so the writes it makes
/// are known, and few, so its traced window stays well under the two seconds
/// after which the harness writes a progress line.
constexpr int TRACED_CALLS = 20;
constexpr int TRACED_REPEATS = 2;

/// Calls of each write version the counting checks make.
constexpr int CALLS = 10;

/// The threaded run: threads, calls per thread and repeats.
constexpr int THREADS = 4;
constexpr int CALLS_PER_THREAD = 25;
constexpr int REPEATS = 3;

/// The kernel's count of the write() calls the calling thread has made (syscw
/// in /proc/thread-self/io); -1 when the file or the count is missing, as on
/// a kernel built without task I/O accounting.
long writeCallsOfThisThread() {
  std::ifstream in("/proc/thread-self/io");
  std::string key;
  long value = 0;
  while (in >> key >> value) {
    if (key == "syscw:") {
      return value;
    }
  }
  return -1;
}

/// What @p writeTo writes into a pipe, read back; @p written is what it
/// returned. The text fits in the pipe's buffer, so nothing waits.
std::string throughAPipe(const std::function<std::size_t(int)>& writeTo, std::size_t& written) {
  int ends[2] = {-1, -1};
  if (::pipe(ends) != 0) {
    return "";
  }
  written = writeTo(ends[1]);
  ::close(ends[1]);
  std::string text;
  char buffer[4096];
  for (ssize_t got = ::read(ends[0], buffer, sizeof(buffer)); got > 0;
       got = ::read(ends[0], buffer, sizeof(buffer))) {
    text.append(buffer, static_cast<std::size_t>(got));
  }
  ::close(ends[0]);
  return text;
}

/// Where @p text first differs from @p expected: the offset of the first byte
/// that differs, or the shorter one's length when one is the start of the
/// other; -1 when they are the same.
long firstDifference(const std::string& text, const std::string& expected) {
  const std::size_t SHORTER = std::min(text.size(), expected.size());
  for (std::size_t i = 0; i < SHORTER; ++i) {
    if (text[i] != expected[i]) {
      return static_cast<long>(i);
    }
  }
  return text.size() == expected.size() ? -1 : static_cast<long>(SHORTER);
}

/// A configuration under which contentionRun() makes exactly
/// THREADS x CALLS_PER_THREAD x REPEATS calls.
PerfConfig fixedRun() {
  PerfConfig cfg;
  cfg.threads = THREADS;
  cfg.cycles = CALLS_PER_THREAD;
  cfg.repeats = REPEATS;
  return cfg;
}

/// Every call's joined length, added up.
std::size_t everyCallsLength(const std::vector<std::string>& words) {
  return static_cast<std::size_t>(THREADS) * CALLS_PER_THREAD * REPEATS * demo::joinedSize(words);
}

/// The demo binary's canonical path, or an empty string when it is missing.
std::string demoPath() {
  std::error_code ec;
  const fs::path PATH = fs::canonical(DEMO_BINARY, ec);
  return ec ? std::string() : PATH.string();
}

/// A new temporary directory for the traced run's files; empty when it
/// cannot be made.
fs::path scratchDir() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo09-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

} // namespace

/* ----------------------------- Writes ----------------------------- */

/** @test The per-line version makes one write() per line, every call */
TEST(BpftraceWritesTest, PerLineMakesOneWritePerLine) {
  const auto LINES = linesOf(demo::makeParts(PART_COUNT, PART_SEED));
  const int FD = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
  ASSERT_GE(FD, 0) << "cannot open /dev/null";
  const long BEFORE = writeCallsOfThisThread();
  if (BEFORE < 0) {
    ::close(FD);
    GTEST_SKIP() << "/proc/thread-self/io has no syscw count here (a kernel without task I/O "
                    "accounting), so the kernel's count of this thread's write() calls cannot "
                    "be read";
  }

  for (int call = 0; call < CALLS; ++call) {
    writeEachLine(FD, LINES);
  }
  const long AFTER = writeCallsOfThisThread();
  ::close(FD);

  EXPECT_EQ(AFTER - BEFORE, static_cast<long>(CALLS) * static_cast<long>(PART_COUNT));
}

/** @test The batched version makes one write() per call */
TEST(BpftraceWritesTest, BatchedMakesOneWrite) {
  const std::string TEXT = demo::joinV1(demo::makeParts(PART_COUNT, PART_SEED), LINE_END);
  const int FD = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
  ASSERT_GE(FD, 0) << "cannot open /dev/null";
  const long BEFORE = writeCallsOfThisThread();
  if (BEFORE < 0) {
    ::close(FD);
    GTEST_SKIP() << "/proc/thread-self/io has no syscw count here (a kernel without task I/O "
                    "accounting), so the kernel's count of this thread's write() calls cannot "
                    "be read";
  }

  for (int call = 0; call < CALLS; ++call) {
    writeBatched(FD, TEXT);
  }
  const long AFTER = writeCallsOfThisThread();
  ::close(FD);

  EXPECT_EQ(AFTER - BEFORE, static_cast<long>(CALLS));
}

/** @test Both versions write the same text: every line, in order, end to end */
TEST(BpftraceWritesTest, BothWriteTheLinesEndToEnd) {
  const auto WORDS = demo::makeParts(PART_COUNT, PART_SEED);
  const auto LINES = linesOf(WORDS);
  std::string expected;
  for (const std::string& line : LINES) {
    expected += line;
  }
  ASSERT_EQ(firstDifference(demo::joinV1(WORDS, LINE_END), expected), -1)
      << "the lines end to end are not joinV1(words, LINE_END)";

  std::size_t perLineWritten = 0;
  const std::string PER_LINE =
      throughAPipe([&](int fd) { return writeEachLine(fd, LINES); }, perLineWritten);
  std::size_t batchedWritten = 0;
  const std::string BATCHED =
      throughAPipe([&](int fd) { return writeBatched(fd, expected); }, batchedWritten);

  EXPECT_EQ(firstDifference(PER_LINE, expected), -1)
      << "the per-line version's text differs from the lines end to end (it wrote "
      << PER_LINE.size() << " of " << expected.size() << " bytes)";
  EXPECT_EQ(firstDifference(BATCHED, expected), -1)
      << "the batched version's text differs from the lines end to end (it wrote " << BATCHED.size()
      << " of " << expected.size() << " bytes)";
  EXPECT_EQ(perLineWritten, expected.size());
  EXPECT_EQ(batchedWritten, expected.size());
}

/* ----------------------------- Totals ----------------------------- */

/** @test The coarse-lock version's shared total holds every call's joined length */
TEST(BpftraceTotalsTest, CoarseLockCountsEveryCall) {
  const auto WORDS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  PerfCase perf{"BpftraceTotalsTest.CoarseLock", fixedRun()};
  perf.contentionRun([&] { addUnderCoarseLock(total, WORDS); }, "coarse_lock");

  EXPECT_EQ(total.value, everyCallsLength(WORDS));
}

/** @test Every no-sharing thread has handed its total over when contentionRun returns */
TEST(BpftraceTotalsTest, NoSharingCountsEveryCallOnceItsThreadsEnd) {
  const auto WORDS = demo::makeParts(PART_COUNT, PART_SEED);
  finishedThreadsTotal = 0;

  PerfCase perf{"BpftraceTotalsTest.NoSharing", fixedRun()};
  perf.contentionRun([&] { addToThreadTotal(WORDS); }, "no_sharing");

  EXPECT_EQ(finishedThreadsTotal.load(), everyCallsLength(WORDS));
}

/* ----------------------------- Traced ----------------------------- */

/**
 * @test write_latency.bt counts every write() WritePerLine makes in its
 *       measured calls.
 *
 * Runs the demo binary on WritePerLine with fixed calls and repeats, under
 * `--profile bpftrace --bpf write_latency` with BENCH_SUDO=1, as the
 * walkthrough does. The case must run to its end and pass, the backend must
 * print nothing about its tracer (a tracer that failed, ended by itself or
 * was killed says so), its report must hold the capture window's two
 * acknowledgements once each, for the demo's pid and the very threads the
 * run's copy names (the arm thread, not the main one, and the thread that ran
 * the test), and its histogram must count at least lines x calls writes and
 * fewer than one more call's lines: the tracer sees every write() the
 * process makes while it runs, so a write of the harness's own in that
 * window counts too. The measured repeats start only once the tracer has
 * acknowledged its arm, so a busy machine only makes the run wait longer.
 *
 * Skipped where the bundled scripts are not supported or bpftrace cannot
 * run: in a PID namespace other than the host's (a container without
 * --pid=host), without bpftrace, and without root or sudo, each known before
 * anything runs; and where the run says bpftrace could not trace here, for
 * want of a privilege (no sudo grant, or bpftrace refused) or of a kernel
 * probe, quoting the run's line, which ends with sudo's or bpftrace's own. A
 * run that fails for any other reason fails, with what it printed.
 */
TEST(Bpftrace, CountsEveryWrite) {
  const std::string NESTED = vernier::bench::bpftrace_tool::foreignPidNamespace();
  if (!NESTED.empty()) {
    GTEST_SKIP() << "this process runs in a PID namespace other than the host's (" << NESTED
                 << "), where the bundled scripts are not supported: run it natively, or in a "
                    "container started with --pid=host";
  }
  if (!vernier::bench::profiler_env::isOnPath("bpftrace")) {
    GTEST_SKIP() << "bpftrace is not installed; this test traces the demo's writes with it";
  }
  if (::geteuid() != 0 && !vernier::bench::profiler_env::isOnPath("sudo")) {
    GTEST_SKIP() << "sudo is not installed, and this test runs bpftrace through sudo -n "
                    "(BENCH_SUDO=1) when it is not root";
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const check::TracedRun RUN = check::runTraced(
      DEMO, TRACED_CASE, {"bpftrace", "--bpf", "write_latency"},
      {"--cycles", std::to_string(TRACED_CALLS), "--repeats", std::to_string(TRACED_REPEATS)}, DIR);
  const std::string CANNOT = check::cannotTraceHere(RUN.output, "bpftrace");
  if (!CANNOT.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "bpftrace could not trace here. The run printed:\n" << CANNOT;
  }
  ASSERT_TRUE(vg::testPassed(RUN.output, TRACED_CASE) && vg::oneTestPassed(RUN.output))
      << "WritePerLine did not run to its end under bpftrace: the demo " << vg::describe(RUN.end)
      << " (its files are in " << DIR << "). It printed:\n"
      << vg::lastLines(RUN.output, 40);
  EXPECT_TRUE(vg::exitedWith(RUN.end, 0)) << "the demo " << vg::describe(RUN.end);
  const std::string BACKEND = check::backendLines(RUN.output, "bpftrace");
  EXPECT_TRUE(BACKEND.empty()) << "the backend reported a problem with its tracer:\n" << BACKEND;

  const fs::path FOLDER = check::captureFolder(RUN, TRACED_CASE, "bpf");
  for (const char* FILE :
       {"write_latency.tmp.bt", "write_latency.out.text", "write_latency.err.txt"}) {
    EXPECT_TRUE(fs::is_regular_file(FOLDER / FILE)) << "the capture folder has no " << FILE;
  }
  const std::string REPORT = vg::readText(FOLDER / "write_latency.out.text");
  // The capture window: the tracer acknowledged the start of the measured
  // repeats once, for the demo's own pid and the arm thread the run's copy
  // names (not the main thread), and their end once, for the same pid and the
  // thread the copy names for the stop.
  const std::string COPY = vg::readText(FOLDER / "write_latency.tmp.bt");
  const long PID = check::filledInPid(COPY);
  const long ARM_THREAD = check::boundThread(COPY, "vernier-arm");
  const long STOP_THREAD = check::boundThread(COPY, "vernier-stop");
  const std::vector<long> ARMED = check::acknowledgement(REPORT, "bpftrace", "armed");
  const std::vector<long> DISARMED = check::acknowledgement(REPORT, "bpftrace", "disarmed");
  ASSERT_GT(PID, 0) << "the run's copy names no pid";
  ASSERT_GT(ARM_THREAD, 0) << "the run's copy names no arm thread:\n" << COPY;
  ASSERT_GT(STOP_THREAD, 0) << "the run's copy names no stopping thread:\n" << COPY;
  EXPECT_NE(ARM_THREAD, PID) << "the run's copy binds the arm to the main thread";
  ASSERT_EQ(ARMED.size(), 2U) << "no arm acknowledgement in the report:\n" << REPORT;
  EXPECT_EQ(ARMED[0], PID) << "the arm was acknowledged for another process";
  EXPECT_EQ(ARMED[1], ARM_THREAD) << "the arm was acknowledged from another thread";
  ASSERT_EQ(DISARMED.size(), 2U) << "no stop acknowledgement in the report:\n" << REPORT;
  EXPECT_EQ(DISARMED[0], PID) << "the stop was acknowledged for another process";
  EXPECT_EQ(DISARMED[1], STOP_THREAD) << "the stop was acknowledged from another thread";
  EXPECT_EQ(check::linesWith(REPORT, "bpftrace armed "), 1) << REPORT;
  EXPECT_EQ(check::linesWith(REPORT, "bpftrace disarmed "), 1) << REPORT;
  const long TOTAL = check::histogramTotal(REPORT, "@write_latency_us");
  const long MADE = static_cast<long>(PART_COUNT) * TRACED_CALLS * TRACED_REPEATS;
  std::printf("[Bpftrace.CountsEveryWrite]  write_latency.bt counted %ld writes; WritePerLine made "
              "%ld in its measured calls (%d x %d calls of %zu lines)\n",
              TOTAL, MADE, TRACED_REPEATS, TRACED_CALLS, PART_COUNT);
  if (TOTAL == -1) {
    ADD_FAILURE() << "the report holds no @write_latency_us histogram, so the tracer saw no write "
                     "of the process. The report:\n"
                  << REPORT << "\nThe run printed:\n"
                  << vg::lastLines(RUN.output, 40);
  } else {
    EXPECT_GE(TOTAL, MADE) << "the tracer missed writes WritePerLine made";
    EXPECT_LT(TOTAL, MADE + static_cast<long>(PART_COUNT))
        << "the tracer counted a call's worth of writes more than WritePerLine made";
  }

  if (HasFailure()) {
    std::printf("the traced run's files are kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/* ----------------------------- Report Reading Tests ----------------------------- */

// What decides CountsEveryWrite's skips and its count, on text taken from
// real runs, with shortened paths: the Pi rig's traced writes (bpftrace
// 0.23.2) and its refusal without BENCH_SUDO, the laptop's traced writes
// (bpftrace 0.14.0), and the laptop's run under the readiness tests' fake
// sudo, which refuses bpftrace as sudo -n does without a grant.

namespace {

/// The report the rig's bpftrace wrote for 500 x 2 per-byte calls of 1,024
/// writes (1,024,000 writes, and the harness's progress line and its clearing).
constexpr const char* RIG_REPORT =
    "\n\n\n"
    "@write_latency_us:\n"
    "[0]              1023352 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|\n"
    "[1]                  403 |                                                    |\n"
    "[2, 4)                 4 |                                                    |\n"
    "[4, 8)               219 |                                                    |\n"
    "[8, 16)               15 |                                                    |\n"
    "[16, 32)               7 |                                                    |\n"
    "[32, 64)               2 |                                                    |\n"
    "\n";

/// The report the laptop's bpftrace 0.14.0 wrote for CountsEveryWrite's run
/// (40,000 writes), the capture window's two lines first; its map name ends
/// with a space.
constexpr const char* LAPTOP_REPORT =
    "bpftrace armed 1782012 1782070\n"
    "bpftrace disarmed 1782012 1782012\n"
    "\n\n\n\n"
    "@write_latency_us: \n"
    "[0]                39988 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|\n"
    "[1]                    1 |                                                    |\n"
    "[2, 4)                 1 |                                                    |\n"
    "[4, 8)                 6 |                                                    |\n"
    "[8, 16)                4 |                                                    |\n"
    "\n";

/// The capture window at the end of that run's copy of write_latency.bt.
constexpr const char* LAPTOP_RUN_COPY_WINDOW =
    "tracepoint:sched:sched_switch {\n"
    "  if (pid == 1782012 && (args->prev_state == 1 || args->prev_state == 2)) {\n"
    "    if (tid == 1782070 && comm == \"vernier-arm\") {\n"
    "      if (@vernier_window == 0) {\n"
    "        @vernier_window = 1;\n"
    "        printf(\"bpftrace armed %d %d\\n\", pid, tid);\n"
    "      }\n"
    "    } else if (tid == 1782012 && comm == \"vernier-stop\") {\n"
    "      if (@vernier_window == 1) {\n"
    "        @vernier_window = 2;\n"
    "        printf(\"bpftrace disarmed %d %d\\n\", pid, tid);\n"
    "        clear(@vernier_window);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n";

/// The laptop's run under a sudo that refuses bpftrace, first the decision's
/// notice: it could not try the run's own command, so it printed a caveat.
constexpr const char* UNVERIFIED_NOTICE =
    "[WARN] Profiler 'bpftrace': unverified: sudo -n refused the probe command /usr/bin/bpftrace "
    "-q -B none /tmp/p/probe0.bt: sudo: a password is required; the run executes "
    "/usr/bin/bpftrace -q -B none <capture folder>/write_latency.tmp.bt instead, which only the "
    "run can try\n"
    "   The grant must allow /usr/bin/bpftrace with the run's script arguments and /usr/bin/kill "
    "with -2, -15 and -9 for the tracer's pid.\n";

/// Then the run's start, which sudo refused.
constexpr const char* REFUSED_START =
    "[bpftrace] the tracer exited before it acknowledged its arm probe: denied: sudo -n refused "
    "/usr/bin/bpftrace -q -B none /t/captures/BpftraceProfiler.WritePerLine.bpf/"
    "write_latency.tmp.bt: sudo: a password is required\n"
    "[bpftrace] The grant must allow /usr/bin/bpftrace with the run's script arguments and "
    "/usr/bin/kill with -2, -15 and -9 for the tracer's pid.\n";

} // namespace

/** @test Every bucket's count is added, and nothing after the histogram */
TEST(BpftraceReportTest, HistogramTotalAddsEveryBucket) {
  EXPECT_EQ(check::histogramTotal(RIG_REPORT, "@write_latency_us"), 1024002);
  EXPECT_EQ(
      check::histogramTotal(std::string(RIG_REPORT) + "@start[4242]: 17\n", "@write_latency_us"),
      1024002);
}

/** @test bpftrace 0.14's report, whose map name ends with a space, is read the same way */
TEST(BpftraceReportTest, HistogramTotalAfterANameWithASpace) {
  EXPECT_EQ(check::histogramTotal(LAPTOP_REPORT, "@write_latency_us"), 40000);
}

/** @test The capture window's lines give their ids; a report without them gives none */
TEST(BpftraceReportTest, AcknowledgementReadsTheWindowsLines) {
  EXPECT_EQ(check::acknowledgement(LAPTOP_REPORT, "bpftrace", "armed"),
            (std::vector<long>{1782012, 1782070}));
  EXPECT_EQ(check::acknowledgement(LAPTOP_REPORT, "bpftrace", "disarmed"),
            (std::vector<long>{1782012, 1782012}));
  EXPECT_TRUE(check::acknowledgement(RIG_REPORT, "bpftrace", "armed").empty());
  EXPECT_TRUE(check::acknowledgement(LAPTOP_REPORT, "offcpu", "armed").empty());
}

/** @test Each of the window's lines is counted, so a second stop line would show */
TEST(BpftraceReportTest, LinesWithCountsTheWindowsLines) {
  EXPECT_EQ(check::linesWith(LAPTOP_REPORT, "bpftrace armed "), 1);
  EXPECT_EQ(check::linesWith(LAPTOP_REPORT, "bpftrace disarmed "), 1);
  EXPECT_EQ(check::linesWith(std::string(LAPTOP_REPORT) + "bpftrace disarmed 1782012 1782012\n",
                             "bpftrace disarmed "),
            2);
  EXPECT_EQ(check::linesWith(RIG_REPORT, "bpftrace armed "), 0);
}

/** @test The threads a run's copy binds its window to, as the program names them */
TEST(BpftraceReportTest, BoundThreadReadsTheRunCopysWindow) {
  EXPECT_EQ(check::boundThread(LAPTOP_RUN_COPY_WINDOW, "vernier-arm"), 1782070);
  EXPECT_EQ(check::boundThread(LAPTOP_RUN_COPY_WINDOW, "vernier-stop"), 1782012);
  EXPECT_EQ(check::boundThread(LAPTOP_RUN_COPY_WINDOW, "vernier-wait"), -1);
  EXPECT_EQ(
      check::boundThread("tracepoint:syscalls:sys_enter_write /pid == 1782012/ {\n", "vernier-arm"),
      -1);
}

/** @test The pid a run's copy was given: the first "pid == " it holds */
TEST(BpftraceReportTest, FilledInPidReadsTheRunCopy) {
  EXPECT_EQ(check::filledInPid("tracepoint:syscalls:sys_enter_write /pid == 1782012/ {\n"),
            1782012);
  EXPECT_EQ(check::filledInPid("tracepoint:syscalls:sys_enter_write /pid == {{PID}}/ {\n"), -1);
  EXPECT_EQ(check::filledInPid("interval:s:1 { exit(); }\n"), -1);
}

/** @test A map the report does not hold is not a total of zero */
TEST(BpftraceReportTest, HistogramTotalOfAMapThatIsNotThere) {
  EXPECT_EQ(check::histogramTotal("\n\n\n\n", "@fsync_latency_us"), -1);
  EXPECT_EQ(check::histogramTotal(RIG_REPORT, "@fsync_latency_us"), -1);
  EXPECT_EQ(check::histogramTotal(RIG_REPORT, "@write_latency"), -1);
}

/** @test sudo's refusal of the run's tracer is quoted as the run printed it, and nothing else */
TEST(BpftraceReportTest, CannotTraceHereQuotesSudo) {
  const std::string OUTPUT = std::string("[ RUN      ] BpftraceProfiler.WritePerLine\n\n") +
                             UNVERIFIED_NOTICE + "\n" + REFUSED_START +
                             "[BpftraceProfiler.WritePerLine]  205.350 us/call  CV=0.0%\n";

  EXPECT_EQ(check::cannotTraceHere(OUTPUT, "bpftrace"),
            "[bpftrace] the tracer exited before it acknowledged its arm probe: denied: sudo -n "
            "refused /usr/bin/bpftrace -q -B none /t/captures/BpftraceProfiler.WritePerLine.bpf/"
            "write_latency.tmp.bt: sudo: a password is required\n");
}

/** @test The decision's refusal is quoted with its remedy */
TEST(BpftraceReportTest, CannotTraceHereQuotesTheNotice) {
  const std::string NOTICE =
      "[FAIL] Profiler 'bpftrace': denied: script 'write_latency' could not attach as the current "
      "user: ERROR: bpftrace currently only supports running as the root user.\n"
      "   Set BENCH_SUDO=1 with a scoped sudoers grant for /usr/bin/bpftrace and /usr/bin/kill, "
      "or run as root.\n"
      "   Falling back to no-op (measurements will proceed without profiling).\n";
  const std::string OUTPUT = "[ RUN      ] BpftraceProfiler.WritePerLine\n\n" + NOTICE +
                             "\n[BpftraceProfiler.WritePerLine]  530.1 us/call\n";

  EXPECT_EQ(check::cannotTraceHere(OUTPUT, "bpftrace"), NOTICE);
}

/** @test A missing script, a caveat or another backend's refusal is not a reason to skip */
TEST(BpftraceReportTest, CannotTraceHereOnlyForWantOfPrivilegeOrProbe) {
  EXPECT_TRUE(check::cannotTraceHere("[FAIL] Profiler 'bpftrace': missing: bpftrace script "
                                     "'write_latency' not found at /s/bpf/write_latency.bt\n",
                                     "bpftrace")
                  .empty());
  EXPECT_TRUE(check::cannotTraceHere(UNVERIFIED_NOTICE, "bpftrace").empty());
  EXPECT_TRUE(check::cannotTraceHere("[FAIL] Profiler 'offcpu': denied: the off-CPU script could "
                                     "not attach as the current user\n",
                                     "bpftrace")
                  .empty());
  EXPECT_TRUE(check::cannotTraceHere("[bpftrace] script 'write_latency' ended by itself before "
                                     "the measured repeats finished\n",
                                     "bpftrace")
                  .empty());
}
