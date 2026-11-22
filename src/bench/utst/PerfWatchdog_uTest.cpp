/**
 * @file PerfWatchdog_uTest.cpp
 * @brief Unit tests for vernier::bench::perf_watchdog.
 *
 * Notes:
 *  - The watchdog ends the process it guards, so each case in which it fires,
 *    or must not, runs WatchdogHelper as a child and reads how it ended.
 *  - Those children wait on real timers: 0.1 s for a direct arm, 0.3 s past
 *    a deadline that must not fire (twice in the disarm and re-arm case),
 *    and the 1 s of --profile-test-timeout 1 for the hung loops.
 *  - The helper is single-threaded, so the watchdog takes its timer engine
 *    where it is built (VERNIER_WATCHDOG_TIMER); run with --threaded, the
 *    helper first starts a thread of its own and the watchdog takes its
 *    thread. The cases that hold for one engine only say which they run.
 *  - The cases of a later arming on the thread make the helper wait until the
 *    watchdog's thread is parked, or waiting for a deadline, before it arms
 *    again: an arming the thread does not see at once is then never seen.
 *  - The cases in this process arm through a measured loop for 30 s or more
 *    and read the watchdog's flag; none waits for it.
 *  - One case runs the helper four times under valgrind's callgrind, about
 *    half a second each, and one four times under helgrind, about a second
 *    and a half each; they skip where valgrind is not installed and in an
 *    address or thread sanitizer build, which valgrind cannot run.
 */

#include "src/bench/inc/PerfHarness.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"

#include <gtest/gtest.h>

#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#ifndef VERNIER_WATCHDOG_HELPER
#error "VERNIER_WATCHDOG_HELPER must name the WatchdogHelper program"
#endif

namespace perf_watchdog = vernier::bench::perf_watchdog;
using vernier::bench::ProbeResult;

namespace {

/// The program the cases in which the watchdog fires, or must not, run.
constexpr const char* HELPER = VERNIER_WATCHDOG_HELPER;

/// How long a run of the helper may take before it is killed: far past what it waits.
constexpr int HELPER_BOUND_MS = 20000;

/// The build has the thread sanitizer, which ends a child of a multi-threaded
/// fork() that starts a thread (its die_after_fork), and starts a thread of its
/// own in a child of a single-threaded one.
#if defined(__SANITIZE_THREAD__)
constexpr bool THREAD_SANITIZER = true;
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
constexpr bool THREAD_SANITIZER = true;
#else
constexpr bool THREAD_SANITIZER = false;
#endif
#else
constexpr bool THREAD_SANITIZER = false;
#endif

/// The build has the address sanitizer, whose program valgrind cannot run.
#if defined(__SANITIZE_ADDRESS__)
constexpr bool ADDRESS_SANITIZER = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
constexpr bool ADDRESS_SANITIZER = true;
#else
constexpr bool ADDRESS_SANITIZER = false;
#endif
#else
constexpr bool ADDRESS_SANITIZER = false;
#endif

/// One run of the helper with @p args, its stdout and stderr kept apart.
ProbeResult runHelper(std::vector<std::string> args) {
  args.insert(args.begin(), HELPER);
  return vernier::bench::runBoundedProbe(args, HELPER_BOUND_MS,
                                         vernier::bench::ReadinessContext::capture(),
                                         vernier::bench::ProbeStreams::SEPARATE);
}

/// The helper started with --threaded: its own thread first, so the watchdog takes its thread.
ProbeResult runThreaded(std::vector<std::string> args) {
  args.insert(args.begin(), "--threaded");
  return runHelper(std::move(args));
}

/// Why a case of the timer engine cannot run here, or empty when it can.
std::string withoutTheTimer() {
  return VERNIER_WATCHDOG_TIMER ? std::string{}
                                : "this build has no timer engine (Linux with glibc 2.34 or later)";
}

/// The thread count and glibc's single-thread word the helper's threads mode
/// prints, when the body first runs and after the case.
struct ThreadsLine {
  int threads = -1;
  int single = -2;
  int afterThreads = -1;
  int afterSingle = -2;
};

/// Reads the threads mode's line out of @p output, after what a profiler
/// prints for the case; false when it is not there.
bool readThreads(const std::string& output, ThreadsLine& line) {
  const std::size_t AT = output.find("threads=");
  return AT != std::string::npos &&
         std::sscanf(output.c_str() + AT,
                     "threads=%d single_threaded=%d after_threads=%d after_single_threaded=%d",
                     &line.threads, &line.single, &line.afterThreads, &line.afterSingle) == 4;
}

/// The watchdog's whole message for @p test under @p tool.
std::string timeoutMessage(const std::string& test, const std::string& tool) {
  return "\n[bench] watchdog timeout: test '" + test + "' under --profile " + tool +
         " exceeded the configured profile-test-timeout.\n"
         "[bench] Likely cause: drain loop, blocking I/O, or filter that holds the\n"
         "[bench] measured loop. Identify the test, exclude it via --gtest_filter,\n"
         "[bench] or raise --profile-test-timeout. Aborting.\n";
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

namespace {

/// The hung loop's run, its timing and its status checked: after the second
/// of --profile-test-timeout 1, the watchdog's message and exit status 2.
void expectHungLoopEnds(const std::vector<std::string>& args) {
  const auto START = std::chrono::steady_clock::now();
  const ProbeResult RUN = runHelper(args);
  const auto ELAPSED_MS = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - START)
                              .count();
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2);
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Hung", "perf"));
  EXPECT_EQ(RUN.output, "");
  EXPECT_GE(ELAPSED_MS, 1000) << "it ended before its timeout";
}

} // namespace

/**
 * @test A measured loop that does not return, under --profile perf with
 * --profile-test-timeout 1, in the single-threaded helper (the timer engine
 * where it is built): after the second, the watchdog writes its message
 * naming the case and the profiler and ends the program with exit status 2.
 */
TEST(PerfWatchdogTest, HungLoopEndsWithStatusTwo) {
  expectHungLoopEnds({"hang", "--profile", "perf", "--profile-test-timeout", "1"});
}

/** @test The same in a helper with a thread of its own: the watchdog's thread ends it */
TEST(PerfWatchdogTest, HungLoopOnTheThreadEndsWithStatusTwo) {
  expectHungLoopEnds({"--threaded", "hang", "--profile", "perf", "--profile-test-timeout", "1"});
}

/**
 * @test The same in the single-threaded helper with every signal blocked on
 * its thread when the loop arms the watchdog: the timer's signal could not
 * reach it, so the watchdog takes its thread, which ends the loop.
 */
TEST(PerfWatchdogTest, HungLoopWithTheSignalBlockedEndsWithStatusTwo) {
  expectHungLoopEnds({"hang-blocked", "--profile", "perf", "--profile-test-timeout", "1"});
}

/**
 * @test A later measured loop that does not return, once the watchdog's
 * thread has parked after a first case that returned, under --profile perf
 * with --profile-test-timeout 1: after the second, the watchdog ends the
 * program with exit status 2, naming the later case.
 */
TEST(PerfWatchdogTest, LaterHungLoopEndsWithStatusTwo) {
  const ProbeResult RUN =
      runHelper({"hang-later", "--profile", "perf", "--profile-test-timeout", "1"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.HungLater", "perf"));
  EXPECT_EQ(RUN.output.rfind("[Watchdog.Returns]", 0), 0U)
      << "the first case's line: " << RUN.output;
}

/** @test The same on the watchdog's thread, the later case armed once it has parked */
TEST(PerfWatchdogTest, LaterHungLoopOnTheThreadEndsWithStatusTwo) {
  const ProbeResult RUN =
      runThreaded({"hang-later", "--profile", "perf", "--profile-test-timeout", "1"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.HungLater", "perf"));
  EXPECT_EQ(RUN.output.rfind("[Watchdog.Returns]", 0), 0U)
      << "the first case's line: " << RUN.output;
}

/**
 * @test An arming after the watchdog's thread has parked (the helper with a
 * thread of its own): armed for 10 s, disarmed once the thread waits for that
 * deadline, armed for 0.1 s once the thread has parked. It fires, naming the
 * later case.
 */
TEST(PerfWatchdogTest, ArmingAfterTheThreadParkedFires) {
  const ProbeResult RUN = runThreaded({"later"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Later", "perf"));
}

/**
 * @test The same on the timer (the single-threaded helper): armed for 10 s,
 * disarmed, armed for 0.1 s; it fires, naming the later case.
 */
TEST(PerfWatchdogTest, ArmingAfterADisarmFires) {
  const ProbeResult RUN = runHelper({"later"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Later", "perf"));
}

/**
 * @test A re-arm that shortens the deadline the thread is waiting for: armed
 * for 10 s, then for 0.1 s. It fires at the new deadline, naming the new
 * case, before the program's own 5 s wait ends, half the first deadline.
 */
TEST(PerfWatchdogTest, ShorterRearmFiresAtTheNewDeadline) {
  const ProbeResult RUN = runHelper({"shorten"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Shortened", "perf"));
}

/** @test The same on the watchdog's thread, the re-arm made once it waits for the 10 s */
TEST(PerfWatchdogTest, ShorterRearmOnTheThreadFiresAtTheNewDeadline) {
  const ProbeResult RUN = runThreaded({"shorten"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Shortened", "perf"));
}

/** @test Reports the test and the profiler by name, in order, then exits with status 2 */
TEST(PerfWatchdogTest, TimeoutNamesTestAndToolThenExits) {
  const ProbeResult RUN = runHelper({"names", "Suite.SlowCase", "callgrind"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2);
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Suite.SlowCase", "callgrind"));
}

/** @test Substitutes a placeholder for a missing name instead of reading a null pointer */
TEST(PerfWatchdogTest, NullNamesBecomePlaceholders) {
  const ProbeResult RUN = runHelper({"null-names"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2);
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("?", "?"));
}

/**
 * @test A watchdog disarmed before its deadline does not fire, leaves no timer
 * running (its signal, blocked past the deadline, is not pending), and a
 * re-arm replaces the deadline: the program runs past both 0.1 s deadlines and
 * exits 0.
 */
TEST(PerfWatchdogTest, DisarmAndRearmReplaceTheDeadline) {
  const ProbeResult RUN = runHelper({"disarmed"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "still running\n");
  EXPECT_EQ(RUN.errorOutput, "");
}

/**
 * @test A child of fork(), whose parent's timer is not inherited, arms a
 * watchdog of its own: on the timer where it is built, starting no thread
 * (on the thread under the thread sanitizer), and leaving alone a timer the
 * child made itself (it has the handle the parent's had); it fires in the
 * child, naming the child's case, and the child exits with status 2.
 */
TEST(PerfWatchdogTest, ForkedChildArmsItsOwnWatchdog) {
  const ProbeResult RUN = runHelper({"fork"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << "the child's exit status, as the helper passes it on\n"
                             << RUN.errorOutput;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Child", "perf"));
  if (THREAD_SANITIZER) {
    // Its runtime starts a thread in the child, which glibc then counts
    // multi-threaded: the child takes the watchdog's thread.
    EXPECT_NE(RUN.output.find(", own timer kept\n"), std::string::npos) << RUN.output;
  } else {
    EXPECT_EQ(RUN.output, VERNIER_WATCHDOG_TIMER ? "child threads=1, own timer kept\n"
                                                 : "child threads=2, own timer not made\n");
  }
}

/**
 * @test The same in a helper with a thread of its own, whose watchdog thread
 * is not copied into the child either: the child, whose C library still counts
 * it multi-threaded, starts the watchdog's thread.
 */
TEST(PerfWatchdogTest, ForkedChildOfAThreadedProcessArmsItsOwnWatchdog) {
  if (THREAD_SANITIZER) {
    GTEST_SKIP() << "the thread sanitizer ends a child of a multi-threaded fork() that starts a "
                    "thread (die_after_fork), as this child may";
  }
  const ProbeResult RUN = runThreaded({"fork"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << "the child's exit status, as the helper passes it on\n"
                             << RUN.errorOutput;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Child", "perf"));
  EXPECT_EQ(RUN.output, VERNIER_WATCHDOG_TIMER ? "child threads=2, own timer kept\n"
                                               : "child threads=2, own timer not made\n");
}

/**
 * @test The watchdog's thread takes no signal: SIGUSR1, sent to the process
 * while the program's own thread blocks it, stays pending for that thread
 * instead of ending the process through the watchdog's. The first arm(),
 * which starts that thread with every signal blocked, leaves the arming
 * thread's own mask as it was.
 */
TEST(PerfWatchdogTest, ItsThreadTakesNoSignal) {
  const ProbeResult RUN = runThreaded({"signals"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "SIGUSR1 waited for the thread that blocks it\n");
}

/**
 * @test Arming and disarming the timer engine (the single-threaded helper)
 * leave the arming thread's signal mask as it was.
 */
TEST(PerfWatchdogTest, TimerArmingLeavesTheSignalMaskAlone) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"signals"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "SIGUSR1 waited for the thread that blocks it\n");
}

/**
 * @test The timer engine takes a real-time signal whose disposition is the
 * default, never SIGALRM or a signal glibc reserves (below SIGRTMIN): with
 * SIGRTMIN + 4 given a handler of the program's own, an arming takes
 * SIGRTMIN + 5, the only signal the process catches since, and leaves the
 * program's handler in place.
 */
TEST(PerfWatchdogTest, TimerTakesAFreeRealTimeSignal) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"first-signal"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "taken: SIGRTMIN+5; SIGRTMIN+4 kept\n");
}

/**
 * @test The timer engine's signal, given a handler of the program's own
 * between two armings, is not used again: the second arming takes another
 * signal or the thread, the program's handler never runs, and the watchdog
 * ends the run, naming the case armed.
 */
TEST(PerfWatchdogTest, SignalTakenOverBetweenArmingsStillEndsTheRun) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"takeover"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.AfterTakeover", "perf"));
  EXPECT_EQ(RUN.output.find("the program's handler ran"), std::string::npos) << RUN.output;
}

/**
 * @test The timer engine's signal from anywhere but its own timer, sent by
 * kill(), or by sigqueue() with the value its timer carries, or raised by a
 * timer of the program's own, does not end an armed run: the program disarms
 * and exits 0.
 */
TEST(PerfWatchdogTest, ItsSignalFromElsewhereEndsNothing) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"foreign"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "still running\n");
  EXPECT_EQ(RUN.errorOutput, "");
}

/**
 * @test A process that starts a thread between two armings: the first armed
 * and disarmed single-threaded, the second armed for 0.1 s with the thread
 * running. The watchdog fires, naming the second case.
 */
TEST(PerfWatchdogTest, BecomingThreadedBetweenArmingsStillFires) {
  const ProbeResult RUN = runHelper({"switch", "between"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Switched", "perf"));
}

/**
 * @test A process that starts a thread while armed: armed for 0.1 s
 * single-threaded, then the thread started. The watchdog fires all the same.
 */
TEST(PerfWatchdogTest, BecomingThreadedWhileArmedStillFires) {
  const ProbeResult RUN = runHelper({"switch", "during"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Switched", "perf"));
}

/**
 * @test A re-arm after the process started a thread replaces the deadline of
 * the arming before it: armed for 0.1 s single-threaded, the thread started,
 * armed again for 10 s (on the watchdog's thread). Nothing fires at 0.1 s; the
 * program disarms and exits 0.
 */
TEST(PerfWatchdogTest, BecomingThreadedThenRearmingReplacesTheDeadline) {
  const ProbeResult RUN = runHelper({"switch", "rearm"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "still running\n");
  EXPECT_EQ(RUN.errorOutput, "");
}

/**
 * @test A benchmark that starts no thread of its own, under --profile perf at
 * the default watchdog (armed for the measured loop): the process has one
 * thread and glibc counts it single-threaded when the body runs and after
 * the case.
 */
TEST(PerfWatchdogTest, SingleThreadedRunStaysSingleThreaded) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"threads", "--profile", "perf"});
  ASSERT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  ThreadsLine line;
  ASSERT_TRUE(readThreads(RUN.output, line)) << RUN.output;
  EXPECT_EQ(line.threads, 1) << RUN.output;
  EXPECT_EQ(line.single, 1) << RUN.output;
  EXPECT_EQ(line.afterThreads, 1) << RUN.output;
  EXPECT_EQ(line.afterSingle, 1) << RUN.output;
}

/**
 * @test A benchmark with a thread of its own, under --profile perf: at the
 * default watchdog it gains exactly one thread, the watchdog's, which stays
 * after the case; with --profile-test-timeout 0 it gains none.
 */
TEST(PerfWatchdogTest, ThreadedRunGainsTheWatchdogsThread) {
  const ProbeResult ON = runThreaded({"threads", "--profile", "perf"});
  const ProbeResult OFF =
      runThreaded({"threads", "--profile", "perf", "--profile-test-timeout", "0"});
  ASSERT_TRUE(ON.succeeded()) << ON.describe() << "\n" << ON.output << ON.errorOutput;
  ASSERT_TRUE(OFF.succeeded()) << OFF.describe() << "\n" << OFF.output << OFF.errorOutput;
  ThreadsLine on;
  ThreadsLine off;
  ASSERT_TRUE(readThreads(ON.output, on)) << ON.output;
  ASSERT_TRUE(readThreads(OFF.output, off)) << OFF.output;
  EXPECT_EQ(on.threads, off.threads + 1) << ON.output << OFF.output;
  EXPECT_EQ(on.afterThreads, off.afterThreads + 1) << ON.output << OFF.output;
  EXPECT_EQ(off.threads, off.afterThreads) << OFF.output;
}

/**
 * @test A watchdog thread that cannot be started (RLIMIT_NPROC 0) ends the
 * measurement as an exception from the body does: the before hook ran, the
 * body did not, the abort hook ran in place of the after hook, and the
 * exception reached the caller.
 */
TEST(PerfWatchdogTest, ThreadThatCannotStartEndsTheMeasurement) {
  const ProbeResult RUN = runHelper({"no-thread", "--profile", "perf"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  if (RUN.exitCode == 77) {
    GTEST_SKIP() << RUN.output;
  }
  EXPECT_EQ(RUN.exitCode, 3) << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "before hook\nabort hook\nmeasured() threw std::system_error: EAGAIN\n");
}

/**
 * @test A thread that could not start is tried again: a second case whose
 * thread cannot start either ends as the first did, through the abort hook;
 * once threads can start, the next arming starts the thread, and it fires,
 * naming the case armed.
 */
TEST(PerfWatchdogTest, ThreadThatCannotStartIsStartedByALaterArming) {
  const ProbeResult RUN = runHelper({"no-thread-retry", "--profile", "perf"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  if (RUN.exitCode == 77) {
    GTEST_SKIP() << RUN.output;
  }
  const std::string FAILED_CASE =
      "before hook\nabort hook\nmeasured() threw std::system_error: EAGAIN\n";
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.output, FAILED_CASE + FAILED_CASE);
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Retried", "perf"));
}

/**
 * @test Under --profile with a tool that starts no thread in the process,
 * callgrind (the helper run by valgrind as a wrap started by hand, its
 * profiler attached as a benchmark's is): a benchmark that starts no thread
 * gains none at the watchdog's default, when the body runs or after the case,
 * and glibc still counts it single-threaded; one with a thread of its own
 * gains exactly the watchdog's, against --profile-test-timeout 0.
 */
TEST(PerfWatchdogTest, CallgrindRunGainsAThreadOnlyWhenThreaded) {
  if (ADDRESS_SANITIZER || THREAD_SANITIZER) {
    GTEST_SKIP() << "valgrind does not run a sanitizer build";
  }
  const vernier::bench::ReadinessContext CONTEXT = vernier::bench::ReadinessContext::capture();
  const auto VALGRIND = vernier::bench::resolveExecutable("valgrind", CONTEXT);
  if (!VALGRIND || !VALGRIND->executable) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the helper under callgrind";
  }
  const auto UNDER_CALLGRIND = [&](bool threaded, bool watchdog, ThreadsLine& line) {
    std::vector<std::string> args = {VALGRIND->path, "--tool=callgrind", "--instr-atstart=no",
                                     "--callgrind-out-file=/dev/null", HELPER};
    if (threaded) {
      args.emplace_back("--threaded");
    }
    for (const char* arg : {"threads", "--profile", "callgrind"}) {
      args.emplace_back(arg);
    }
    if (!watchdog) {
      args.emplace_back("--profile-test-timeout");
      args.emplace_back("0");
    }
    const ProbeResult RUN = vernier::bench::runBoundedProbe(args, HELPER_BOUND_MS, CONTEXT,
                                                            vernier::bench::ProbeStreams::SEPARATE);
    EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
    EXPECT_TRUE(readThreads(RUN.output, line)) << RUN.output;
  };
  ThreadsLine plainOn;
  ThreadsLine plainOff;
  ThreadsLine threadedOn;
  ThreadsLine threadedOff;
  UNDER_CALLGRIND(false, true, plainOn);
  UNDER_CALLGRIND(false, false, plainOff);
  UNDER_CALLGRIND(true, true, threadedOn);
  UNDER_CALLGRIND(true, false, threadedOff);
  EXPECT_EQ(plainOff.threads, 1);
  EXPECT_EQ(plainOn.threads, VERNIER_WATCHDOG_TIMER ? 1 : 2) << "the threads at the default";
  EXPECT_EQ(plainOn.afterThreads, plainOn.threads);
  if (VERNIER_WATCHDOG_TIMER) {
    EXPECT_EQ(plainOn.single, 1) << "glibc counted the process multi-threaded";
    EXPECT_EQ(plainOn.afterSingle, 1);
  }
  EXPECT_EQ(threadedOn.threads, threadedOff.threads + 1) << "not exactly the watchdog's thread";
  EXPECT_EQ(threadedOn.afterThreads, threadedOff.afterThreads + 1);
}

namespace {

/// One run of the helper under helgrind with @p args, any error helgrind
/// reports giving exit status 97, a status the helper does not give. With
/// @p fair, valgrind's --fair-sched=yes: valgrind runs one thread at a time,
/// and by default a thread that computes keeps the others from running, the
/// watchdog's among them, until it makes a blocking call.
ProbeResult underHelgrind(const std::string& valgrind, bool fair,
                          const std::vector<std::string>& args) {
  std::vector<std::string> command = {valgrind, "--tool=helgrind", "--error-exitcode=97"};
  if (fair) {
    command.emplace_back("--fair-sched=yes");
  }
  command.emplace_back(HELPER);
  command.insert(command.end(), args.begin(), args.end());
  return vernier::bench::runBoundedProbe(command, HELPER_BOUND_MS,
                                         vernier::bench::ReadinessContext::capture(),
                                         vernier::bench::ProbeStreams::SEPARATE);
}

} // namespace

/**
 * @test Under helgrind, the watchdog gives no report in either engine: the
 * helper, run by valgrind's helgrind with an exit status for any error,
 * measures two cases under --profile helgrind (armed, a loop measured,
 * disarmed, armed again), single-threaded and with a thread of its own; and a
 * hung loop is ended by the watchdog in each while it still spins (its repeat
 * never completes, so no progress line), the process ending with the
 * watchdog's status 2. helgrind counts no error in any of the four. The
 * single-threaded runs take valgrind's default scheduling; the threaded ones
 * --fair-sched=yes, without which the watchdog's thread does not run until
 * the hung loop's thread stops computing.
 */
TEST(PerfWatchdogTest, HelgrindReportsNothingOfTheWatchdog) {
  if (ADDRESS_SANITIZER || THREAD_SANITIZER) {
    GTEST_SKIP() << "valgrind does not run a sanitizer build";
  }
  const vernier::bench::ReadinessContext CONTEXT = vernier::bench::ReadinessContext::capture();
  const auto VALGRIND = vernier::bench::resolveExecutable("valgrind", CONTEXT);
  if (!VALGRIND || !VALGRIND->executable) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the helper under helgrind";
  }
  for (const bool THREADED : {false, true}) {
    std::vector<std::string> prefix;
    if (THREADED) {
      prefix.emplace_back("--threaded");
    }
    std::vector<std::string> rearmArgs = prefix;
    for (const char* arg : {"rearm", "--profile", "helgrind"}) {
      rearmArgs.emplace_back(arg);
    }
    std::vector<std::string> hangArgs = prefix;
    for (const char* arg : {"hang", "--profile", "helgrind", "--profile-test-timeout", "1"}) {
      hangArgs.emplace_back(arg);
    }
    const ProbeResult REARM = underHelgrind(VALGRIND->path, THREADED, rearmArgs);
    ASSERT_TRUE(REARM.exited) << REARM.describe() << "\n" << REARM.output << REARM.errorOutput;
    EXPECT_EQ(REARM.exitCode, 0) << (THREADED ? "threaded: " : "single: ") << REARM.output
                                 << REARM.errorOutput;
    EXPECT_NE(REARM.output.find("measured twice\n"), std::string::npos) << REARM.output;
    EXPECT_NE(REARM.errorOutput.find("ERROR SUMMARY: 0 errors from 0 contexts"), std::string::npos)
        << REARM.errorOutput;
    const ProbeResult HANG = underHelgrind(VALGRIND->path, THREADED, hangArgs);
    ASSERT_TRUE(HANG.exited) << HANG.describe() << "\n" << HANG.output << HANG.errorOutput;
    EXPECT_EQ(HANG.exitCode, 2) << (THREADED ? "threaded: " : "single: ") << HANG.output
                                << HANG.errorOutput;
    EXPECT_EQ(HANG.errorOutput.find("[Watchdog.Hung] repeat"), std::string::npos)
        << "the hung loop's repeat completed before the watchdog fired\n"
        << HANG.errorOutput;
    EXPECT_NE(HANG.errorOutput.find("ERROR SUMMARY: 0 errors from 0 contexts"), std::string::npos)
        << HANG.errorOutput;
    EXPECT_NE(HANG.errorOutput.find(timeoutMessage("Watchdog.Hung", "helgrind")), std::string::npos)
        << HANG.errorOutput;
  }
}

namespace {

/// How many times the program's own SIGALRM handler ran.
std::atomic<int> g_programAlarms{0};

/// The program's own SIGALRM handler.
void programAlarm(int /*sig*/) { g_programAlarms.fetch_add(1); }

} // namespace

/**
 * @test A program's own SIGALRM handler and pending alarm survive a measured
 * loop under --profile: inside the loop its timer is still the one running,
 * and afterwards its handler is installed and its alarm still pending.
 */
TEST(PerfWatchdogTest, LeavesTheProgramsAlarmAndHandler) {
  struct sigaction program{};
  program.sa_handler = programAlarm;
  ::sigemptyset(&program.sa_mask);
  struct sigaction previous{};
  ASSERT_EQ(::sigaction(SIGALRM, &program, &previous), 0);
  ::alarm(100);

  vernier::bench::PerfConfig cfg;
  cfg.cycles = 1;
  cfg.repeats = 1;
  cfg.profileTool = "perf"; // a requested profile arms the watchdog; no profiler is made here
  cfg.profileTestTimeoutSecs = 30;
  itimerval inside{};
  bool armed = false;
  {
    vernier::bench::PerfCase perf{"Watchdog.ProgramAlarm", cfg};
    perf.measured([&] {
      ::getitimer(ITIMER_REAL, &inside);
      armed = perf_watchdog::g_armed.load();
    });
  }
  // Read and cancel the alarm at once, and hand the signal back as found: a
  // pending alarm would end this process.
  const unsigned int PENDING = ::alarm(0);
  struct sigaction after{};
  ASSERT_EQ(::sigaction(SIGALRM, &previous, &after), 0);
  EXPECT_TRUE(armed) << "the loop ran without the watchdog";
  EXPECT_GT(inside.it_value.tv_sec, 98) << "inside the loop, the timer was not the program's";
  EXPECT_EQ(after.sa_handler, &programAlarm) << "the program's SIGALRM handler was replaced";
  EXPECT_GT(PENDING, 98U) << "the program's alarm was cancelled";
  EXPECT_EQ(g_programAlarms.load(), 0) << "the program's alarm went off early";
}

namespace {

/** @brief Whether the watchdog was armed while a measured loop of a case parsed from @p flags ran.
 */
bool armedDuringMeasure(std::vector<const char*> flags) {
  std::vector<std::string> storage{"prog"};
  storage.insert(storage.end(), flags.begin(), flags.end());
  std::vector<char*> argv;
  for (std::string& arg : storage) {
    argv.push_back(arg.data());
  }
  int argc = static_cast<int>(argv.size());
  vernier::bench::PerfConfig cfg;
  vernier::bench::parsePerfFlags(cfg, &argc, argv.data());
  cfg.cycles = 1;
  cfg.repeats = 1;
  vernier::bench::PerfCase perf{"Watchdog.Case", cfg};
  bool armed = false;
  perf.measured([&] { armed = perf_watchdog::g_armed.load(); });
  EXPECT_FALSE(perf_watchdog::g_armed.load()) << "the loop left the watchdog armed";
  return armed;
}

} // namespace

/**
 * @test An explicit --profile-test-timeout 0 under --profile arms nothing; the
 * omitted flag (300 s) and a given one arm the watchdog for the measured loop.
 */
TEST(PerfWatchdogTest, ExplicitZeroDoesNotArm) {
  EXPECT_FALSE(armedDuringMeasure({"--profile", "perf", "--profile-test-timeout", "0"}));
  EXPECT_TRUE(armedDuringMeasure({"--profile", "perf"}));
  EXPECT_TRUE(armedDuringMeasure({"--profile", "perf", "--profile-test-timeout", "30"}));
  EXPECT_FALSE(armedDuringMeasure({"--profile-test-timeout", "30"})) << "armed without --profile";
}

namespace {

/** @brief The workload's own exception type and value, to see both arrive. */
struct WorkloadError {
  int code;
};

} // namespace

/**
 * @test A measured callback that throws: its exception reaches the caller
 * unchanged, the watchdog is off, the after hook does not run and no row is
 * published; the next case of the process measures, publishes and turns its
 * watchdog off as usual.
 */
TEST(PerfWatchdogTest, ThrowingCallbackTurnsTheWatchdogOff) {
  using vernier::bench::PerfCase;
  using vernier::bench::PerfRegistry;

  vernier::bench::PerfConfig cfg;
  cfg.cycles = 1;
  cfg.repeats = 3;
  cfg.profileTool = "perf"; // a requested profile arms the watchdog; no profiler is made here
  cfg.profileTestTimeoutSecs = 30;
  (void)PerfRegistry::instance().take();
  const std::size_t PUBLISHED = PerfRegistry::instance().summary().size();

  int befores = 0;
  int afters = 0;
  int code = 0;
  {
    PerfCase perf{"Watchdog.Throws", cfg};
    perf.setBeforeMeasureHook([&](const PerfCase&) { ++befores; });
    perf.setAfterMeasureHook([&](const PerfCase&, const vernier::bench::Stats&) { ++afters; });
    try {
      perf.measured([] { throw WorkloadError{7}; });
      ADD_FAILURE() << "the exception did not reach the caller";
    } catch (const WorkloadError& e) {
      code = e.code;
    }
  }
  const bool ARMED = perf_watchdog::g_armed.load();
  perf_watchdog::disarm(); // left armed, it would end this process in 30 s
  EXPECT_FALSE(ARMED) << "the watchdog still reports itself armed";
  EXPECT_EQ(code, 7) << "the workload's exception arrived changed";
  EXPECT_EQ(befores, 1);
  EXPECT_EQ(afters, 0) << "the after hook ran for a measurement that ended by an exception";
  EXPECT_FALSE(PerfRegistry::instance().take().has_value()) << "a row was published";
  EXPECT_EQ(PerfRegistry::instance().summary().size(), PUBLISHED);

  bool armedInside = false;
  {
    PerfCase next{"Watchdog.Next", cfg};
    next.setAfterMeasureHook([&](const PerfCase&, const vernier::bench::Stats&) { ++afters; });
    next.measured([&] { armedInside = perf_watchdog::g_armed.load(); });
  }
  EXPECT_TRUE(armedInside) << "the next case measured without its watchdog";
  EXPECT_FALSE(perf_watchdog::g_armed.load());
  EXPECT_EQ(afters, 1) << "the next case's after hook";
  const auto ROW = PerfRegistry::instance().take();
  ASSERT_TRUE(ROW.has_value()) << "the next case published no row";
  EXPECT_EQ(ROW->testName, "Watchdog.Next");
}
