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
 *  - The helper defines timer_settime() itself, recording each call and
 *    passing it on to the C library's; one case reads the order of the timer
 *    engine's calls from it.
 *  - The cases in this process arm through a measured loop for 30 s or more
 *    and read the watchdog's flag; none waits for it.
 *  - The valgrind cases run the helper through the line `bench run` writes:
 *    one four times under callgrind, under a second each; four once each,
 *    a hung loop with a thread of its own under callgrind or memcheck, about
 *    two seconds each; one four times under helgrind, about a second and a
 *    half each. They skip where valgrind is not installed and in an address
 *    or thread sanitizer build, which valgrind cannot run.
 */

#include "src/bench/inc/PerfHarness.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/ProfilerCallgrind.hpp"
#include "src/bench/inc/ProfilerHelgrind.hpp"
#include "src/bench/inc/ProfilerMemcheck.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

#include <gtest/gtest.h>

#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
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
 * @test The timer engine's timer_settime() calls, as the helper records them
 * ("set" a deadline, "stop" a zero value): a re-arm before a disarm stops the
 * timer before it sets the new deadline (armed for 10 s, armed again, disarmed:
 * set, stop, set, stop), and an arming after a disarm stops nothing first
 * (armed, disarmed, armed, disarmed: set, stop, set, stop).
 */
TEST(PerfWatchdogTest, RearmBeforeADisarmStopsTheTimerFirst) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"settime"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "armed twice: set stop set stop\n"
                        "disarmed between: set stop set stop\n");
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
 * @test A timer signal left pending by an overrun does not end a later loop:
 * the timer armed for 0.1 s, its signal blocked and the deadline overrun, so
 * the signal stays pending; disarmed; armed for 10 s with the signal still
 * blocked (on the watchdog's thread), and the signal unblocked inside that
 * loop. The pending signal arrives and ends nothing; the program disarms and
 * exits 0.
 */
TEST(PerfWatchdogTest, StaleTimerSignalEndsNoLaterLoop) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"stale", "thread"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "still running\n");
  EXPECT_EQ(RUN.errorOutput, "");
}

/**
 * @test The same, the later loop armed for 0.1 s on the thread and hung: the
 * watchdog ends it with exit status 2, naming the later case, not the one
 * whose signal was left pending.
 */
TEST(PerfWatchdogTest, LoopHungAfterAStaleTimerSignalEndsUnderItsOwnName) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"stale", "hang"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Second", "perf"));
}

/**
 * @test The same overrun, the signal unblocked before the later arming, which
 * takes the timer again for 10 s: the pending signal arrives while nothing is
 * armed and ends nothing; the program disarms and exits 0.
 */
TEST(PerfWatchdogTest, StaleTimerSignalBeforeATimerArmingEndsNothing) {
  if (const std::string WHY = withoutTheTimer(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const ProbeResult RUN = runHelper({"stale", "timer"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "still running\n");
  EXPECT_EQ(RUN.errorOutput, "");
}

/**
 * @test A process that starts a thread between two armings moves to the
 * watchdog's thread: armed for 0.1 s single-threaded (the timer), the thread
 * started, armed again for 10 s (the watchdog's thread, the timer stopped):
 * the first deadline passes and nothing fires; armed again for 0.1 s: the
 * watchdog fires, naming that case.
 */
TEST(PerfWatchdogTest, BecomingThreadedBetweenArmingsMovesToTheThread) {
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

namespace {

/// valgrind's path for a case that runs the helper under it; empty, with
/// @p why saying so, where valgrind is not installed or cannot run this build.
std::string valgrindForTheHelper(std::string& why) {
  if (ADDRESS_SANITIZER || THREAD_SANITIZER) {
    why = "valgrind does not run a sanitizer build";
    return {};
  }
  const auto VALGRIND =
      vernier::bench::resolveExecutable("valgrind", vernier::bench::ReadinessContext::capture());
  if (!VALGRIND || !VALGRIND->executable) {
    why = "valgrind is not installed; this test runs the helper under it";
    return {};
  }
  return VALGRIND->path;
}

/// The mode valgrind's @p tool runs in for a request without --profile-args,
/// as its backend reads it.
vernier::bench::valgrind_tool::ValgrindMode modeOf(const std::string& tool) {
  vernier::bench::valgrind_tool::ValgrindMode mode;
  const auto REFUSED = tool == "callgrind"  ? vernier::bench::parseCallgrindMode("", mode)
                       : tool == "memcheck" ? vernier::bench::parseMemcheckMode("", mode)
                                            : vernier::bench::parseHelgrindMode("", mode);
  EXPECT_FALSE(REFUSED.has_value()) << tool;
  return mode;
}

/// A directory of its own for one wrapped run's output, removed with it.
class WrapDir {
public:
  WrapDir() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "vernier-watchdog-wrap-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    }
  }
  WrapDir(const WrapDir&) = delete;
  WrapDir& operator=(const WrapDir&) = delete;
  ~WrapDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  const std::string& path() const { return path_; }

private:
  std::string path_;
};

/// The text of the file at @p path; empty when it cannot be read.
std::string fileText(const std::string& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// The helper with @p args, run by valgrind (@p valgrind) through the line
/// `bench run` writes for @p tool, writing into @p dir: wrapArguments(), which
/// the shared route table holds equal to the runner's route.
ProbeResult underWrap(const std::string& valgrind, const std::string& tool, const std::string& dir,
                      const std::vector<std::string>& args) {
  std::vector<std::string> command{valgrind};
  const std::vector<std::string> WRAP =
      vernier::bench::valgrind_tool::wrapArguments(tool, modeOf(tool), dir);
  command.insert(command.end(), WRAP.begin(), WRAP.end());
  command.emplace_back(HELPER);
  command.insert(command.end(), args.begin(), args.end());
  return vernier::bench::runBoundedProbe(command, HELPER_BOUND_MS,
                                         vernier::bench::ReadinessContext::capture(),
                                         vernier::bench::ProbeStreams::SEPARATE);
}

} // namespace

/**
 * @test Under --profile with a tool that starts no thread in the process,
 * callgrind (the helper run through the line `bench run` writes, its profiler
 * attached as a benchmark's is): a benchmark that starts no thread gains none
 * at the watchdog's default, when the body runs or after the case, and glibc
 * still counts it single-threaded; one with a thread of its own gains exactly
 * the watchdog's, against --profile-test-timeout 0.
 */
TEST(PerfWatchdogTest, CallgrindRunGainsAThreadOnlyWhenThreaded) {
  std::string why;
  const std::string VALGRIND = valgrindForTheHelper(why);
  if (VALGRIND.empty()) {
    GTEST_SKIP() << why;
  }
  const auto UNDER_CALLGRIND = [&](bool threaded, bool watchdog, ThreadsLine& line) {
    std::vector<std::string> args;
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
    const WrapDir DIR;
    const ProbeResult RUN = underWrap(VALGRIND, "callgrind", DIR.path(), args);
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

/// How long after its loop started a hung case under a valgrind tool may end:
/// the second of --profile-test-timeout 1, and one more for valgrind to run
/// the watchdog and end the process.
constexpr std::int64_t ON_TIME_NS = 2000000000;

/// CLOCK_MONOTONIC now, in nanoseconds: the clock the helper prints its
/// loop's start on.
std::int64_t monotonicNs() {
  timespec now{};
  ::clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::int64_t>(now.tv_sec) * 1000000000 + now.tv_nsec;
}

/// The helper with a thread of its own, its case hung in @p mode under
/// --profile <tool> --profile-test-timeout 1, run by @p tool through the line
/// `bench run` writes: it ends with the watchdog's status 2 and message within
/// ON_TIME_NS of the loop's start.
void expectHungUnderToolEndsOnTime(const std::string& valgrind, const std::string& tool,
                                   const char* mode) {
  const WrapDir DIR;
  const ProbeResult RUN =
      underWrap(valgrind, tool, DIR.path(),
                {"--threaded", mode, "--profile", tool, "--profile-test-timeout", "1"});
  const std::int64_t ENDED = monotonicNs();
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << RUN.output << RUN.errorOutput;
  EXPECT_NE(RUN.errorOutput.find(timeoutMessage("Watchdog.Hung", tool)), std::string::npos)
      << RUN.errorOutput;
  const std::size_t AT = RUN.output.find("loop started at ");
  ASSERT_NE(AT, std::string::npos) << RUN.output;
  const std::int64_t STARTED = std::stoll(RUN.output.substr(AT + 16));
  EXPECT_LE(ENDED - STARTED, ON_TIME_NS)
      << "it ended " << (ENDED - STARTED) / 1000000 << " ms after its loop started";
}

} // namespace

/**
 * @test Under callgrind, through the line `bench run` writes: a hung loop in
 * a process with a thread of its own (idle), so on the watchdog's thread, is
 * ended at its timeout, status 2 within two seconds of the loop's start.
 * valgrind runs one thread at a time; with its default scheduler a thread
 * that computes keeps the others from running, the watchdog's among them,
 * and the wrap asks for valgrind's fair scheduler.
 */
TEST(PerfWatchdogTest, HungThreadedLoopUnderCallgrindEndsOnTime) {
  std::string why;
  const std::string VALGRIND = valgrindForTheHelper(why);
  if (VALGRIND.empty()) {
    GTEST_SKIP() << why;
  }
  expectHungUnderToolEndsOnTime(VALGRIND, "callgrind", "hang-timed");
}

/**
 * @test Under callgrind, through the line `bench run` writes: a measured loop
 * waiting to join a thread that computes, in a process with a thread of its
 * own, is ended at its timeout, status 2 within two seconds of the loop's
 * start.
 */
TEST(PerfWatchdogTest, LoopJoiningABusyThreadUnderCallgrindEndsOnTime) {
  std::string why;
  const std::string VALGRIND = valgrindForTheHelper(why);
  if (VALGRIND.empty()) {
    GTEST_SKIP() << why;
  }
  expectHungUnderToolEndsOnTime(VALGRIND, "callgrind", "hang-joined");
}

/**
 * @test Under memcheck, through the line `bench run` writes: a hung loop in a
 * process with a thread of its own is ended at its timeout, status 2 within
 * two seconds of the loop's start.
 */
TEST(PerfWatchdogTest, HungThreadedLoopUnderMemcheckEndsOnTime) {
  std::string why;
  const std::string VALGRIND = valgrindForTheHelper(why);
  if (VALGRIND.empty()) {
    GTEST_SKIP() << why;
  }
  expectHungUnderToolEndsOnTime(VALGRIND, "memcheck", "hang-timed");
}

/**
 * @test Under memcheck, through the line `bench run` writes: a measured loop
 * waiting to join a thread that computes is ended at its timeout, status 2
 * within two seconds of the loop's start.
 */
TEST(PerfWatchdogTest, LoopJoiningABusyThreadUnderMemcheckEndsOnTime) {
  std::string why;
  const std::string VALGRIND = valgrindForTheHelper(why);
  if (VALGRIND.empty()) {
    GTEST_SKIP() << why;
  }
  expectHungUnderToolEndsOnTime(VALGRIND, "memcheck", "hang-joined");
}

/**
 * @test Under helgrind, the watchdog gives no report in either engine: the
 * helper, run by helgrind through the line `bench run` writes, measures two
 * cases under --profile helgrind (armed, a loop measured, disarmed, armed
 * again), single-threaded and with a thread of its own; and a hung loop is
 * ended by the watchdog in each while it still spins (its repeat never
 * completes, so no progress line), the process ending with the watchdog's
 * status 2. helgrind's log counts no error in any of the four.
 */
TEST(PerfWatchdogTest, HelgrindReportsNothingOfTheWatchdog) {
  std::string why;
  const std::string VALGRIND = valgrindForTheHelper(why);
  if (VALGRIND.empty()) {
    GTEST_SKIP() << why;
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
    const WrapDir REARM_DIR;
    const ProbeResult REARM = underWrap(VALGRIND, "helgrind", REARM_DIR.path(), rearmArgs);
    const std::string REARM_LOG = fileText(REARM_DIR.path() + "/helgrind.log");
    ASSERT_TRUE(REARM.exited) << REARM.describe() << "\n" << REARM.output << REARM.errorOutput;
    EXPECT_EQ(REARM.exitCode, 0) << (THREADED ? "threaded: " : "single: ") << REARM.output
                                 << REARM.errorOutput;
    EXPECT_NE(REARM.output.find("measured twice\n"), std::string::npos) << REARM.output;
    EXPECT_NE(REARM_LOG.find("ERROR SUMMARY: 0 errors from 0 contexts"), std::string::npos)
        << REARM_LOG;
    const WrapDir HANG_DIR;
    const ProbeResult HANG = underWrap(VALGRIND, "helgrind", HANG_DIR.path(), hangArgs);
    const std::string HANG_LOG = fileText(HANG_DIR.path() + "/helgrind.log");
    ASSERT_TRUE(HANG.exited) << HANG.describe() << "\n" << HANG.output << HANG.errorOutput;
    EXPECT_EQ(HANG.exitCode, 2) << (THREADED ? "threaded: " : "single: ") << HANG.output
                                << HANG.errorOutput;
    EXPECT_EQ(HANG.errorOutput.find("[Watchdog.Hung] repeat"), std::string::npos)
        << "the hung loop's repeat completed before the watchdog fired\n"
        << HANG.errorOutput;
    EXPECT_NE(HANG_LOG.find("ERROR SUMMARY: 0 errors from 0 contexts"), std::string::npos)
        << HANG_LOG;
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
