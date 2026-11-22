/**
 * @file PerfWatchdog_uTest.cpp
 * @brief Unit tests for vernier::bench::perf_watchdog.
 *
 * Notes:
 *  - The watchdog ends the process it guards, so each case in which it fires,
 *    or must not, runs WatchdogHelper as a child and reads how it ended.
 *  - Those children wait on real timers: 0.1 s for a direct arm, 0.6 s for
 *    the disarm and re-arm case, and the 1 s of --profile-test-timeout 1 for
 *    the two hung loops.
 *  - The cases of a later arming make the helper wait until the watchdog's
 *    thread is parked, or waiting for a deadline, before it arms again: an
 *    arming the thread does not see at once is then never seen.
 *  - The cases in this process arm through a measured loop for 30 s or more
 *    and read the watchdog's flag; none waits for it.
 *  - One case runs the helper twice under valgrind's callgrind, about half a
 *    second each, and one once under helgrind; they skip where valgrind is
 *    not installed and in an address or thread sanitizer build, which
 *    valgrind cannot run.
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
/// fork() that starts a thread (its die_after_fork).
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

/**
 * @test A measured loop that does not return, under --profile perf with
 * --profile-test-timeout 1: after the second, the watchdog writes its message
 * naming the case and the profiler and ends the program with exit status 2.
 */
TEST(PerfWatchdogTest, HungLoopEndsWithStatusTwo) {
  const auto START = std::chrono::steady_clock::now();
  const ProbeResult RUN = runHelper({"hang", "--profile", "perf", "--profile-test-timeout", "1"});
  const auto ELAPSED_MS = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - START)
                              .count();
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2);
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Hung", "perf"));
  EXPECT_EQ(RUN.output, "");
  EXPECT_GE(ELAPSED_MS, 1000) << "it ended before its timeout";
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

/**
 * @test An arming after the watchdog's thread has parked: armed for 10 s,
 * disarmed once the thread waits for that deadline, armed for 0.1 s once the
 * thread has parked. It fires, naming the later case.
 */
TEST(PerfWatchdogTest, ArmingAfterTheThreadParkedFires) {
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
 * @test A watchdog disarmed before its deadline does not fire, and a re-arm
 * replaces the deadline the thread is waiting for: the program runs past
 * both 0.1 s deadlines and exits 0.
 */
TEST(PerfWatchdogTest, DisarmAndRearmReplaceTheDeadline) {
  const ProbeResult RUN = runHelper({"disarmed"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "still running\n");
  EXPECT_EQ(RUN.errorOutput, "");
}

/**
 * @test A child of fork(), whose parent's watchdog thread is not copied into
 * it, arms a watchdog of its own: it fires in the child, naming the child's
 * case, and the child exits with status 2.
 */
TEST(PerfWatchdogTest, ForkedChildArmsItsOwnWatchdog) {
  if (THREAD_SANITIZER) {
    GTEST_SKIP() << "the thread sanitizer ends a child of a multi-threaded fork() that starts a "
                    "thread (die_after_fork), as this child must";
  }
  const ProbeResult RUN = runHelper({"fork"});
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 2) << "the child's exit status, as the helper passes it on\n"
                             << RUN.errorOutput;
  EXPECT_EQ(RUN.errorOutput, timeoutMessage("Watchdog.Child", "perf"));
}

/**
 * @test The watchdog's thread takes no signal: SIGUSR1, sent to the process
 * while the program's own thread blocks it, stays pending for that thread
 * instead of ending the process through the watchdog's. The first arm(),
 * which starts that thread with every signal blocked, leaves the arming
 * thread's own mask as it was.
 */
TEST(PerfWatchdogTest, ItsThreadTakesNoSignal) {
  const ProbeResult RUN = runHelper({"signals"});
  EXPECT_TRUE(RUN.succeeded()) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_EQ(RUN.output, "SIGUSR1 waited for the thread that blocks it\n");
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
 * profiler attached as a benchmark's is), the measured body first runs in a
 * process of one thread with --profile-test-timeout 0, and of two at the
 * watchdog's default: the second is the watchdog's. glibc then no longer
 * counts the process as single-threaded.
 */
TEST(PerfWatchdogTest, OnlyTheWatchdogAddsAThreadUnderCallgrind) {
  if (ADDRESS_SANITIZER || THREAD_SANITIZER) {
    GTEST_SKIP() << "valgrind does not run a sanitizer build";
  }
  const vernier::bench::ReadinessContext CONTEXT = vernier::bench::ReadinessContext::capture();
  const auto VALGRIND = vernier::bench::resolveExecutable("valgrind", CONTEXT);
  if (!VALGRIND || !VALGRIND->executable) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the helper under callgrind";
  }
  const auto UNDER_CALLGRIND = [&](const std::vector<std::string>& timeout) {
    std::vector<std::string> args = {
        VALGRIND->path, "--tool=callgrind", "--instr-atstart=no", "--callgrind-out-file=/dev/null",
        HELPER,         "threads",          "--profile",          "callgrind"};
    args.insert(args.end(), timeout.begin(), timeout.end());
    return vernier::bench::runBoundedProbe(args, HELPER_BOUND_MS, CONTEXT,
                                           vernier::bench::ProbeStreams::SEPARATE);
  };
  const ProbeResult OFF = UNDER_CALLGRIND({"--profile-test-timeout", "0"});
  const ProbeResult ON = UNDER_CALLGRIND({});
  ASSERT_TRUE(OFF.succeeded()) << OFF.describe() << "\n" << OFF.output << OFF.errorOutput;
  ASSERT_TRUE(ON.succeeded()) << ON.describe() << "\n" << ON.output << ON.errorOutput;
  // The helper's line, after what the callgrind backend prints for the case.
  const auto READ = [](const std::string& output, int& threads, int& single) {
    const std::size_t AT = output.find("threads=");
    return AT != std::string::npos &&
           std::sscanf(output.c_str() + AT, "threads=%d single_threaded=%d", &threads, &single) ==
               2;
  };
  int offThreads = -1;
  int offSingle = -2;
  int onThreads = -1;
  int onSingle = -2;
  ASSERT_TRUE(READ(OFF.output, offThreads, offSingle)) << OFF.output;
  ASSERT_TRUE(READ(ON.output, onThreads, onSingle)) << ON.output;
  EXPECT_EQ(offThreads, 1) << "a thread with the watchdog off:\n" << OFF.errorOutput;
  EXPECT_EQ(onThreads, 2) << "not one more thread with the watchdog on:\n" << ON.errorOutput;
  if (offSingle != -1) { // glibc 2.32 and later say whether a thread was ever started
    EXPECT_EQ(offSingle, 1);
    EXPECT_EQ(onSingle, 0);
  }
}

/**
 * @test Under helgrind, a benchmark that starts no thread of its own gets no
 * report from the watchdog: the helper, run by valgrind's helgrind with an
 * exit status for any error, measures two cases under --profile helgrind with
 * the watchdog at its default, so the watchdog is armed, a loop measured, the
 * watchdog disarmed and armed again, its thread then parked; helgrind counts
 * no error.
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
  // Any exit status the helper does not give.
  constexpr int HELGRIND_ERRORS = 97;
  const ProbeResult RUN = vernier::bench::runBoundedProbe(
      {VALGRIND->path, "--tool=helgrind", "--error-exitcode=" + std::to_string(HELGRIND_ERRORS),
       HELPER, "rearm", "--profile", "helgrind"},
      HELPER_BOUND_MS, CONTEXT, vernier::bench::ProbeStreams::SEPARATE);
  ASSERT_TRUE(RUN.exited) << RUN.describe() << "\n" << RUN.output << RUN.errorOutput;
  EXPECT_NE(RUN.exitCode, HELGRIND_ERRORS) << "helgrind reported the watchdog:\n"
                                           << RUN.errorOutput;
  EXPECT_EQ(RUN.exitCode, 0) << RUN.output << RUN.errorOutput;
  EXPECT_NE(RUN.output.find("measured twice\n"), std::string::npos) << RUN.output;
  EXPECT_NE(RUN.errorOutput.find("ERROR SUMMARY: 0 errors from 0 contexts"), std::string::npos)
      << RUN.errorOutput;
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
