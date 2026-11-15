/**
 * @file WatchdogHelper.cpp
 * @brief The program the watchdog tests run as a child (PerfWatchdog_uTest.cpp).
 *
 * The watchdog ends the process it guards, so each case in which it fires, or
 * must not, runs here and the test reads how the program ended. A test
 * fixture, never installed. One mode per run:
 *
 *   hang <flags...>       a case, Watchdog.Hung, measured under <flags> (the
 *                         test gives --profile perf --profile-test-timeout 1)
 *                         whose body spins and does not return
 *   names <test> <tool>   the watchdog armed for 0.1 s with these names
 *   null-names            the watchdog armed for 0.1 s with no names
 *   disarmed              armed for 0.1 s and disarmed; then armed for 0.1 s
 *                         and, before that passes, armed again for 10 s; a
 *                         wait past 0.1 s after each; prints "still running"
 *   fork                  the watchdog's thread started here and parked, then
 *                         a child of fork() arms it for 0.1 s as
 *                         Watchdog.Child; exits with the child's status
 *   signals               the watchdog's thread started and parked, then
 *                         SIGUSR1, at its default action (it ends the
 *                         process) and blocked on this thread, sent to the
 *                         process: exits 0 when it is still pending here
 *   no-thread <flags...>  no thread can be started (RLIMIT_NPROC 0); a case,
 *                         Watchdog.NoThread, measured under <flags>: prints
 *                         which hooks ran and what measured() threw
 *
 * Where the watchdog should end the program, the program waits 5 s for it,
 * then says it did not and exits 0.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace perf_watchdog = vernier::bench::perf_watchdog;
using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;

namespace {

/// The deadline of the direct arms: short, so their cases stay fast.
constexpr std::chrono::milliseconds SHORT{100};

/// A wait well past SHORT.
constexpr std::chrono::milliseconds PAST_SHORT{300};

/// How long the program waits for a watchdog that should end it.
constexpr std::chrono::seconds FIRE_WAIT{5};

/// The flags after the mode, read as a benchmark reads its command line; one
/// cycle, one repeat.
PerfConfig configFrom(int argc, char** argv) {
  std::vector<char*> args{argv[0]};
  args.insert(args.end(), argv + 2, argv + argc);
  int count = static_cast<int>(args.size());
  PerfConfig cfg;
  vernier::bench::parsePerfFlags(cfg, &count, args.data());
  cfg.cycles = 1;
  cfg.repeats = 1;
  return cfg;
}

/// Waits for the watchdog to end the program; says it did not, and returns 0.
int waitForTheWatchdog() {
  std::this_thread::sleep_for(FIRE_WAIT);
  std::puts("the watchdog did not end the program");
  return 0;
}

int hang(int argc, char** argv) {
  PerfCase perf{"Watchdog.Hung", configFrom(argc, argv)};
  perf.measured([] {
    const auto UNTIL = std::chrono::steady_clock::now() + FIRE_WAIT;
    while (std::chrono::steady_clock::now() < UNTIL) {
    }
  });
  std::puts("the watchdog did not end the program");
  return 0;
}

int names(const char* test, const char* tool) {
  perf_watchdog::arm(test, tool, SHORT);
  return waitForTheWatchdog();
}

int disarmed() {
  perf_watchdog::arm("Watchdog.Disarmed", "perf", SHORT);
  perf_watchdog::disarm();
  std::this_thread::sleep_for(PAST_SHORT);
  perf_watchdog::arm("Watchdog.Rearmed", "perf", SHORT);
  // Long enough for the thread to start waiting for the first deadline.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  perf_watchdog::arm("Watchdog.Rearmed", "perf", std::chrono::seconds(10));
  std::this_thread::sleep_for(PAST_SHORT);
  perf_watchdog::disarm();
  std::puts("still running");
  return 0;
}

int forked() {
  perf_watchdog::arm("Watchdog.Parent", "perf", std::chrono::seconds(10));
  perf_watchdog::disarm();
  const pid_t CHILD = ::fork();
  if (CHILD < 0) {
    std::perror("fork");
    return 1;
  }
  if (CHILD == 0) {
    perf_watchdog::arm("Watchdog.Child", "perf", SHORT);
    std::this_thread::sleep_for(FIRE_WAIT);
    ::_exit(0);
  }
  int status = 0;
  while (::waitpid(CHILD, &status, 0) < 0) {
    if (errno != EINTR) {
      std::perror("waitpid");
      return 1;
    }
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

int signals() {
  sigset_t usr1;
  ::sigemptyset(&usr1);
  ::sigaddset(&usr1, SIGUSR1);
  // Not blocked here when the watchdog's thread starts, so the thread's mask
  // is the watchdog's own.
  ::signal(SIGUSR1, SIG_DFL);
  ::pthread_sigmask(SIG_UNBLOCK, &usr1, nullptr);
  perf_watchdog::arm("Watchdog.Signals", "perf", std::chrono::seconds(10));
  perf_watchdog::disarm();
  ::pthread_sigmask(SIG_BLOCK, &usr1, nullptr);
  ::kill(::getpid(), SIGUSR1);
  const timespec BOUND{1, 0};
  if (::sigtimedwait(&usr1, nullptr, &BOUND) != SIGUSR1) {
    std::puts("SIGUSR1 was not pending");
    return 1;
  }
  std::puts("SIGUSR1 waited for the thread that blocks it");
  return 0;
}

/// noThread()'s measurement, once no thread can be started.
int measureWithoutThreads(int argc, char** argv) {
  try {
    std::thread([] {}).join();
    std::puts("a thread started past RLIMIT_NPROC 0 (root, or CAP_SYS_RESOURCE)");
    return 77;
  } catch (const std::system_error&) {
  }
  PerfCase perf{"Watchdog.NoThread", configFrom(argc, argv)};
  perf.setBeforeMeasureHook([](const PerfCase&) { std::puts("before hook"); });
  perf.setAbortMeasureHook([](const PerfCase&) { std::puts("abort hook"); });
  perf.setAfterMeasureHook(
      [](const PerfCase&, const vernier::bench::Stats&) { std::puts("after hook"); });
  try {
    perf.measured([] { std::puts("body"); });
  } catch (const std::system_error& e) {
    std::printf("measured() threw std::system_error: %s\n",
                e.code() == std::errc::resource_unavailable_try_again ? "EAGAIN" : e.what());
    return 3;
  }
  std::puts("measured() returned");
  return 0;
}

int noThread(int argc, char** argv) {
  rlimit original{};
  if (::getrlimit(RLIMIT_NPROC, &original) != 0) {
    std::perror("getrlimit");
    return 1;
  }
  rlimit none = original;
  none.rlim_cur = 0;
  if (::setrlimit(RLIMIT_NPROC, &none) != 0) {
    std::perror("setrlimit");
    return 1;
  }
  const int STATUS = measureWithoutThreads(argc, argv);
  // Restored, so the program ends as any other: a leak checker starts a
  // thread at exit.
  if (::setrlimit(RLIMIT_NPROC, &original) != 0) {
    std::perror("setrlimit");
    return 1;
  }
  return STATUS;
}

} // namespace

int main(int argc, char** argv) {
  const std::string_view MODE = argc > 1 ? argv[1] : "";
  if (MODE == "hang") {
    return hang(argc, argv);
  }
  if (MODE == "names" && argc == 4) {
    return names(argv[2], argv[3]);
  }
  if (MODE == "null-names") {
    return names(nullptr, nullptr);
  }
  if (MODE == "disarmed") {
    return disarmed();
  }
  if (MODE == "fork") {
    return forked();
  }
  if (MODE == "signals") {
    return signals();
  }
  if (MODE == "no-thread") {
    return noThread(argc, argv);
  }
  std::fprintf(stderr,
               "usage: %s hang <flags>|names <test> <tool>|null-names|disarmed|fork|signals|"
               "no-thread <flags>\n",
               argv[0]);
  return 64;
}
