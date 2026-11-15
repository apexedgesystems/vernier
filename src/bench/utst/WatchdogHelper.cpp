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
 *   hang-later <flags...> a case, Watchdog.Returns, measured under <flags> and
 *                         returning; once the watchdog's thread has parked, a
 *                         second, Watchdog.HungLater, whose body spins
 *   later                 armed for 10 s as Watchdog.First, the thread seen
 *                         waiting for that deadline, disarmed, the thread
 *                         seen parked, then armed for 0.1 s as Watchdog.Later
 *   shorten               armed for 10 s as Watchdog.Long, the thread seen
 *                         waiting for that deadline, then armed for 0.1 s as
 *                         Watchdog.Shortened
 *   names <test> <tool>   the watchdog armed for 0.1 s with these names
 *   null-names            the watchdog armed for 0.1 s with no names
 *   disarmed              armed for 0.1 s and disarmed; then armed for 0.1 s
 *                         and, before that passes, armed again for 10 s; a
 *                         wait past 0.1 s after each; prints "still running"
 *   fork                  the watchdog's thread started here and parked, then
 *                         a child of fork() arms it for 0.1 s as
 *                         Watchdog.Child; exits with the child's status
 *   signals               the watchdog's thread started and parked, this
 *                         thread's signal mask checked unchanged, then
 *                         SIGUSR1, at its default action (it ends the
 *                         process) and blocked on this thread, sent to the
 *                         process: exits 0 when it is still pending here
 *   no-thread <flags...>  no thread can be started (RLIMIT_NPROC 0); a case,
 *                         Watchdog.NoThread, measured under <flags>: prints
 *                         which hooks ran and what measured() threw
 *   no-thread-retry <flags...>
 *                         as no-thread, then a second case,
 *                         Watchdog.NoThreadAgain; then, threads allowed
 *                         again, the watchdog armed for 0.1 s as
 *                         Watchdog.Retried
 *
 * Where the watchdog should end the program, the program waits 5 s for it,
 * then says it did not and exits 0. Where it waits for the watchdog's thread
 * to park or to wait for a deadline (Linux: /proc/self/task/<tid>/syscall
 * shows the system call a blocked thread is in), it waits 2 s at most, then
 * says what it saw and exits 4.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <linux/futex.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
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

/// A deadline the thread is seen waiting for: twice FIRE_WAIT, so a watchdog
/// that fires only there leaves the program to say it did not end it.
constexpr std::chrono::seconds LONG{10};

/// How long the program waits for the watchdog's thread to reach a wait.
constexpr std::chrono::seconds WAIT_BOUND{2};

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

/// A measured body that spins for FIRE_WAIT.
void spin() {
  const auto UNTIL = std::chrono::steady_clock::now() + FIRE_WAIT;
  while (std::chrono::steady_clock::now() < UNTIL) {
  }
}

int hang(int argc, char** argv) {
  PerfCase perf{"Watchdog.Hung", configFrom(argc, argv)};
  perf.measured(spin);
  std::puts("the watchdog did not end the program");
  return 0;
}

/// The threads of this process, by id, in order.
std::vector<pid_t> threads() {
  std::vector<pid_t> ids;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task", ec)) {
    ids.push_back(static_cast<pid_t>(std::stol(entry.path().filename().string())));
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

/// The one thread in @p after that @p before does not hold: the watchdog's,
/// when the first arm() came between the two listings; -1 when not one.
pid_t startedThread(const std::vector<pid_t>& before, const std::vector<pid_t>& after) {
  std::vector<pid_t> started;
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                      std::back_inserter(started));
  return started.size() == 1 ? started.front() : -1;
}

/// Where a thread is blocked, as /proc/self/task/<tid>/syscall shows it: in a
/// condition variable's wait (futex, FUTEX_WAIT_BITSET) without a deadline,
/// parked until the next arm(); in one with a deadline; or elsewhere.
enum class Blocked { PARKED, ON_A_DEADLINE, ELSEWHERE };

Blocked blockedOf(pid_t tid, std::string& shown) {
  std::ifstream in("/proc/self/task/" + std::to_string(tid) + "/syscall");
  shown.clear();
  std::getline(in, shown);
  std::istringstream fields(shown);
  long number = -1;
  std::string address;
  std::string op;
  std::string value;
  std::string timeout;
  if (!(fields >> number >> address >> op >> value >> timeout) || number != SYS_futex) {
    return Blocked::ELSEWHERE;
  }
  if ((std::stoul(op, nullptr, 16) & FUTEX_CMD_MASK) != FUTEX_WAIT_BITSET) {
    return Blocked::ELSEWHERE;
  }
  return std::stoul(timeout, nullptr, 16) == 0 ? Blocked::PARKED : Blocked::ON_A_DEADLINE;
}

/// Waits up to WAIT_BOUND for @p tid to be blocked as @p wanted; says what
/// /proc last showed, and returns false, when it is not.
bool waitUntilBlocked(pid_t tid, Blocked wanted, const char* what) {
  const auto UNTIL = std::chrono::steady_clock::now() + WAIT_BOUND;
  std::string shown;
  do {
    if (blockedOf(tid, shown) == wanted) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < UNTIL);
  std::printf("the watchdog's thread did not %s within 2 s: /proc showed '%s'\n", what,
              shown.c_str());
  return false;
}

/// The watchdog armed for LONG as @p first, the first arming of this
/// process, and its thread seen waiting for that deadline: the thread, or -1
/// after saying why not.
pid_t armAndWatch(const char* first) {
  const std::vector<pid_t> BEFORE = threads();
  perf_watchdog::arm(first, "perf", LONG);
  const pid_t WATCHDOG = startedThread(BEFORE, threads());
  if (WATCHDOG < 0) {
    std::puts("the first arm() started no thread of its own");
    return -1;
  }
  return waitUntilBlocked(WATCHDOG, Blocked::ON_A_DEADLINE, "wait for the deadline") ? WATCHDOG
                                                                                     : -1;
}

int hangLater(int argc, char** argv) {
  const PerfConfig CFG = configFrom(argc, argv);
  const std::vector<pid_t> BEFORE = threads();
  {
    PerfCase first{"Watchdog.Returns", CFG};
    first.measured([] {});
  }
  // The first case's line, kept when the watchdog ends the program.
  std::fflush(stdout);
  const pid_t WATCHDOG = startedThread(BEFORE, threads());
  if (WATCHDOG < 0) {
    std::puts("the first case started no watchdog thread of its own");
    return 4;
  }
  if (!waitUntilBlocked(WATCHDOG, Blocked::PARKED, "park after the first case")) {
    return 4;
  }
  PerfCase second{"Watchdog.HungLater", CFG};
  second.measured(spin);
  std::puts("the watchdog did not end the program");
  return 0;
}

int later() {
  const pid_t WATCHDOG = armAndWatch("Watchdog.First");
  if (WATCHDOG < 0) {
    return 4;
  }
  perf_watchdog::disarm();
  if (!waitUntilBlocked(WATCHDOG, Blocked::PARKED, "park after disarm()")) {
    return 4;
  }
  perf_watchdog::arm("Watchdog.Later", "perf", SHORT);
  return waitForTheWatchdog();
}

int shorten() {
  if (armAndWatch("Watchdog.Long") < 0) {
    return 4;
  }
  perf_watchdog::arm("Watchdog.Shortened", "perf", SHORT);
  return waitForTheWatchdog();
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
  sigset_t before;
  ::pthread_sigmask(SIG_SETMASK, nullptr, &before);
  perf_watchdog::arm("Watchdog.Signals", "perf", std::chrono::seconds(10));
  perf_watchdog::disarm();
  // The first arm() blocks every signal while it starts the thread: this
  // thread's own mask is to be as it was.
  sigset_t after;
  ::pthread_sigmask(SIG_SETMASK, nullptr, &after);
  for (int sig = 1; sig < NSIG; ++sig) {
    if (::sigismember(&before, sig) != ::sigismember(&after, sig)) {
      std::printf("arm() left signal %d %s on the thread that armed it\n", sig,
                  ::sigismember(&after, sig) == 1 ? "blocked" : "unblocked");
      return 1;
    }
  }
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

/// A case, @p name, measured under @p cfg with hooks that say they ran: 3
/// when measured() threw std::system_error (said too), 0 when it returned.
int measureSaying(const char* name, const PerfConfig& cfg) {
  PerfCase perf{name, cfg};
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

/// noThread()'s measurements, once no thread can be started: one case, and
/// with @p again a second.
int measureWithoutThreads(int argc, char** argv, bool again) {
  try {
    std::thread([] {}).join();
    std::puts("a thread started past RLIMIT_NPROC 0 (root, or CAP_SYS_RESOURCE)");
    return 77;
  } catch (const std::system_error&) {
  }
  const PerfConfig CFG = configFrom(argc, argv);
  const int STATUS = measureSaying("Watchdog.NoThread", CFG);
  return again && STATUS == 3 ? measureSaying("Watchdog.NoThreadAgain", CFG) : STATUS;
}

/// The no-thread mode; with @p retry, the no-thread-retry mode.
int noThread(int argc, char** argv, bool retry) {
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
  const int STATUS = measureWithoutThreads(argc, argv, retry);
  // Restored, so the program ends as any other (a leak checker starts a
  // thread at exit), and so that the next arming can start the thread.
  if (::setrlimit(RLIMIT_NPROC, &original) != 0) {
    std::perror("setrlimit");
    return 1;
  }
  if (!retry || STATUS != 3) {
    return STATUS;
  }
  // The lines so far, kept when the watchdog ends the program.
  std::fflush(stdout);
  perf_watchdog::arm("Watchdog.Retried", "perf", SHORT);
  return waitForTheWatchdog();
}

} // namespace

int main(int argc, char** argv) {
  const std::string_view MODE = argc > 1 ? argv[1] : "";
  if (MODE == "hang") {
    return hang(argc, argv);
  }
  if (MODE == "hang-later") {
    return hangLater(argc, argv);
  }
  if (MODE == "later") {
    return later();
  }
  if (MODE == "shorten") {
    return shorten();
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
    return noThread(argc, argv, false);
  }
  if (MODE == "no-thread-retry") {
    return noThread(argc, argv, true);
  }
  std::fprintf(stderr,
               "usage: %s hang <flags>|hang-later <flags>|later|shorten|names <test> <tool>|"
               "null-names|disarmed|fork|signals|no-thread <flags>|no-thread-retry <flags>\n",
               argv[0]);
  return 64;
}
