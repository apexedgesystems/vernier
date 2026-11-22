/**
 * @file WatchdogHelper.cpp
 * @brief The program the watchdog tests run as a child (PerfWatchdog_uTest.cpp).
 *
 * The watchdog ends the process it guards, so each case in which it fires, or
 * must not, runs here and the test reads how the program ended. A test
 * fixture, never installed. The program is single-threaded, so the watchdog
 * takes its timer engine where it is built; "--threaded" before the mode
 * first starts a thread of the program's own (every signal blocked, waiting
 * forever), so it takes its thread. One mode per run:
 *
 *   hang <flags...>       a case, Watchdog.Hung, measured under <flags> (the
 *                         test gives --profile perf --profile-test-timeout 1)
 *                         whose body spins and does not return
 *   hang-blocked <flags...>
 *                         as hang, with every signal blocked on this thread
 *                         before the case
 *   hang-timed <flags...> as hang, the body first printing "loop started at
 *                         <ns>" (CLOCK_MONOTONIC), so the test can time how
 *                         long after it the program ended
 *   hang-joined <flags...>
 *                         as hang-timed, the body then starting a thread that
 *                         spins and joining it: the measuring thread waits
 *                         while a thread computes
 *   hang-later <flags...> a case, Watchdog.Returns, measured under <flags> and
 *                         returning; once the watchdog's thread has parked
 *                         (the timer stops at once), a second,
 *                         Watchdog.HungLater, whose body spins
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
 *                         wait past 0.1 s after each, with the timer's
 *                         signal blocked after a disarm so one it raised
 *                         stays pending and is named; prints "still running"
 *   fork                  the watchdog armed and disarmed here, then a child
 *                         of fork() makes a timer of its own (the timer
 *                         engine's builds) and arms the watchdog for 0.1 s
 *                         as Watchdog.Child: the child prints its threads
 *                         and whether its own timer is kept, deleted or not
 *                         made; exits with the child's status
 *   signals               the watchdog armed and disarmed, this thread's
 *                         signal mask checked unchanged, then SIGUSR1, at its
 *                         default action (it ends the process) and blocked on
 *                         this thread, sent to the process: exits 0 when it
 *                         is still pending here
 *   first-signal          SIGRTMIN + 4 given a handler of the program's own;
 *                         armed for 10 s and disarmed: prints the signals the
 *                         process catches since (SigCgt, /proc/self/status),
 *                         from SIGRTMIN up as SIGRTMIN+n, and whether
 *                         SIGRTMIN + 4 kept the program's handler
 *   takeover              armed for 10 s and disarmed; the signal the timer
 *                         engine took (SigCgt, /proc/self/status) then given
 *                         a handler of the program's own; armed for 0.1 s as
 *                         Watchdog.AfterTakeover
 *   stale thread|hang|timer
 *                         (the timer engine's builds) armed for 0.1 s as
 *                         Watchdog.Overrun, its signal blocked here and the
 *                         deadline overrun, so the signal stays pending (exits
 *                         4 if it does not), disarmed; then armed as
 *                         Watchdog.Second: thread, for 10 s with the signal
 *                         still blocked (on the watchdog's thread), the signal
 *                         unblocked, disarmed: prints "still running"; hang,
 *                         the same armed for 0.1 s and left to the watchdog;
 *                         timer, the signal unblocked before that arming,
 *                         which takes the timer again for 10 s, disarmed:
 *                         prints "still running"
 *   foreign               (the timer engine's builds) armed for 10 s; the
 *                         signal it took sent to the process by kill(), and
 *                         by sigqueue() with the value its own timer carries,
 *                         and raised by a timer of the program's own; then
 *                         disarmed: prints "still running"
 *   switch between|during
 *                         between: armed for 0.1 s as Watchdog.First, a thread
 *                         of the program's own started, armed again for 10 s
 *                         as Watchdog.Rearmed, a wait past 0.1 s, then armed
 *                         for 0.1 s as Watchdog.Switched; during: armed for
 *                         0.1 s as Watchdog.Switched, then the thread started
 *   no-thread <flags...>  no thread can be started (RLIMIT_NPROC 0), one asked
 *                         for and refused, and every signal is blocked on this
 *                         thread, so the watchdog takes its thread; a case,
 *                         Watchdog.NoThread, measured under <flags>: prints
 *                         which hooks ran and what measured() threw
 *   no-thread-retry <flags...>
 *                         as no-thread, then a second case,
 *                         Watchdog.NoThreadAgain; then, threads allowed
 *                         again, the watchdog armed for 0.1 s as
 *                         Watchdog.Retried
 *   threads <flags...>    a case, Watchdog.Threads, with its profiler
 *                         attached as a benchmark's is, measured under
 *                         <flags>: prints how many threads the process has
 *                         and glibc's __libc_single_threaded (-1 without it)
 *                         when the body first runs and after the case
 *   rearm <flags...>      two cases, Watchdog.First and Watchdog.Second, each
 *                         with its profiler attached as a benchmark's is and
 *                         measured under <flags>: the watchdog armed, a
 *                         measured loop, the watchdog disarmed, armed again;
 *                         with the thread, each body waits for it to wait for
 *                         its deadline, and the second arming for it to park;
 *                         prints "measured twice"
 *
 * Where the watchdog should end the program, the program waits 5 s for it,
 * then says it did not and exits 0. Where it waits for the watchdog's thread
 * to park or to wait for a deadline (Linux: /proc/self/task/<tid>/syscall
 * shows the system call a blocked thread is in), it waits 2 s at most, then
 * says what it saw and exits 4; with the timer engine no thread starts, and
 * arm() and disarm() are complete when they return, so there is nothing to
 * wait for.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

#include <linux/futex.h>
#include <sys/resource.h>
#if __has_include(<sys/single_threaded.h>)
#include <sys/single_threaded.h>
#endif
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
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

/// Prints when the measured loop starts, once: "loop started at <ns>", on
/// CLOCK_MONOTONIC, the clock the test reads when the program has ended.
void printLoopStart() {
  static bool printed = false;
  if (printed) {
    return;
  }
  printed = true;
  timespec now{};
  ::clock_gettime(CLOCK_MONOTONIC, &now);
  std::printf("loop started at %lld\n",
              static_cast<long long>(now.tv_sec) * 1000000000LL + now.tv_nsec);
  std::fflush(stdout);
}

/// The hang-timed mode, and with @p joined the hang-joined mode.
int hangTimed(int argc, char** argv, bool joined) {
  PerfCase perf{"Watchdog.Hung", configFrom(argc, argv)};
  if (joined) {
    perf.measured([] {
      printLoopStart();
      std::thread worker(spin);
      worker.join();
    });
  } else {
    perf.measured([] {
      printLoopStart();
      spin();
    });
  }
  std::puts("the watchdog did not end the program");
  return 0;
}

/// The hang-blocked mode: every signal blocked on this thread, then hang.
int hangBlocked(int argc, char** argv) {
  sigset_t all;
  ::sigfillset(&all);
  ::pthread_sigmask(SIG_SETMASK, &all, nullptr);
  return hang(argc, argv);
}

/// Starts a thread of the program's own that blocks every signal and waits
/// forever: the process is then multi-threaded, as a benchmark with threads
/// of its own is.
void startOwnThread() {
  sigset_t all;
  sigset_t previous;
  ::sigfillset(&all);
  ::pthread_sigmask(SIG_SETMASK, &all, &previous);
  std::thread([] {
    for (;;) {
      ::pause();
    }
  }).detach();
  ::pthread_sigmask(SIG_SETMASK, &previous, nullptr);
}

/// The signals this process catches (SigCgt in /proc/self/status): bit n - 1
/// set for signal n.
std::uint64_t caughtSignals() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("SigCgt:", 0) == 0) {
      return std::stoull(line.substr(7), nullptr, 16);
    }
  }
  return 0;
}

/// The lowest real-time signal (SIGRTMIN or above, past the ones glibc
/// reserves) this process catches now and did not in @p before: the one the
/// timer engine took; 0 when there is none. (The first thread a process starts
/// gives glibc's SIGSETXID, below SIGRTMIN, a handler.)
int takenSignal(std::uint64_t before) {
  const std::uint64_t NEW = caughtSignals() & ~before;
  for (int sig = SIGRTMIN; sig <= SIGRTMAX && sig <= 64; ++sig) {
    if ((NEW >> (sig - 1)) & 1U) {
      return sig;
    }
  }
  return 0;
}

/// The program's own handler for the signals the first-signal and takeover
/// modes give it: says it ran, with write(2) alone.
void programHandler(int /*sig*/) {
  static const char kRan[] = "the program's handler ran\n";
  const ssize_t WRITTEN = ::write(STDOUT_FILENO, kRan, sizeof(kRan) - 1);
  (void)WRITTEN;
}

int firstSignal() {
  struct sigaction program{};
  program.sa_handler = programHandler;
  ::sigemptyset(&program.sa_mask);
  ::sigaction(SIGRTMIN + 4, &program, nullptr);
  const std::uint64_t CAUGHT = caughtSignals();
  perf_watchdog::arm("Watchdog.First", "perf", LONG);
  perf_watchdog::disarm();
  const std::uint64_t NEW = caughtSignals() & ~CAUGHT;
  std::string taken;
  for (int sig = 1; sig <= 64; ++sig) {
    if ((NEW >> (sig - 1)) & 1U) {
      taken += taken.empty() ? "" : " ";
      taken += sig >= SIGRTMIN ? "SIGRTMIN+" + std::to_string(sig - SIGRTMIN) : std::to_string(sig);
    }
  }
  struct sigaction now{};
  ::sigaction(SIGRTMIN + 4, nullptr, &now);
  std::printf("taken: %s; SIGRTMIN+4 %s\n", taken.empty() ? "none" : taken.c_str(),
              now.sa_handler == programHandler ? "kept" : "replaced");
  return 0;
}

int takeover() {
  const std::uint64_t CAUGHT = caughtSignals();
  perf_watchdog::arm("Watchdog.First", "perf", LONG);
  perf_watchdog::disarm();
  const int TAKEN = takenSignal(CAUGHT);
  if (TAKEN == 0) {
    std::puts("the watchdog took no signal");
    return 4;
  }
  struct sigaction program{};
  program.sa_handler = programHandler;
  ::sigemptyset(&program.sa_mask);
  ::sigaction(TAKEN, &program, nullptr);
  perf_watchdog::arm("Watchdog.AfterTakeover", "perf", SHORT);
  return waitForTheWatchdog();
}

#if VERNIER_WATCHDOG_TIMER
int foreign() {
  const std::uint64_t CAUGHT = caughtSignals();
  perf_watchdog::arm("Watchdog.Foreign", "perf", LONG);
  const int TAKEN = takenSignal(CAUGHT);
  if (TAKEN == 0) {
    std::puts("the watchdog took no signal");
    return 4;
  }
  ::kill(::getpid(), TAKEN);
  // Queued with the value the watchdog's own timer carries: still not its timer.
  sigval forged{};
  forged.sival_ptr = &perf_watchdog::g_timer;
  ::sigqueue(::getpid(), TAKEN, forged);
  sigevent event{};
  event.sigev_notify = SIGEV_SIGNAL;
  event.sigev_signo = TAKEN;
  timer_t own{};
  if (::timer_create(CLOCK_MONOTONIC, &event, &own) != 0) {
    std::perror("timer_create");
    return 1;
  }
  itimerspec when{};
  when.it_value.tv_nsec = 10'000'000;
  ::timer_settime(own, 0, &when, nullptr);
  std::this_thread::sleep_for(PAST_SHORT);
  ::timer_delete(own);
  perf_watchdog::disarm();
  std::puts("still running");
  return 0;
}

int stale(std::string_view second) {
  const std::uint64_t CAUGHT = caughtSignals();
  perf_watchdog::arm("Watchdog.Overrun", "perf", SHORT);
  const int TAKEN = takenSignal(CAUGHT);
  if (TAKEN == 0) {
    std::puts("the watchdog took no signal");
    return 4;
  }
  // The first loop blocks the timer's signal and overruns its deadline.
  sigset_t timerSignal;
  ::sigemptyset(&timerSignal);
  ::sigaddset(&timerSignal, TAKEN);
  ::pthread_sigmask(SIG_BLOCK, &timerSignal, nullptr);
  std::this_thread::sleep_for(PAST_SHORT);
  perf_watchdog::disarm();
  sigset_t pending;
  ::sigpending(&pending);
  if (::sigismember(&pending, TAKEN) != 1) {
    std::puts("the overrun left no signal pending");
    return 4;
  }
  if (second == "timer") {
    ::pthread_sigmask(SIG_UNBLOCK, &timerSignal, nullptr);
    perf_watchdog::arm("Watchdog.Second", "perf", LONG);
  } else {
    // Armed with the signal still blocked: the watchdog's thread. The second
    // loop unblocks the signal well inside its own deadline.
    perf_watchdog::arm("Watchdog.Second", "perf", second == "hang" ? SHORT : LONG);
    ::pthread_sigmask(SIG_UNBLOCK, &timerSignal, nullptr);
    if (second == "hang") {
      return waitForTheWatchdog();
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  perf_watchdog::disarm();
  std::puts("still running");
  return 0;
}
#endif

int switched(std::string_view when) {
  if (when == "between") {
    perf_watchdog::arm("Watchdog.First", "perf", SHORT);
    startOwnThread();
    perf_watchdog::arm("Watchdog.Rearmed", "perf", LONG);
    std::this_thread::sleep_for(PAST_SHORT);
    perf_watchdog::arm("Watchdog.Switched", "perf", SHORT);
  } else if (when == "during") {
    perf_watchdog::arm("Watchdog.Switched", "perf", SHORT);
    startOwnThread();
  } else {
    std::fputs("switch takes between or during\n", stderr);
    return 64;
  }
  return waitForTheWatchdog();
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

/// The threads in @p after that @p before does not hold: when the first
/// arm() came between the two listings, the watchdog's, and the thread
/// sanitizer's own, which its runtime starts with a program's first thread.
std::vector<pid_t> startedThreads(const std::vector<pid_t>& before,
                                  const std::vector<pid_t>& after) {
  std::vector<pid_t> started;
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                      std::back_inserter(started));
  return started;
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

/// Waits up to WAIT_BOUND for one of @p candidates to be blocked as @p wanted:
/// that thread, the watchdog's (no other thread here waits on a condition
/// variable); or -1 after saying what /proc last showed.
pid_t waitUntilBlocked(const std::vector<pid_t>& candidates, Blocked wanted, const char* what) {
  const auto UNTIL = std::chrono::steady_clock::now() + WAIT_BOUND;
  std::string shown;
  do {
    for (const pid_t TID : candidates) {
      if (blockedOf(TID, shown) == wanted) {
        return TID;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < UNTIL);
  std::printf("the watchdog's thread did not %s within 2 s: /proc showed '%s'\n", what,
              shown.c_str());
  return -1;
}

/// The watchdog armed for LONG as @p first, the first arming of this
/// process, and its thread seen waiting for that deadline: the thread; 0 when
/// no thread started (the timer engine, armed when arm() returns); or -1
/// after saying why not.
pid_t armAndWatch(const char* first) {
  const std::vector<pid_t> BEFORE = threads();
  perf_watchdog::arm(first, "perf", LONG);
  const std::vector<pid_t> STARTED = startedThreads(BEFORE, threads());
  if (STARTED.empty()) {
    return 0;
  }
  return waitUntilBlocked(STARTED, Blocked::ON_A_DEADLINE, "wait for the deadline");
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
  const std::vector<pid_t> STARTED = startedThreads(BEFORE, threads());
  if (!STARTED.empty() &&
      waitUntilBlocked(STARTED, Blocked::PARKED, "park after the first case") < 0) {
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
  if (WATCHDOG > 0 && waitUntilBlocked({WATCHDOG}, Blocked::PARKED, "park after disarm()") < 0) {
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
  const std::uint64_t CAUGHT = caughtSignals();
  perf_watchdog::arm("Watchdog.Disarmed", "perf", SHORT);
  perf_watchdog::disarm();
  // The timer engine's signal, blocked past the deadline: a timer left
  // running raises it, and it stays pending here.
  const int TAKEN = takenSignal(CAUGHT);
  sigset_t timerSignal;
  ::sigemptyset(&timerSignal);
  if (TAKEN > 0) {
    ::sigaddset(&timerSignal, TAKEN);
    ::pthread_sigmask(SIG_BLOCK, &timerSignal, nullptr);
  }
  std::this_thread::sleep_for(PAST_SHORT);
  if (TAKEN > 0) {
    sigset_t pending;
    ::sigpending(&pending);
    if (::sigismember(&pending, TAKEN) == 1) {
      std::printf("signal %d pending after disarm()\n", TAKEN);
    }
    const timespec NONE{0, 0};
    while (::sigtimedwait(&timerSignal, nullptr, &NONE) == TAKEN) {
    }
    ::pthread_sigmask(SIG_UNBLOCK, &timerSignal, nullptr);
  }
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
  // Armed and disarmed here: the parent's timer, or its thread, is not the
  // child's.
  perf_watchdog::arm("Watchdog.Parent", "perf", std::chrono::seconds(10));
  perf_watchdog::disarm();
  const pid_t CHILD = ::fork();
  if (CHILD < 0) {
    std::perror("fork");
    return 1;
  }
  if (CHILD == 0) {
    const char* own = "not made";
#if VERNIER_WATCHDOG_TIMER
    // A timer of the child's own, made before it arms: the first of this
    // process, so it has the handle the parent's first timer had, which the
    // child's arming must not take for its own and delete.
    sigevent none{};
    none.sigev_notify = SIGEV_NONE;
    timer_t timer{};
    const bool MADE = ::timer_create(CLOCK_MONOTONIC, &none, &timer) == 0;
#endif
    perf_watchdog::arm("Watchdog.Child", "perf", SHORT);
#if VERNIER_WATCHDOG_TIMER
    itimerspec left{};
    if (MADE) {
      own = ::timer_gettime(timer, &left) == 0 ? "kept" : "deleted";
    }
#endif
    std::printf("child threads=%zu, own timer %s\n", threads().size(), own);
    std::fflush(stdout);
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
  // The first arm() blocks every signal while it starts the thread, and the
  // timer engine reads the mask: this thread's own mask is to be as it was.
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
  // The watchdog takes its thread, which cannot start, for two reasons:
  // measureWithoutThreads() first asks for a thread, and the refused
  // pthread_create() still clears glibc's __libc_single_threaded; and every
  // signal is blocked on this thread, so the timer's could not reach it.
  sigset_t all;
  ::sigfillset(&all);
  ::pthread_sigmask(SIG_SETMASK, &all, nullptr);
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

/// glibc's word on whether this process ever started a second thread: 1 or
/// 0, -1 without it.
int singleThreaded() {
#if __has_include(<sys/single_threaded.h>)
  return __libc_single_threaded ? 1 : 0;
#else
  return -1;
#endif
}

/// The threads mode: the process's threads when the measured body first runs
/// and after the case.
int threadsInBody(int argc, char** argv) {
  std::size_t count = 0;
  int single = -1;
  {
    PerfCase perf =
        vernier::bench::makePerfCaseWithProfiler("Watchdog.Threads", configFrom(argc, argv));
    perf.measured([&] {
      if (count == 0) {
        count = threads().size();
        single = singleThreaded();
      }
    });
  }
  std::printf("threads=%zu single_threaded=%d after_threads=%zu after_single_threaded=%d\n", count,
              single, threads().size(), singleThreaded());
  return 0;
}

/// The rearm mode: two measured loops, so the watchdog is armed, disarmed and
/// armed again. Each body waits until the watchdog's thread waits for its
/// deadline, and the second arming comes once it is parked, so every
/// notification after the first arming finds the thread waiting.
int rearm(int argc, char** argv) {
  const PerfConfig CFG = configFrom(argc, argv);
  const std::vector<pid_t> BEFORE = threads();
  std::vector<pid_t> started;
  bool seen = true;
  {
    PerfCase perf = vernier::bench::makePerfCaseWithProfiler("Watchdog.First", CFG);
    perf.measured([&] {
      started = startedThreads(BEFORE, threads());
      seen =
          started.empty() ||
          (waitUntilBlocked(started, Blocked::ON_A_DEADLINE, "wait for its deadline") >= 0 && seen);
    });
  }
  seen = started.empty() || (waitUntilBlocked(started, Blocked::PARKED, "park") >= 0 && seen);
  {
    PerfCase perf = vernier::bench::makePerfCaseWithProfiler("Watchdog.Second", CFG);
    perf.measured([&] {
      seen =
          started.empty() ||
          (waitUntilBlocked(started, Blocked::ON_A_DEADLINE, "wait for its deadline") >= 0 && seen);
    });
  }
  if (!seen) {
    return 4;
  }
  std::puts("measured twice");
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--threaded") {
    startOwnThread();
    --argc;
    ++argv;
  }
  const std::string_view MODE = argc > 1 ? argv[1] : "";
  if (MODE == "hang") {
    return hang(argc, argv);
  }
  if (MODE == "hang-blocked") {
    return hangBlocked(argc, argv);
  }
  if (MODE == "hang-timed") {
    return hangTimed(argc, argv, false);
  }
  if (MODE == "hang-joined") {
    return hangTimed(argc, argv, true);
  }
  if (MODE == "first-signal") {
    return firstSignal();
  }
  if (MODE == "takeover") {
    return takeover();
  }
#if VERNIER_WATCHDOG_TIMER
  if (MODE == "foreign") {
    return foreign();
  }
  if (MODE == "stale" && argc == 3 &&
      (std::string_view(argv[2]) == "thread" || std::string_view(argv[2]) == "hang" ||
       std::string_view(argv[2]) == "timer")) {
    return stale(argv[2]);
  }
#endif
  if (MODE == "switch" && argc == 3) {
    return switched(argv[2]);
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
  if (MODE == "threads") {
    return threadsInBody(argc, argv);
  }
  if (MODE == "rearm") {
    return rearm(argc, argv);
  }
  std::fprintf(stderr,
               "usage: %s [--threaded] hang <flags>|hang-blocked <flags>|hang-timed <flags>|"
               "hang-joined <flags>|hang-later <flags>|later|"
               "shorten|names <test> <tool>|null-names|disarmed|fork|signals|first-signal|takeover|"
               "foreign|"
               "switch between|during|stale thread|hang|timer|no-thread <flags>|"
               "no-thread-retry <flags>|"
               "threads <flags>|"
               "rearm <flags>\n",
               argv[0]);
  return 64;
}
