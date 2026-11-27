#ifndef VERNIER_PERFHARNESS_HPP
#define VERNIER_PERFHARNESS_HPP
/**
 * @file PerfHarness.hpp
 * @brief High-level perf test constructs layered over GoogleTest.
 *
 * Includes console printing, metadata capture, memory bandwidth analysis,
 * and the PerfCase class that drives warmup/measure/teardown phases.
 */

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <thread>

#include <unistd.h> // gethostname, getpid, _exit, write

// The watchdog's timer engine is built on Linux with glibc 2.34 or later: its
// C library has timer_create and SIGEV_THREAD_ID, and __libc_single_threaded
// says whether the process has ever started a second thread. Elsewhere the
// watchdog is its parked thread alone.
#if defined(__linux__) && defined(__GLIBC__) && defined(__GLIBC_MINOR__) &&                        \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34)) &&                                \
    __has_include(<sys/single_threaded.h>)
#define VERNIER_WATCHDOG_TIMER 1
#include <signal.h>
#include <sys/single_threaded.h>
#include <time.h>
#else
#define VERNIER_WATCHDOG_TIMER 0
#endif

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/PerfUtils.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfValidation.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp" // runBoundedProbe, for git describe

namespace vernier {
namespace bench {

/* ----------------------------- Console Printing ----------------------------- */

/**
 * @brief Format a large number with K/M/G suffix for readability.
 * @note RT-safe (pure computation).
 */
inline void formatCompact(char* buf, std::size_t bufLen, double val) {
  if (val >= 1e9) {
    std::snprintf(buf, bufLen, "%.1fG", val / 1e9);
  } else if (val >= 1e6) {
    std::snprintf(buf, bufLen, "%.1fM", val / 1e6);
  } else if (val >= 1e3) {
    std::snprintf(buf, bufLen, "%.1fK", val / 1e3);
  } else {
    std::snprintf(buf, bufLen, "%.0f", val);
  }
}

/**
 * @brief Print a compact, scannable result line for a test.
 *
 * Format: [TestName] 3.456 us/call  CV=7.2%  ~289K calls/s  (p10=3.10 p90=3.80 sd=0.25)
 *
 * Key metrics (median, CV, calls/s) are first for quick scanning.
 * Secondary stats (percentiles, stddev) are in parentheses.
 *
 * @note NOT RT-safe (console I/O).
 */
inline void printStats(const char* label, const Stats& s, double callsPerSecond, bool stable) {
  char cpsBuf[16];
  formatCompact(cpsBuf, sizeof(cpsBuf), callsPerSecond);

  const char* STABILITY = stable ? "" : " [UNSTABLE]";

  std::printf("%s  %.3f us/call  CV=%.1f%%  ~%s calls/s  (p10=%.3f p90=%.3f sd=%.3f)%s\n", label,
              s.median, s.cv * 100.0, cpsBuf, s.p10, s.p90, s.stddev, STABILITY);
}

/**
 * @brief Print actionable hints based on benchmark results.
 * @note NOT RT-safe (console I/O).
 */
inline void printHints(const Stats& s, const PerfConfig& cfg) {
  if (cfg.quickMode && s.cv > 0.15) {
    std::fprintf(stderr,
                 "  [INFO] High CV%% (%.1f%%) in quick mode is expected.\n"
                 "         For stable baselines, run without --quick\n",
                 s.cv * 100.0);
    return;
  }

  if (s.cv > 0.20 && cfg.profileTool.empty()) {
    std::fprintf(stderr,
                 "  [!] High variance (CV=%.1f%%). Consider:\n"
                 "      --profile perf    # CPU counters + flamegraph\n"
                 "      --profile nsight  # GPU kernel analysis\n",
                 s.cv * 100.0);
  } else if (!cfg.quickMode && s.cv > 0.10 && cfg.profileTool.empty()) {
    std::fprintf(stderr, "  [WARN] Moderate variance (CV=%.1f%%). Try --repeats 20 or --profile\n",
                 s.cv * 100.0);
  }
}

/**
 * @brief Print stats with automatic hints (convenience wrapper).
 * @note NOT RT-safe (console I/O).
 */
inline void printStatsWithHints(const char* label, const Stats& s, double callsPerSecond,
                                const PerfConfig& cfg, bool stable) {
  printStats(label, s, callsPerSecond, stable);
  printHints(s, cfg);
}

/* ----------------------------- Under Nsight Compute ----------------------------- */

/**
 * @brief Whether Nsight Compute (ncu) profiles this process: bench run's ncu
 *        route, nsight's compute route or an ncu wrap typed by hand
 *        (profiler_env::nsightSessionTool()).
 *
 * Its instrumentation adds to what host timers and CUDA events measure,
 * whatever metrics it collects (whether it also replays a launch depends on
 * them), so neither harness times a measurement under it: a CPU case whose
 * work launches kernels is slowed as a GPU case is. Nsight Systems is not ncu.
 *
 * @note NOT RT-safe (reads the environment); called after a measurement.
 */
inline bool profiledByNcu() { return profiler_env::nsightSessionTool() == "ncu"; }

/**
 * @brief Print the console line of a measurement the harness did not time, in
 *        place of printStatsWithHints(): `<label>  not timed: ...`.
 * @note NOT RT-safe (console I/O).
 */
inline void printNotTimed(const char* label) {
  std::printf("%s  not timed: Nsight Compute (ncu) profiles this process\n", label);
}

/**
 * @brief Under Nsight Compute (profiledByNcu()), leave @p row's timing cells
 *        empty and say why, once per process, on stderr; otherwise do nothing.
 *
 * The row is marked untimed (PerfRow::timed: its wall*, callsPerSecond and
 * stable cells stay empty), and the GPU values computed from the timing are
 * reset. The other cells, and ncu's own measurements in its report, stay.
 *
 * @note NOT RT-safe (environment, console I/O); called after a measurement.
 */
inline void untimeUnderNcu(PerfRow& row) {
  if (!profiledByNcu()) {
    return;
  }
  row.timed = false;
  row.kernelTimeUs.reset();
  row.transferTimeUs.reset();
  row.speedupVsCpu.reset();
  row.memBandwidthGBs.reset();
  row.multiGpuEfficiency.reset();
  row.p2pBandwidthGBs.reset();
  static std::atomic<bool> stated{false};
  if (!stated.exchange(true)) {
    std::fprintf(stderr,
                 "[bench] Nsight Compute (ncu) profiles this process, and its instrumentation "
                 "adds to what host timers and CUDA events measure, so no measurement is "
                 "timed: every row's wallMedian, wallP10, wallP90, wallP99, wallP999, wallMin, "
                 "wallMax, wallMean, wallStddev, wallCV, callsPerSecond and stable cells stay "
                 "empty, and so do a GPU row's kernelTimeUs, transferTimeUs, speedupVsCpu, "
                 "memBandwidthGBs, multiGpuEfficiency and p2pBandwidthGBs. ncu's own "
                 "measurements are in its report.\n");
  }
}

/**
 * @brief Print progress during long measurement runs.
 *
 * Emits a carriage-return overwrite on stderr every ~2 seconds.
 * Only prints if the measurement has been running for >2 seconds.
 * The line is cleared before the final result is printed.
 *
 * @note NOT RT-safe (console I/O).
 */
inline void printProgress(const char* testName, int repeat, int totalRepeats,
                          std::chrono::steady_clock::time_point startTime) {
  auto now = std::chrono::steady_clock::now();
  double elapsedS =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
  double estTotalS = elapsedS * totalRepeats / (repeat + 1);
  double remainS = estTotalS - elapsedS;
  std::fprintf(stderr, "\r  [%s] repeat %d/%d (%.0f%%)  elapsed=%.1fs  remaining~%.1fs   ",
               testName, repeat + 1, totalRepeats, 100.0 * (repeat + 1) / totalRepeats, elapsedS,
               remainS);
}

/** @brief Clear progress line from stderr. */
inline void clearProgress() { std::fprintf(stderr, "\r%*s\r", 80, ""); }

/* ----------------------------- Watchdog ----------------------------- */

/**
 * The per-test watchdog under --profile, armed by measured() around its
 * repeats: when a deadline passes while it is still armed, it writes a message
 * naming the case and the profiler and ends the process with exit status 2.
 * It uses no alarm(), no SIGALRM and no interval timer, so a benchmark's own
 * alarm() and SIGALRM handler, and a profiler that samples on SIGALRM, work
 * under it.
 *
 * Each arming takes one of two engines. Where the timer engine is built
 * (VERNIER_WATCHDOG_TIMER) and glibc still counts the process single-threaded,
 * a POSIX timer signals the arming thread with a real-time signal the watchdog
 * took at its default disposition: no thread is started, so the C library and
 * libstdc++ keep their single-threaded paths in the measured loop. Otherwise
 * (the process has another thread, the signal is blocked in the arming thread,
 * the signal is no longer the watchdog's and no other one is free, the timer
 * cannot be made, or the engine is not built) one thread per process, started
 * by the first arming that needs it and parked on a condition variable between
 * arms. One gap the thread does not have: a single-threaded benchmark that
 * blocks the watchdog's signal, or takes its disposition, inside its measured
 * loop after arming is not ended.
 */
namespace perf_watchdog {

/// Whether a measured loop is under the watchdog: set by arm(), cleared by disarm().
inline std::atomic<bool> g_armed{false};

// Writes the whole buffer to stderr, resuming after a short write or EINTR.
// Any other failure ends the attempt: the caller is about to _exit and has no
// other channel to report on. write(2) alone, without stdio or the allocator,
// whose locks the hung thread may hold; async-signal-safe.
inline void writeStderr(const char* data, std::size_t len) noexcept {
  while (len > 0) {
    const ssize_t WRITTEN = ::write(STDERR_FILENO, data, len);
    if (WRITTEN < 0 && errno == EINTR) {
      continue;
    }
    if (WRITTEN <= 0) {
      return;
    }
    data += WRITTEN;
    len -= static_cast<std::size_t>(WRITTEN);
  }
}

// Writes the message for the case @p testName under @p profileTool, both
// fixed buffers, and ends the process with exit status 2: write(2) and
// _exit(2) only, so the timer engine's signal handler may call it.
[[noreturn]] inline void expire(const char* testName, const char* profileTool) noexcept {
  static const char kPrefix[] = "\n[bench] watchdog timeout: test '";
  static const char kMid[] = "' under --profile ";
  static const char kSuffix[] =
      " exceeded the configured profile-test-timeout.\n"
      "[bench] Likely cause: drain loop, blocking I/O, or filter that holds the\n"
      "[bench] measured loop. Identify the test, exclude it via --gtest_filter,\n"
      "[bench] or raise --profile-test-timeout. Aborting.\n";
  writeStderr(kPrefix, sizeof(kPrefix) - 1);
  writeStderr(testName, std::strlen(testName));
  writeStderr(kMid, sizeof(kMid) - 1);
  writeStderr(profileTool, std::strlen(profileTool));
  writeStderr(kSuffix, sizeof(kSuffix) - 1);
  ::_exit(2);
}

/* --- The thread engine --- */

/**
 * @brief The thread engine's state in one process, made by its first arming
 * and never destroyed, so the detached thread waiting on it never sees it go
 * at exit. A child of fork() makes its own and leaves the copied one as it is:
 * the parent's thread is not copied, and the copied lock may be held.
 */
struct State {
  std::mutex mutex;
  std::condition_variable changed; ///< Notified by arm() and disarm().
  bool watching = false;           ///< The thread is started.
  bool armed = false;
  std::uint64_t arming = 0; ///< Counts arm() calls: a re-arm restarts the wait.
  std::chrono::steady_clock::time_point deadline{};
  char testName[160] = {0};   ///< A fixed buffer, so the expiry does not allocate.
  char profileTool[32] = {0}; ///< Likewise.
  pid_t pid = 0;              ///< The process this state belongs to.
};

/// This process's state: none before its first arm(), the parent's in a child of fork().
inline std::atomic<State*> g_state{nullptr};

// The watchdog's thread: parked until armed, then waiting for the deadline, a
// disarm or a re-arm. A deadline that passes while still armed ends the
// process, with the names copied and the lock released first, so the thread
// that ends the process holds no lock.
inline void watch(State* state) {
  std::unique_lock<std::mutex> lock(state->mutex);
  for (;;) {
    state->changed.wait(lock, [state] { return state->armed; });
    const std::uint64_t ARMING = state->arming;
    const auto DEADLINE = state->deadline;
    const bool ENDED = state->changed.wait_until(
        lock, DEADLINE, [state, ARMING] { return !state->armed || state->arming != ARMING; });
    if (!ENDED) {
      char testName[sizeof(state->testName)];
      char profileTool[sizeof(state->profileTool)];
      std::memcpy(testName, state->testName, sizeof(testName));
      std::memcpy(profileTool, state->profileTool, sizeof(profileTool));
      lock.unlock();
      expire(testName, profileTool);
    }
  }
}

// Starts the thread for @p state with every signal blocked on it (a thread
// takes the mask of the thread that starts it), so a signal meant for the
// benchmark, one its own threads block among them, is never delivered to it.
inline void startWatching(State* state) {
  sigset_t all;
  sigset_t previous;
  ::sigfillset(&all);
  ::pthread_sigmask(SIG_SETMASK, &all, &previous);
  struct Restore {
    const sigset_t& mask;
    ~Restore() { ::pthread_sigmask(SIG_SETMASK, &mask, nullptr); }
  } const RESTORE{previous};
  std::thread(watch, state).detach();
}

// This process's state, made on the first call in it.
inline State* processState() {
  const pid_t SELF = ::getpid();
  State* current = g_state.load(std::memory_order_acquire);
  while (current == nullptr || current->pid != SELF) {
    auto* made = new State;
    made->pid = SELF;
    if (g_state.compare_exchange_strong(current, made, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return made;
    }
    delete made; // another thread of this process made it first
  }
  return current;
}

// Arms the thread engine, starting its thread at the first arming that needs
// it; throws std::system_error when the thread cannot be started.
inline void armThread(const char* testName, const char* profileTool,
                      std::chrono::steady_clock::duration timeout) {
  State* state = processState();
  const std::lock_guard<std::mutex> LOCK(state->mutex);
  if (!state->watching) {
    startWatching(state);
    state->watching = true;
  }
  std::strncpy(state->testName, testName != nullptr ? testName : "?", sizeof(state->testName) - 1);
  std::strncpy(state->profileTool, profileTool != nullptr ? profileTool : "?",
               sizeof(state->profileTool) - 1);
  state->deadline = std::chrono::steady_clock::now() + timeout;
  ++state->arming;
  state->armed = true;
  // Signalled with the lock held, as disarm() does: a race checker the
  // benchmark runs under (helgrind) then has nothing of the watchdog's to
  // report.
  state->changed.notify_one();
}

// Disarms the thread engine, if this process armed it: its thread parks.
inline void disarmThread() {
  State* state = g_state.load(std::memory_order_acquire);
  if (state == nullptr || state->pid != ::getpid()) {
    return;
  }
  const std::lock_guard<std::mutex> LOCK(state->mutex);
  state->armed = false;
  state->changed.notify_one();
}

#if VERNIER_WATCHDOG_TIMER

/* --- The timer engine --- */

/**
 * @brief The timer engine's state in one process. A child of fork() inherits
 * the signal's disposition but not the timer, so it makes its own timer.
 */
struct TimerState {
  pid_t pid = 0;        ///< The process the timer belongs to.
  int signo = 0;        ///< The real-time signal taken; 0 before the first.
  bool created = false; ///< The timer exists in this process.
  timer_t timer{};
  pid_t tid = 0;              ///< The thread the timer signals: the last arming thread.
  char testName[160] = {0};   ///< Fixed buffers, read by the signal handler.
  char profileTool[32] = {0}; ///< Likewise.
  /// The timer is the engine armed now: set by armTimer(), cleared by
  /// disarmTimer(). Not g_armed, which an arming on the thread sets too, so a
  /// signal an overrun left pending ends no later loop armed on the thread.
  std::atomic<bool> armed{false};
};

// The handler loads the flag: a signal handler may perform only lock-free
// atomic operations on it.
static_assert(std::atomic<bool>::is_always_lock_free,
              "the timer engine needs a lock-free std::atomic<bool>");

/// This process's timer state, touched by arm() and disarm() and read by the handler.
inline TimerState g_timer;

// The timer's signal: the watchdog's expiry ends the process when the timer is
// the engine armed; anything else (a disarm that raced the expiry, a signal an
// earlier overrun left pending, the signal sent by someone else or raised by
// another timer) returns at once, touching nothing. The flag is loaded with
// acquire: the arming's release fence after its copy of the names then
// synchronizes with it, so the names are read after they were written.
inline void onTimer(int /*signo*/, siginfo_t* info, void* /*context*/) {
  if (info != nullptr && info->si_code == SI_TIMER && info->si_value.sival_ptr == &g_timer &&
      g_timer.armed.load(std::memory_order_acquire)) {
    expire(g_timer.testName, g_timer.profileTool);
  }
}

// Whether @p signo's disposition is the watchdog's handler.
inline bool ownsSignal(int signo) {
  struct sigaction current{};
  return ::sigaction(signo, nullptr, &current) == 0 && (current.sa_flags & SA_SIGINFO) != 0 &&
         current.sa_sigaction == onTimer;
}

// Takes the first real-time signal from SIGRTMIN + 4 whose disposition is the
// default, with the handler installed; 0 when none is free.
inline int takeSignal() {
  for (int s = SIGRTMIN + 4; s <= SIGRTMAX; ++s) {
    struct sigaction current{};
    if (::sigaction(s, nullptr, &current) != 0 || (current.sa_flags & SA_SIGINFO) != 0 ||
        current.sa_handler != SIG_DFL) {
      continue;
    }
    struct sigaction mine{};
    mine.sa_sigaction = onTimer;
    mine.sa_flags = SA_SIGINFO | SA_RESTART;
    ::sigemptyset(&mine.sa_mask);
    if (::sigaction(s, &mine, nullptr) == 0) {
      return s;
    }
  }
  return 0;
}

// Arms the timer engine for the calling thread; false, with nothing armed,
// when it cannot be used now and the caller takes the thread.
inline bool armTimer(const char* testName, const char* profileTool,
                     std::chrono::steady_clock::duration timeout) {
  TimerState& state = g_timer;
  if (state.pid != ::getpid()) {
    state.pid = ::getpid();
    state.created = false; // a child of fork(): the parent's timer is not its own
  }
  if (state.signo != 0 && !ownsSignal(state.signo)) {
    // Taken by someone else since the last arming: left to its new owner.
    if (state.created) {
      ::timer_delete(state.timer);
      state.created = false;
    }
    state.signo = 0;
  }
  if (state.signo == 0 && (state.signo = takeSignal()) == 0) {
    return false;
  }
  sigset_t mask;
  if (::pthread_sigmask(SIG_BLOCK, nullptr, &mask) != 0 || ::sigismember(&mask, state.signo) != 0) {
    return false; // blocked here: the signal could not end the loop
  }
  const pid_t TID = ::gettid();
  if (state.created && state.tid != TID) {
    ::timer_delete(state.timer);
    state.created = false;
  }
  if (!state.created) {
    struct sigevent event{};
    event.sigev_notify = SIGEV_THREAD_ID;
    event.sigev_signo = state.signo;
    event.sigev_value.sival_ptr = &g_timer; // how the handler knows its own timer
    event._sigev_un._tid = TID;             // glibc's member for SIGEV_THREAD_ID
    if (::timer_create(CLOCK_MONOTONIC, &event, &state.timer) != 0) {
      return false;
    }
    state.created = true;
    state.tid = TID;
  }
  if (state.armed.load(std::memory_order_relaxed)) {
    // A re-arm before a disarm stops the earlier deadline first: a signal it
    // raised is handled as this call returns, naming the earlier arming, and
    // the timer raises nothing more until its new deadline is set, so no
    // expiry meets the copy below or names this arming for the earlier one.
    struct itimerspec stopped{};
    ::timer_settime(state.timer, 0, &stopped, nullptr);
  }
  // The names are copied disarmed, and with the timer stopped (by disarm() or
  // above) no signal of it is handled meanwhile; the fences keep the copy
  // between the two stores, and the second is the release half of the
  // handler's acquire load. The handler runs on this thread.
  state.armed.store(false, std::memory_order_relaxed);
  std::atomic_signal_fence(std::memory_order_seq_cst);
  std::strncpy(state.testName, testName != nullptr ? testName : "?", sizeof(state.testName) - 1);
  std::strncpy(state.profileTool, profileTool != nullptr ? profileTool : "?",
               sizeof(state.profileTool) - 1);
  std::atomic_signal_fence(std::memory_order_seq_cst);
  state.armed.store(true, std::memory_order_relaxed); // before the timer can fire
  const long long NS = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count();
  struct itimerspec when{};
  when.it_value.tv_sec = static_cast<time_t>(NS / 1000000000LL);
  when.it_value.tv_nsec = static_cast<long>(NS % 1000000000LL);
  if (when.it_value.tv_sec == 0 && when.it_value.tv_nsec <= 0) {
    when.it_value.tv_nsec = 1; // a zero value would disarm the timer
  }
  if (::timer_settime(state.timer, 0, &when, nullptr) != 0) {
    state.armed.store(false, std::memory_order_relaxed);
    return false;
  }
  return true;
}

// Disarms the timer engine: the timer is no longer the engine armed, so a
// signal it raised before stopping, or one still pending from an overrun,
// ends nothing; and the timer stops, if this process made it.
inline void disarmTimer() {
  TimerState& state = g_timer;
  state.armed.store(false, std::memory_order_relaxed);
  if (!state.created || state.pid != ::getpid()) {
    return;
  }
  struct itimerspec stopped{};
  ::timer_settime(state.timer, 0, &stopped, nullptr);
}

#endif // VERNIER_WATCHDOG_TIMER

/**
 * @brief Arm the watchdog: unless disarm() comes first, @p timeout from now
 * it writes its message naming @p testName and @p profileTool ("?" for a null
 * name) and ends the process with exit status 2. A later arm() replaces the
 * deadline and the names. The engine is chosen at each arming (see the
 * namespace's comment). When it is the thread and the thread cannot be
 * started, throws std::system_error; nothing is armed then.
 */
inline void arm(const char* testName, const char* profileTool,
                std::chrono::steady_clock::duration timeout) {
#if VERNIER_WATCHDOG_TIMER
  if (__libc_single_threaded != 0 && armTimer(testName, profileTool, timeout)) {
    g_armed.store(true, std::memory_order_relaxed);
    return;
  }
  disarmTimer(); // a timer an earlier arming set is not left running, nor armed
#endif
  armThread(testName, profileTool, timeout);
  g_armed.store(true, std::memory_order_relaxed);
}

/** @brief arm() for @p timeoutSecs whole seconds, as --profile-test-timeout gives them. */
inline void arm(const char* testName, const char* profileTool, unsigned int timeoutSecs) {
  arm(testName, profileTool, std::chrono::seconds(timeoutSecs));
}

/** @brief Disarm the watchdog: its timer stops, or its thread parks until the next arm(). */
inline void disarm() {
  g_armed.store(false, std::memory_order_relaxed);
#if VERNIER_WATCHDOG_TIMER
  disarmTimer();
#endif
  disarmThread();
}

} // namespace perf_watchdog

/* ----------------------------- Metadata Capture ----------------------------- */

/**
 * @brief Capture current timestamp in ISO 8601 format.
 * @note NOT RT-safe (system calls, heap allocation).
 */
inline std::string captureTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};

#if defined(__unix__) || defined(__APPLE__)
  gmtime_r(&t, &tm);
#else
  std::tm* tmp = std::gmtime(&t);
  if (tmp)
    tm = *tmp;
#endif

  std::array<char, 32> buf{};
  std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buf.data());
}

/// How long the run's `git describe` may take; past it, it is ended and the
/// hash is "unknown".
inline constexpr int GIT_DESCRIBE_TIMEOUT_MS = 10000;

/**
 * @brief Name the working directory's commit the way `git describe` does.
 *
 * Runs `git describe --always --dirty --tags` in the working directory as a
 * bounded child without a shell (runBoundedProbe()), in the probe environment
 * (ReadinessContext::probeEnvironment()): without LD_PRELOAD, MALLOC_CONF and
 * the other variables that put a profiler into a process, so a benchmark that
 * a tool's wrap started (bench run's jemalloc preload, for one) starts git
 * without that tool. "unknown" when git is not on PATH, fails, prints
 * nothing or outlives GIT_DESCRIBE_TIMEOUT_MS.
 * @note NOT RT-safe (starts a process).
 */
inline std::string captureGitHash() {
  const ReadinessContext CTX = ReadinessContext::capture();
  const std::optional<ResolvedTool> GIT = resolveExecutable("git", CTX);
  if (!GIT || !GIT->executable) {
    return "unknown";
  }
  const ProbeResult RUN = runBoundedProbe({GIT->path, "describe", "--always", "--dirty", "--tags"},
                                          GIT_DESCRIBE_TIMEOUT_MS, CTX, ProbeStreams::SEPARATE);
  if (!RUN.succeeded()) {
    return "unknown";
  }
  std::string result = RUN.output.substr(0, RUN.output.find('\n'));
  if (!result.empty() && result.back() == '\r') {
    result.pop_back();
  }
  return result.empty() ? "unknown" : result;
}

/**
 * @brief Capture system hostname.
 * @note NOT RT-safe (system call, heap allocation).
 */
inline std::string captureHostname() {
  std::array<char, 256> buf{};

#if defined(__unix__) || defined(__APPLE__)
  if (gethostname(buf.data(), buf.size()) == 0) {
    buf.back() = '\0';
    return std::string(buf.data());
  }
#endif

  return "unknown";
}

/**
 * @brief Detect platform architecture.
 * @note RT-safe (compile-time constant string).
 */
inline std::string capturePlatform() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__riscv)
  return "riscv";
#else
  return "unknown";
#endif
}

/**
 * @brief Capture all metadata at once (cached for performance).
 * @param cacheMetadata If true, capture once and reuse. If false, always fresh.
 * @return Tuple of (timestamp, gitHash, hostname, platform).
 * @note NOT RT-safe (system calls, subprocess, heap allocation).
 */
inline std::tuple<std::string, std::string, std::string, std::string>
captureMetadata(bool cacheMetadata = true) {
  if (!cacheMetadata) {
    return std::make_tuple(captureTimestamp(), captureGitHash(), captureHostname(),
                           capturePlatform());
  }
  // Magic-static initialization IS the synchronization: the first caller
  // captures under the compiler's init guard, everyone else waits, and
  // afterwards the tuple is immutable. The GPU harness publishes from worker
  // threads as well as from the test's thread, and captureGitHash() runs a
  // subprocess, so a first capture lasts long enough for a second caller to
  // arrive during it.
  static const std::tuple<std::string, std::string, std::string> CACHED =
      std::make_tuple(captureGitHash(), captureHostname(), capturePlatform());
  return std::make_tuple(captureTimestamp(), std::get<0>(CACHED), std::get<1>(CACHED),
                         std::get<2>(CACHED));
}

/* ----------------------------- MemoryProfile ----------------------------- */

/**
 * @brief Memory profile for bandwidth analysis.
 *
 * Describes memory access patterns for an operation to enable
 * bandwidth calculations and CPU-bound vs memory-bound analysis.
 */
struct MemoryProfile {
  size_t bytesRead = 0;      ///< Input data size per operation (bytes)
  size_t bytesWritten = 0;   ///< Output data size per operation (bytes)
  size_t bytesAllocated = 0; ///< Heap allocations per operation (optional)

  /**
   * @brief Calculate total bandwidth in MB/s.
   * @param durationUs Duration in microseconds
   * @return Bandwidth in MB/s (read + write)
   * @note RT-safe (pure calculation, no allocations).
   */
  [[nodiscard]] double bandwidthMBs(double durationUs) const {
    if (durationUs <= 0.0)
      return 0.0;
    const double TOTAL_BYTES = static_cast<double>(bytesRead + bytesWritten);
    const double DURATION_S = durationUs / 1e6;
    return (TOTAL_BYTES / DURATION_S) / 1e6; // MB/s
  }

  /**
   * @brief Calculate efficiency vs theoretical peak.
   * @param durationUs Duration in microseconds
   * @param peakMBs Theoretical peak bandwidth (MB/s)
   * @return Efficiency percentage [0-100]
   * @note RT-safe (pure calculation, no allocations).
   */
  [[nodiscard]] double efficiency(double durationUs, double peakMBs) const {
    if (peakMBs <= 0.0)
      return 0.0;
    const double ACHIEVED = bandwidthMBs(durationUs);
    return (ACHIEVED / peakMBs) * 100.0;
  }
};

/* ------------------------------ PerfResult ------------------------------ */

/** @brief Result of a measured section. */
struct PerfResult {
  Stats stats{};           ///< Summarized per-call wall us
  double callsPerSecond{}; ///< Derived from median
  std::string label;       ///< Section label (e.g., "measured")

  // Memory bandwidth analysis (optional)
  std::optional<MemoryProfile> memoryProfile{};

  // Stability assessment
  bool stable{true};        ///< CV% below adaptive threshold
  double cvThreshold{0.05}; ///< Threshold used (from recommendedCVThreshold)
};

/* ----------------------------- Row Builder ----------------------------- */

/**
 * @brief Build a PerfRow from test state and measurement results.
 *
 * Centralizes PerfRow construction so callers don't need 38 positional fields.
 * GPU/multi-GPU/Unified Memory fields remain at default (nullopt) for CPU tests.
 * @p label is the measurement's label, which names the row when its case
 * measures more than once in a test.
 *
 * @note NOT RT-safe (captures metadata via subprocess).
 */
inline PerfRow buildPerfRow(const std::string& testName, const PerfConfig& cfg, int actualWarmup,
                            int threadCount, const Stats& stats, double callsPerSecond,
                            std::string label = {}) {
  auto [timestamp, gitHash, hostname, platform] = captureMetadata(/*cache=*/true);

  const double CV_THRESHOLD = recommendedCVThreshold(cfg);

  PerfRow row;
  row.testName = testName;
  row.cycles = cfg.cycles;
  row.repeats = cfg.repeats;
  row.warmup = actualWarmup;
  row.threads = threadCount;
  row.msgBytes = cfg.msgBytes;
  row.console = cfg.console;
  row.nonBlocking = cfg.nonBlocking;
  row.minLevel = cfg.minLevel;
  row.stats = stats;
  row.callsPerSecond = callsPerSecond;
  row.timestamp = timestamp;
  row.gitHash = gitHash;
  row.hostname = hostname;
  row.platform = platform;
  row.stable = stats.cv < CV_THRESHOLD;
  row.cvThreshold = CV_THRESHOLD;
  row.label = std::move(label);
  return row;
}

/**
 * @brief Cycles that make one repeat span ~targetUs given an estimated
 * per-call cost. Pure math for testability; clamped to [1, 50e6]. The floor
 * is one cycle: a call that already outlasts the target runs once per
 * repeat instead of stretching the repeat to a multiple of it.
 */
inline int calibratedCycles(long long targetUs, double estimatedPerCallUs) {
  const double EST = estimatedPerCallUs > 0.001 ? estimatedPerCallUs : 0.001;
  double cycles = static_cast<double>(targetUs) / EST;
  if (cycles < 1.0) {
    cycles = 1.0;
  }
  if (cycles > 50000000.0) {
    cycles = 50000000.0;
  }
  return static_cast<int>(cycles);
}

/* ------------------------------- PerfCase ------------------------------- */

/**
 * @brief A convenience facade to structure a perf test into phases.
 *
 * Typical usage:
 *   PerfCase perf{"Suite.Name", cfg};
 *   perf.setup(...);
 *   perf.warmup([&]{ body(cycles); });
 *   const PerfResult r = perf.measured([&]{ body(cycles); });
 *   perf.teardown(...);
 *
 * @note NOT RT-safe (heap allocation, threads, console/file I/O).
 */
class PerfCase {
public:
  using Fn = std::function<void()>;
  using BeforeHook = std::function<void(const PerfCase&)>;
  using AfterHook = std::function<void(const PerfCase&, const Stats&)>;
  using AbortHook = std::function<void(const PerfCase&)>;

  PerfCase(std::string testName, PerfConfig cfg)
      : testName_(std::move(testName)), cfg_(std::move(cfg)) {}

  /** @brief Optional one-time setup before warmup/measure. */
  void setup(Fn fn) {
    if (fn) {
      fn();
    }
  }

  /**
   * @brief Run warmup repeats (user provides the inner loop).
   *
   * Auto-scaling warmup: If cfg.warmup == 0, automatically determine warmup count
   * based on the number of cycles:
   *   - cycles < 1000:  warmup = 5 (small workloads need more warmup)
   *   - cycles < 10000: warmup = 3 (medium workloads)
   *   - cycles >= 10000: warmup = 1 (large workloads already warm caches)
   *
   * Otherwise, use the explicitly specified warmup count.
   */
  void warmup(Fn fn) {
    if (!fn) {
      return;
    }

    // Determine warmup count
    actualWarmup_ = cfg_.warmup;

    // Auto-scaling: if warmup is 0, calculate based on cycles
    if (actualWarmup_ == 0) {
      if (cfg_.cycles < 1000) {
        actualWarmup_ = 5; // Small workloads need more warmup
      } else if (cfg_.cycles < 10000) {
        actualWarmup_ = 3; // Medium workloads
      } else {
        actualWarmup_ = 1; // Large workloads (already warmed by sheer volume)
      }
    }

    for (int w = 0; w < actualWarmup_; ++w) {
      fn();
    }
  }

  /** @brief Install hooks bracketing the measured phase (optional). */
  void setBeforeMeasureHook(BeforeHook h) { beforeHook_ = std::move(h); }
  void setAfterMeasureHook(AfterHook h) { afterHook_ = std::move(h); }

  /**
   * @brief Install the hook measured() runs in place of the after hook when
   * its callback throws (optional): it ends what the before hook started,
   * with nothing summarized or published. An exception it throws is dropped,
   * so the caller gets the callback's.
   */
  void setAbortMeasureHook(AbortHook h) { abortHook_ = std::move(h); }

  /**
   * @brief Measure repeats. User-provided body should perform exactly `cycles()` ops.
   * Records wall us per call per repeat, then summarizes.
   *
   * The body runs on the calling thread, so the row records one thread
   * whatever `threads()` says; contentionRun() is the construct that starts
   * threads.
   *
   * Each completed call publishes its own row. When the case measures more
   * than once in a test, each row is named `<case>/<label>` (see
   * PerfRegistry); a body that throws publishes nothing.
   *
   * Progress is printed to stderr every ~2 seconds for long-running measurements.
   *
   * An exception from @p fn, or from starting the watchdog's thread under
   * --profile, ends the measurement: the watchdog is turned off, the abort
   * hook runs (attachProfilerHooks(): the profiler ends what its before hook
   * started, without analysis), nothing is summarized or published, the after
   * hook does not run, and the exception reaches the caller unchanged. The
   * case can then measure again.
   */
  PerfResult measured(Fn fn, std::string label = "measured") {
    if (beforeHook_) {
      beforeHook_(*this);
    }

    const auto MEASURE_START = std::chrono::steady_clock::now();
    auto lastProgressTime = MEASURE_START;
    bool showedProgress = false;

    // Arm the per-test watchdog only when profiling AND the timeout is above
    // zero (0 turns it off). It ends a hung fn() that never returns to be
    // checked, a drain loop or a blocking recv, by its timer's signal or its
    // thread (perf_watchdog).
    const bool watchdogActive = (cfg_.profileTestTimeoutSecs > 0) && !cfg_.profileTool.empty();

    std::vector<double> perCall;
    try {
      // Inside the try: a watchdog thread that cannot be started ends the
      // measurement as an exception from fn() does.
      if (watchdogActive) {
        perf_watchdog::arm(testName_.c_str(), cfg_.profileTool.c_str(),
                           static_cast<unsigned int>(cfg_.profileTestTimeoutSecs));
      }
      perCall.reserve(static_cast<std::size_t>(cfg_.repeats));
      for (int r = 0; r < cfg_.repeats; ++r) {
        // Time points, not nowUs(), so the repeat keeps the clock's own
        // resolution instead of two readings cut to whole microseconds.
        const auto T0 = std::chrono::steady_clock::now();
        fn();
        const auto T1 = std::chrono::steady_clock::now();
        perCall.push_back(std::chrono::duration<double, std::micro>(T1 - T0).count() /
                          static_cast<double>(cfg_.cycles));

        // Progress reporting: print every ~2 seconds for long runs
        auto now = std::chrono::steady_clock::now();
        auto sinceLast =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime);
        if (sinceLast.count() >= 2000) {
          printProgress(testName_.c_str(), r, cfg_.repeats, MEASURE_START);
          lastProgressTime = now;
          showedProgress = true;
        }
      }
    } catch (...) {
      // Left armed, the watchdog would end the process during a later case.
      if (showedProgress) {
        clearProgress();
      }
      if (watchdogActive) {
        perf_watchdog::disarm();
      }
      // What the before hook started ends now, not with the case: the caller
      // may keep the case, measure it again or measure another.
      runAbortHook();
      throw;
    }

    if (showedProgress) {
      clearProgress();
    }

    if (watchdogActive) {
      perf_watchdog::disarm();
    }

    auto vals = perCall; // copy so summarize() can sort
    const Stats S = summarize(vals);
    const double CPS = (S.median > 0.0) ? 1e6 / S.median : 0.0;
    const double CV_THRESHOLD = recommendedCVThreshold(cfg_);
    const bool IS_STABLE = S.cv < CV_THRESHOLD;

    // The row, untimed under Nsight Compute; its line, stats with automatic
    // hints or why there are none.
    PerfRow row = buildPerfRow(testName_, cfg_, actualWarmup_, /*threadCount=*/1, S, CPS, label);
    untimeUnderNcu(row);
    const std::string LABEL_STR = "[" + testName_ + "]";
    if (row.timed) {
      printStatsWithHints(LABEL_STR.c_str(), S, CPS, cfg_, IS_STABLE);
    } else {
      printNotTimed(LABEL_STR.c_str());
    }

    PerfRegistry::instance().set(std::move(row));

    if (afterHook_) {
      afterHook_(*this, S);
    }

    return PerfResult{S, CPS, std::move(label), std::nullopt, IS_STABLE, CV_THRESHOLD};
  }

  /** @brief Optional teardown after measurement. */
  void teardown(Fn fn) {
    if (fn) {
      fn();
    }
  }

  // ---------------------------------------------------------------------------
  // Pre-baked constructs (lightweight wrappers around measured())
  // ---------------------------------------------------------------------------

  /**
   * @brief Throughput loop: runs a lambda `op()` exactly `cycles()` times.
   * The lambda should perform one operation and be allocation-free in hot paths.
   *
   * Optional memory profile for bandwidth analysis.
   */
  PerfResult throughputLoop(const std::function<void()>& op, std::string label = "throughput",
                            std::optional<MemoryProfile> memProfile = std::nullopt) {
    maybeCalibrate(op);
    auto loop = [&]() {
      for (int i = 0; i < cfg_.cycles; ++i) {
        op();
      }
    };

    auto result = measured(loop, std::move(label));
    result.memoryProfile = memProfile;

    // Print bandwidth analysis if provided: every figure in it divides by the
    // median, so a run under Nsight Compute, which measured() left untimed,
    // prints none of it. The returned result keeps its numbers.
    if (memProfile.has_value() && !profiledByNcu()) {
      const double BW = memProfile->bandwidthMBs(result.stats.median);
      std::printf("  Memory bandwidth: %.1f MB/s (%.1f MB read, %.1f MB written per call)\n", BW,
                  memProfile->bytesRead / 1e6, memProfile->bytesWritten / 1e6);

      // Estimate theoretical peak (rough heuristic: DDR4-2400 ~19 GB/s per channel)
      // Users should provide actual peak for their system
      const double ESTIMATED_PEAK = 12000.0; // MB/s (conservative DDR4 estimate)
      const double EFF = memProfile->efficiency(result.stats.median, ESTIMATED_PEAK);
      std::printf("  Estimated efficiency: %.1f%% of theoretical peak (~%.0f MB/s)\n", EFF,
                  ESTIMATED_PEAK);

      if (EFF < 10.0) {
        std::printf(
            "  Hint: Low bandwidth utilization -> CPU-bound (algorithm optimization needed)\n");
      } else if (EFF > 50.0) {
        std::printf(
            "  Hint: High bandwidth utilization -> Memory-bound (consider memory layout)\n");
      }
    }

    return result;
  }

  /**
   * @brief Throughput loop with simple memory tracking (read-only).
   *
   * Simplified overload for common case of read-only memory operations.
   *
   * @param op Operation to benchmark (called cycles() times)
   * @param label Label for output and CSV
   * @param bytesRead Bytes read per operation
   * @return PerfResult with memory bandwidth analysis
   *
   * Example:
   * @code
   * auto result = perf.throughputLoop([&]{ sum(data); }, "sum", data.size());
   * @endcode
   */
  PerfResult throughputLoop(const std::function<void()>& op, std::string label,
                            std::size_t bytesRead) {
    return throughputLoop(op, std::move(label), MemoryProfile{bytesRead, 0, 0});
  }

  /**
   * @brief Throughput loop with read/write memory tracking.
   *
   * Simplified overload for read/write memory operations.
   *
   * @param op Operation to benchmark (called cycles() times)
   * @param label Label for output and CSV
   * @param bytesRead Bytes read per operation
   * @param bytesWritten Bytes written per operation
   * @return PerfResult with memory bandwidth analysis
   *
   * Example:
   * @code
   * auto result = perf.throughputLoop([&]{ copy(src, dst); }, "copy", srcSize, dstSize);
   * @endcode
   */
  PerfResult throughputLoop(const std::function<void()>& op, std::string label,
                            std::size_t bytesRead, std::size_t bytesWritten) {
    return throughputLoop(op, std::move(label), MemoryProfile{bytesRead, bytesWritten, 0});
  }

  /**
   * @brief Contention run: start `threads()` threads together; each calls `worker` `cycles()`
   * times per repeat, so `worker` is one operation, not a loop. The threads call it at the same
   * time, so it keeps its own shared state safe. Caller is free to compute extras (e.g., drop%).
   * This returns only timing stats (per-call across all operations). For drop%, derive externally.
   * Publishes its own row, which records the workers it started, named as measured() names its
   * rows.
   *
   * An exception from a worker, or a worker thread that cannot be started, ends the run as an
   * exception from measured()'s callback ends that: the workers started are joined, the abort
   * hook runs, nothing is summarized or published, the after hook does not run, and the first
   * exception reaches the caller on the calling thread. The case can then measure again.
   */
  PerfResult contentionRun(const std::function<void()>& worker, std::string label = "contention") {
    // Calibrates from one uncontended call; contention only lengthens the
    // repeat, which errs the window in the safe (longer) direction.
    maybeCalibrate(worker);
    const int THREAD_COUNT = threads();
    std::vector<double> perCall;
    perCall.reserve(static_cast<std::size_t>(cfg_.repeats));

    if (beforeHook_) {
      beforeHook_(*this);
    }

    const auto MEASURE_START = std::chrono::steady_clock::now();
    auto lastProgressTime = MEASURE_START;
    bool showedProgress = false;

    try {
      for (int r = 0; r < cfg_.repeats; ++r) {
        perCall.push_back(contendOnce(worker, THREAD_COUNT));

        // Progress reporting
        auto now = std::chrono::steady_clock::now();
        auto sinceLast =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime);
        if (sinceLast.count() >= 2000) {
          printProgress(testName_.c_str(), r, cfg_.repeats, MEASURE_START);
          lastProgressTime = now;
          showedProgress = true;
        }
      }
    } catch (...) {
      if (showedProgress) {
        clearProgress();
      }
      // What the before hook started ends now, as measured() ends it.
      runAbortHook();
      throw;
    }

    if (showedProgress) {
      clearProgress();
    }

    auto vals = perCall;
    const Stats S = summarize(vals);
    const double CPS = (S.median > 0.0) ? 1e6 / S.median : 0.0;
    const double CV_THRESHOLD = recommendedCVThreshold(cfg_);
    const bool IS_STABLE = S.cv < CV_THRESHOLD;

    // The row, untimed under Nsight Compute; its line, stats with automatic
    // hints or why there are none.
    PerfRow row = buildPerfRow(testName_, cfg_, actualWarmup_, THREAD_COUNT, S, CPS, label);
    untimeUnderNcu(row);
    const std::string LABEL_STR = "[" + testName_ + "]";
    if (row.timed) {
      printStatsWithHints(LABEL_STR.c_str(), S, CPS, cfg_, IS_STABLE);
    } else {
      printNotTimed(LABEL_STR.c_str());
    }

    PerfRegistry::instance().set(std::move(row));

    if (afterHook_) {
      afterHook_(*this, S);
    }

    return PerfResult{S, CPS, std::move(label), std::nullopt, IS_STABLE, CV_THRESHOLD};
  }

  // Accessors
  [[nodiscard]] int cycles() const noexcept { return cfg_.cycles; }
  [[nodiscard]] int repeats() const noexcept { return cfg_.repeats; }
  /// Workers contentionRun() starts: the case's thread count (`--threads`), at least 1.
  [[nodiscard]] int threads() const noexcept { return (cfg_.threads > 0) ? cfg_.threads : 1; }
  [[nodiscard]] int warmup() const noexcept { return actualWarmup_; }
  [[nodiscard]] const PerfConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const std::string& testName() const noexcept { return testName_; }

private:
  /**
   * @brief One repeat of contentionRun(): @p threadCount workers released at
   * once, each making cycles() calls of @p worker; the wall microseconds per
   * call. A worker's exception is taken on its own thread, and the first one
   * rethrown here once every worker has been joined. A worker thread that
   * cannot be started ends the repeat once those started have made their
   * calls and been joined, with its error: none is left joinable, which would
   * end the process.
   */
  double contendOnce(const std::function<void()>& worker, int threadCount) {
    StartGate gate(threadCount);
    std::mutex failureMutex;
    std::exception_ptr workerFailure;
    std::exception_ptr startFailure;
    std::vector<std::thread> ts;
    ts.reserve(static_cast<std::size_t>(threadCount));
    const auto T0 = std::chrono::steady_clock::now();
    try {
      for (int t = 0; t < threadCount; ++t) {
        ts.emplace_back([&]() {
          gate.start();
          try {
            for (int i = 0; i < cfg_.cycles; ++i) {
              worker();
            }
          } catch (...) {
            const std::lock_guard<std::mutex> LOCK(failureMutex);
            if (!workerFailure) {
              workerFailure = std::current_exception();
            }
          }
        });
      }
    } catch (...) {
      startFailure = std::current_exception();
    }
    gate.releaseWhenReady(static_cast<int>(ts.size()));
    for (auto& th : ts) {
      th.join();
    }
    const auto T1 = std::chrono::steady_clock::now();
    if (startFailure) {
      std::rethrow_exception(startFailure);
    }
    if (workerFailure) {
      std::rethrow_exception(workerFailure);
    }
    const double TOTAL_CALLS = static_cast<double>(threadCount) * cfg_.cycles;
    return std::chrono::duration<double, std::micro>(T1 - T0).count() / TOTAL_CALLS;
  }

  /// Run the abort hook, if any. What it throws is dropped, so the exception
  /// that ended the measurement is the one the caller gets.
  void runAbortHook() noexcept {
    if (!abortHook_) {
      return;
    }
    try {
      abortHook_(*this);
    } catch (...) {
      // Dropped: the measured callback's exception is the one rethrown.
    }
  }

  /**
   * @brief --target-time auto-sizing: estimate one call's cost, then set
   * this instance's cycle count so a repeat targets the requested wall time.
   *
   * A single call of a sub-microsecond operation costs little more than the
   * two clock reads around it, and a call of a few nanoseconds can be
   * shorter than the clock's tick. The estimate instead times batches of
   * calls, doubling the batch until a sample spans CALIBRATION_SAMPLE_US, and
   * divides. Calibration finishes before the measured phase starts, so none
   * of it is inside a timed window.
   *
   * The estimate is best effort. A sample's wall time also counts time the
   * thread did not run, so the estimate is the least per-call time among the
   * samples long enough to resolve it (CALIBRATION_RESOLVED_US), and a sample
   * that spans the sample span ends the doubling only after a resolved one.
   * This reduces isolated scheduling interference. A first call stretched
   * past the requested time, or consecutive stretched samples, can still size
   * the repeats short: 300 us of work with a 50 ms interruption on the first
   * call against a 40 ms target selects one cycle, and the repeats then take
   * about 300 us.
   *
   * A single call ends the calibration at once only when it spans the
   * requested time, so a slow operation is called once; an operation between
   * the sample span and the requested time costs three calibration calls
   * instead of one.
   *
   * cfg_ is a per-case copy, so the division in measured(), the CSV row, and
   * the loops all follow automatically; raw measured() users keep explicit
   * --cycles (they own their loop bodies).
   */
  void maybeCalibrate(const std::function<void()>& op) {
    if (cfg_.targetTimeUs <= 0 || calibrated_) {
      return;
    }
    calibrated_ = true;

    // 1 ms sample.
    constexpr double CALIBRATION_SAMPLE_US = 1000.0;
    // A quarter of the sample: the two clock reads and the clock's tick are
    // then a negligible share of the sample, apart from scheduling and
    // workload variation.
    constexpr double CALIBRATION_RESOLVED_US = CALIBRATION_SAMPLE_US / 4.0;
    // Bounds the doubling when the operation costs next to nothing.
    constexpr long long CALIBRATION_MAX_BATCH = 1LL << 26;

    long long batch = 1;
    double elapsedUs = 0.0;
    double previousUs = 0.0;
    bool resolved = false;
    double leastPerCallUs = 0.0;
    for (;;) {
      const auto T0 = std::chrono::steady_clock::now();
      for (long long i = 0; i < batch; ++i) {
        op();
      }
      elapsedUs =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - T0).count();
      if (elapsedUs >= CALIBRATION_RESOLVED_US) {
        const double PER_CALL_US = elapsedUs / static_cast<double>(batch);
        if (!resolved || PER_CALL_US < leastPerCallUs) {
          leastPerCallUs = PER_CALL_US;
        }
        resolved = true;
      }
      // A single call that spans the requested time sets a one-cycle window.
      const bool SPANS_THE_TARGET =
          batch == 1 && elapsedUs >= static_cast<double>(cfg_.targetTimeUs);
      const bool GREW_INTO_IT = previousUs >= CALIBRATION_RESOLVED_US;
      if (SPANS_THE_TARGET || (elapsedUs >= CALIBRATION_SAMPLE_US && GREW_INTO_IT) ||
          batch >= CALIBRATION_MAX_BATCH) {
        break;
      }
      previousUs = elapsedUs;
      batch *= 2;
    }

    const double EST = resolved ? leastPerCallUs : elapsedUs / static_cast<double>(batch);
    cfg_.cycles = calibratedCycles(cfg_.targetTimeUs, EST);
    std::fprintf(stderr,
                 "[target-time] %.3f ms -> cycles=%d (calibrated %.4f us/call, batch of %lld)\n",
                 static_cast<double>(cfg_.targetTimeUs) / 1000.0, cfg_.cycles, EST, batch);
  }

  std::string testName_;
  PerfConfig cfg_{};
  bool calibrated_ = false;
  int actualWarmup_ = 1;

  // Optional hooks (no-ops unless set)
  BeforeHook beforeHook_{};
  AfterHook afterHook_{};
  AbortHook abortHook_{};
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFHARNESS_HPP