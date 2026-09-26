#ifndef VERNIER_DEMO_HARDWARECOUNTER_HPP
#define VERNIER_DEMO_HARDWARECOUNTER_HPP
/**
 * @file HardwareCounter.hpp
 * @brief One hardware event counted on the calling thread, through the
 *        kernel's perf interface, for the demos' effect checks.
 *
 * A check that a demo still demonstrates what its walkthrough shows needs the
 * counts the walkthrough reads, on exactly the calls it makes: instructions
 * per call of one version against another, branch misses on random input
 * against sorted. perf stat reports those for a whole process from outside;
 * this counter reads the same hardware event from inside, around one
 * callable, with no perf binary involved. It counts user-space work only, as
 * an unprivileged perf stat does.
 *
 * Linux only. Opening the counter can fail: the kernel refuses an unprivileged
 * process above kernel.perf_event_paranoid 2, a container's default seccomp
 * profile refuses the system call, a virtual machine may expose no PMU, and a
 * PMU may lack the event (the Cortex-A72 has no branch-instructions event).
 * failure() says which, so a test can skip with the reason.
 *
 * An open counter can still miss the work: the kernel schedules it on the
 * PMU only while the thread runs on a core whose PMU it belongs to (a hybrid
 * processor's two kinds of core have two), and takes it off when more events
 * are open than the PMU has counters. A count read without knowing that is a
 * number that could be zero for a loop that ran. Every reading therefore
 * carries the time the counter was enabled and the time it was running, as
 * perf stat reads them: the count is scaled by their ratio when the counter
 * ran part of the time, and a reading whose counter ran less than
 * MIN_RUN_FRACTION of the time is refused, with the reason.
 */

#include <cstddef>
#include <cstdint>

#include <array>
#include <initializer_list>
#include <string>
#include <utility>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include <fstream>
#endif

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- HardwareEvent ----------------------------- */

/// The generic hardware events perf names cpu-cycles, instructions,
/// branch-misses and cache-misses.
enum class HardwareEvent : std::uint8_t { CPU_CYCLES, INSTRUCTIONS, BRANCH_MISSES, CACHE_MISSES };

/// perf's name for @p event, as perf stat prints it.
inline const char* toString(HardwareEvent event) noexcept {
  switch (event) {
  case HardwareEvent::CPU_CYCLES:
    return "cpu-cycles";
  case HardwareEvent::INSTRUCTIONS:
    return "instructions";
  case HardwareEvent::BRANCH_MISSES:
    return "branch-misses";
  case HardwareEvent::CACHE_MISSES:
    return "cache-misses";
  }
  return "unknown";
}

/* ----------------------------- Constants ----------------------------- */

/// A reading whose counter was on the PMU for less than this share of the
/// time the thread ran is refused: the scaled count would rest more on the
/// assumption that the uncounted part looked like the counted part than on
/// the count. Above it the count is scaled and the share reported.
inline constexpr double MIN_RUN_FRACTION = 0.5;

/* ----------------------------- Reading ----------------------------- */

/// One count, with the share of the thread's time the counter was on the
/// PMU for.
struct Reading {
  double value = 0.0;       ///< The count (per call from perCall()), scaled by enabled/running
  double runFraction = 0.0; ///< Time running over time enabled: 1 throughout, 0 never on the PMU
  std::uint64_t raw = 0;    ///< The count as read, unscaled

  /** @return True when the counter ran at least MIN_RUN_FRACTION of the time. */
  [[nodiscard]] bool counted() const noexcept { return runFraction >= MIN_RUN_FRACTION; }

  /**
   * @brief Why this reading cannot be used, or an empty string when it can.
   * @param zeroIsImpossible True when the work counted cannot count zero (any
   *        code retires instructions; a loop with a data-dependent branch on
   *        random input mispredicts), so a zero is a counter that did not
   *        count, whatever its times say.
   */
  [[nodiscard]] std::string whyNotCounted(bool zeroIsImpossible) const {
    if (!counted()) {
      return "the counter was on the PMU for " +
             std::to_string(static_cast<int>(runFraction * 100.0)) +
             "% of the time the thread ran (a hybrid processor counts an event on one kind of core "
             "only, and a PMU with more events open than counters takes turns): pin the run to one "
             "core of the counted kind";
    }
    if (zeroIsImpossible && raw == 0) {
      return "the counter ran and counted nothing for work that cannot count zero";
    }
    return {};
  }
};

/// The reasons in @p reasons that are not empty, joined with "; ": one
/// string for a test that skips on any of several readings.
inline std::string joinReasons(std::initializer_list<std::string> reasons) {
  std::string joined;
  for (const std::string& reason : reasons) {
    if (reason.empty()) {
      continue;
    }
    joined += (joined.empty() ? "" : "; ") + reason;
  }
  return joined;
}

/// " (counted N% of the time)" when @p reading was scaled, else empty, for
/// a printed figure.
inline std::string scalingNote(const Reading& reading) {
  if (reading.runFraction >= 0.999) {
    return {};
  }
  return " (counted " + std::to_string(static_cast<int>(reading.runFraction * 100.0)) +
         "% of the time)";
}

/* ----------------------------- HardwareCounter ----------------------------- */

/**
 * @brief A counter for one hardware event on the calling thread's user-space
 *        work, opened on construction.
 * @note NOT RT-safe: a system call to open, and three per count.
 */
class HardwareCounter {
public:
  explicit HardwareCounter(HardwareEvent event);
  ~HardwareCounter();
  HardwareCounter(const HardwareCounter&) = delete;
  HardwareCounter& operator=(const HardwareCounter&) = delete;

  /** @return True when the counter opened and count() can be used. */
  [[nodiscard]] bool isOpen() const noexcept { return fd_ >= 0; }

  /** @return Why the counter did not open; empty when it did. */
  [[nodiscard]] const std::string& failure() const noexcept { return failure_; }

  /**
   * @brief The event's count over one run of @p op, with the share of the
   *        thread's time the counter was on the PMU for.
   * @return A reading; empty (never counted) when the counter is not open.
   */
  template <typename Op> Reading count(Op&& op) {
    Reading reading;
#ifdef __linux__
    if (fd_ < 0) {
      op();
      return reading;
    }
    // The count resets; the two times only accumulate, so the window's
    // share is the difference of two readings around it.
    std::array<std::uint64_t, 3> before{};
    std::array<std::uint64_t, 3> after{};
    ::ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
    const bool READ_BEFORE = readCounter(before);
    ::ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
    op();
    ::ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    if (!READ_BEFORE || !readCounter(after)) {
      return reading;
    }
    const std::uint64_t ENABLED = after[1] - before[1];
    const std::uint64_t RUNNING = after[2] - before[2];
    reading.raw = after[0];
    if (ENABLED == 0 || RUNNING == 0) {
      return reading;
    }
    reading.runFraction = static_cast<double>(RUNNING) / static_cast<double>(ENABLED);
    reading.value = static_cast<double>(after[0]) / reading.runFraction;
    return reading;
#else
    op();
    return reading;
#endif
  }

  /**
   * @brief The event's count per call over @p calls calls of @p op, after one
   *        call that is not counted and warms the caches and the allocator.
   */
  template <typename Op> Reading perCall(int calls, Op&& op) {
    op();
    Reading reading = count([&] {
      for (int call = 0; call < calls; ++call) {
        op();
      }
    });
    reading.value /= static_cast<double>(calls);
    return reading;
  }

private:
  /// The count, the time enabled and the time running, in that order.
  bool readCounter(std::array<std::uint64_t, 3>& out) const noexcept {
#ifdef __linux__
    return ::read(fd_, out.data(), sizeof(out)) == static_cast<ssize_t>(sizeof(out));
#else
    (void)out;
    return false;
#endif
  }

  int fd_ = -1;
  std::string failure_;
};

/* ----------------------------- HardwareCounter Methods ----------------------------- */

#ifdef __linux__

inline HardwareCounter::HardwareCounter(HardwareEvent event) {
  perf_event_attr attr{};
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof(attr);
  switch (event) {
  case HardwareEvent::CPU_CYCLES:
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    break;
  case HardwareEvent::INSTRUCTIONS:
    attr.config = PERF_COUNT_HW_INSTRUCTIONS;
    break;
  case HardwareEvent::BRANCH_MISSES:
    attr.config = PERF_COUNT_HW_BRANCH_MISSES;
    break;
  case HardwareEvent::CACHE_MISSES:
    attr.config = PERF_COUNT_HW_CACHE_MISSES;
    break;
  }
  attr.disabled = 1;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

  // This thread (pid 0), any CPU (-1), no group, no flags.
  const long FD = ::syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
  if (FD >= 0) {
    fd_ = static_cast<int>(FD);
    return;
  }

  const int ERR = errno;
  failure_ = std::string("perf_event_open for ") + toString(event) + ": " + std::strerror(ERR);
  if (ERR == EACCES || ERR == EPERM) {
    std::ifstream paranoid("/proc/sys/kernel/perf_event_paranoid");
    int level = 0;
    if (paranoid >> level) {
      failure_ += " (kernel.perf_event_paranoid=" + std::to_string(level) +
                  "; 2 or lower lets a user count its own process)";
    }
  } else if (ERR == ENOENT || ERR == EOPNOTSUPP || ERR == ENODEV) {
    failure_ += " (this processor's PMU has no such event)";
  }
}

inline HardwareCounter::~HardwareCounter() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

#else

inline HardwareCounter::HardwareCounter(HardwareEvent event)
    : failure_(std::string("no hardware counter for ") + toString(event) +
               ": perf_event_open is Linux only") {}

inline HardwareCounter::~HardwareCounter() = default;

#endif

} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_HARDWARECOUNTER_HPP
