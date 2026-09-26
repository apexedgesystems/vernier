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
 */

#include <cstddef>
#include <cstdint>

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

/* ----------------------------- HardwareCounter ----------------------------- */

/**
 * @brief A counter for one hardware event on the calling thread's user-space
 *        work, opened on construction.
 * @note NOT RT-safe: a system call to open, and one per count.
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
   * @brief The event's count over one run of @p op.
   * @return The count, or 0 when the counter is not open.
   */
  template <typename Op> std::uint64_t count(Op&& op) {
#ifdef __linux__
    if (fd_ < 0) {
      op();
      return 0;
    }
    ::ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
    ::ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
    op();
    ::ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    std::uint64_t value = 0;
    if (::read(fd_, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value))) {
      return 0;
    }
    return value;
#else
    op();
    return 0;
#endif
  }

  /**
   * @brief The event's count per call over @p calls calls of @p op, after one
   *        call that is not counted and warms the caches and the allocator.
   */
  template <typename Op> double perCall(int calls, Op&& op) {
    op();
    const std::uint64_t TOTAL = count([&] {
      for (int call = 0; call < calls; ++call) {
        op();
      }
    });
    return static_cast<double>(TOTAL) / static_cast<double>(calls);
  }

private:
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
