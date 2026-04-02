#ifndef VERNIER_MONITORSCOPEGUARD_HPP
#define VERNIER_MONITORSCOPEGUARD_HPP
/**
 * @file MonitorScopeGuard.hpp
 * @brief RAII scope timer that auto-records duration on destruction.
 */

#include "src/monitor/inc/MonitorTag.hpp"

namespace vernier {
namespace monitor {

/* ----------------------------- Forward Declarations ----------------------------- */

class Monitor;

/* ----------------------------- ScopeGuard ----------------------------- */

/**
 * @brief RAII guard that records a SCOPE sample from construction to destruction.
 * @note RT-safe: Construction and destruction are lock-free and allocation-free.
 */
class ScopeGuard {
public:
  ScopeGuard(Monitor& mon, const char* scopeName, const MonitorTag& tag) noexcept
      : monitor_{mon}, tag_{tag}, startNs_{nowNs()} {
    if (scopeName) {
      std::strncpy(scope_, scopeName, sizeof(scope_) - 1);
      scope_[sizeof(scope_) - 1] = '\0';
    }
  }

  ~ScopeGuard() noexcept; // Defined in Monitor.hpp after Monitor is complete

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

private:
  Monitor& monitor_;
  MonitorTag tag_;
  std::uint64_t startNs_;
  char scope_[32]{};
};

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORSCOPEGUARD_HPP
