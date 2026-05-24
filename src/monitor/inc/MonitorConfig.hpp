#ifndef VERNIER_MONITORCONFIG_HPP
#define VERNIER_MONITORCONFIG_HPP
/**
 * @file MonitorConfig.hpp
 * @brief Configuration for the runtime performance monitor.
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <string>

namespace vernier {
namespace monitor {

/* ----------------------------- AlertLevel ----------------------------- */

/** @brief Alert level for filtering which samples reach which sinks. */
enum class AlertLevel : std::uint8_t {
  INFO = 0, ///< All samples
  WARNING,  ///< Only threshold breaches and above
  CRITICAL, ///< Reserved for future use
};

/* ----------------------------- Sink ----------------------------- */

/** @brief Bitmask for selecting output sinks. */
enum Sink : std::uint8_t {
  SINK_NONE = 0,
  SINK_CONSOLE = 1 << 0,
  SINK_FILE = 1 << 1,
};

/* ----------------------------- MonitorConfig ----------------------------- */

/** @brief Monitor configuration. All fields have sensible defaults. */
struct MonitorConfig {
  std::size_t queueCapacity{4096};  ///< Ring buffer capacity (must be power of 2, minimum 64).
  std::uint8_t sinks{SINK_CONSOLE}; ///< Which sinks to enable (bitwise OR of Sink values).
  std::string filePath{}; ///< File path for the file sink (only used if SINK_FILE is set).
  AlertLevel consoleLevel{AlertLevel::INFO}; ///< Minimum alert level that reaches the console sink.
  AlertLevel fileLevel{AlertLevel::INFO};    ///< Minimum alert level that reaches the file sink.

  /// Global enable/disable switch. When false, all instrumentation
  /// becomes a no-op (checked once per sample, not per-call).
  bool enabled{true};
};

/* ----------------------------- API ----------------------------- */

/**
 * @brief Round up to next power of two (minimum 64).
 * @param v Input value.
 * @return Next power of two >= max(v, 64).
 * @note RT-safe: Bounded computation, no allocation.
 */
[[nodiscard]] inline std::size_t roundUpPow2(std::size_t v) noexcept {
  if (v < 64)
    v = 64;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

/* ----------------------------- Env-var helpers ----------------------------- */

namespace detail {

inline bool envTruthy(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return false;
  if (v[0] == '0' && v[1] == '\0') return false;
  return std::strcmp(v, "false") != 0 && std::strcmp(v, "off") != 0 &&
         std::strcmp(v, "no") != 0;
}

inline AlertLevel parseAlertLevel(const char* s, AlertLevel fallback) {
  if (!s || !*s) return fallback;
  if (std::strcmp(s, "INFO") == 0 || std::strcmp(s, "info") == 0) return AlertLevel::INFO;
  if (std::strcmp(s, "WARNING") == 0 || std::strcmp(s, "warning") == 0 ||
      std::strcmp(s, "WARN") == 0 || std::strcmp(s, "warn") == 0) {
    return AlertLevel::WARNING;
  }
  if (std::strcmp(s, "CRITICAL") == 0 || std::strcmp(s, "critical") == 0) {
    return AlertLevel::CRITICAL;
  }
  return fallback;
}

} // namespace detail

/**
 * @brief Build a MonitorConfig from environment variables.
 *
 * Recognized env vars:
 *
 *   VERNIER_MONITOR           truthy enables the monitor (default: same as the
 *                             struct default, currently true)
 *   VERNIER_MONITOR_DISABLE   truthy forces disabled (overrides VERNIER_MONITOR)
 *   VERNIER_MONITOR_FILE      path; sets filePath and turns on the file sink
 *   VERNIER_MONITOR_CONSOLE   INFO | WARNING | CRITICAL | off; controls
 *                             console sink and its minimum alert level
 *   VERNIER_MONITOR_QUEUE     ring-buffer capacity; rounded up to pow2
 *
 * Designed for zero-code-change enablement: a binary instrumented with the
 * VERNIER_MONITOR_* macros stays silent until the operator sets the env var.
 */
[[nodiscard]] inline MonitorConfig configFromEnv() {
  MonitorConfig cfg{};

  if (const char* v = std::getenv("VERNIER_MONITOR")) {
    cfg.enabled = detail::envTruthy("VERNIER_MONITOR");
    (void)v;
  }
  if (detail::envTruthy("VERNIER_MONITOR_DISABLE")) {
    cfg.enabled = false;
  }

  if (const char* path = std::getenv("VERNIER_MONITOR_FILE")) {
    if (*path) {
      cfg.filePath = path;
      cfg.sinks = static_cast<std::uint8_t>(cfg.sinks | SINK_FILE);
    }
  }

  if (const char* level = std::getenv("VERNIER_MONITOR_CONSOLE")) {
    if (std::strcmp(level, "off") == 0 || std::strcmp(level, "OFF") == 0) {
      cfg.sinks = static_cast<std::uint8_t>(cfg.sinks & ~SINK_CONSOLE);
    } else {
      cfg.sinks = static_cast<std::uint8_t>(cfg.sinks | SINK_CONSOLE);
      cfg.consoleLevel = detail::parseAlertLevel(level, cfg.consoleLevel);
    }
  }

  if (const char* q = std::getenv("VERNIER_MONITOR_QUEUE")) {
    const long parsed = std::strtol(q, nullptr, 10);
    if (parsed > 0) {
      cfg.queueCapacity = roundUpPow2(static_cast<std::size_t>(parsed));
    }
  }

  return cfg;
}

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORCONFIG_HPP
