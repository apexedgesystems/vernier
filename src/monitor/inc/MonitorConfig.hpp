#ifndef VERNIER_MONITORCONFIG_HPP
#define VERNIER_MONITORCONFIG_HPP
/**
 * @file MonitorConfig.hpp
 * @brief Configuration for the runtime performance monitor.
 */

#include <cstddef>
#include <cstdint>

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

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORCONFIG_HPP
