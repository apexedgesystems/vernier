#ifndef VERNIER_MONITORTAG_HPP
#define VERNIER_MONITORTAG_HPP
/**
 * @file MonitorTag.hpp
 * @brief Fixed-size tag and sample types for runtime performance monitoring.
 */

#include <cstdint>
#include <cstring>

#include <chrono>

namespace vernier {
namespace monitor {

/* ----------------------------- MonitorTag ----------------------------- */

/**
 * @brief Lightweight label attached to every monitor instance and sample.
 *
 * Fixed-size, trivially copyable, zero heap allocation.
 */
struct MonitorTag {
  char name[32]{};     ///< Human-readable label, e.g. "decoder"
  std::uint16_t id{0}; ///< Numeric tag for fast filtering/grouping

  MonitorTag() = default;

  MonitorTag(const char* n, std::uint16_t i) noexcept : id{i} {
    if (n) {
      std::strncpy(name, n, sizeof(name) - 1);
      name[sizeof(name) - 1] = '\0';
    }
  }
};

/* ----------------------------- SampleKind ----------------------------- */

/** @brief Type of metric being recorded. */
enum class SampleKind : std::uint8_t {
  SCOPE = 0,        ///< Timed scope (duration in value)
  COUNTER,          ///< Monotonic counter increment
  GAUGE,            ///< Point-in-time value
  THRESHOLD_BREACH, ///< Scope that exceeded its threshold
};

/* ----------------------------- Sample ----------------------------- */

/**
 * @brief A single metric sample. Fixed-size, trivially copyable.
 *
 * Fits in two cache lines (~96 bytes).
 */
struct Sample {
  std::uint64_t timestampNs{0}; ///< steady_clock nanoseconds since epoch
  std::uint64_t durationNs{0};  ///< Duration (0 for counters/gauges)
  MonitorTag tag{};             ///< Instance label + id
  char scope[32]{};             ///< Scope name, e.g. "pipeline_stage"
  SampleKind kind{SampleKind::SCOPE};
  double value{0.0}; ///< Gauge value, counter delta, or duration (ns)

  Sample() = default;
};

/* ----------------------------- API ----------------------------- */

/**
 * @brief Current time as nanoseconds (steady_clock).
 * @return Nanosecond timestamp.
 * @note RT-safe: Bounded computation, no allocation.
 */
[[nodiscard]] inline std::uint64_t nowNs() noexcept {
  return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
}

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORTAG_HPP
