#ifndef VERNIER_MONITORCONSOLESINK_HPP
#define VERNIER_MONITORCONSOLESINK_HPP
/**
 * @file MonitorConsoleSink.hpp
 * @brief Console output sink using snprintf (no fmt dependency).
 */

#include "src/monitor/inc/MonitorSink.hpp"

#include <cstdio>

namespace vernier {
namespace monitor {

/* ----------------------------- MonitorConsoleSink ----------------------------- */

/** @brief Console output sink that writes formatted samples to stderr. */
class MonitorConsoleSink : public MonitorSink {
public:
  explicit MonitorConsoleSink(AlertLevel minLevel = AlertLevel::INFO) : minLevel_{minLevel} {}

  /**
   * @brief Write a sample to stderr.
   * @param sample The sample to write.
   * @note NOT RT-safe: Console I/O.
   */
  void write(const Sample& sample) override {
    // Filter by alert level
    if (sample.kind == SampleKind::THRESHOLD_BREACH) {
      // Breaches always pass
    } else if (minLevel_ >= AlertLevel::WARNING) {
      return; // Only breaches at WARNING level
    }

    char buf[256];

    switch (sample.kind) {
    case SampleKind::SCOPE: {
      const double MS = static_cast<double>(sample.durationNs) / 1e6;
      std::snprintf(buf, sizeof(buf), "[monitor:%s/%u] %-24s %.3f ms\n", sample.tag.name,
                    sample.tag.id, sample.scope, MS);
      break;
    }
    case SampleKind::THRESHOLD_BREACH: {
      const double MS = static_cast<double>(sample.durationNs) / 1e6;
      const double THRESH_MS = sample.value / 1e3; // value stores threshold in us
      std::snprintf(buf, sizeof(buf), "[monitor:%s/%u] %-24s SLOW %.3f ms (threshold: %.3f ms)\n",
                    sample.tag.name, sample.tag.id, sample.scope, MS, THRESH_MS);
      break;
    }
    case SampleKind::COUNTER:
      std::snprintf(buf, sizeof(buf), "[monitor:%s/%u] %-24s +%.0f\n", sample.tag.name,
                    sample.tag.id, sample.scope, sample.value);
      break;
    case SampleKind::GAUGE:
      std::snprintf(buf, sizeof(buf), "[monitor:%s/%u] %-24s = %.3f\n", sample.tag.name,
                    sample.tag.id, sample.scope, sample.value);
      break;
    }

    std::fputs(buf, stderr);
  }

  /**
   * @brief Flush stderr.
   * @note NOT RT-safe: Console I/O.
   */
  void flush() override { std::fflush(stderr); }

private:
  AlertLevel minLevel_;
};

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORCONSOLESINK_HPP
