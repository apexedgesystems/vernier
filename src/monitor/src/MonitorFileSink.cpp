/**
 * @file MonitorFileSink.cpp
 * @brief Implementation of MonitorFileSink operations.
 */

#include "src/monitor/inc/MonitorFileSink.hpp"

#include <cerrno>
#include <cstring>

namespace vernier {
namespace monitor {

/* ----------------------------- MonitorFileSink Methods ----------------------------- */

MonitorFileSink::MonitorFileSink(const std::string& path, AlertLevel minLevel)
    : minLevel_{minLevel} {
  // Open with append mode for atomic writes
  fp_ = std::fopen(path.c_str(), "a");
  if (!fp_) {
    std::fprintf(stderr, "vernier::monitor: failed to open '%s': %s\n", path.c_str(),
                 std::strerror(errno));
  }
}

MonitorFileSink::~MonitorFileSink() {
  if (fp_) {
    std::fflush(fp_);
    std::fclose(fp_);
  }
}

void MonitorFileSink::write(const Sample& sample) {
  if (!fp_)
    return;

  // Filter by alert level
  if (sample.kind == SampleKind::THRESHOLD_BREACH) {
    // Breaches always pass
  } else if (minLevel_ >= AlertLevel::WARNING) {
    return;
  }

  // Tab-delimited: timestamp kind tag/id scope duration value [flags]
  char buf[512];
  int n = std::snprintf(buf, sizeof(buf), "%lu\t%s\t%s/%u\t%s\t%lu\t%.6f",
                        static_cast<unsigned long>(sample.timestampNs), kindStr(sample.kind),
                        sample.tag.name, sample.tag.id, sample.scope,
                        static_cast<unsigned long>(sample.durationNs), sample.value);

  if (sample.kind == SampleKind::THRESHOLD_BREACH && n > 0 &&
      static_cast<std::size_t>(n) < sizeof(buf) - 20) {
    n += std::snprintf(buf + n, sizeof(buf) - static_cast<std::size_t>(n), "\tTHRESHOLD_BREACH");
  }

  if (n > 0 && static_cast<std::size_t>(n) < sizeof(buf) - 1) {
    buf[n] = '\n';
    // Single write for atomic append
    std::fwrite(buf, 1, static_cast<std::size_t>(n + 1), fp_);
  }
}

void MonitorFileSink::flush() {
  if (fp_)
    std::fflush(fp_);
}

const char* MonitorFileSink::kindStr(SampleKind k) noexcept {
  switch (k) {
  case SampleKind::SCOPE:
    return "SCOPE";
  case SampleKind::COUNTER:
    return "COUNTER";
  case SampleKind::GAUGE:
    return "GAUGE";
  case SampleKind::THRESHOLD_BREACH:
    return "SCOPE";
  }
  return "UNKNOWN";
}

} // namespace monitor
} // namespace vernier
