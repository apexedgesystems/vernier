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
    : minLevel_{minLevel}, path_{path} {
  // Open with append mode for atomic writes
  fp_ = std::fopen(path.c_str(), "a");
  if (!fp_) {
    std::fprintf(stderr, "vernier::monitor: failed to open '%s': %s\n", path.c_str(),
                 std::strerror(errno));
  }
}

MonitorFileSink::~MonitorFileSink() {
  if (!fp_) {
    return;
  }
  const bool FLUSHED = std::fflush(fp_) == 0;
  settle(FLUSHED, errno);
  const bool CLOSED = std::fclose(fp_) == 0;
  if (!CLOSED) {
    settle(false, errno);
  }
  fp_ = nullptr;
  if (lost_ > 0) {
    std::fprintf(stderr, "vernier::monitor: %llu of %llu records to '%s' were not written\n",
                 static_cast<unsigned long long>(lost_), static_cast<unsigned long long>(given_),
                 path_.c_str());
  }
}

void MonitorFileSink::settle(bool ok, int err) noexcept {
  if (!ok) {
    // The C library drops what was still buffered when a write failed.
    lost_ += given_ - settled_;
    if (!failed_) {
      failed_ = true;
      std::fprintf(stderr, "vernier::monitor: cannot write '%s': %s\n", path_.c_str(),
                   std::strerror(err));
    }
  }
  settled_ = given_;
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
    const std::size_t LEN = static_cast<std::size_t>(n + 1);
    ++given_;
    if (std::fwrite(buf, 1, LEN, fp_) != LEN) {
      settle(false, errno);
    }
  }
}

void MonitorFileSink::flush() {
  if (!fp_) {
    return;
  }
  const bool FLUSHED = std::fflush(fp_) == 0;
  settle(FLUSHED, errno);
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
