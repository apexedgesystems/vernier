#ifndef VERNIER_MONITORFILESINK_HPP
#define VERNIER_MONITORFILESINK_HPP
/**
 * @file MonitorFileSink.hpp
 * @brief File output sink with atomic appends (O_APPEND).
 *
 * Writes tab-delimited records, one per line, for post-run analysis.
 * Format: timestamp_ns \t kind \t tag/id \t scope \t duration_ns \t value
 */

#include "src/monitor/inc/MonitorSink.hpp"

#include <cstdio>

#include <string>

namespace vernier {
namespace monitor {

/* ----------------------------- MonitorFileSink ----------------------------- */

/** @brief File output sink that writes tab-delimited sample records. */
class MonitorFileSink : public MonitorSink {
public:
  /**
   * @brief Construct a file sink, opening the given path for append.
   * @param path File path to write to.
   * @param minLevel Minimum alert level for filtering.
   * @note NOT RT-safe: File I/O.
   */
  explicit MonitorFileSink(const std::string& path, AlertLevel minLevel = AlertLevel::INFO);
  ~MonitorFileSink() override;

  MonitorFileSink(const MonitorFileSink&) = delete;
  MonitorFileSink& operator=(const MonitorFileSink&) = delete;

  /**
   * @brief Write a sample to the file.
   * @param sample The sample to write.
   * @note NOT RT-safe: File I/O.
   */
  void write(const Sample& sample) override;

  /**
   * @brief Flush buffered file output.
   * @note NOT RT-safe: File I/O.
   */
  void flush() override;

  /**
   * @brief True if the file was opened successfully.
   * @return Open status.
   * @note RT-safe: No I/O, reads cached pointer.
   */
  [[nodiscard]] bool isOpen() const noexcept { return fp_ != nullptr; }

private:
  [[nodiscard]] static const char* kindStr(SampleKind k) noexcept;

  FILE* fp_{nullptr};
  AlertLevel minLevel_;
};

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORFILESINK_HPP
