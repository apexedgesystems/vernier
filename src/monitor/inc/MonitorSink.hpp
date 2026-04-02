#ifndef VERNIER_MONITORSINK_HPP
#define VERNIER_MONITORSINK_HPP
/**
 * @file MonitorSink.hpp
 * @brief Abstract sink interface for monitor output.
 */

#include "src/monitor/inc/MonitorConfig.hpp"
#include "src/monitor/inc/MonitorTag.hpp"

namespace vernier {
namespace monitor {

/* ----------------------------- MonitorSink ----------------------------- */

/** @brief Base class for monitor output sinks. */
class MonitorSink {
public:
  virtual ~MonitorSink() = default;

  /**
   * @brief Write a sample to this sink. Called from the I/O thread only.
   * @param sample The sample to write.
   * @note NOT RT-safe: I/O operations.
   */
  virtual void write(const Sample& sample) = 0;

  /**
   * @brief Flush any buffered output.
   * @note NOT RT-safe: I/O operations.
   */
  virtual void flush() = 0;
};

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORSINK_HPP
