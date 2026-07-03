// CI fast-path probe: measurement-only change; PR closes unmerged.
/**
 * @file MonitorEnvVarExample.cpp
 * @brief End-to-end example: env-var-driven Monitor configuration.
 *
 * The same binary stays silent in production and emits a full report once
 * the operator sets the relevant env var:
 *
 *   ./MonitorEnvVarExample                       # quiet (default config)
 *   VERNIER_MONITOR=1 ./MonitorEnvVarExample     # console output
 *
 *   VERNIER_MONITOR_FILE=/tmp/run.vmon \
 *   VERNIER_MONITOR_CONSOLE=WARNING \
 *       ./MonitorEnvVarExample                   # warnings on console, all
 *                                                # samples in /tmp/run.vmon
 *
 *   VERNIER_MONITOR_DISABLE=1 ./MonitorEnvVarExample  # hard-disable
 *
 * The application code does not need to change between these modes.
 */

#include "src/monitor/inc/Monitor.hpp"
#include "src/monitor/inc/MonitorConfig.hpp"
#include "src/monitor/inc/MonitorTag.hpp"

#include <chrono>
#include <thread>

namespace mon = vernier::monitor;

// MonitorTag's strncpy constructor is not constexpr; static const is fine.
static const mon::MonitorTag DECODER_TAG{"decoder", 1};
static const mon::MonitorTag IO_TAG{"io", 2};

// Simulates two distinct phases per frame, plus a periodic gauge sample.
static void processFrame(mon::Monitor& monitor, int frameIdx) {
  {
    VERNIER_MONITOR_SCOPE(monitor, "decode", DECODER_TAG);
    std::this_thread::sleep_for(std::chrono::microseconds(800));
  }
  {
    VERNIER_MONITOR_SCOPE(monitor, "render", DECODER_TAG);
    std::this_thread::sleep_for(std::chrono::microseconds(400));
  }
  VERNIER_MONITOR_INCREMENT(monitor, "frames", DECODER_TAG);
  VERNIER_MONITOR_GAUGE(monitor, "queue_depth", IO_TAG, frameIdx % 16);
}

int main() {
  auto cfg = mon::configFromEnv();
  mon::Monitor monitor(cfg);

  // Threshold: warn if any decode phase exceeds 5 ms. setThreshold takes the
  // numeric tag id (not the full MonitorTag) so the hot path can match it
  // against the per-sample tag.id without a string compare.
  monitor.setThreshold("decode", DECODER_TAG.id, 5000);

  // start() spins up the I/O drain thread and freezes the threshold table
  // for the RT-safe hot path. stop() (called by the destructor) joins the
  // thread, flushes, and prints the summary table to stderr.
  monitor.start();
  for (int i = 0; i < 50; ++i) {
    processFrame(monitor, i);
  }
  return 0;
}
