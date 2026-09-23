/**
 * @file MonitorEnvVarExample.cpp
 * @brief End-to-end example: env-var-driven Monitor configuration.
 *
 * The same binary serves several deployments; only the environment changes:
 *
 *   ./MonitorEnvVarExample                       # default config: a line per
 *                                                # sample plus the summary
 *                                                # table, both on stderr
 *
 *   VERNIER_MONITOR_CONSOLE=off \
 *       ./MonitorEnvVarExample                   # nothing on the console; the
 *                                                # samples are still collected
 *
 *   VERNIER_MONITOR_FILE=/tmp/run.vmon \
 *   VERNIER_MONITOR_CONSOLE=WARNING \
 *       ./MonitorEnvVarExample                   # threshold breaches and the
 *                                                # summary on the console,
 *                                                # every sample in the file
 *
 *   VERNIER_MONITOR_DISABLE=1 \
 *       ./MonitorEnvVarExample                   # records nothing, starts no
 *                                                # thread, writes no file
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

  // start() opens the configured sinks, freezes the threshold table the
  // recording path reads, and spins up the I/O drain thread. A disabled
  // configuration makes it do none of that.
  monitor.start();
  for (int i = 0; i < 50; ++i) {
    processFrame(monitor, i);
  }

  // The producer is done, so stop() can drain: it writes every queued sample
  // to the sinks, joins the thread, and prints the summary table when the
  // console sink is configured. The destructor would do the same; calling it
  // here keeps the reported window to the instrumented work.
  monitor.stop();
  return 0;
}
