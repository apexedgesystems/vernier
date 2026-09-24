/**
 * @file MonitorConsumer.cpp
 * @brief A program instrumented with the monitor as a consumer of the
 * installed package writes it.
 *
 * Built once per include style, like BenchConsumer.cpp. The file sink is the
 * part of the monitor compiled into libmonitor, so the sample file appearing
 * shows the installed library was loaded and ran.
 *
 * Usage: MonitorConsumer <sample-file>
 */

#if defined(VERNIER_CONSUMER_QUALIFIED_INCLUDES)
#include "src/monitor/inc/Monitor.hpp"
#else
#include "Monitor.hpp"
#endif

#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <sample-file>\n", argv[0]);
    return 2;
  }

  vernier::monitor::MonitorConfig cfg;
  cfg.sinks = vernier::monitor::SINK_FILE;
  cfg.filePath = argv[1];

  vernier::monitor::Monitor monitor(cfg);
  monitor.start();
  const vernier::monitor::MonitorTag consumer{"consumer", 1};
  VERNIER_MONITOR_INCREMENT(monitor, "runs", consumer);
  monitor.stop();
  return 0;
}
