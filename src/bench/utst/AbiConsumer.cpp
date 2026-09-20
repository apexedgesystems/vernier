/**
 * @file AbiConsumer.cpp
 * @brief A benchmark with its own main() that builds one profiler, used to
 * check the start-up layout check with really different headers.
 *
 * Compiled twice against the same libbench: once with the tree's headers
 * (control) and once with a copy in which PerfConfig has one more member. It
 * calls no check itself: Profiler::make() is the first contact with libbench.
 *
 * Usage: AbiConsumer <artifact-root>
 */

#include "src/bench/inc/Profiler.hpp"

#include <cstdio>

int main(int argc, char** argv) {
  vernier::bench::PerfConfig cfg;
  cfg.profileTool = "jemalloc";
  cfg.artifactRoot = (argc > 1) ? argv[1] : ".";
  const vernier::bench::PerfConfig COPIED = cfg;

  std::fprintf(stderr, "consumer: sizeof(PerfConfig)=%zu\n", sizeof(vernier::bench::PerfConfig));
  // A long test name keeps the strings involved off the small-string path.
  const auto PROFILER = vernier::bench::Profiler::make(
      COPIED, "AbiConsumer.ATestNameLongEnoughToBeAllocatedOnTheHeapByStdString");
  std::fprintf(stderr, "consumer: constructed profiler '%s'\n", PROFILER->toolName().c_str());
  return 0;
}
