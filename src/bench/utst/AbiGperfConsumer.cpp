/**
 * @file AbiGperfConsumer.cpp
 * @brief A benchmark translation unit that reports how it lays out
 * GperfProfiler, against the size libbench was compiled with.
 *
 * Compiled twice against the same libbench: with the default include path,
 * and with a copy of it in which gperftools' headers are missing, as on a
 * host with the gperftools library but not its development package
 * (AbiGperfLayout.cmake). A benchmark may construct a GperfProfiler itself,
 * in storage of the size its own headers give, through a constructor libbench
 * runs, so both must give libbench's size.
 *
 * Usage: AbiGperfConsumer; prints one line and exits 0 when the sizes agree.
 */

#include "src/bench/inc/ProfilerGperf.hpp"

#include <cstddef>
#include <cstdio>

int main() {
#if defined(__has_include)
#if __has_include(<gperftools/profiler.h>)
  const char* const HEADERS = "visible";
#else
  const char* const HEADERS = "hidden";
#endif
#else
  const char* const HEADERS = "unknown";
#endif
  const std::size_t MINE = sizeof(vernier::bench::GperfProfiler);
  const std::size_t LIBRARY = vernier::bench::detail::gperfProfilerSize();
  std::printf("consumer: gperftools headers %s; sizeof(GperfProfiler) %zu here, %zu in libbench\n",
              HEADERS, MINE, LIBRARY);
  return MINE == LIBRARY ? 0 : 1;
}
