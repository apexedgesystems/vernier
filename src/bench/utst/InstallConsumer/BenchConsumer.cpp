/**
 * @file BenchConsumer.cpp
 * @brief A benchmark as a consumer of the installed package writes it: the
 * convenience header, one case, and the supplied main.
 *
 * Built once per include style: the short entry point, or the path from the
 * source root that the headers use among themselves.
 */

#if defined(VERNIER_CONSUMER_QUALIFIED_INCLUDES)
#include "src/bench/inc/Perf.hpp"
#else
#include "Perf.hpp"
#endif

#include <cstdint>

PERF_TEST(InstallConsumer, Sum) {
  PERF_GUARD(perf);
  volatile std::uint64_t acc = 0;
  perf.warmup([&] { acc = acc + 1; });
  perf.throughputLoop([&] { acc = acc + 1; }, "sum");
}

PERF_MAIN()
