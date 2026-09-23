/**
 * @file ConsumerTranslationUnit.cpp
 * @brief A benchmark file as a consumer writes it: the convenience header,
 * one case, and the supplied main.
 *
 * Compiled with warnings as errors under an optimized, fortified
 * configuration (see the CMake target), so a public header that warns in a
 * consumer's build fails this one. Never linked or run.
 */

#include "src/bench/inc/Perf.hpp"

#include <cstdint>

PERF_TEST(Consumer, Sum) {
  UB_PERF_GUARD(perf);
  volatile std::uint64_t acc = 0;
  perf.warmup([&] { acc = acc + 1; });
  perf.throughputLoop([&] { acc = acc + 1; });
}

PERF_MAIN()
