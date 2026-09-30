/**
 * @file ReadinessFixtureTarget.cpp
 * @brief A benchmark with two guarded cases, for the readiness CLI tests.
 *
 * Built for the tests only, never installed. Both cases create their profiler
 * through the profiler guard, so a --profile request is decided, reported
 * and, when it can run, attached exactly as in any benchmark; two cases show
 * that one decision serves every case of a run.
 */

#include "src/bench/inc/Perf.hpp"

#include <cstdint>

PERF_TEST(ReadinessFixture, First) {
  UB_PERF_GUARD(perf);
  volatile std::uint64_t acc = 0;
  perf.throughputLoop([&] { acc = acc + 1; });
}

PERF_TEST(ReadinessFixture, Second) {
  UB_PERF_GUARD(perf);
  volatile std::uint64_t acc = 0;
  perf.throughputLoop([&] { acc = acc + 2; });
}

PERF_MAIN()
