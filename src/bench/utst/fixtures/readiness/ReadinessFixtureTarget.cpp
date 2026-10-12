/**
 * @file ReadinessFixtureTarget.cpp
 * @brief A benchmark with two guarded cases and one bare case, for the
 * readiness CLI tests.
 *
 * Built for the tests only, never installed. The guarded cases create their
 * profiler through the profiler guard, so a --profile request is decided,
 * reported and, when it can run, attached exactly as in any benchmark; two
 * cases show that one decision serves every case of a run. The third case is
 * built without the guard and creates no profiler: run alone, it shows what a
 * run reports when no case could be profiled.
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

PERF_TEST(ReadinessFixture, Bare) {
  UB_PERF_GUARD_NOPROFILE(perf);
  volatile std::uint64_t acc = 0;
  perf.throughputLoop([&] { acc = acc + 3; });
}

PERF_MAIN()
