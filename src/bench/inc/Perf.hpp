#ifndef VERNIER_PERF_HPP
#define VERNIER_PERF_HPP
/**
 * @file Perf.hpp
 * @brief All-in-one convenience header for performance benchmarking utilities.
 *
 * Scope: Includes all public-facing headers for writing performance tests.
 * Users can include only this file instead of individual component headers.
 *
 * Typical usage:
 * @code{.cpp}
 *   #include "src/bench/inc/Perf.hpp"
 *
 *   PERF_TEST(MyLib, Throughput) {
 *     UB_PERF_GUARD(perf);
 *     perf.warmup([&]{ ... });
 *     perf.throughputLoop([&]{ ... });
 *   }
 *
 *   PERF_MAIN()
 * @endcode
 */

// Core configuration and flag parsing
#include "src/bench/inc/PerfConfig.hpp"

// Test harness (includes console printing, metadata capture)
#include "src/bench/inc/PerfHarness.hpp"

// Test declaration macros (PERF_TEST, PERF_MAIN, etc.)
#include "src/bench/inc/PerfTestMacros.hpp"

// CSV output listener for GoogleTest
#include "src/bench/inc/PerfListener.hpp"

// Optional profiler integration (perf, gperftools, bpftrace, callgrind)
#include "src/bench/inc/Profiler.hpp"

// Validation helpers (CV% thresholds, EXPECT_STABLE_CV)
#include "src/bench/inc/PerfValidation.hpp"

#endif // VERNIER_PERF_HPP