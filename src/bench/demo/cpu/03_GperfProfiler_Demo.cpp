/**
 * @file 03_GperfProfiler_Demo.cpp
 * @brief Demo 03: gperftools CPU profiler -- which function has the time
 *
 * Measures the two versions of the shared join example, one measurement per
 * test, so each writes its own CSV row and each can be profiled on its own:
 * run one of them under --profile gperf to get that version's profile, then
 * read it with google-pprof.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_03_GperfProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Profile one version: writes GperfProfiler.JoinV0.gperf/cpu.prof
 *   bench run ./BenchDemo_03_GperfProfiler --profile gperf -- \
 *     --gtest_filter=GperfProfiler.JoinV0 --target-time 200ms --repeats 10
 *
 *   # Read it
 *   google-pprof --text ./BenchDemo_03_GperfProfiler GperfProfiler.JoinV0.gperf/cpu.prof
 *   @endcode
 *
 * @see docs/03_GPERF_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, the same input as demo 01.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(GperfProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(GperfProfiler, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
