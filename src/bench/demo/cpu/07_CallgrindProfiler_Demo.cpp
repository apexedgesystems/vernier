/**
 * @file 07_CallgrindProfiler_Demo.cpp
 * @brief Demo 07: exact instruction counts with callgrind
 *
 * Measures the two versions of the shared join example:
 *  1. JoinV0 and JoinV1 each measure one version, so each writes its own row
 *  2. Profiled under `bench run --profile callgrind`, the same two tests show
 *     which functions and source lines execute the instructions
 *
 * The check that V0 keeps executing several times V1's instructions per call
 * is not part of this demo: it is the join example's JoinInstructionCounts
 * (examples/join/utst/JoinInstructionCounts.cpp), which ctest runs.
 *
 * Usage:
 *   @code{.sh}
 *   # Time both versions
 *   ./BenchDemo_07_CallgrindProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Count their instructions under callgrind, then read the counts
 *   bench run ./BenchDemo_07_CallgrindProfiler --profile callgrind --cycles 10 --repeats 1
 *   callgrind_annotate bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out
 *   @endcode
 *
 * @see docs/07_CALLGRIND_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, the size demo 01 measures. V0's cost grows with its square.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(CallgrindProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(CallgrindProfiler, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
