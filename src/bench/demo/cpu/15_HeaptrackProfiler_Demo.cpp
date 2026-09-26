/**
 * @file 15_HeaptrackProfiler_Demo.cpp
 * @brief Demo 15: heaptrack -- who allocates, and how often
 *
 * Measures the two versions of the shared join example, one version per test,
 * so each can be run under heaptrack on its own:
 *  1. One measurement per test, so each writes its own CSV row
 *  2. A profiled run makes a known number of calls: one to check the answer,
 *     the warmup, then cycles x repeats; heaptrack's counts divide by it
 *  3. Each test calls only its own version, so its trace holds no allocation
 *     of the other one
 *
 * The allocation counts this demo shows are guarded by the example's unit
 * tests (examples/join/utst/Join_uTest.cpp), which fail when V0 stops
 * allocating for its parts or V1 stops allocating once.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure, sizing each repeat to about 50 ms
 *   ./BenchDemo_15_HeaptrackProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Record V0 under heaptrack: 102 calls (check, warmup, 100 measured)
 *   bench run ./BenchDemo_15_HeaptrackProfiler --profile heaptrack --cycles 100 \
 *     --repeats 1 --profile-output-dir heaptrack-v0 -- --gtest_filter='Heaptrack.JoinV0'
 *
 *   # Read the trace (heaptrack writes .gz or .zst, depending on its build)
 *   heaptrack_print heaptrack-v0/BenchDemo_15_HeaptrackProfiler.heaptrack/run.*
 *   @endcode
 *
 * @see docs/21_HEAPTRACK_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, as in demo 01, so the two walkthroughs measure the same call.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: two temporaries and a full copy per part. */
PERF_THROUGHPUT(Heaptrack, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: one allocation per call. */
PERF_THROUGHPUT(Heaptrack, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
