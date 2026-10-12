/**
 * @file 01_BasicWorkflow_Demo.cpp
 * @brief Demo 01: Basic benchmarking workflow
 *
 * Measures the two versions of the shared join example, one measurement per
 * test, so each writes its own CSV row: export with --csv, read the rows with
 * bench summary, and compare two runs with bench compare.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure, sizing each repeat to about 50 ms
 *   ./BenchDemo_01_BasicWorkflow --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Read the rows
 *   bench summary run.csv
 *
 *   # Compare two runs
 *   bench compare baseline.csv run.csv
 *   @endcode
 *
 * @see docs/01_BASIC_WORKFLOW.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join. The cost of V0 grows with the square of this number.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(BasicWorkflow, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(BasicWorkflow, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
