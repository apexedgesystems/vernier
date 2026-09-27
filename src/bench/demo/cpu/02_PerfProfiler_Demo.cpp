/**
 * @file 02_PerfProfiler_Demo.cpp
 * @brief Demo 02: perf hardware counters -- what the processor did, at full speed
 *
 * Measures two shared examples, one version per test, so each can be run
 * under --profile perf on its own:
 *  1. JoinV0 and JoinV1 measure the join example: V0 does far more work
 *  2. FilterBranchyRandom, FilterBranchySorted and FilterBranchless measure
 *     the filter example: the same work, with and without a branch the
 *     processor has to guess
 *
 * What the counters show about each is checked by the examples' unit tests,
 * not here: see the walkthrough's "What Keeps This Page True".
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_02_PerfProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Count one version's events over a known number of calls: writes
 *   # PerfProfiler.JoinV0.perf/stat.txt
 *   bench run ./BenchDemo_02_PerfProfiler --profile perf -- \
 *     --gtest_filter=PerfProfiler.JoinV0 --cycles 100 --repeats 10
 *
 *   # Read it
 *   cat PerfProfiler.JoinV0.perf/stat.txt
 *   @endcode
 *
 * @see docs/02_PERF_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/filter/inc/Filter.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, the same input as demo 01.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// Values per filter call: 800 KB of input, long enough that a call's fixed
/// cost is nothing next to its per-value work.
static constexpr std::size_t VALUE_COUNT = 100000;

/// Fixed seed: every run filters the same values.
static constexpr unsigned VALUE_SEED = 42;

/// Half the values are above it, so on random input the branch is taken
/// half the time in no order a predictor can learn.
static constexpr double THRESHOLD = 0.5;

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(PerfProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR), demo::joinV1(PARTS, SEPARATOR));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(PerfProfiler, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR), demo::joinV0(PARTS, SEPARATOR));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/** @test Throughput of the branchy filter on random-order values: an unlearnable branch. */
PERF_THROUGHPUT(PerfProfiler, FilterBranchyRandom) {
  PERF_GUARD(perf);

  const auto VALUES = demo::makeValues(VALUE_COUNT, VALUE_SEED);
  std::vector<double> out(VALUE_COUNT);
  ASSERT_EQ(demo::filterBranchy(VALUES, THRESHOLD, out),
            demo::filterBranchless(VALUES, THRESHOLD, out));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::filterBranchy(VALUES, THRESHOLD, out); });
  perf.throughputLoop([&] { sink = demo::filterBranchy(VALUES, THRESHOLD, out); },
                      "filter_branchy_random");
}

/** @test Throughput of the same filter on the same values ascending: the branch turns once. */
PERF_THROUGHPUT(PerfProfiler, FilterBranchySorted) {
  PERF_GUARD(perf);

  const auto VALUES = demo::makeSortedValues(VALUE_COUNT, VALUE_SEED);
  std::vector<double> out(VALUE_COUNT);
  ASSERT_EQ(demo::filterBranchy(VALUES, THRESHOLD, out),
            demo::filterBranchless(VALUES, THRESHOLD, out));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::filterBranchy(VALUES, THRESHOLD, out); });
  perf.throughputLoop([&] { sink = demo::filterBranchy(VALUES, THRESHOLD, out); },
                      "filter_branchy_sorted");
}

/** @test Throughput of the branchless filter on values in random order: nothing to predict. */
PERF_THROUGHPUT(PerfProfiler, FilterBranchless) {
  PERF_GUARD(perf);

  const auto VALUES = demo::makeValues(VALUE_COUNT, VALUE_SEED);
  std::vector<double> out(VALUE_COUNT);
  ASSERT_EQ(demo::filterBranchless(VALUES, THRESHOLD, out),
            demo::filterBranchy(VALUES, THRESHOLD, out));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::filterBranchless(VALUES, THRESHOLD, out); });
  perf.throughputLoop([&] { sink = demo::filterBranchless(VALUES, THRESHOLD, out); },
                      "filter_branchless");
}

PERF_MAIN()
