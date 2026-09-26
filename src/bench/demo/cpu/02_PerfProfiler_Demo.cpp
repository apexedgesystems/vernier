/**
 * @file 02_PerfProfiler_Demo.cpp
 * @brief Demo 02: perf hardware counters -- what the processor did, at full speed
 *
 * Measures two shared examples, one version per test, so each can be run
 * under --profile perf on its own, and checks what the counters say:
 *  1. JoinV0 and JoinV1 measure the join example; JoinInstructions counts
 *     the instructions each version retires per call and fails when V0 stops
 *     retiring many times more than V1
 *  2. FilterBranchyRandom, FilterBranchySorted and FilterBranchless measure
 *     the filter example; FilterBranchMisses counts the branch misses per
 *     call of each case and fails when the random input stops mispredicting
 *     far more than the sorted input and than the branchless version
 *  3. The two checks count through the kernel's perf interface on their own
 *     calls (helpers/HardwareCounter.hpp), write no CSV row, and skip where
 *     the counter cannot be opened, and under --profile
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
#include <cstdio>

#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/filter/inc/Filter.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/demo/helpers/HardwareCounter.hpp"

namespace ub = vernier::bench;
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

/// Calls counted per version or case in the checks.
static constexpr int COUNTED_CALLS = 10;

/// joinV0 must retire at least this many times the instructions joinV1
/// retires per call. On the reference rig the ratio is about 33; joinV0
/// building like joinV1 gives 1.
static constexpr double MIN_INSTRUCTION_RATIO = 10.0;

/// filterBranchy on random input must mispredict at least this many
/// branches per value: a branch taken half the time in no learnable order
/// mispredicts about half the time, and a loop whose test became a
/// conditional select mispredicts almost nothing.
static constexpr double MIN_MISPREDICTS_PER_VALUE = 0.1;

/// And at least this many times as many per call as the same function on
/// sorted input, and as filterBranchless on the same random input. On the
/// reference rig both ratios are in the thousands.
static constexpr double MIN_MISPREDICT_RATIO = 10.0;

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

/**
 * @test joinV0 retires many times the instructions joinV1 retires, per call.
 *
 * Counts the instructions each version retires over COUNTED_CALLS calls,
 * through the same kernel interface perf stat reads, and fails when V0's
 * count per call is no longer MIN_INSTRUCTION_RATIO times V1's: the copying
 * and allocating that V0's one-liner asks for is the work the counters show.
 * Writes no CSV row.
 */
PERF_TEST(PerfProfiler, JoinInstructions) {
  if (!ub::detail::getPerfConfig().profileTool.empty()) {
    GTEST_SKIP() << "counts on its own; run it without --profile";
  }
  demo::HardwareCounter instructions(demo::HardwareEvent::INSTRUCTIONS);
  if (!instructions.isOpen()) {
    GTEST_SKIP() << "cannot count instructions here: " << instructions.failure();
  }

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR), demo::joinV1(PARTS, SEPARATOR));

  volatile std::size_t sink = 0;
  const double V0 =
      instructions.perCall(COUNTED_CALLS, [&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  const double V1 =
      instructions.perCall(COUNTED_CALLS, [&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });

  std::printf("[PerfProfiler.JoinInstructions]  V0 %.0f instructions/call  V1 %.0f "
              "instructions/call  %.1fx\n",
              V0, V1, V0 / V1);

  EXPECT_GE(V0, MIN_INSTRUCTION_RATIO * V1)
      << "joinV0 retired " << V0 << " instructions per call and joinV1 " << V1
      << ": V0 no longer does " << MIN_INSTRUCTION_RATIO
      << " times V1's work, and the demo has stopped demonstrating";
}

/**
 * @test The branchy filter mispredicts on random input, and neither the same
 *       filter on sorted input nor the branchless filter does.
 *
 * Counts the branch misses per call of the three cases, through the same
 * kernel interface perf stat reads. Fails when the random case mispredicts
 * fewer than MIN_MISPREDICTS_PER_VALUE per value (the branch is gone: a
 * conditional select in its place) or fewer than MIN_MISPREDICT_RATIO times
 * the sorted case or the branchless case (the contrast the walkthrough shows
 * is gone). Writes no CSV row.
 */
PERF_TEST(PerfProfiler, FilterBranchMisses) {
  if (!ub::detail::getPerfConfig().profileTool.empty()) {
    GTEST_SKIP() << "counts on its own; run it without --profile";
  }
  demo::HardwareCounter misses(demo::HardwareEvent::BRANCH_MISSES);
  if (!misses.isOpen()) {
    GTEST_SKIP() << "cannot count branch-misses here: " << misses.failure();
  }

  const auto RANDOM = demo::makeValues(VALUE_COUNT, VALUE_SEED);
  const auto SORTED = demo::makeSortedValues(VALUE_COUNT, VALUE_SEED);
  std::vector<double> out(VALUE_COUNT);
  ASSERT_EQ(demo::filterBranchy(RANDOM, THRESHOLD, out),
            demo::filterBranchless(RANDOM, THRESHOLD, out));

  volatile std::size_t sink = 0;
  const double BRANCHY_RANDOM =
      misses.perCall(COUNTED_CALLS, [&] { sink = demo::filterBranchy(RANDOM, THRESHOLD, out); });
  const double BRANCHY_SORTED =
      misses.perCall(COUNTED_CALLS, [&] { sink = demo::filterBranchy(SORTED, THRESHOLD, out); });
  const double BRANCHLESS =
      misses.perCall(COUNTED_CALLS, [&] { sink = demo::filterBranchless(RANDOM, THRESHOLD, out); });

  std::printf("[PerfProfiler.FilterBranchMisses]  branchy random %.0f  branchy sorted %.0f  "
              "branchless %.0f  branch-misses/call\n",
              BRANCHY_RANDOM, BRANCHY_SORTED, BRANCHLESS);

  const double PER_VALUE = BRANCHY_RANDOM / static_cast<double>(VALUE_COUNT);
  EXPECT_GE(PER_VALUE, MIN_MISPREDICTS_PER_VALUE)
      << "filterBranchy mispredicted " << PER_VALUE
      << " branches per value on random input: its test of each value is no longer a branch";
  EXPECT_GE(BRANCHY_RANDOM, MIN_MISPREDICT_RATIO * BRANCHY_SORTED)
      << "filterBranchy mispredicted " << BRANCHY_RANDOM << " branches per call on random input "
      << "and " << BRANCHY_SORTED << " on sorted input: the order of the data no longer matters";
  EXPECT_GE(BRANCHY_RANDOM, MIN_MISPREDICT_RATIO * BRANCHLESS)
      << "filterBranchy mispredicted " << BRANCHY_RANDOM << " branches per call and "
      << "filterBranchless " << BRANCHLESS << ": the branchless version no longer removes them";
}

PERF_MAIN()
