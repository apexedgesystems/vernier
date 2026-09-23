/**
 * @file 01_BasicWorkflow_Demo.cpp
 * @brief Demo 01: Basic benchmarking workflow
 *
 * Measures the two versions of the shared join example and guards the
 * difference between them:
 *  1. One measurement per test, so each writes its own CSV row
 *  2. Export with --csv, read with bench summary, compare two runs
 *  3. JoinSpeedup fails when V0 stops being the slower version
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
#include <cstdio>

#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join. The cost of V0 grows with the square of this number.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// Calls per sample and samples per version in the guard test.
static constexpr int GUARD_CALLS = 10;
static constexpr int GUARD_SAMPLES = 5;

/// V0 must stay at least this many times slower than V1. The smallest ratio
/// measured while choosing this number came from an unoptimized build, and
/// was still close to twice it; an optimized build on either reference rig
/// measures several times more. Low enough not to flake on a loaded machine,
/// high enough that the two versions cannot converge without failing.
static constexpr double MIN_SPEEDUP = 3.0;

/* ----------------------------- File Helpers ----------------------------- */

namespace {

/// Bytes a joined string holds: every part plus one separator each.
std::size_t joinedSize(const std::vector<std::string>& parts) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }
  return total;
}

/// Median microseconds per call over GUARD_SAMPLES short batches. Used only
/// by the guard test, which compares two versions within one test and so
/// must not publish a row of its own.
double medianUsPerCall(const std::function<void()>& op) {
  std::vector<double> perCall;
  perCall.reserve(GUARD_SAMPLES);

  for (int sample = 0; sample < GUARD_SAMPLES; ++sample) {
    const auto START = std::chrono::steady_clock::now();
    for (int call = 0; call < GUARD_CALLS; ++call) {
      op();
    }
    const auto END = std::chrono::steady_clock::now();
    const double ELAPSED_US =
        std::chrono::duration<double, std::micro>(END - START).count() / GUARD_CALLS;
    perCall.push_back(ELAPSED_US);
  }

  return vernier::bench::summarize(perCall).median;
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(BasicWorkflow, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(BasicWorkflow, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/**
 * @test V0 stays the slower version by at least MIN_SPEEDUP.
 *
 * The two measured tests above cannot make this comparison between them
 * without carrying state from one test to the next, which --gtest_shuffle
 * would break. This test times both versions itself and writes no CSV row.
 */
PERF_TEST(BasicWorkflow, JoinSpeedup) {
  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR), demo::joinV1(PARTS, SEPARATOR));

  volatile std::size_t sink = 0;
  const double V0_US = medianUsPerCall([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  const double V1_US = medianUsPerCall([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });

  std::printf("[BasicWorkflow.JoinSpeedup]  V0 %.3f us/call  V1 %.3f us/call  %.1fx\n", V0_US,
              V1_US, V0_US / V1_US);

  EXPECT_GT(V0_US, MIN_SPEEDUP * V1_US)
      << "V0 " << V0_US << " us/call is not " << MIN_SPEEDUP << "x slower than V1 " << V1_US
      << " us/call: the demo has stopped demonstrating";
}

PERF_MAIN()
