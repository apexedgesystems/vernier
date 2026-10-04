/**
 * @file JoinSpeedup_pTest.cpp
 * @brief The timing check behind walkthrough 01: joinV0 stays at least three
 *        times slower than joinV1 at demo 01's 1,000 words.
 *
 * Demo 01 measures each version in a test of its own, one CSV row each, and
 * two tests cannot compare their timings without carrying state from one to
 * the next, which --gtest_shuffle would break. This program times both
 * versions in one test, with a timer of its own, and writes no CSV row.
 *
 * A timing belongs to the machine that takes it, so this is a performance
 * test: built into bin/ptests, never registered with ctest, and run by hand on
 * the reference rig before a release, as walkthrough 01 says.
 *
 * Usage:
 *   @code{.sh}
 *   taskset -c 3 ./build/bin/ptests/JoinSpeedup
 *   @endcode
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>

#include <chrono>
#include <functional>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Demo 01's input: 1,000 words from a fixed seed, joined by ','.
static constexpr std::size_t PART_COUNT = 1000;
static constexpr unsigned PART_SEED = 42;
static constexpr char SEPARATOR = ',';

/// Calls per sample and samples per version.
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

/// Median microseconds per call over GUARD_SAMPLES short batches. The check
/// compares two versions within one test, so it times them itself instead of
/// publishing a row per version.
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

/** @test V0 stays the slower version by at least MIN_SPEEDUP. */
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
