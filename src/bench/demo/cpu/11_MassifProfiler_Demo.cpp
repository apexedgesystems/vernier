/**
 * @file 11_MassifProfiler_Demo.cpp
 * @brief Demo 11: Valgrind Massif heap profiler -- peak heap, and who owns it
 *
 * Measures the two versions of the shared join example at 20,000 words, where
 * the strings are large enough to dominate the heap. One measurement per
 * test, so each writes its own CSV row, and each test can be run under massif
 * on its own to see its version's peak heap and the call sites that hold it.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_11_MassifProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Profile one version; --time-unit=B puts the bytes allocated and freed,
 *   # not instructions, on the graph's x-axis
 *   valgrind --tool=massif --time-unit=B --massif-out-file=massif.v0.out \
 *     ./BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 \
 *     --gtest_filter=Massif.JoinV0
 *   ms_print massif.v0.out
 *   @endcode
 *
 * @see docs/14_MASSIF_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Words per join. Large enough that the joined string and the words
/// themselves hold almost all of the heap, small enough that V0, whose cost
/// grows with the square of this number, stays under half a second per call
/// on the reference rig. The walkthrough records how it was chosen.
static constexpr std::size_t PART_COUNT = 20000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner at PART_COUNT words. */
PERF_THROUGHPUT(Massif, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version at PART_COUNT words. */
PERF_THROUGHPUT(Massif, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
