/**
 * @file 08_RaplProfiler_Demo.cpp
 * @brief Demo 08: Intel RAPL -- the energy a call costs
 *
 * Measures the two versions of the shared join example, one measurement per
 * test, so each can be measured under --profile rapl on its own. RAPL reads
 * the package's energy counter, a model-specific register, before and after
 * the measured repeats, and the case's report gives the energy per call as
 * the window's average package power times the median time per call. The
 * counter covers the whole package (every core, the caches and the rest of
 * the package), not the case's thread alone.
 *
 * Reading the counter needs an Intel CPU, the msr driver's /dev/cpu/0/msr and
 * a process that may read it: root, or CAP_SYS_RAWIO together with read
 * access to the device.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure, about two seconds a case
 *   ./BenchDemo_08_RaplProfiler --target-time 100ms --repeats 20 --csv run.csv
 *
 *   # Measure the energy too, with access to the counter: each case's report
 *   # is energy.txt in its folder, RaplProfiler.JoinV0.rapl and RaplProfiler.JoinV1.rapl
 *   bench run ./BenchDemo_08_RaplProfiler --profile rapl --target-time 100ms --repeats 20 \
 *     --csv run.csv
 *   @endcode
 *
 * @see docs/08_RAPL_PROFILER.md for the step-by-step walkthrough
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
PERF_THROUGHPUT(RaplProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(RaplProfiler, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
