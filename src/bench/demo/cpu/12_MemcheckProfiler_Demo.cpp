/**
 * @file 12_MemcheckProfiler_Demo.cpp
 * @brief Demo 12: memcheck -- a memory error a timer cannot see
 *
 * Measures the two versions of the shared join example and carries a third,
 * deliberately wrong join for memcheck to find:
 *  1. JoinV0 and JoinV1 each measure one version, one CSV row each
 *  2. JoinOffByOne calls the wrong join, which returns the right string and
 *     writes one byte past its buffer; it runs only under valgrind and skips
 *     itself anywhere else
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_12_MemcheckProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Find the bug: memcheck's log lands in
 *   # bench-out/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log
 *   bench run ./BenchDemo_12_MemcheckProfiler --profile memcheck -- \
 *     --gtest_filter=Memcheck.JoinOffByOne
 *
 *   # Read it
 *   cat bench-out/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log
 *   @endcode
 *
 * What memcheck reports for this binary is checked apart from it, by
 * utst/12_MemcheckProfiler_uTest.cpp.
 *
 * @see docs/15_MEMCHECK_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/cpu/12_MemcheckProfiler_OffByOne.hpp"
#include "src/bench/demo/cpu/12_MemcheckProfiler_Workload.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"

namespace demo = vernier::bench::demo;
namespace wrong = vernier::bench::demo::memcheck_demo;

using vernier::bench::demo::memcheck_demo::OFF_BY_ONE_CALLS;
using vernier::bench::demo::memcheck_demo::PART_COUNT;
using vernier::bench::demo::memcheck_demo::PART_SEED;
using vernier::bench::demo::memcheck_demo::SEPARATOR;

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: two temporaries and a full copy per part. */
PERF_THROUGHPUT(Memcheck, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: one allocation per call. */
PERF_THROUGHPUT(Memcheck, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/**
 * @test The wrong join returns the right string OFF_BY_ONE_CALLS times, and
 *       writes past its buffer every time. Runs only under valgrind.
 *
 * Measures nothing: under valgrind a timing means nothing, and a fixed number
 * of calls is what memcheck's error count divides by. Anywhere else the case
 * skips itself and says how to run it.
 */
PERF_TEST(Memcheck, JoinOffByOne) {
  DEMO_SKIP_UNLESS_UNDER_VALGRIND();

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  const std::string EXPECTED = demo::joinV1(PARTS, SEPARATOR);
  for (long call = 0; call < OFF_BY_ONE_CALLS; ++call) {
    EXPECT_EQ(wrong::joinOffByOne(PARTS, SEPARATOR), EXPECTED) << "call " << call;
  }
}

PERF_MAIN()
