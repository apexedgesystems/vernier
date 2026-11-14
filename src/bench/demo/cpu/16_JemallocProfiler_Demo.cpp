/**
 * @file 16_JemallocProfiler_Demo.cpp
 * @brief Demo 16: jemalloc's heap profiler -- which function allocates the bytes
 *
 * Measures the two versions of the shared join example, one version per test,
 * so each can be run under jemalloc's profiler on its own:
 *  1. One measurement per test, so each writes its own CSV row
 *  2. A profiled run makes a known number of calls: one to check the answer,
 *     the warmup, then cycles x repeats
 *  3. Each test calls only its own version, so its dump holds no allocation
 *     of the other one
 *
 * jemalloc samples allocations inside the allocator, about one per 512 KiB
 * allocated, with their stacks, and writes them when the process exits;
 * `jeprof --alloc_space` reads the bytes allocated over the run. It needs a
 * jemalloc built with profiling (configure --enable-prof). What this demo
 * shows is checked by TestDemoJemalloc (utst/16_JemallocProfiler_uTest.cpp),
 * which fails when JoinV0's bytes stop being sampled under joinV0 or JoinV1
 * stops allocating a small fraction of them.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure, sizing each repeat to about 50 ms
 *   ./BenchDemo_16_JemallocProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Profile V0: 102 calls (check, warmup, 100 measured); bench run preloads
 *   # jemalloc with prof:true,prof_final:true,prof_accum:true and checks the dump
 *   bench run ./BenchDemo_16_JemallocProfiler --profile jemalloc --cycles 100 \
 *     --repeats 1 --profile-output-dir jemalloc-v0 -- --gtest_filter='Jemalloc.JoinV0'
 *
 *   # Read the bytes allocated over the run, by function
 *   jeprof --text --alloc_space --cum ./BenchDemo_16_JemallocProfiler \
 *     jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.*.f.heap
 *   @endcode
 *
 * @see docs/22_JEMALLOC_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>

#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, as in demos 01 and 15, so the walkthroughs measure the same call.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: two temporaries and a full copy per part. */
PERF_THROUGHPUT(Jemalloc, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: one allocation per call. */
PERF_THROUGHPUT(Jemalloc, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

PERF_MAIN()
