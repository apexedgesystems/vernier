/**
 * @file 09_BpftraceProfiler_Demo.cpp
 * @brief Demo 09: bpftrace -- what the kernel did while a test ran
 *
 * Two pairs of cases, one CSV row each (09_BpftraceProfiler_Workload.hpp):
 *  1. WritePerLine writes the join example's 1,000 words to /dev/null as
 *     lines, one write() per line; WriteBatched writes the same text with one
 *     write(). bpftrace's write_latency.bt counts and times the writes.
 *  2. CoarseLock: threads join the words and add the length to one total,
 *     holding its lock for the whole call; NoSharing: each thread adds to a
 *     total of its own and hands it over once, when the thread ends. The lock
 *     makes threads sleep until another wakes them, which wakeup_latency.bt
 *     records.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure the writes, on one core
 *   taskset -c 3 ./BenchDemo_09_BpftraceProfiler --gtest_filter='BpftraceProfiler.Write*' \
 *     --target-time 50ms --repeats 10 --csv writes.csv
 *
 *   # Trace them; the histogram lands in
 *   # BpftraceProfiler.WritePerLine.bpf/write_latency.out.text
 *   BENCH_SUDO=1 ./BenchDemo_09_BpftraceProfiler --profile bpftrace --bpf write_latency \
 *     --gtest_filter=BpftraceProfiler.WritePerLine --cycles 20 --repeats 2
 *
 *   # Wakeups while three threads take turns at the lock
 *   BENCH_SUDO=1 taskset -c 0-3 ./BenchDemo_09_BpftraceProfiler --profile bpftrace \
 *     --bpf wakeup_latency --gtest_filter=BpftraceProfiler.CoarseLock --threads 3
 *   @endcode
 *
 * What the writes and the totals do, and what write_latency.bt counts for
 * WritePerLine, is checked apart from the demo, by
 * utst/09_BpftraceProfiler_uTest.cpp.
 *
 * @see docs/09_BPFTRACE_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <string>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/cpu/09_BpftraceProfiler_Workload.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

using vernier::bench::demo::bpftrace_demo::addToThreadTotal;
using vernier::bench::demo::bpftrace_demo::addUnderCoarseLock;
using vernier::bench::demo::bpftrace_demo::LINE_END;
using vernier::bench::demo::bpftrace_demo::linesOf;
using vernier::bench::demo::bpftrace_demo::PART_COUNT;
using vernier::bench::demo::bpftrace_demo::PART_SEED;
using vernier::bench::demo::bpftrace_demo::SharedTotal;
using vernier::bench::demo::bpftrace_demo::writeBatched;
using vernier::bench::demo::bpftrace_demo::writeEachLine;

/* ----------------------------- Tests ----------------------------- */

/** @test 1,000 lines to /dev/null, one write() per line. */
PERF_IO(BpftraceProfiler, WritePerLine) {
  PERF_GUARD(perf);

  const auto LINES = linesOf(demo::makeParts(PART_COUNT, PART_SEED));
  const int FD = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
  ASSERT_GE(FD, 0) << "cannot open /dev/null";

  perf.warmup([&] { writeEachLine(FD, LINES); });
  perf.throughputLoop([&] { writeEachLine(FD, LINES); }, "write_per_line");
  ::close(FD);
}

/** @test The same 1,000 lines, end to end, with one write(). */
PERF_IO(BpftraceProfiler, WriteBatched) {
  PERF_GUARD(perf);

  const std::string TEXT = demo::joinV1(demo::makeParts(PART_COUNT, PART_SEED), LINE_END);
  const int FD = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
  ASSERT_GE(FD, 0) << "cannot open /dev/null";

  perf.warmup([&] { writeBatched(FD, TEXT); });
  perf.throughputLoop([&] { writeBatched(FD, TEXT); }, "write_batched");
  ::close(FD);
}

/** @test Threads add joined lengths to one total, holding its lock for the whole call. */
PERF_CONTENTION(BpftraceProfiler, CoarseLock) {
  PERF_GUARD(perf);

  const auto WORDS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  perf.warmup([&] { addUnderCoarseLock(total, WORDS); });
  perf.contentionRun([&] { addUnderCoarseLock(total, WORDS); }, "coarse_lock");
}

/** @test Threads add joined lengths to totals of their own, sharing nothing while they run. */
PERF_CONTENTION(BpftraceProfiler, NoSharing) {
  PERF_GUARD(perf);

  const auto WORDS = demo::makeParts(PART_COUNT, PART_SEED);

  perf.warmup([&] { addToThreadTotal(WORDS); });
  perf.contentionRun([&] { addToThreadTotal(WORDS); }, "no_sharing");
}

PERF_MAIN()
