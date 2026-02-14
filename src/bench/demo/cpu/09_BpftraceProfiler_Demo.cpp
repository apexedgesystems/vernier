/**
 * @file 09_BpftraceProfiler_Demo.cpp
 * @brief Demo 09: bpftrace for syscall overhead analysis
 *
 * Demonstrates using bpftrace to identify syscall overhead from
 * inefficient I/O patterns. Shows how batching writes eliminates
 * per-byte syscall overhead.
 *
 * Slow: One write() syscall per byte (N syscalls for N bytes)
 * Fast: Single batched write() syscall (1 syscall for N bytes)
 *
 * Usage:
 *   @code{.sh}
 *   # Baseline measurement
 *   ./BenchDemo_09_BpftraceProfiler --csv baseline.csv
 *
 *   # Profile syscall latency (requires root)
 *   sudo ./BenchDemo_09_BpftraceProfiler --profile bpftrace --bpf syslat \
 *     --gtest_filter="*ManySmallWrites*"
 *
 *   # Compare
 *   bench summary baseline.csv
 *   @endcode
 *
 * @note Requires root/sudo for bpftrace. Falls back gracefully if unavailable.
 *
 * @see docs/09_BPFTRACE_PROFILER.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

// 1 KB payload: small enough for many-writes test to finish quickly
static constexpr std::size_t PAYLOAD_SIZE = 1024;

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: One write() syscall per byte.
 *
 * Each byte triggers a separate write() syscall to /dev/null.
 * The overhead is not in the data transfer but in the kernel
 * context switch for each syscall (~1-5 us per syscall).
 *
 * bpftrace with the syslat script will show a histogram of
 * write() latencies, with 1024 calls per measurement iteration.
 */
PERF_IO(BpftraceProfiler, ManySmallWrites) {
  UB_PERF_GUARD(perf);

  std::vector<std::uint8_t> data(PAYLOAD_SIZE, 0xAB);

  // Open /dev/null for writing
  int fd = ::open("/dev/null", O_WRONLY);
  ASSERT_GE(fd, 0) << "Failed to open /dev/null";

  perf.warmup([&] { demo::manySmallWrites(fd, data.data(), data.size()); });

  auto result = perf.throughputLoop([&] { demo::manySmallWrites(fd, data.data(), data.size()); },
                                    "many_small_writes");

  EXPECT_GT(result.callsPerSecond, 1.0);

  ::close(fd);
}

/**
 * @test Fast: Single batched write() syscall.
 *
 * All 1024 bytes go in a single write() syscall. One context switch
 * instead of 1024. The kernel handles the buffer in one pass.
 *
 * bpftrace will show 1 write() call per iteration instead of 1024.
 * The latency histogram shows a single peak instead of a cloud.
 *
 * Expected improvement: 100-1000x (dominated by syscall overhead reduction).
 */
PERF_IO(BpftraceProfiler, SingleBatchedWrite) {
  UB_PERF_GUARD(perf);

  std::vector<std::uint8_t> data(PAYLOAD_SIZE, 0xAB);

  int fd = ::open("/dev/null", O_WRONLY);
  ASSERT_GE(fd, 0) << "Failed to open /dev/null";

  perf.warmup([&] { demo::singleBatchedWrite(fd, data.data(), data.size()); });

  auto result = perf.throughputLoop([&] { demo::singleBatchedWrite(fd, data.data(), data.size()); },
                                    "single_batched_write");

  EXPECT_GT(result.callsPerSecond, 100.0);

  ::close(fd);
}

PERF_MAIN()
