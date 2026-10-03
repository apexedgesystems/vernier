/**
 * @file 14_HelgrindProfiler_Demo.cpp
 * @brief Demo 14: helgrind -- a data race a check of the answer cannot see
 *
 * Threads add the length of the shared join example's result to one shared
 * total, two ways:
 *  1. LockedTotal takes a mutex for each addition and measures the calls with
 *     contentionRun, on the threads --threads asks for; one CSV row
 *  2. RacyTotal adds without the mutex, on four threads of its own; it runs
 *     only under valgrind and skips itself anywhere else
 *
 * Usage:
 *   @code{.sh}
 *   # Measure the locked version on four threads
 *   ./BenchDemo_14_HelgrindProfiler --threads 4 --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Find the race: helgrind's log lands in
 *   # bench-out/BenchDemo_14_HelgrindProfiler.helgrind/helgrind.log
 *   bench run ./BenchDemo_14_HelgrindProfiler --profile helgrind -- \
 *     --gtest_filter=Helgrind.RacyTotal
 *
 *   # Read it
 *   cat bench-out/BenchDemo_14_HelgrindProfiler.helgrind/helgrind.log
 *   @endcode
 *
 * What helgrind reports for this binary is checked apart from it, by
 * utst/14_HelgrindProfiler_uTest.cpp.
 *
 * @see docs/20_HELGRIND_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/demo/cpu/14_HelgrindProfiler_Racy.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;
namespace racy = vernier::bench::demo::helgrind_demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, the size demo 01 measures.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// Threads RacyTotal starts, each adding once.
static constexpr std::size_t RACY_THREADS = 4;

/// What RacyTotal prints when it skips: the race is helgrind's to find.
static constexpr const char* RACY_SKIP_REASON =
    "this case makes a data race for helgrind to find, so it runs only under valgrind: run this "
    "binary under `valgrind --tool=helgrind`, or through `bench run --profile helgrind`";

/* ----------------------------- Tests ----------------------------- */

/** @test Threads add joined lengths to one total, each addition under a mutex. */
PERF_CONTENTION(Helgrind, LockedTotal) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  std::mutex totalMutex;
  std::size_t total = 0;
  std::size_t calls = 0;
  const auto addJoinedLength = [&] {
    std::lock_guard<std::mutex> lock(totalMutex);
    total += demo::joinV1(PARTS, SEPARATOR).size();
    ++calls;
  };

  perf.warmup(addJoinedLength);
  perf.contentionRun(addJoinedLength, "locked_total");
  EXPECT_EQ(total, calls * demo::joinedSize(PARTS));
}

/**
 * @test Four threads add joined lengths to one total with no lock. Runs only
 *       under valgrind.
 *
 * Measures nothing: under valgrind a timing means nothing, and a fixed number
 * of additions is what the total is checked against. Anywhere else the case
 * skips itself and says how to run it.
 */
PERF_TEST(Helgrind, RacyTotal) {
  if (!vernier::bench::profiler_env::isRunningUnderValgrind()) {
    GTEST_SKIP() << RACY_SKIP_REASON;
  }

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  std::size_t total = 0;
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < RACY_THREADS; ++t) {
    threads.emplace_back([&] { racy::addJoinedLength(total, PARTS, SEPARATOR); });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(total, RACY_THREADS * demo::joinedSize(PARTS));
}

PERF_MAIN()
