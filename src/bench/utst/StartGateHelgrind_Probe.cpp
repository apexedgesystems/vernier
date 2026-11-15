/**
 * @file StartGateHelgrind_Probe.cpp
 * @brief The racy probe StartGateHelgrind_uTest.cpp runs under helgrind as a
 *        child of this program: workers that add to a counter without a lock.
 *
 * A worker source of TestStartGateHelgrind (vernier_add_gtest's
 * WORKER_SOURCES): compiled in, never registered with CTest, since outside
 * valgrind it only skips.
 */

#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/utst/StartGateProbe.hpp"

#include <cstdint>

#include <gtest/gtest.h>

namespace {

/// Adds one to @p counter without a lock: the racy probe's race. The counter
/// is eight bytes wide, which neither of the gate's flags is.
[[gnu::noinline]] void addUnlocked(std::uint64_t& counter) { ++counter; }

} // namespace

/** @test The racy probe: workers add without a lock; runs only under valgrind */
TEST(StartGateProbe, RacyWorkers) {
  if (!vernier::bench::profiler_env::isRunningUnderValgrind()) {
    GTEST_SKIP() << "this probe races on purpose, for helgrind to find, so it runs only under "
                    "valgrind";
  }
  vernier::bench::PerfCase perf{vernier::bench::test::RACY_PROBE,
                                vernier::bench::test::probeConfig()};
  std::uint64_t counter = 0;

  perf.contentionRun([&] { addUnlocked(counter); }, "racy");
}
