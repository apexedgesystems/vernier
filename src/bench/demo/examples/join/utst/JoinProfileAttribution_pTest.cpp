/**
 * @file JoinProfileAttribution_pTest.cpp
 * @brief The profile check behind walkthrough 03: gperftools puts V0's time in
 *        the copying and allocating joinV0 calls, and V1's in joinV1's own
 *        code.
 *
 * Demo 03 measures the two versions, and `--profile gperf` profiles them the
 * way a reader does. This program profiles both versions itself, through the
 * same gperf backend, and reads the profiles with google-pprof; the plumbing
 * is in GperfProfiles.hpp.
 *
 * A profile's shares are samples, so this is a performance test: built into
 * bin/ptests, never registered with ctest, and run by hand on the reference
 * rig before a release, as walkthrough 03 says.
 *
 * Usage:
 *   @code{.sh}
 *   taskset -c 3 ./build/bin/ptests/JoinProfileAttribution
 *   @endcode
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>

#include <string>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/demo/examples/join/utst/GperfProfiles.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::gperf_check;

/* ----------------------------- Constants ----------------------------- */

/// Demo 03's input: 1,000 words from a fixed seed, joined by ','.
static constexpr std::size_t PART_COUNT = 1000;
static constexpr unsigned PART_SEED = 42;
static constexpr char SEPARATOR = ',';

/// CPU seconds each version runs under the profiler. gperftools samples 100
/// times per CPU second by default, so this gives about 200 samples per
/// version.
static constexpr double PROFILE_CPU_SECONDS = 2.0;

/// Fewer samples than this and a percentage says little about the program.
static constexpr long MIN_SAMPLES = 100;

/// A version's function must be on the stack of at least this share of its
/// samples: the profile attributes the version's time to it.
static constexpr double MIN_TOTAL_PERCENT = 80.0;

/// The line between "the time is in what the function calls" and "the time
/// is in the function's own code". On the reference rig joinV0's own code
/// holds a few percent of V0's samples and joinV1's a large share of V1's, so
/// the line sits far from both readings.
static constexpr double SELF_PERCENT_SPLIT = 25.0;

static constexpr const char* JOIN_V0 = "vernier::bench::demo::joinV0";
static constexpr const char* JOIN_V1 = "vernier::bench::demo::joinV1";

/* ----------------------------- Tests ----------------------------- */

/**
 * @test The profile attributes each version's time where the walkthrough says.
 *
 * Profiles both versions with the gperf backend and reads the profiles with
 * google-pprof. joinV0 must be on the stack of most of V0's samples while its
 * own code holds few of them: the time is in the copying and allocating it
 * calls. joinV1 must be on the stack of most of V1's samples and its own code
 * must hold a real share of them. A function's own code is its machine code,
 * the string code the compiler inlined into it included. Fails if either name
 * drops out of the profile (an inlined version) or if the versions stop
 * differing. Writes no CSV row.
 */
PERF_TEST(GperfProfiler, ProfileAttribution) {
  if (!ub::detail::getPerfConfig().profileTool.empty()) {
    GTEST_SKIP() << "runs its own profiles; run it without --profile";
  }
  const ub::EnvReport GPERF = ub::ProfilerRegistry::instance().runCheck("gperf");
  if (GPERF.status == ub::EnvReport::Status::Error) {
    GTEST_SKIP() << "gperf backend unavailable: " << GPERF.message;
  }
  if (!ub::profiler_env::isOnPath("google-pprof")) {
    GTEST_SKIP() << "google-pprof is not on PATH; it reads the profiles";
  }

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR), demo::joinV1(PARTS, SEPARATOR));

  const check::ScratchDir SCRATCH;
  ASSERT_FALSE(SCRATCH.path().empty()) << "could not create a scratch directory";

  volatile std::size_t sink = 0;
  const std::string V0_PROFILE = check::profileWithGperf(
      SCRATCH.path(), "GperfProfiler.ProfileAttribution.V0", PROFILE_CPU_SECONDS,
      [&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  const std::string V1_PROFILE = check::profileWithGperf(
      SCRATCH.path(), "GperfProfiler.ProfileAttribution.V1", PROFILE_CPU_SECONDS,
      [&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });

  const check::FunctionShare V0 = check::readShare(V0_PROFILE, JOIN_V0);
  const check::FunctionShare V1 = check::readShare(V1_PROFILE, JOIN_V1);

  std::printf("[GperfProfiler.ProfileAttribution]  V0: %ld samples, joinV0 %.1f%% self, %.1f%% "
              "total\n",
              V0.samples, V0.selfPercent, V0.totalPercent);
  std::printf("[GperfProfiler.ProfileAttribution]  V1: %ld samples, joinV1 %.1f%% self, %.1f%% "
              "total\n",
              V1.samples, V1.selfPercent, V1.totalPercent);

  ASSERT_GE(V0.samples, MIN_SAMPLES) << "google-pprof counted " << V0.samples << " samples in "
                                     << V0_PROFILE << "; the shares need more";
  ASSERT_GE(V1.samples, MIN_SAMPLES) << "google-pprof counted " << V1.samples << " samples in "
                                     << V1_PROFILE << "; the shares need more";
  // The shares are read from the stacks; stacks that miss samples are a
  // report this check cannot read.
  ASSERT_EQ(V0.stackSamples, V0.samples)
      << "google-pprof --stacks listed stacks for " << V0.stackSamples << " of " << V0.samples
      << " samples in " << V0_PROFILE;
  ASSERT_EQ(V1.stackSamples, V1.samples)
      << "google-pprof --stacks listed stacks for " << V1.stackSamples << " of " << V1.samples
      << " samples in " << V1_PROFILE;

  // A function's own share means something only once the profile names it.
  EXPECT_GE(V0.totalPercent, MIN_TOTAL_PERCENT)
      << JOIN_V0 << " is on the stack of " << V0.totalPercent
      << "% of V0's samples: the profile no longer names it";
  if (V0.totalPercent >= MIN_TOTAL_PERCENT) {
    EXPECT_LE(V0.selfPercent, SELF_PERCENT_SPLIT)
        << JOIN_V0 << "'s own code holds " << V0.selfPercent
        << "% of V0's samples: the time is no longer in the copying and allocating it calls";
  }
  EXPECT_GE(V1.totalPercent, MIN_TOTAL_PERCENT)
      << JOIN_V1 << " is on the stack of " << V1.totalPercent
      << "% of V1's samples: the profile no longer names it";
  if (V1.totalPercent >= MIN_TOTAL_PERCENT) {
    EXPECT_GE(V1.selfPercent, SELF_PERCENT_SPLIT)
        << JOIN_V1 << "'s own code holds only " << V1.selfPercent
        << "% of V1's samples: V1 spends its time in calls, as V0 does";
  }
}

PERF_MAIN()
