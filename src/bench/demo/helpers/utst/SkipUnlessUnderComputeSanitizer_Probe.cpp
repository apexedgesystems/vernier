/**
 * @file SkipUnlessUnderComputeSanitizer_Probe.cpp
 * @brief The probe case SkipUnlessUnderComputeSanitizer_uTest.cpp runs as a
 *        child of this program, once plainly and once under compute-sanitizer.
 *
 * A worker source of TestDemoHelpers (vernier_add_gtest's WORKER_SOURCES):
 * compiled in, never registered with CTest, since run alone it only skips.
 */

#include "src/bench/demo/helpers/SkipUnlessUnderComputeSanitizer.hpp"

#include "src/bench/demo/helpers/utst/SkipUnlessProbe.hpp"

#include <cstdio>

#include <gtest/gtest.h>

/** @test The probe the child runs select: skips unless under the tool, then reports that it ran */
TEST(SkipUnlessUnderComputeSanitizerProbe, RunsOnlyUnderTheTool) {
  DEMO_SKIP_UNLESS_UNDER_COMPUTE_SANITIZER();
  EXPECT_TRUE(vernier::bench::demo::reasonToSkipUnlessUnderComputeSanitizer().empty());
  std::puts(vernier::bench::demo::PROBE_RAN);
}
