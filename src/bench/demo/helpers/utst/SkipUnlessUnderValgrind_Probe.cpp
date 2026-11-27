/**
 * @file SkipUnlessUnderValgrind_Probe.cpp
 * @brief The probe case SkipUnlessUnderValgrind_uTest.cpp runs as a child of
 *        this program, once plainly and once under valgrind.
 *
 * A worker source of TestDemoHelpers (vernier_add_gtest's WORKER_SOURCES):
 * compiled in, never registered with CTest, since run alone it only skips.
 */

#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"

#include "src/bench/demo/helpers/utst/SkipUnlessProbe.hpp"

#include <cstdio>

#include <gtest/gtest.h>

/** @test The probe the child runs select: skips unless under valgrind, then reports that it ran */
TEST(SkipUnlessUnderValgrindProbe, RunsOnlyUnderValgrind) {
  DEMO_SKIP_UNLESS_UNDER_VALGRIND();
  EXPECT_TRUE(vernier::bench::demo::reasonToSkipUnlessUnderValgrind().empty());
  std::puts(vernier::bench::demo::PROBE_RAN);
}
