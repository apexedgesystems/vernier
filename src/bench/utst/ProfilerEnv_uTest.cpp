/**
 * @file ProfilerEnv_uTest.cpp
 * @brief Unit tests for profiler_env helpers: externalWrapTool() and
 *        cuptiMustYield().
 *
 * The helpers read process environment variables, so each test scrubs
 * the variables it touches via an RAII guard to stay order-independent.
 */

#include "src/bench/inc/ProfilerEnv.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

using vernier::bench::profiler_env::cuptiMustYield;
using vernier::bench::profiler_env::externalWrapTool;

namespace {

/** @brief Unsets the given variables on construction and destruction. */
class EnvScrub {
public:
  explicit EnvScrub(std::initializer_list<const char*> names) : names_(names.begin(), names.end()) {
    for (const auto& n : names_) {
      ::unsetenv(n.c_str());
    }
  }
  ~EnvScrub() {
    for (const auto& n : names_) {
      ::unsetenv(n.c_str());
    }
  }

private:
  std::vector<std::string> names_;
};

/** @test Without the env var, externalWrapTool() is empty. */
TEST(ProfilerEnv, ExternalWrapToolDefaultsEmpty) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP"};
  EXPECT_TRUE(externalWrapTool().empty());
}

/** @test externalWrapTool() returns the env var verbatim. */
TEST(ProfilerEnv, ExternalWrapToolReadsEnv) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP"};
  ::setenv("VERNIER_EXTERNAL_WRAP", "nsight", 1);
  EXPECT_EQ(externalWrapTool(), "nsight");
}

/** @test No env, non-Nsight tool: CUPTI stays on. */
TEST(ProfilerEnv, CuptiOnByDefault) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI"};
  EXPECT_FALSE(cuptiMustYield(""));
  EXPECT_FALSE(cuptiMustYield("massif"));
  EXPECT_FALSE(cuptiMustYield("perf"));
}

/** @test An active nsight/ncu profile tool forces the yield. */
TEST(ProfilerEnv, CuptiYieldsToNsightProfileTool) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI"};
  EXPECT_TRUE(cuptiMustYield("nsight"));
  EXPECT_TRUE(cuptiMustYield("ncu"));
}

/** @test A runner wrap with nsys/ncu forces the yield regardless of tool. */
TEST(ProfilerEnv, CuptiYieldsToExternalWrap) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI"};
  ::setenv("VERNIER_EXTERNAL_WRAP", "nsight", 1);
  EXPECT_TRUE(cuptiMustYield(""));
  ::setenv("VERNIER_EXTERNAL_WRAP", "ncu", 1);
  EXPECT_TRUE(cuptiMustYield(""));
  // A non-Nsight wrap (e.g. massif) does not disturb CUPTI.
  ::setenv("VERNIER_EXTERNAL_WRAP", "massif", 1);
  EXPECT_FALSE(cuptiMustYield(""));
}

/** @test VERNIER_DISABLE_CUPTI is an explicit override with truthy parsing. */
TEST(ProfilerEnv, CuptiDisableEnvOverride) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI"};
  ::setenv("VERNIER_DISABLE_CUPTI", "1", 1);
  EXPECT_TRUE(cuptiMustYield(""));
  ::setenv("VERNIER_DISABLE_CUPTI", "0", 1);
  EXPECT_FALSE(cuptiMustYield(""));
  ::setenv("VERNIER_DISABLE_CUPTI", "false", 1);
  EXPECT_FALSE(cuptiMustYield(""));
  ::setenv("VERNIER_DISABLE_CUPTI", "", 1);
  EXPECT_FALSE(cuptiMustYield(""));
}

} // namespace
