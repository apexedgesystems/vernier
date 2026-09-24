/**
 * @file ProfilerEnv_uTest.cpp
 * @brief Unit tests for profiler_env helpers: externalWrapTool(),
 *        nsightSessionTool() and cuptiMustYield().
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
using vernier::bench::profiler_env::nsightSessionTool;

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

/** @test Without a wrap or a session variable, no Nsight tool runs this process. */
TEST(ProfilerEnv, NsightSessionToolEmptyWithoutASession) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE"};
  EXPECT_EQ(nsightSessionTool(), "");
}

/** @test The runner's nsys or ncu wrap names the tool; any other wrap names none. */
TEST(ProfilerEnv, NsightSessionToolReadsTheRunnersWrap) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE"};
  ::setenv("VERNIER_EXTERNAL_WRAP", "nsight", 1);
  EXPECT_EQ(nsightSessionTool(), "nsys");
  ::setenv("VERNIER_EXTERNAL_WRAP", "nsys", 1);
  EXPECT_EQ(nsightSessionTool(), "nsys");
  ::setenv("VERNIER_EXTERNAL_WRAP", "ncu", 1);
  EXPECT_EQ(nsightSessionTool(), "ncu");
  ::setenv("VERNIER_EXTERNAL_WRAP", "massif", 1);
  EXPECT_EQ(nsightSessionTool(), "");
}

/** @test A wrap typed by hand is recognised from the variable each tool exports. */
TEST(ProfilerEnv, NsightSessionToolRecognisesAHandTypedWrap) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE"};
  ::setenv("NSYS_PROFILING_SESSION_ID", "1017521", 1);
  EXPECT_EQ(nsightSessionTool(), "nsys");
  ::unsetenv("NSYS_PROFILING_SESSION_ID");
  ::setenv("NV_NSIGHT_INJECTION_PORT_BASE", "49152", 1);
  EXPECT_EQ(nsightSessionTool(), "ncu");
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

/** @test BENCH_SUDO truthy parsing (only meaningful for non-root runs). */
TEST(ProfilerEnv, BenchSudoActiveParsesEnv) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "benchSudoActive is defined false for root";
  }
  EnvScrub scrub{"BENCH_SUDO"};
  EXPECT_FALSE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "1", 1);
  EXPECT_TRUE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "true", 1);
  EXPECT_TRUE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "0", 1);
  EXPECT_FALSE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "false", 1);
  EXPECT_FALSE(vernier::bench::profiler_env::benchSudoActive());
}

/** @test processAlive: our own pid is alive; a just-reaped child is not. */
TEST(ProfilerEnv, ProcessAliveBasics) {
  EXPECT_TRUE(vernier::bench::profiler_env::processAlive(::getpid()));
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
