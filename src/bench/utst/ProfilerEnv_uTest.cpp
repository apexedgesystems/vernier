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

/** @test Without a session or the override, CUPTI stays on. */
TEST(ProfilerEnv, CuptiOnByDefault) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  EXPECT_FALSE(cuptiMustYield());
}

/** @test A CUDA injection library alone is not an Nsight session. */
TEST(ProfilerEnv, CuptiIgnoresAGenericInjection) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  ::setenv("CUDA_INJECTION64_PATH", "/opt/tool/libToolsInjection64.so", 1);
  EXPECT_FALSE(cuptiMustYield());
}

/** @test The runner's nsys or ncu wrap forces the yield, under each spelling. */
TEST(ProfilerEnv, CuptiYieldsToExternalWrap) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  for (const char* tool : {"nsight", "nsys", "ncu"}) {
    ::setenv("VERNIER_EXTERNAL_WRAP", tool, 1);
    EXPECT_TRUE(cuptiMustYield()) << "VERNIER_EXTERNAL_WRAP=" << tool;
  }
  // A non-Nsight wrap (e.g. massif) does not disturb CUPTI.
  ::setenv("VERNIER_EXTERNAL_WRAP", "massif", 1);
  EXPECT_FALSE(cuptiMustYield());
}

/** @test A session typed by hand forces the yield: nsys's and ncu's variables. */
TEST(ProfilerEnv, CuptiYieldsToAHandTypedSession) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  ::setenv("NSYS_PROFILING_SESSION_ID", "1017521", 1);
  EXPECT_TRUE(cuptiMustYield());
  ::unsetenv("NSYS_PROFILING_SESSION_ID");
  ::setenv("NV_NSIGHT_INJECTION_PORT_BASE", "49152", 1);
  EXPECT_TRUE(cuptiMustYield());
}

/** @test An explicit 0 or false does not keep CUPTI on inside a session. */
TEST(ProfilerEnv, CuptiDisableFalseDoesNotOverrideASession) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  ::setenv("VERNIER_DISABLE_CUPTI", "false", 1);
  ::setenv("NSYS_PROFILING_SESSION_ID", "1017521", 1);
  EXPECT_TRUE(cuptiMustYield());
  ::unsetenv("NSYS_PROFILING_SESSION_ID");
  ::setenv("VERNIER_DISABLE_CUPTI", "0", 1);
  ::setenv("VERNIER_EXTERNAL_WRAP", "ncu", 1);
  EXPECT_TRUE(cuptiMustYield());
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

/** @test VERNIER_DISABLE_CUPTI disables when set; empty, 0 and false do not. */
TEST(ProfilerEnv, CuptiDisableEnvOverride) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  ::setenv("VERNIER_DISABLE_CUPTI", "1", 1);
  EXPECT_TRUE(cuptiMustYield());
  ::setenv("VERNIER_DISABLE_CUPTI", "true", 1);
  EXPECT_TRUE(cuptiMustYield());
  ::setenv("VERNIER_DISABLE_CUPTI", "0", 1);
  EXPECT_FALSE(cuptiMustYield());
  ::setenv("VERNIER_DISABLE_CUPTI", "false", 1);
  EXPECT_FALSE(cuptiMustYield());
  ::setenv("VERNIER_DISABLE_CUPTI", "", 1);
  EXPECT_FALSE(cuptiMustYield());
}

} // namespace
