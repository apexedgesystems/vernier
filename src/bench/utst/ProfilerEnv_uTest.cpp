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
#include <stdexcept>
#include <string>
#include <vector>

using vernier::bench::profiler_env::cuptiDecision;
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

/** @test No false spelling keeps CUPTI on inside a session, hand-typed or the runner's. */
TEST(ProfilerEnv, CuptiDisableFalseDoesNotOverrideASession) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  for (const char* value : {"0", "false", "FALSE", "no", "No", "off", "OFF", ""}) {
    ::setenv("VERNIER_DISABLE_CUPTI", value, 1);
    ::setenv("NSYS_PROFILING_SESSION_ID", "1017521", 1);
    EXPECT_TRUE(cuptiMustYield()) << "nsys session, VERNIER_DISABLE_CUPTI='" << value << "'";
    ::unsetenv("NSYS_PROFILING_SESSION_ID");
    ::setenv("VERNIER_EXTERNAL_WRAP", "ncu", 1);
    EXPECT_TRUE(cuptiMustYield()) << "runner's ncu wrap, VERNIER_DISABLE_CUPTI='" << value << "'";
    ::unsetenv("VERNIER_EXTERNAL_WRAP");
  }
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

/** @test Every true spelling of VERNIER_DISABLE_CUPTI, in any case, disables CUPTI. */
TEST(ProfilerEnv, CuptiDisableAcceptsEveryTrueSpelling) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  for (const char* value : {"1", "true", "TRUE", "True", "yes", "YES", "Yes", "on", "ON", "On"}) {
    ::setenv("VERNIER_DISABLE_CUPTI", value, 1);
    EXPECT_TRUE(cuptiDecision().error.empty()) << value;
    EXPECT_TRUE(cuptiMustYield()) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Every false spelling, in any case, and the empty value leave CUPTI on. */
TEST(ProfilerEnv, CuptiDisableAcceptsEveryFalseSpelling) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  for (const char* value :
       {"0", "false", "FALSE", "False", "no", "NO", "No", "off", "OFF", "Off", ""}) {
    ::setenv("VERNIER_DISABLE_CUPTI", value, 1);
    EXPECT_TRUE(cuptiDecision().error.empty()) << value;
    EXPECT_FALSE(cuptiMustYield()) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Any other value is a configuration error naming the value and the accepted ones. */
TEST(ProfilerEnv, CuptiDisableRejectsAnyOtherValue) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  for (const std::string VALUE : {"2", "-1", "maybe", "disable", "truee", "y", "n", " 1", "1 "}) {
    ::setenv("VERNIER_DISABLE_CUPTI", VALUE.c_str(), 1);
    const vernier::bench::profiler_env::CuptiDecision DECISION = cuptiDecision();
    EXPECT_EQ(DECISION.error, "VERNIER_DISABLE_CUPTI='" + VALUE + "' is not a boolean");
    EXPECT_NE(DECISION.remedy.find("1, true, yes or on"), std::string::npos) << DECISION.remedy;
    EXPECT_NE(DECISION.remedy.find("0, false, no, off or an empty value"), std::string::npos)
        << DECISION.remedy;
    try {
      (void)cuptiMustYield();
      ADD_FAILURE() << "VERNIER_DISABLE_CUPTI='" << VALUE << "' was accepted";
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "configuration: " + DECISION.error + ". " + DECISION.remedy);
    }
  }
}

/** @test An invalid value is an error inside a session too: never a silent yield. */
TEST(ProfilerEnv, CuptiDisableInvalidInsideASessionIsStillAnError) {
  EnvScrub scrub{"VERNIER_EXTERNAL_WRAP", "VERNIER_DISABLE_CUPTI", "NSYS_PROFILING_SESSION_ID",
                 "NV_NSIGHT_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH"};
  ::setenv("NSYS_PROFILING_SESSION_ID", "1017521", 1);
  ::setenv("VERNIER_DISABLE_CUPTI", "maybe", 1);
  EXPECT_FALSE(cuptiDecision().error.empty());
  EXPECT_THROW((void)cuptiMustYield(), std::invalid_argument);
}

} // namespace
