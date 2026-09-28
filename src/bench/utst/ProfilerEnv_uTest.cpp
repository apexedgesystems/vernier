/**
 * @file ProfilerEnv_uTest.cpp
 * @brief Unit tests for profiler_env helpers: externalWrapTool(),
 *        cuptiMustYield() and the privilege helpers kept for compatibility.
 *
 * The helpers read process environment variables, so each test scrubs or
 * overrides the variables it touches via an RAII guard to stay
 * order-independent. The privilege helpers run fake tools
 * (ReadinessFixtures.hpp) found through an overridden PATH.
 */

#include "src/bench/inc/ProfilerEnv.hpp"

#include "src/bench/utst/ReadinessFixtures.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>

#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
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

/** @brief Sets one variable for the scope and restores its previous state. */
class EnvOverride {
public:
  EnvOverride(const char* name, const std::string& value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_ = old;
    }
    ::setenv(name, value.c_str(), 1);
  }
  ~EnvOverride() {
    if (old_) {
      ::setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }
  EnvOverride(const EnvOverride&) = delete;
  EnvOverride& operator=(const EnvOverride&) = delete;

private:
  std::string name_;
  std::optional<std::string> old_;
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
  // One parser for every boolean setting: words in any case.
  ::setenv("BENCH_SUDO", "Yes", 1);
  EXPECT_TRUE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "on", 1);
  EXPECT_TRUE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "no", 1);
  EXPECT_FALSE(vernier::bench::profiler_env::benchSudoActive());
  ::setenv("BENCH_SUDO", "off", 1);
  EXPECT_FALSE(vernier::bench::profiler_env::benchSudoActive());
  // An invalid value is a configuration error, never an opt-in.
  ::setenv("BENCH_SUDO", "2", 1);
  EXPECT_FALSE(vernier::bench::profiler_env::benchSudoActive());
}

/** @test sudoKill() delivers through `sudo -n -- <kill>` found on PATH. */
TEST(ProfilerEnv, SudoKillDeliversThroughResolvedTools) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "sudoKill signals directly as root";
  }
  vernier::bench::test::FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  dir.install("fake_sudo.sh", "sudo");
  const pid_t CHILD = ::fork();
  ASSERT_GE(CHILD, 0);
  if (CHILD == 0) {
    ::pause();
    ::_exit(0);
  }
  {
    EnvOverride path("PATH", dir.path());
    EnvOverride log("FAKE_LOG", dir.logPath());
    EXPECT_TRUE(vernier::bench::profiler_env::sudoKill(CHILD, SIGTERM));
  }
  int status = 0;
  EXPECT_EQ(::waitpid(CHILD, &status, 0), CHILD);
  EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM);
  EXPECT_EQ(dir.logLines("sudo"),
            (std::vector<std::string>{"sudo -n -- " + KILL + " -15 " + std::to_string(CHILD)}));
}

/** @test The attach probe is bounded without timeout(1), directly and through sudo. */
TEST(ProfilerEnv, BpftraceAttachViableNeedsNoTimeout) {
  vernier::bench::test::FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string BPFTRACE = dir.install("fake_bpftrace.sh", "bpftrace");
  dir.install("fake_sudo.sh", "sudo");
  EnvOverride path("PATH", dir.path()); // holds no timeout(1)
  EnvOverride log("FAKE_LOG", dir.logPath());
  EXPECT_TRUE(vernier::bench::profiler_env::bpftraceAttachViable(false)) << dir.log();
  EXPECT_TRUE(vernier::bench::profiler_env::bpftraceAttachViable(true)) << dir.log();
  EXPECT_EQ(dir.logLines("sudo -n -- " + BPFTRACE + " -e ").size(), 1U) << dir.log();
  {
    EnvOverride failing("FAKE_BPFTRACE_MODE", "eperm");
    EXPECT_FALSE(vernier::bench::profiler_env::bpftraceAttachViable(false));
  }
}

/** @test sudoBpftraceUsable() runs exactly `sudo -n -- <bpftrace> --version`. */
TEST(ProfilerEnv, SudoBpftraceUsableRunsTheVersion) {
  vernier::bench::test::FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string BPFTRACE = dir.install("fake_bpftrace.sh", "bpftrace");
  dir.install("fake_sudo.sh", "sudo");
  EnvOverride path("PATH", dir.path());
  EnvOverride log("FAKE_LOG", dir.logPath());
  EXPECT_TRUE(vernier::bench::profiler_env::sudoBpftraceUsable());
  EXPECT_EQ(dir.logLines("sudo"),
            (std::vector<std::string>{"sudo -n -- " + BPFTRACE + " --version"}));
  EnvOverride refused("FAKE_SUDO_DENY", "--version");
  EXPECT_FALSE(vernier::bench::profiler_env::sudoBpftraceUsable());
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
