/**
 * @file ProfilerNsight_uTest.cpp
 * @brief Unit tests for the Nsight backend: it never starts nsys or ncu, prints
 *        the command that captures the run, and stays passive under a session.
 *
 * Notes:
 *  - Fake nsys and ncu executables, first on PATH, record every invocation;
 *    the backend must never run them. The real tools are not needed.
 *  - Each test sets or clears the variables it depends on and restores them.
 *  - The backend makes no CUDA call, so no device is needed; it lives in the
 *    CUDA library, so the test exists only where that library is built.
 */

#include "src/bench/inc/ProfilerNsight.hpp"

#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>

using vernier::bench::NsightProfiler;
using vernier::bench::PerfConfig;
using vernier::bench::ReplayMetrics;
using vernier::bench::Stats;
using vernier::bench::test::StderrCapture;

namespace {

/* ----------------------------- Helpers ----------------------------- */

/** @brief Sets (or, with nullptr, clears) one variable for the scope; restores it after. */
class ScopedEnv {
public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_ = old;
    }
    if (value != nullptr) {
      ::setenv(name, value, 1);
    } else {
      ::unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_) {
      ::setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
  std::string name_;
  std::optional<std::string> old_;
};

/** @brief A private directory with fake nsys and ncu that log their arguments. */
class FakeNsightTools {
public:
  /** @param name Makes the directory this test's own: one per test name and process. */
  explicit FakeNsightTools(const std::string& name)
      : dir_((std::filesystem::temp_directory_path() /
              ("vernier_nsight_" + name + "_" + std::to_string(::getpid())))
                 .string()),
        log_(dir_ + "/invocations.log") {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    if (!std::filesystem::create_directories(dir_, ec)) {
      dir_.clear();
      return;
    }
    for (const char* tool : {"nsys", "ncu"}) {
      const std::string TOOL_PATH = dir_ + "/" + tool;
      std::ofstream(TOOL_PATH) << "#!/bin/sh\necho \"" << tool << " $*\" >> '" << log_ << "'\n";
      ::chmod(TOOL_PATH.c_str(), 0755);
    }
  }
  ~FakeNsightTools() {
    std::error_code ec;
    if (!dir_.empty()) {
      std::filesystem::remove_all(dir_, ec);
    }
  }
  FakeNsightTools(const FakeNsightTools&) = delete;
  FakeNsightTools& operator=(const FakeNsightTools&) = delete;

  [[nodiscard]] bool ok() const { return !dir_.empty(); }
  [[nodiscard]] const std::string& dir() const { return dir_; }

  /** @brief Every invocation of a fake so far, one per line; empty when none. */
  [[nodiscard]] std::string invocations() const {
    std::ifstream in(log_);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }

private:
  std::string dir_;
  std::string log_;
};

} // namespace

/* ----------------------------- Fixture ----------------------------- */

/** @brief Fakes first on PATH, no session variables, artifacts in a private root. */
class NsightProfilerTest : public ::testing::Test {
protected:
  static constexpr const char* TEST_NAME = "NsightSuite.Case";

  void SetUp() override {
    tools_.emplace(::testing::UnitTest::GetInstance()->current_test_info()->name());
    ASSERT_TRUE(tools_->ok()) << "could not create the fake tool directory";
    const char* oldPath = std::getenv("PATH");
    const std::string SEARCH_PATH = tools_->dir() + ":" + (oldPath != nullptr ? oldPath : "");
    path_.emplace("PATH", SEARCH_PATH.c_str());
    wrap_.emplace("VERNIER_EXTERNAL_WRAP", nullptr);
    wrapDir_.emplace("VERNIER_EXTERNAL_WRAP_DIR", nullptr);
    nsysSession_.emplace("NSYS_PROFILING_SESSION_ID", nullptr);
    ncuSession_.emplace("NV_NSIGHT_INJECTION_PORT_BASE", nullptr);
    cfg_.artifactRoot = tools_->dir() + "/artifacts";
  }

  void TearDown() override {
    ncuSession_.reset();
    nsysSession_.reset();
    wrapDir_.reset();
    wrap_.reset();
    path_.reset();
    tools_.reset();
  }

  /** @brief One test's life of the backend: create it, bracket one window, destroy it. */
  std::string runBackend(const std::string& tool, const std::string& args) {
    cfg_.profileTool = tool;
    cfg_.profileArgs = args;
    StderrCapture capture;
    {
      NsightProfiler prof(cfg_, TEST_NAME);
      prof.beforeMeasure();
      prof.afterMeasure(Stats{});
    }
    return capture.text();
  }

  /** @brief The per-test folder the backend reports for @p suffix. */
  [[nodiscard]] std::string folder(const char* suffix) const {
    return cfg_.artifactRoot + "/" + TEST_NAME + "." + suffix;
  }

  std::optional<FakeNsightTools> tools_;
  PerfConfig cfg_{};
  std::optional<ScopedEnv> path_;
  std::optional<ScopedEnv> wrap_;
  std::optional<ScopedEnv> wrapDir_;
  std::optional<ScopedEnv> nsysSession_;
  std::optional<ScopedEnv> ncuSession_;
};

/* ----------------------------- API Tests ----------------------------- */

/** @test Systems mode starts no process and prints the nsys wrap and the bench run form */
TEST_F(NsightProfilerTest, SystemsModePrintsTheNsysWrap) {
  const std::string ERR = runBackend("nsight", "");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("nsys profile -o " + folder("nsight") +
                     "/profile -t cuda,nvtx --force-overwrite true"),
            std::string::npos)
      << ERR;
  EXPECT_NE(ERR.find("<this-binary> --profile nsight [...]"), std::string::npos) << ERR;
  EXPECT_NE(ERR.find("bench run <this-binary> --profile nsight -- [...]"), std::string::npos)
      << ERR;
}

/** @test The nsys alias selects Systems mode and is repeated in the printed command */
TEST_F(NsightProfilerTest, NsysAliasPrintsTheNsysWrap) {
  const std::string ERR = runBackend("nsys", "");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("nsys profile -o " + folder("nsight") + "/profile"), std::string::npos) << ERR;
  EXPECT_NE(ERR.find("<this-binary> --profile nsys [...]"), std::string::npos) << ERR;
}

/** @test Compute mode starts no process and prints a real ncu command, right-sized */
TEST_F(NsightProfilerTest, ComputeModePrintsTheNcuWrap) {
  const std::string ERR = runBackend("ncu", "");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("ncu -o " + folder("ncu") + "/kernel_profile -f --target-processes all"),
            std::string::npos)
      << ERR;
  EXPECT_NE(ERR.find("<this-binary> --profile ncu --cycles 3 --repeats 1 [...]"), std::string::npos)
      << ERR;
  EXPECT_EQ(ERR.find("ncu profile"), std::string::npos) << "ncu has no profile subcommand\n" << ERR;
}

/** @test Compute mode through the nsight name keeps its flags and its folder */
TEST_F(NsightProfilerTest, ComputeModeThroughTheNsightName) {
  const std::string ERR = runBackend("nsight", "compute");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("ncu -o " + folder("nsight") + "/kernel_profile"), std::string::npos) << ERR;
  EXPECT_NE(ERR.find("<this-binary> --profile nsight --profile-args 'compute' --cycles 3"),
            std::string::npos)
      << ERR;
}

/** @test Replay mode starts no process and prints the ncu command with its metric list */
TEST_F(NsightProfilerTest, ReplayModePrintsTheMetricList) {
  const std::string ERR = runBackend("ncu", "replay");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("ncu --metrics " + ReplayMetrics{}.toNcuMetricString() + " -o " +
                     folder("ncu") + "/kernel_replay -f --target-processes all"),
            std::string::npos)
      << ERR;
}

/** @test Under the runner's nsys wrap the backend starts nothing and prints no wrap */
TEST_F(NsightProfilerTest, RunnersWrapKeepsItPassive) {
  const ScopedEnv WRAP("VERNIER_EXTERNAL_WRAP", "nsight");
  const ScopedEnv DIR("VERNIER_EXTERNAL_WRAP_DIR", (tools_->dir() + "/bin.nsight").c_str());
  const std::string ERR = runBackend("nsight", "");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("this process runs under nsys"), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find("nsys profile"), std::string::npos) << ERR;
}

/** @test A wrap typed by hand is recognised: nothing started, no wrap printed */
TEST_F(NsightProfilerTest, HandTypedNsysWrapKeepsItPassive) {
  const ScopedEnv SESSION("NSYS_PROFILING_SESSION_ID", "1017521");
  const std::string ERR = runBackend("nsight", "");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("this process runs under nsys"), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find("nsys profile"), std::string::npos) << ERR;
}

/** @test An ncu wrap typed by hand is recognised: nothing started, no wrap printed */
TEST_F(NsightProfilerTest, HandTypedNcuWrapKeepsItPassive) {
  const ScopedEnv SESSION("NV_NSIGHT_INJECTION_PORT_BASE", "49152");
  const std::string ERR = runBackend("ncu", "");

  EXPECT_EQ(tools_->invocations(), "") << "the backend started a tool";
  EXPECT_NE(ERR.find("this process runs under ncu"), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find("ncu -o"), std::string::npos) << ERR;
}
