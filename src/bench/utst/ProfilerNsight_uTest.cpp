/**
 * @file ProfilerNsight_uTest.cpp
 * @brief Unit tests for the Nsight backend: it never starts nsys or ncu, prints
 *        the command that captures the run, and stays passive under a session.
 *
 * Notes:
 *  - Fake nsys and ncu executables, first on PATH, record every invocation;
 *    the backend must never run them. The real tools are not needed.
 *  - The quoting tests run the printed command through /bin/sh, placeholders
 *    filled in, and compare the arguments the fake received with the ones
 *    the run was given: a folder or an argument with a space or a quote must
 *    arrive as one argument.
 *  - Each test sets or clears the variables it depends on and restores them.
 *  - The backend makes no CUDA call, so no device is needed; it lives in the
 *    CUDA library, so the test exists only where that library is built.
 */

#include "src/bench/inc/ProfilerNsight.hpp"

#include "src/bench/utst/ScopedEnv.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using vernier::bench::NsightProfiler;
using vernier::bench::PerfConfig;
using vernier::bench::ReplayMetrics;
using vernier::bench::Stats;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

namespace {

/* ----------------------------- Helpers ----------------------------- */

/**
 * @brief A private directory with fake nsys and ncu that log their arguments:
 * every invocation joined on one line, and the last one's arguments one per
 * line in <dir>/<tool>.argv.
 */
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
      std::ofstream(TOOL_PATH) << "#!/bin/sh\necho \"" << tool << " $*\" >> '" << log_ << "'\n"
                               << "for a in \"$@\"; do printf '%s\\n' \"$a\"; done > '" << dir_
                               << "/" << tool << ".argv'\n";
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

  /** @brief The arguments of @p tool's last invocation, as it received them. */
  [[nodiscard]] std::vector<std::string> argv(const std::string& tool) const {
    std::vector<std::string> args;
    std::ifstream in(dir_ + "/" + tool + ".argv");
    for (std::string line; std::getline(in, line);) {
      args.push_back(line);
    }
    return args;
  }

private:
  std::string dir_;
  std::string log_;
};

/// What the quoting tests put in place of the printed placeholders.
constexpr const char* BINARY = "./bench-binary";
constexpr const char* REST = "--gtest_filter=Suite.Case";

/**
 * @brief Run the command the backend printed for @p tool through /bin/sh: its
 * two lines from the tool name on, `<this-binary>` and `[...]` filled in,
 * written as a script in @p dir. @return /bin/sh's exit status.
 */
int runPrintedCommand(const std::string& err, const std::string& tool, const std::string& dir) {
  std::istringstream lines(err);
  std::string script;
  for (std::string line; std::getline(lines, line);) {
    const std::string PREFIX = "[nsight]   " + tool + " ";
    if (line.rfind(PREFIX, 0) != 0) {
      continue;
    }
    std::string next;
    std::getline(lines, next);
    script = line.substr(line.find(tool)) + "\n" + next.substr(next.find_first_not_of(' ', 8));
    break;
  }
  if (script.empty()) {
    return -1;
  }
  for (const auto& [placeholder, value] :
       {std::pair<std::string, std::string>{"<this-binary>", BINARY},
        std::pair<std::string, std::string>{"[...]", REST}}) {
    const std::size_t AT = script.find(placeholder);
    if (AT != std::string::npos) {
      script.replace(AT, placeholder.size(), value);
    }
  }
  const std::string SCRIPT_PATH = dir + "/printed.sh";
  std::ofstream(SCRIPT_PATH) << script << "\n";
  return std::system(("/bin/sh '" + SCRIPT_PATH + "'").c_str());
}

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
  EXPECT_NE(ERR.find("<this-binary> --profile nsight --profile-args compute --cycles 3"),
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

/* ----------------------------- Quoting Tests ----------------------------- */

/** @test The printed nsys command keeps a folder and args with spaces and quotes whole */
TEST_F(NsightProfilerTest, SystemsWrapKeepsArgumentBoundaries) {
  cfg_.artifactRoot = tools_->dir() + "/GPU's captures";
  const std::string ERR = runBackend("nsight", "trace it's");
  ASSERT_EQ(tools_->invocations(), "") << "the backend started a tool";

  ASSERT_EQ(runPrintedCommand(ERR, "nsys", tools_->dir()), 0) << ERR;
  const std::vector<std::string> EXPECTED = {"profile",
                                             "-o",
                                             folder("nsight") + "/profile",
                                             "-t",
                                             "cuda,nvtx",
                                             "--force-overwrite",
                                             "true",
                                             BINARY,
                                             "--profile",
                                             "nsight",
                                             "--profile-args",
                                             "trace it's",
                                             REST};
  EXPECT_EQ(tools_->argv("nsys"), EXPECTED) << ERR;
}

/** @test The printed ncu command keeps a folder with spaces and quotes, and the args, whole */
TEST_F(NsightProfilerTest, ComputeWrapKeepsArgumentBoundaries) {
  cfg_.artifactRoot = tools_->dir() + "/captures with spaces";
  const std::string ERR = runBackend("nsight", "compute GPU's");
  ASSERT_EQ(tools_->invocations(), "") << "the backend started a tool";

  ASSERT_EQ(runPrintedCommand(ERR, "ncu", tools_->dir()), 0) << ERR;
  const std::vector<std::string> EXPECTED = {"-o",
                                             folder("nsight") + "/kernel_profile",
                                             "-f",
                                             "--target-processes",
                                             "all",
                                             BINARY,
                                             "--profile",
                                             "nsight",
                                             "--profile-args",
                                             "compute GPU's",
                                             "--cycles",
                                             "3",
                                             "--repeats",
                                             "1",
                                             REST};
  EXPECT_EQ(tools_->argv("ncu"), EXPECTED) << ERR;
}

/** @test The printed replay command keeps the metric list, the folder and the args whole */
TEST_F(NsightProfilerTest, ReplayWrapKeepsArgumentBoundaries) {
  cfg_.artifactRoot = tools_->dir() + "/GPU's captures";
  const std::string ERR = runBackend("ncu", "replay it's");
  ASSERT_EQ(tools_->invocations(), "") << "the backend started a tool";

  ASSERT_EQ(runPrintedCommand(ERR, "ncu", tools_->dir()), 0) << ERR;
  const std::vector<std::string> EXPECTED = {"--metrics",
                                             ReplayMetrics{}.toNcuMetricString(),
                                             "-o",
                                             folder("ncu") + "/kernel_replay",
                                             "-f",
                                             "--target-processes",
                                             "all",
                                             BINARY,
                                             "--profile",
                                             "ncu",
                                             "--profile-args",
                                             "replay it's",
                                             "--cycles",
                                             "3",
                                             "--repeats",
                                             "1",
                                             REST};
  EXPECT_EQ(tools_->argv("ncu"), EXPECTED) << ERR;
}
