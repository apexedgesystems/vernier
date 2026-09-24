/**
 * @file ProfilerReadiness_uTest.cpp
 * @brief Unit tests for the shared readiness mechanisms.
 *
 * Covers the pieces every backend's decision is built from: boolean settings,
 * the privilege precedence, the context snapshot, executable resolution,
 * bounded probes, owned helper processes and the result memo. Tools are fakes
 * from ReadinessFixtures.hpp, reached through an explicit context; no test
 * changes the process environment.
 */

#include "src/bench/inc/ProfilerReadiness.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using vernier::bench::decidePrivilege;
using vernier::bench::EnvBool;
using vernier::bench::EnvReport;
using vernier::bench::HelperStart;
using vernier::bench::HelperStopPolicy;
using vernier::bench::HelperStopResult;
using vernier::bench::LaunchContext;
using vernier::bench::outputTail;
using vernier::bench::OwnedHelper;
using vernier::bench::parseEnvBool;
using vernier::bench::PerfConfig;
using vernier::bench::PrivilegeDecision;
using vernier::bench::PrivilegeRoute;
using vernier::bench::PROBE_OUTPUT_LIMIT;
using vernier::bench::ProbeResult;
using vernier::bench::ProbeScratch;
using vernier::bench::ProbeStreams;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessMemo;
using vernier::bench::ReadinessRequest;
using vernier::bench::readinessRequestFor;
using vernier::bench::ReadinessResult;
using vernier::bench::readinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::resolveExecutable;
using vernier::bench::runBoundedProbe;
using vernier::bench::test::FakeToolDir;

namespace {

/** @brief A context with exactly @p env and effective uid @p euid. */
ReadinessContext contextWith(std::map<std::string, std::string> env, uid_t euid = 1000) {
  return ReadinessContext(euid, ::getpid(), std::move(env));
}

/** @brief Fast stop waits so helper tests stay short. */
HelperStopPolicy quickPolicy(PrivilegeRoute route = PrivilegeRoute::CURRENT_USER) {
  HelperStopPolicy policy;
  policy.route = route;
  policy.interruptWaitMs = 400;
  policy.terminateWaitMs = 400;
  policy.killWaitMs = 400;
  return policy;
}

/** @brief Kill and reap every pid the fake helper logged ("... pid=N"). */
void killLoggedChildren(const FakeToolDir& dir) {
  for (const std::string& line : dir.logLines("helper child pid=")) {
    const pid_t PID = static_cast<pid_t>(std::stol(line.substr(line.find('=') + 1)));
    (void)::kill(PID, SIGKILL);
  }
}

/** @brief The pids of the children the fake helper logged ("helper child pid=N"). */
std::vector<pid_t> loggedChildren(const FakeToolDir& dir) {
  std::vector<pid_t> pids;
  for (const std::string& line : dir.logLines("helper child pid=")) {
    pids.push_back(static_cast<pid_t>(std::stol(line.substr(line.find('=') + 1))));
  }
  return pids;
}

/** @brief True when @p pid has ended: no such process, or a zombie awaiting its parent. */
bool ended(pid_t pid) {
  std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
  std::string text;
  if (!std::getline(stat, text)) {
    return true;
  }
  const std::size_t COMM_END = text.rfind(')');
  return COMM_END != std::string::npos && COMM_END + 2 < text.size() &&
         (text[COMM_END + 2] == 'Z' || text[COMM_END + 2] == 'X');
}

/** @brief How a probe that leaves a child behind came out, observed as it returned. */
struct OrphanOutcome {
  bool returned = false; ///< It returned within the watchdog.
  ProbeResult result;
  std::chrono::milliseconds elapsed{0};
  pid_t child = -1;        ///< The child the helper left behind.
  bool childEnded = false; ///< That child had ended when the probe returned.
};

/**
 * @brief Run `helper orphan KIND STATUS` bounded by @p timeoutMs on another
 * thread. A probe that does not return within 10 s is reported, and its
 * child killed so that it can; every logged child is killed afterwards.
 */
OrphanOutcome runOrphanProbe(const FakeToolDir& dir, const std::string& helper,
                             const std::string& kind, const std::string& status, int timeoutMs) {
  const ReadinessContext CTX = dir.context();
  auto future = std::async(std::launch::async, [&] {
    OrphanOutcome out;
    const auto START = std::chrono::steady_clock::now();
    out.result = runBoundedProbe({helper, "orphan", kind, status}, timeoutMs, CTX);
    out.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - START);
    const std::vector<pid_t> CHILDREN = loggedChildren(dir);
    if (!CHILDREN.empty()) {
      out.child = CHILDREN.front();
      out.childEnded = ended(out.child);
    }
    return out;
  });
  const bool RETURNED = future.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
  if (!RETURNED) {
    killLoggedChildren(dir);
  }
  OrphanOutcome out = future.get();
  out.returned = RETURNED;
  killLoggedChildren(dir);
  return out;
}

/** @brief Every thread arrives, then all proceed; false when @p count never arrive in time. */
class Rendezvous {
public:
  explicit Rendezvous(int count) : count_(count) {}

  bool arriveAndWait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrived_;
    cv_.notify_all();
    return cv_.wait_for(lock, timeout, [this] { return arrived_ >= count_; });
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_;
  int arrived_ = 0;
};

} // namespace

/* ----------------------------- Settings ----------------------------- */

/** @test Boolean settings: case-insensitive words, empty is false, the rest invalid. */
TEST(ReadinessSettings, ParseEnvBoolTable) {
  EXPECT_EQ(parseEnvBool(std::nullopt), EnvBool::ABSENT);
  for (const char* v : {"1", "true", "TRUE", "yes", "Yes", "on", "ON"}) {
    EXPECT_EQ(parseEnvBool(std::string{v}), EnvBool::TRUE_VALUE) << v;
  }
  for (const char* v : {"", "0", "false", "False", "no", "NO", "off", "Off"}) {
    EXPECT_EQ(parseEnvBool(std::string{v}), EnvBool::FALSE_VALUE) << "'" << v << "'";
  }
  for (const char* v : {"2", "maybe", " 1", "1 ", "y", "enable"}) {
    EXPECT_EQ(parseEnvBool(std::string{v}), EnvBool::INVALID) << "'" << v << "'";
  }
}

/* ----------------------------- Privilege Policy ----------------------------- */

namespace {

struct PrecedenceRow {
  const char* name;
  std::map<std::string, std::string> env;
  PrivilegeRoute userRoute; ///< For a non-root user.
  bool optIn;
  const char* source;     ///< Exact source text ("" when CONFIG_ERROR).
  const char* warningHas; ///< Substring of the warning, "" for none.
  const char* errorHas;   ///< Substring of the error, "" for none.
};

const char* const ALIAS = "PERF_BPF_SUDO";

std::vector<PrecedenceRow> precedenceRows() {
  return {
      {"1 neither set",
       {},
       PrivilegeRoute::CURRENT_USER,
       false,
       "no opt-in (BENCH_SUDO unset)",
       "",
       ""},
      {"2 alias true",
       {{"PERF_BPF_SUDO", "1"}},
       PrivilegeRoute::SCOPED_SUDO,
       true,
       "PERF_BPF_SUDO=1 (deprecated alias)",
       "PERF_BPF_SUDO is deprecated; set BENCH_SUDO instead. PERF_BPF_SUDO=1 selects: the probe "
       "tool runs with sudo -n.",
       ""},
      {"3 alias false",
       {{"PERF_BPF_SUDO", "no"}},
       PrivilegeRoute::CURRENT_USER,
       false,
       "PERF_BPF_SUDO=no (deprecated alias)",
       "PERF_BPF_SUDO=no selects: the probe tool runs as the current user.",
       ""},
      {"4 alias invalid",
       {{"PERF_BPF_SUDO", "maybe"}},
       PrivilegeRoute::CONFIG_ERROR,
       false,
       "",
       "",
       "PERF_BPF_SUDO='maybe' is not a boolean"},
      {"5 bench true",
       {{"BENCH_SUDO", "1"}},
       PrivilegeRoute::SCOPED_SUDO,
       true,
       "BENCH_SUDO=1",
       "",
       ""},
      {"6 bench false",
       {{"BENCH_SUDO", "0"}},
       PrivilegeRoute::CURRENT_USER,
       false,
       "BENCH_SUDO=0",
       "",
       ""},
      {"7 bench true, alias false",
       {{"BENCH_SUDO", "1"}, {"PERF_BPF_SUDO", "0"}},
       PrivilegeRoute::SCOPED_SUDO,
       true,
       "BENCH_SUDO=1",
       "BENCH_SUDO=1 and PERF_BPF_SUDO=0 disagree; BENCH_SUDO wins (the probe tool runs with sudo "
       "-n). PERF_BPF_SUDO is deprecated: remove it.",
       ""},
      {"7 bench false, alias true",
       {{"BENCH_SUDO", "off"}, {"PERF_BPF_SUDO", "true"}},
       PrivilegeRoute::CURRENT_USER,
       false,
       "BENCH_SUDO=off",
       "disagree; BENCH_SUDO wins (the probe tool runs as the current user)",
       ""},
      {"8 equal values",
       {{"BENCH_SUDO", "1"}, {"PERF_BPF_SUDO", "yes"}},
       PrivilegeRoute::SCOPED_SUDO,
       true,
       "BENCH_SUDO=1",
       "",
       ""},
      {"9 bench valid, alias invalid",
       {{"BENCH_SUDO", "0"}, {"PERF_BPF_SUDO", "sometimes"}},
       PrivilegeRoute::CURRENT_USER,
       false,
       "BENCH_SUDO=0",
       "PERF_BPF_SUDO='sometimes' is not a boolean and is ignored; BENCH_SUDO=0 wins.",
       ""},
      {"10 bench invalid",
       {{"BENCH_SUDO", "2"}, {"PERF_BPF_SUDO", "1"}},
       PrivilegeRoute::CONFIG_ERROR,
       false,
       "",
       "",
       "BENCH_SUDO='2' is not a boolean"},
  };
}

} // namespace

/** @test Every precedence row, for an ordinary user and for root. */
TEST(ReadinessPrivilege, PrecedenceRows) {
  for (const PrecedenceRow& row : precedenceRows()) {
    for (const uid_t EUID : {uid_t{1000}, uid_t{0}}) {
      const PrivilegeDecision D = decidePrivilege(contextWith(row.env, EUID), ALIAS);
      PrivilegeRoute expected = row.userRoute;
      if (EUID == 0 && expected != PrivilegeRoute::CONFIG_ERROR) {
        expected = PrivilegeRoute::ALREADY_ROOT; // root never calls sudo
      }
      EXPECT_EQ(D.route, expected) << row.name << " euid=" << EUID;
      if (expected == PrivilegeRoute::CONFIG_ERROR) {
        EXPECT_NE(D.error.find(row.errorHas), std::string::npos) << row.name << ": " << D.error;
        EXPECT_NE(D.remedy.find("Use 1, true, yes or on"), std::string::npos) << row.name;
        continue;
      }
      EXPECT_EQ(D.optIn, row.optIn) << row.name;
      EXPECT_EQ(D.source, row.source) << row.name;
      if (std::string{row.warningHas}.empty()) {
        EXPECT_TRUE(D.warning.empty()) << row.name << ": " << D.warning;
      } else {
        EXPECT_NE(D.warning.find(row.warningHas), std::string::npos)
            << row.name << ": " << D.warning;
      }
    }
  }
}

/** @test Without a legacy alias the alias is never read (offcpu's collapsed rows). */
TEST(ReadinessPrivilege, NoAliasIgnoresLegacySetting) {
  const PrivilegeDecision D =
      decidePrivilege(contextWith({{"PERF_BPF_SUDO", "1"}, {"BENCH_SUDO", "0"}}));
  EXPECT_EQ(D.route, PrivilegeRoute::CURRENT_USER);
  EXPECT_TRUE(D.warning.empty()) << D.warning;
  const PrivilegeDecision ONLY_ALIAS = decidePrivilege(contextWith({{"PERF_BPF_SUDO", "bogus"}}));
  EXPECT_EQ(ONLY_ALIAS.route, PrivilegeRoute::CURRENT_USER);
  EXPECT_EQ(ONLY_ALIAS.source, "no opt-in (BENCH_SUDO unset)");
}

/* ----------------------------- Context ----------------------------- */

/** @test capture() snapshots the live environment, uid and pid. */
TEST(ReadinessContextTest, CaptureSnapshotsProcess) {
  const ReadinessContext CTX = ReadinessContext::capture();
  EXPECT_EQ(CTX.euid(), ::geteuid());
  EXPECT_EQ(CTX.self(), ::getpid());
  const char* path = std::getenv("PATH");
  if (path != nullptr) {
    ASSERT_TRUE(CTX.get("PATH").has_value());
    EXPECT_EQ(*CTX.get("PATH"), path);
  }
  EXPECT_FALSE(CTX.get("VERNIER_READINESS_NO_SUCH_VARIABLE").has_value());
}

/** @test The fingerprint follows the decision inputs and ignores everything else. */
TEST(ReadinessContextTest, FingerprintCoversDecisionInputsOnly) {
  const std::map<std::string, std::string> BASE{{"PATH", "/a"}, {"HOME", "/h"}};
  const std::string F0 = contextWith(BASE).fingerprint();

  auto changed = BASE;
  changed["HOME"] = "/other";
  changed["CPUPROFILE_FREQUENCY"] = "1000";
  EXPECT_EQ(contextWith(changed).fingerprint(), F0) << "unrelated variables must not re-key";

  for (const char* KEY :
       {"PATH", "BENCH_SUDO", "VERNIER_EXTERNAL_WRAP", "VERNIER_EXTERNAL_WRAP_DIR"}) {
    auto with = BASE;
    with[KEY] = "x";
    EXPECT_NE(contextWith(with).fingerprint(), F0) << KEY;
  }
  EXPECT_NE(contextWith(BASE, 0).fingerprint(), F0) << "euid";

  auto extra = BASE;
  extra["PERF_BPF_SUDO"] = "1";
  EXPECT_EQ(contextWith(extra).fingerprint(), F0) << "undeclared key";
  EXPECT_NE(contextWith(extra).fingerprint({"PERF_BPF_SUDO"}),
            contextWith(BASE).fingerprint({"PERF_BPF_SUDO"}))
      << "declared key";
  // Unset and empty differ.
  auto empty = BASE;
  empty["BENCH_SUDO"] = "";
  EXPECT_NE(contextWith(empty).fingerprint(), F0);
}

/** @test Probe children never inherit the benchmark's wrap and run in the C locale. */
TEST(ReadinessContextTest, ProbeEnvironmentDropsInjectionVariables) {
  const ReadinessContext CTX = contextWith({{"PATH", "/p"},
                                            {"LD_PRELOAD", "/x.so"},
                                            {"MALLOC_CONF", "prof:true"},
                                            {"HEAPTRACK_OUTPUT", "/h"},
                                            {"CUDA_INJECTION64_PATH", "/c.so"},
                                            {"ROCP_TOOL_LIB", "/r.so"},
                                            {"ROCPROFILER_LIBRARY", "/r2.so"},
                                            {"VERNIER_EXTERNAL_WRAP", "massif"},
                                            {"VERNIER_EXTERNAL_WRAP_DIR", "/d"},
                                            {"LC_ALL", "de_DE.UTF-8"},
                                            {"KEEP_ME", "1"}});
  const std::vector<std::string> ENV = CTX.probeEnvironment();
  const auto HAS = [&](const std::string& entry) {
    return std::find(ENV.begin(), ENV.end(), entry) != ENV.end();
  };
  EXPECT_TRUE(HAS("PATH=/p"));
  EXPECT_TRUE(HAS("KEEP_ME=1"));
  EXPECT_TRUE(HAS("LC_ALL=C"));
  EXPECT_FALSE(HAS("LC_ALL=de_DE.UTF-8"));
  for (const std::string& entry : ENV) {
    for (const char* DROPPED :
         {"LD_PRELOAD=", "MALLOC_CONF=", "HEAPTRACK_", "CUDA_INJECTION64_PATH=", "ROCP_TOOL_LIB=",
          "ROCPROFILER_LIBRARY=", "VERNIER_EXTERNAL_WRAP"}) {
      EXPECT_NE(entry.rfind(DROPPED, 0), 0U) << entry;
    }
  }
}

/** @test The request mirrors the configuration; a runner wrap of the same backend is noticed. */
TEST(ReadinessContextTest, RequestForConfiguration) {
  PerfConfig cfg;
  cfg.profileTool = "nsys";
  cfg.profileArgs = "--stats";
  cfg.bpfScripts = {"a", "b"};
  cfg.profileAnalyze = true;
  const ReadinessRequest PLAIN = readinessRequestFor(cfg, ReadinessScope::RUNTIME, contextWith({}));
  EXPECT_EQ(PLAIN.backend, "nsight");
  EXPECT_EQ(PLAIN.profileArgs, "--stats");
  EXPECT_EQ(PLAIN.bpfScripts, (std::vector<std::string>{"a", "b"}));
  EXPECT_TRUE(PLAIN.analyze);
  EXPECT_EQ(PLAIN.scope, ReadinessScope::RUNTIME);
  EXPECT_EQ(PLAIN.launch, LaunchContext::IN_PROCESS);

  const ReadinessRequest WRAPPED = readinessRequestFor(
      cfg, ReadinessScope::RUNTIME, contextWith({{"VERNIER_EXTERNAL_WRAP", "nsight"}}));
  EXPECT_EQ(WRAPPED.launch, LaunchContext::RUNNER_WRAPPED);
  const ReadinessRequest OTHER = readinessRequestFor(
      cfg, ReadinessScope::RUNTIME, contextWith({{"VERNIER_EXTERNAL_WRAP", "massif"}}));
  EXPECT_EQ(OTHER.launch, LaunchContext::IN_PROCESS);
}

/* ----------------------------- Results ----------------------------- */

/** @test Each cause maps to one status and one leading word; analysis says so. */
TEST(ReadinessResultTest, CauseWordsAndStatus) {
  struct Row {
    ReadinessCause cause;
    EnvReport::Status status;
    const char* message;
  };
  const Row ROWS[] = {
      {ReadinessCause::READY, EnvReport::Status::Ok, "d"},
      {ReadinessCause::CAVEAT, EnvReport::Status::Warning, "d"},
      {ReadinessCause::UNVERIFIED, EnvReport::Status::Warning, "unverified: d"},
      {ReadinessCause::MISSING, EnvReport::Status::Error, "missing: d"},
      {ReadinessCause::UNUSABLE, EnvReport::Status::Error, "unusable: d"},
      {ReadinessCause::UNSUPPORTED, EnvReport::Status::Error, "unsupported: d"},
      {ReadinessCause::DENIED, EnvReport::Status::Error, "denied: d"},
      {ReadinessCause::CONFIGURATION, EnvReport::Status::Error, "configuration: d"},
      {ReadinessCause::MISSING_HELPER, EnvReport::Status::Error, "missing helper: d"},
      {ReadinessCause::INTERNAL, EnvReport::Status::Error, "internal: d"},
  };
  for (const Row& row : ROWS) {
    const ReadinessResult R = readinessResult(row.cause, "d", "fix it");
    EXPECT_EQ(R.report.status, row.status) << row.message;
    EXPECT_EQ(R.report.message, row.message);
    EXPECT_EQ(R.report.hint, "fix it");
    EXPECT_EQ(R.cause, row.cause);
    EXPECT_EQ(R.stage, ReadinessStage::COLLECTION);
  }
  const ReadinessResult ANALYSIS = readinessResult(ReadinessCause::MISSING, "no analyzer",
                                                   "install one", ReadinessStage::ANALYSIS);
  EXPECT_EQ(ANALYSIS.report.message, "analysis: missing: no analyzer");
  EXPECT_TRUE(ANALYSIS.collectionReady()) << "an analysis error still collects";
  EXPECT_FALSE(readinessResult(ReadinessCause::DENIED, "d", "").collectionReady());
  EXPECT_TRUE(readinessResult(ReadinessCause::CAVEAT, "d", "").collectionReady());
}

/** @test outputTail keeps the last non-empty lines, trimmed, on one line. */
TEST(ReadinessResultTest, OutputTail) {
  EXPECT_EQ(outputTail("first\n\n  second  \nthird\n\n"), "second | third");
  EXPECT_EQ(outputTail("only"), "only");
  EXPECT_EQ(outputTail(""), "");
  EXPECT_EQ(outputTail("a\nb\nc", 1), "c");
}

/* ----------------------------- Executables ----------------------------- */

/** @test Resolution follows PATH, skips non-executables, reports them when nothing else exists. */
TEST(ReadinessExecutables, ResolveOnSnapshotPath) {
  FakeToolDir first;
  FakeToolDir second;
  ASSERT_TRUE(first.ok() && second.ok());
  const std::string EXEC_SECOND = second.install("fake_helper.sh", "tool");
  first.install("fake_helper.sh", "tool", 0644); // not executable: skipped
  first.makeDirectory("dirtool");                // a directory named like a tool
  second.install("fake_helper.sh", "plain", 0644);

  const ReadinessContext CTX =
      contextWith({{"PATH", first.path() + ":" + second.path()}}, ::geteuid());
  const auto TOOL = resolveExecutable("tool", CTX);
  ASSERT_TRUE(TOOL.has_value());
  EXPECT_TRUE(TOOL->executable);
  EXPECT_EQ(TOOL->path, EXEC_SECOND);

  const auto DIR = resolveExecutable("dirtool", CTX);
  ASSERT_TRUE(DIR.has_value());
  EXPECT_FALSE(DIR->executable) << "a directory is not an executable";

  const auto PLAIN = resolveExecutable("plain", CTX);
  ASSERT_TRUE(PLAIN.has_value());
  EXPECT_FALSE(PLAIN->executable);
  EXPECT_EQ(PLAIN->path, second.path() + "/plain");

  EXPECT_FALSE(resolveExecutable("absent-tool", CTX).has_value());
  EXPECT_FALSE(resolveExecutable("", CTX).has_value());

  const auto BY_PATH = resolveExecutable(EXEC_SECOND, contextWith({{"PATH", "/nonexistent"}}));
  ASSERT_TRUE(BY_PATH.has_value());
  EXPECT_TRUE(BY_PATH->executable);
  EXPECT_EQ(BY_PATH->path, EXEC_SECOND);
}

/* ----------------------------- Bounded Probes ----------------------------- */

/** @test A probe reports the exit status and the output, without timeout(1) on PATH. */
TEST(ReadinessProbes, ExitStatusAndOutput) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context(); // PATH holds the fakes only: no timeout(1)

  const ProbeResult OK = runBoundedProbe({HELPER, "exit", "0"}, 5000, CTX);
  EXPECT_TRUE(OK.succeeded()) << OK.describe();
  EXPECT_NE(OK.output.find("args: exit 0"), std::string::npos) << OK.output;

  const ProbeResult FAIL = runBoundedProbe({HELPER, "fail", "broken build"}, 5000, CTX);
  EXPECT_TRUE(FAIL.started);
  EXPECT_TRUE(FAIL.exited);
  EXPECT_EQ(FAIL.exitCode, 3);
  EXPECT_FALSE(FAIL.succeeded());
  EXPECT_EQ(outputTail(FAIL.output), "broken build");
  EXPECT_EQ(FAIL.describe(), "exit status 3");
}

/** @test stderr is interleaved by default and kept apart on request. */
TEST(ReadinessProbes, SeparateStreams) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ProbeResult MERGED = runBoundedProbe({HELPER, "both"}, 5000, dir.context());
  EXPECT_NE(MERGED.output.find("to stdout"), std::string::npos);
  EXPECT_NE(MERGED.output.find("to stderr"), std::string::npos);
  EXPECT_TRUE(MERGED.errorOutput.empty());
  const ProbeResult APART =
      runBoundedProbe({HELPER, "both"}, 5000, dir.context(), ProbeStreams::SEPARATE);
  EXPECT_TRUE(APART.succeeded()) << APART.describe();
  EXPECT_EQ(APART.output, "to stdout\n");
  EXPECT_EQ(APART.errorOutput, "to stderr\n");
}

/** @test Output beyond the limit is dropped, and the probe still ends. */
TEST(ReadinessProbes, OutputIsBounded) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ProbeResult R = runBoundedProbe({HELPER, "flood"}, 10000, dir.context());
  EXPECT_TRUE(R.succeeded()) << R.describe();
  EXPECT_EQ(R.output.size(), PROBE_OUTPUT_LIMIT);
}

/** @test A missing or non-executable program is a start failure with its errno. */
TEST(ReadinessProbes, StartFailures) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string PLAIN = dir.install("fake_helper.sh", "plain", 0644);
  const ProbeResult MISSING = runBoundedProbe({dir.path() + "/absent"}, 2000, dir.context());
  EXPECT_FALSE(MISSING.started);
  EXPECT_EQ(MISSING.spawnErrno, ENOENT);
  const ProbeResult DENIED = runBoundedProbe({PLAIN}, 2000, dir.context());
  EXPECT_FALSE(DENIED.started);
  EXPECT_EQ(DENIED.spawnErrno, EACCES);
  EXPECT_NE(DENIED.describe().find("could not be started"), std::string::npos);
}

/** @test A probe that outlives its bound is killed, with its process group, near the bound. */
TEST(ReadinessProbes, TimeoutKills) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const auto START = std::chrono::steady_clock::now();
  const ProbeResult R = runBoundedProbe({HELPER, "run"}, 300, dir.context());
  const auto ELAPSED = std::chrono::steady_clock::now() - START;
  EXPECT_TRUE(R.timedOut);
  EXPECT_FALSE(R.succeeded());
  EXPECT_LT(ELAPSED, std::chrono::seconds(5)) << "the bound was not enforced";
  const std::vector<std::string> STARTS = dir.logLines("helper run pid=");
  ASSERT_EQ(STARTS.size(), 1U);
  const pid_t PID =
      static_cast<pid_t>(std::stol(STARTS.front().substr(STARTS.front().find('=') + 1)));
  EXPECT_NE(::kill(PID, 0), 0) << "the probe is still running after its bound";
}

/** @test A child the program leaves behind ends with the probe, whether it exits, fails or times
 * out. */
TEST(ReadinessProbes, LeftChildEndsWithTheProbe) {
  const struct {
    const char* kind;
    const char* status;
    int timeoutMs;
    bool succeeded;
    bool timedOut;
  } ROWS[] = {{"quiet", "0", 5000, true, false},
              {"quiet", "3", 5000, false, false},
              {"quiet", "stay", 300, false, true},
              {"detached", "0", 5000, true, false}};
  for (const auto& row : ROWS) {
    SCOPED_TRACE(std::string{"orphan "} + row.kind + " " + row.status);
    FakeToolDir dir;
    ASSERT_TRUE(dir.ok());
    const std::string HELPER = dir.install("fake_helper.sh", "helper");
    const OrphanOutcome OUT = runOrphanProbe(dir, HELPER, row.kind, row.status, row.timeoutMs);
    ASSERT_TRUE(OUT.returned) << "the probe did not return";
    EXPECT_EQ(OUT.result.succeeded(), row.succeeded) << OUT.result.describe();
    EXPECT_EQ(OUT.result.timedOut, row.timedOut);
    EXPECT_LT(OUT.elapsed, std::chrono::milliseconds(row.timeoutMs + 3000));
    ASSERT_GT(OUT.child, 0) << dir.log();
    EXPECT_TRUE(OUT.childEnded) << "child " << OUT.child << " outlived the probe";
  }
}

/** @test A child that keeps writing does not hold the probe open, and ends with it. */
TEST(ReadinessProbes, WritingChildEndsWithTheProbe) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const OrphanOutcome OUT = runOrphanProbe(dir, HELPER, "writer", "0", 5000);
  ASSERT_TRUE(OUT.returned) << "the probe kept reading its child's output";
  EXPECT_TRUE(OUT.result.succeeded()) << OUT.result.describe();
  EXPECT_LT(OUT.elapsed, std::chrono::milliseconds(3000));
  ASSERT_GT(OUT.child, 0) << dir.log();
  EXPECT_TRUE(OUT.childEnded) << "child " << OUT.child << " outlived the probe";
}

/** @test A writer that left the probe's group cannot hold it open past the drain bound. */
TEST(ReadinessProbes, DrainIsBoundedWhenAWriterLeavesTheGroup) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const auto SETSID = resolveExecutable("setsid", contextWith({{"PATH", "/usr/bin:/bin"}}));
  if (!SETSID || !SETSID->executable) {
    GTEST_SKIP() << "setsid(1) is not installed";
  }
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const OrphanOutcome OUT = runOrphanProbe(dir, HELPER, "escaped", "0", 5000);
  ASSERT_TRUE(OUT.returned) << "the probe kept reading output from outside its group";
  EXPECT_TRUE(OUT.result.succeeded()) << OUT.result.describe();
  EXPECT_FALSE(OUT.result.output.empty());
  EXPECT_LT(OUT.elapsed, std::chrono::milliseconds(3000));
}

/** @test The probe's environment is the snapshot's probe environment, not the process's. */
TEST(ReadinessProbes, RunsInProbeEnvironment) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context({{"LD_PRELOAD", "/nonexistent/inject.so"},
                                            {"VERNIER_EXTERNAL_WRAP", "massif"},
                                            {"MARKER_FROM_SNAPSHOT", "present"}});
  const ProbeResult R = runBoundedProbe({HELPER, "env"}, 5000, CTX);
  ASSERT_TRUE(R.succeeded()) << R.describe() << "\n" << R.output;
  EXPECT_NE(R.output.find("MARKER_FROM_SNAPSHOT=present"), std::string::npos) << R.output;
  EXPECT_NE(R.output.find("LC_ALL=C"), std::string::npos) << R.output;
  EXPECT_EQ(R.output.find("LD_PRELOAD"), std::string::npos) << R.output;
  EXPECT_EQ(R.output.find("VERNIER_EXTERNAL_WRAP"), std::string::npos) << R.output;
}

/** @test A probe scratch directory lives under the snapshot's TMPDIR and goes away with it. */
TEST(ReadinessProbes, ScratchDirectoryIsPrivateAndRemoved) {
  FakeToolDir base;
  ASSERT_TRUE(base.ok());
  std::string where;
  {
    const ProbeScratch SCRATCH(contextWith({{"TMPDIR", base.path()}}));
    ASSERT_TRUE(SCRATCH.ok());
    where = SCRATCH.path();
    EXPECT_EQ(where.rfind(base.path() + "/vernier_probe_", 0), 0U) << where;
    struct stat st{};
    ASSERT_EQ(::stat(where.c_str(), &st), 0);
    EXPECT_EQ(st.st_mode & 0777, 0700U);
    const std::string FILE = SCRATCH.write("copy.bt", "text");
    EXPECT_EQ(FILE, where + "/copy.bt");
  }
  struct stat gone{};
  EXPECT_NE(::stat(where.c_str(), &gone), 0) << "the scratch directory outlived its owner";
}

/* ----------------------------- Owned Helpers ----------------------------- */

/** @test A running helper stops on SIGINT, delivered directly on the current-user route. */
TEST(ReadinessOwnedHelper, StopsOnInterrupt) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  HelperStopPolicy policy = quickPolicy();
  policy.sudoPath = dir.install("fake_sudo.sh", "sudo"); // present, but not on this route
  policy.killPath = dir.install("fake_kill.sh", "kill");
  policy.context = std::make_shared<const ReadinessContext>(CTX);
  OwnedHelper helper(policy);
  const HelperStart START = helper.start({HELPER, "run"}, "", dir.path() + "/err.txt", 200, &CTX);
  ASSERT_TRUE(START.running()) << START.errorTail;
  const HelperStopResult STOP = helper.stop();
  EXPECT_TRUE(STOP.wasRunning);
  EXPECT_TRUE(STOP.reaped);
  EXPECT_FALSE(STOP.stillAlive);
  EXPECT_EQ(STOP.stoppedBy, SIGINT);
  ASSERT_EQ(STOP.deliveries.size(), 1U);
  EXPECT_TRUE(STOP.deliveries[0].delivered);
  EXPECT_EQ(STOP.deliveries[0].target, START.started ? helper.pid() : -1);
  EXPECT_TRUE(dir.logLines("sudo").empty()) << "the current-user route never calls sudo";
  EXPECT_TRUE(dir.logLines("kill").empty()) << dir.log();
}

/** @test A helper that ends within the grace is reported with its stderr and never signalled. */
TEST(ReadinessOwnedHelper, EarlyExitIsReportedAndNotSignalled) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  OwnedHelper helper(quickPolicy());
  const HelperStart START =
      helper.start({HELPER, "fail", "attach refused"}, "", dir.path() + "/err.txt", 2000, &CTX);
  EXPECT_TRUE(START.started);
  EXPECT_TRUE(START.exitedEarly);
  EXPECT_NE(START.errorTail.find("attach refused"), std::string::npos) << START.errorTail;
  const HelperStopResult STOP = helper.stop();
  EXPECT_FALSE(STOP.wasRunning);
  EXPECT_TRUE(STOP.deliveries.empty()) << "a reaped helper must never be signalled";
}

/** @test A helper that ignores SIGINT is stopped by the escalation, each step recorded. */
TEST(ReadinessOwnedHelper, EscalatesWhenInterruptIsIgnored) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  OwnedHelper helper(quickPolicy());
  ASSERT_TRUE(helper.start({HELPER, "ignore-int"}, "", "", 200, &CTX).running());
  const HelperStopResult STOP = helper.stop();
  EXPECT_TRUE(STOP.reaped);
  EXPECT_EQ(STOP.stoppedBy, SIGTERM);
  ASSERT_EQ(STOP.deliveries.size(), 2U);
  EXPECT_EQ(STOP.deliveries[0].signal, SIGINT);
  EXPECT_EQ(STOP.deliveries[1].signal, SIGTERM);
}

/** @test On the sudo route every stop signal goes through `sudo -n -- kill`, to the owned pid. */
TEST(ReadinessOwnedHelper, SudoRouteDeliversThroughSudoKill) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  auto ctx = std::make_shared<const ReadinessContext>(dir.context());
  HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
  policy.sudoPath = SUDO;
  policy.killPath = KILL;
  policy.context = ctx;
  OwnedHelper helper(policy);
  ASSERT_TRUE(helper.start({SUDO, "-n", "--", HELPER, "run"}, "", "", 300, ctx.get()).running());
  const pid_t CHILD = helper.pid();
  const HelperStopResult STOP = helper.stop();
  EXPECT_TRUE(STOP.reaped);
  EXPECT_EQ(STOP.stoppedBy, SIGINT);
  ASSERT_EQ(STOP.deliveries.size(), 1U);
  EXPECT_EQ(STOP.deliveries[0].command, SUDO + " -n -- " + KILL + " -2 " + std::to_string(CHILD));
  EXPECT_EQ(dir.logLines("kill"), (std::vector<std::string>{"kill -2 " + std::to_string(CHILD)}));
}

/** @test A refused stop signal is reported with the exact command and sudo's words. */
TEST(ReadinessOwnedHelper, RefusedDeliveryIsReported) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  auto ctx = std::make_shared<const ReadinessContext>(dir.context({{"FAKE_SUDO_DENY", "kill -2"}}));
  HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
  policy.sudoPath = SUDO;
  policy.killPath = KILL;
  policy.context = ctx;
  OwnedHelper helper(policy);
  ASSERT_TRUE(helper.start({SUDO, "-n", "--", HELPER, "run"}, "", "", 300, ctx.get()).running());
  const HelperStopResult STOP = helper.stop();
  ASSERT_GE(STOP.deliveries.size(), 2U);
  EXPECT_FALSE(STOP.deliveries[0].delivered);
  EXPECT_EQ(STOP.deliveries[0].detail, "sudo: a password is required");
  EXPECT_NE(STOP.deliveries[0].command.find(" -2 "), std::string::npos);
  EXPECT_FALSE(STOP.allDelivered());
  EXPECT_TRUE(STOP.deliveries[1].delivered) << "SIGTERM is still tried";
  EXPECT_EQ(STOP.stoppedBy, SIGTERM);
  EXPECT_TRUE(STOP.reaped);
}

/** @test When every stop signal is refused the helper is reported alive, not flushed. */
TEST(ReadinessOwnedHelper, AllDeliveriesRefusedLeavesItReportedAlive) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  auto ctx = std::make_shared<const ReadinessContext>(dir.context({{"FAKE_SUDO_DENY", "kill -"}}));
  HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
  policy.sudoPath = SUDO;
  policy.killPath = KILL;
  policy.context = ctx;
  OwnedHelper helper(policy);
  ASSERT_TRUE(helper.start({SUDO, "-n", "--", HELPER, "run"}, "", "", 300, ctx.get()).running());
  const pid_t CHILD = helper.pid();
  const HelperStopResult STOP = helper.stop();
  EXPECT_TRUE(STOP.stillAlive);
  EXPECT_FALSE(STOP.reaped);
  EXPECT_EQ(STOP.deliveries.size(), 3U);
  EXPECT_EQ(STOP.stoppedBy, 0);
  // Clean up the fake directly (same user); the helper then reaps it.
  (void)::kill(CHILD, SIGKILL);
  for (int i = 0; i < 100 && helper.running(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_FALSE(helper.running());
}

/** @test On the sudo route the single child under the direct child is the one signalled. */
TEST(ReadinessOwnedHelper, SudoRouteTargetsTheOnlyGrandchild) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  auto ctx = std::make_shared<const ReadinessContext>(dir.context());
  HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
  policy.sudoPath = SUDO;
  policy.killPath = KILL;
  policy.context = ctx;
  OwnedHelper helper(policy);
  ASSERT_TRUE(helper.start({SUDO, "-n", "--", HELPER, "parent"}, "", "", 400, ctx.get()).running());
  const std::vector<std::string> CHILDREN = dir.logLines("helper child pid=");
  ASSERT_EQ(CHILDREN.size(), 1U) << dir.log();
  const std::string GRANDCHILD = CHILDREN.front().substr(CHILDREN.front().find('=') + 1);
  const HelperStopResult STOP = helper.stop();
  killLoggedChildren(dir);
  EXPECT_TRUE(STOP.reaped);
  ASSERT_FALSE(STOP.deliveries.empty());
  for (const auto& delivery : STOP.deliveries) {
    EXPECT_EQ(std::to_string(delivery.target), GRANDCHILD) << delivery.command;
  }
}

/** @test Two children under the direct child are ambiguous: the direct child is signalled. */
TEST(ReadinessOwnedHelper, AmbiguousChildrenFallBackToDirectChild) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  auto ctx = std::make_shared<const ReadinessContext>(dir.context());
  HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
  policy.sudoPath = SUDO;
  policy.killPath = KILL;
  policy.context = ctx;
  OwnedHelper helper(policy);
  ASSERT_TRUE(
      helper.start({SUDO, "-n", "--", HELPER, "parents"}, "", "", 400, ctx.get()).running());
  const pid_t CHILD = helper.pid();
  const HelperStopResult STOP = helper.stop();
  killLoggedChildren(dir);
  ASSERT_FALSE(STOP.deliveries.empty());
  EXPECT_EQ(STOP.deliveries[0].target, CHILD);
  EXPECT_TRUE(STOP.reaped);
}

/** @test Destroying a running helper stops it: no helper outlives its owner. */
TEST(ReadinessOwnedHelper, DestructorStopsRunningHelper) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  pid_t pid = -1;
  {
    OwnedHelper helper(quickPolicy());
    ASSERT_TRUE(helper.start({HELPER, "run"}, "", "", 200, &CTX).running());
    pid = helper.pid();
  }
  errno = 0;
  EXPECT_NE(::kill(pid, 0), 0);
  EXPECT_EQ(errno, ESRCH) << "the helper outlived its owner";
}

/** @test A capture file that cannot be opened is a start failure naming it. */
TEST(ReadinessOwnedHelper, UnwritableCaptureFileFailsToStart) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  OwnedHelper helper(quickPolicy());
  const HelperStart START =
      helper.start({HELPER, "run"}, dir.path() + "/no/such/dir/out.txt", "", 200, nullptr);
  EXPECT_FALSE(START.started);
  EXPECT_EQ(START.spawnErrno, ENOENT);
  EXPECT_NE(START.errorTail.find("could not open"), std::string::npos) << START.errorTail;
}

/* ----------------------------- Memo ----------------------------- */

/** @test Concurrent callers of one key compute it once and all see that result. */
TEST(ReadinessMemoTest, OneComputationPerKeyUnderConcurrency) {
  ReadinessMemo memo;
  std::atomic<int> computations{0};
  const auto COMPUTE = [&] {
    computations.fetch_add(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return readinessResult(ReadinessCause::CAVEAT, "computed", "");
  };
  std::vector<std::thread> threads;
  std::vector<std::string> messages(8);
  for (std::size_t i = 0; i < messages.size(); ++i) {
    threads.emplace_back([&, i] { messages[i] = memo.getOrCompute("k", COMPUTE).report.message; });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  EXPECT_EQ(computations.load(), 1);
  for (const std::string& m : messages) {
    EXPECT_EQ(m, "computed");
  }
  EXPECT_EQ(memo.size(), 1U);
}

/** @test Independent keys compute at the same time: no global lock is held while computing. */
TEST(ReadinessMemoTest, IndependentKeysComputeConcurrently) {
  ReadinessMemo memo;
  Rendezvous both(2);
  std::atomic<int> metInside{0};
  const auto COMPUTE = [&] {
    if (both.arriveAndWait(std::chrono::seconds(5))) {
      metInside.fetch_add(1);
    }
    return readinessResult(ReadinessCause::READY, "ok", "");
  };
  std::thread a([&] { (void)memo.getOrCompute("a", COMPUTE); });
  std::thread b([&] { (void)memo.getOrCompute("b", COMPUTE); });
  a.join();
  b.join();
  EXPECT_EQ(metInside.load(), 2) << "the two computations never ran at the same time";
}

/** @test Exactly one concurrent claimant prints a notice per key. */
TEST(ReadinessMemoTest, NoticeClaimedOnceUnderConcurrency) {
  ReadinessMemo memo;
  std::atomic<int> claims{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      if (memo.claimNotice("k")) {
        claims.fetch_add(1);
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  EXPECT_EQ(claims.load(), 1);
  EXPECT_TRUE(memo.claimNotice("other"));
}

/** @test Every outcome is kept until reset(); after it the key is computed again. */
TEST(ReadinessMemoTest, ResetIsTheOnlyRecompute) {
  ReadinessMemo memo;
  int calls = 0;
  const auto FAILING = [&] {
    ++calls;
    return readinessResult(ReadinessCause::MISSING, "tool", "install it");
  };
  EXPECT_EQ(memo.getOrCompute("k", FAILING).report.status, EnvReport::Status::Error);
  EXPECT_EQ(memo.getOrCompute("k", FAILING).report.status, EnvReport::Status::Error);
  EXPECT_EQ(calls, 1) << "an error is kept like any other outcome";
  EXPECT_TRUE(memo.claimNotice("k"));
  memo.reset();
  EXPECT_EQ(memo.size(), 0U);
  EXPECT_TRUE(memo.claimNotice("k")) << "reset also forgets printed notices";
  const auto REPAIRED = [&] {
    ++calls;
    return readinessResult(ReadinessCause::READY, "tool found", "");
  };
  EXPECT_EQ(memo.getOrCompute("k", REPAIRED).report.status, EnvReport::Status::Ok);
  EXPECT_EQ(calls, 2);
}

/** @test A computation that throws is kept as an internal error, not rethrown. */
TEST(ReadinessMemoTest, ThrowingComputationIsKeptAsError) {
  ReadinessMemo memo;
  int calls = 0;
  const auto THROWS = [&]() -> ReadinessResult {
    ++calls;
    throw std::runtime_error("boom");
  };
  const ReadinessResult R = memo.getOrCompute("k", THROWS);
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  EXPECT_EQ(R.cause, ReadinessCause::INTERNAL);
  EXPECT_NE(R.report.message.find("boom"), std::string::npos) << R.report.message;
  EXPECT_EQ(memo.getOrCompute("k", THROWS).report.message, R.report.message);
  EXPECT_EQ(calls, 1);
}
