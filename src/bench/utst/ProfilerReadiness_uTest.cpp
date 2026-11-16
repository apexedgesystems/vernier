/**
 * @file ProfilerReadiness_uTest.cpp
 * @brief Unit tests for the shared readiness mechanisms.
 *
 * Covers the pieces every backend's decision is built from: boolean settings,
 * the privilege precedence, the context snapshot, executable resolution, the
 * quoting and escaping of a launch's words, a process's children and the
 * result memo; and the count of this process's children left to reap
 * (ZombieChildren.hpp) that the bounded-probe tests take. Tools are fakes from
 * ReadinessFixtures.hpp, reached through an explicit context; no test changes
 * the process environment.
 */

#include "src/bench/inc/ProfilerReadiness.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/ZombieChildren.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

using vernier::bench::childProcesses;
using vernier::bench::decidePrivilege;
using vernier::bench::EnvBool;
using vernier::bench::EnvReport;
using vernier::bench::LaunchContext;
using vernier::bench::outputTail;
using vernier::bench::parseEnvBool;
using vernier::bench::PerfConfig;
using vernier::bench::PrivilegeDecision;
using vernier::bench::PrivilegeRoute;
using vernier::bench::ProbeResult;
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
using vernier::bench::test::ChildObservation;
using vernier::bench::test::contextWith;
using vernier::bench::test::CountedChild;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::namedZombieChildren;
using vernier::bench::test::ProcessState;
using vernier::bench::test::VANISH_WAIT;
using vernier::bench::test::zombieChildren;

namespace {

/**
 * @brief Every thread arrives, then all proceed; false when @p count never arrive in time.
 *
 * Waits by polling an atomic count. A timed std::condition_variable wait calls
 * pthread_cond_clockwait, which GCC 11's ThreadSanitizer does not intercept: it
 * sees the waiting thread keep the mutex and reports the next arrival's lock
 * as a double lock and a data race.
 */
class Rendezvous {
public:
  explicit Rendezvous(int count) : count_(count) {}

  bool arriveAndWait(std::chrono::milliseconds timeout) {
    arrived_.fetch_add(1);
    const auto DEADLINE = std::chrono::steady_clock::now() + timeout;
    while (arrived_.load() < count_) {
      if (std::chrono::steady_clock::now() >= DEADLINE) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }

private:
  const int count_;
  std::atomic<int> arrived_{0};
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
  cfg.profileFrequency = 250;
  const ReadinessRequest PLAIN = readinessRequestFor(cfg, ReadinessScope::RUNTIME, contextWith({}));
  EXPECT_EQ(PLAIN.backend, "nsight");
  EXPECT_EQ(PLAIN.profileArgs, "--stats");
  EXPECT_EQ(PLAIN.bpfScripts, (std::vector<std::string>{"a", "b"}));
  EXPECT_TRUE(PLAIN.analyze);
  EXPECT_EQ(PLAIN.profileFrequency, 250);
  EXPECT_EQ(PLAIN.scope, ReadinessScope::RUNTIME);
  EXPECT_EQ(PLAIN.launch, LaunchContext::IN_PROCESS);

  const ReadinessRequest WRAPPED = readinessRequestFor(
      cfg, ReadinessScope::RUNTIME, contextWith({{"VERNIER_EXTERNAL_WRAP", "nsight"}}));
  EXPECT_EQ(WRAPPED.launch, LaunchContext::RUNNER_WRAPPED);
  const ReadinessRequest OTHER = readinessRequestFor(
      cfg, ReadinessScope::RUNTIME, contextWith({{"VERNIER_EXTERNAL_WRAP", "massif"}}));
  EXPECT_EQ(OTHER.launch, LaunchContext::IN_PROCESS);
}

/* ----------------------------- Request Identity ----------------------------- */

/**
 * @test The request has exactly the fields identity() keys.
 *
 * The binding below stops compiling when a field is added to
 * ReadinessRequest: key it in identity(), vary it in EveryFieldChangesTheKey,
 * then name it here.
 */
TEST(ReadinessRequestIdentityTest, NamesEveryField) {
  ReadinessRequest request;
  request.bpfScripts = {"s"};
  const auto& [backend, profileArgs, bpfScripts, analyze, scope, launch, profileFrequency] =
      request;
  EXPECT_TRUE(backend.empty());
  EXPECT_TRUE(profileArgs.empty());
  EXPECT_EQ(bpfScripts.size(), 1U);
  EXPECT_FALSE(analyze);
  EXPECT_EQ(scope, ReadinessScope::DEFAULT_INVENTORY);
  EXPECT_EQ(launch, LaunchContext::IN_PROCESS);
  EXPECT_EQ(profileFrequency, 0) << "no rate is asked for";
}

/** @test A change to any one field is a different identity, and no value imitates a separator. */
TEST(ReadinessRequestIdentityTest, EveryFieldChangesTheKey) {
  ReadinessRequest base;
  base.backend = "b";
  base.profileArgs = "m";
  base.bpfScripts = {"x"};
  std::vector<ReadinessRequest> variants(7, base);
  variants[0].backend = "c";
  variants[1].profileArgs = "n";
  variants[2].bpfScripts = {"y"};
  variants[3].analyze = true;
  variants[4].scope = ReadinessScope::RUNTIME;
  variants[5].launch = LaunchContext::RUNNER_WRAPPED;
  variants[6].profileFrequency = 250;
  std::vector<std::string> keys{base.identity()};
  for (const ReadinessRequest& variant : variants) {
    keys.push_back(variant.identity());
  }
  // Values that a separator-based key would confuse.
  ReadinessRequest joined = base;
  joined.bpfScripts = {"x;y"};
  ReadinessRequest split = base;
  split.bpfScripts = {"x", "y"};
  ReadinessRequest none = base;
  none.bpfScripts = {};
  ReadinessRequest emptyName = base;
  emptyName.bpfScripts = {""};
  ReadinessRequest shifted = base;
  shifted.backend = "b;1:m";
  shifted.profileArgs = "";
  for (const ReadinessRequest* request : {&joined, &split, &none, &emptyName, &shifted}) {
    keys.push_back(request->identity());
  }
  for (std::size_t i = 0; i < keys.size(); ++i) {
    for (std::size_t j = i + 1; j < keys.size(); ++j) {
      EXPECT_NE(keys[i], keys[j]) << "requests " << i << " and " << j << " share a key";
    }
  }
  EXPECT_EQ(ReadinessRequest{base}.identity(), base.identity()) << "equal requests, equal keys";
}

/* ----------------------------- Results ----------------------------- */

/** @test Each cause maps to one status and one leading word; analysis and completion say so. */
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
  const ReadinessResult COMPLETION =
      readinessResult(ReadinessCause::MISSING, "no output", "rerun", ReadinessStage::COMPLETION);
  EXPECT_EQ(COMPLETION.report.message, "completion: missing: no output");
  EXPECT_EQ(COMPLETION.report.status, EnvReport::Status::Error);
  EXPECT_TRUE(COMPLETION.collectionReady()) << "a completion outcome concerns a capture that ran";
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

/**
 * @test One shell word: a safe word as it is, any other single-quoted with a
 * quote written as '\''; and /bin/sh reads every quoted word back as it was.
 */
TEST(ReadinessShellQuote, OneWordTheShellReadsBack) {
  using vernier::bench::detail::shellQuote;
  EXPECT_EQ(shellQuote("/usr/bin/perf"), "/usr/bin/perf");
  EXPECT_EQ(shellQuote("fifo:/tmp/ctl,/tmp/ack"), "fifo:/tmp/ctl,/tmp/ack");
  EXPECT_EQ(shellQuote("a b"), "'a b'");
  EXPECT_EQ(shellQuote("it's"), "'it'\\''s'");
  EXPECT_EQ(shellQuote(""), "''");
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessContext CTX = dir.context();
  for (const std::string WORD :
       {"plain", "two words", "it's", "$HOME;`id`", "a\\b*?[c]", "\"q\" & <r>", "", "x'y'z"}) {
    const ProbeResult READ = runBoundedProbe({"/bin/sh", "-c", "printf '%s' " + shellQuote(WORD)},
                                             5000, CTX, ProbeStreams::SEPARATE);
    EXPECT_TRUE(READ.succeeded()) << WORD << ": " << READ.describe();
    EXPECT_EQ(READ.output, WORD) << shellQuote(WORD);
  }
}

/** @test A path as a macro-expanding tool reads it: each '%' doubled, nothing else changed. */
TEST(ReadinessEscapePercent, DoublesEachPercent) {
  using vernier::bench::detail::escapePercent;
  EXPECT_EQ(escapePercent("/tmp/out/sanitizer.log"), "/tmp/out/sanitizer.log");
  EXPECT_EQ(escapePercent("/tmp/a%b/x.log"), "/tmp/a%%b/x.log");
  EXPECT_EQ(escapePercent("%%q{X}%"), "%%%%q{X}%%");
  EXPECT_EQ(escapePercent(""), "");
}

/** @brief The pid zombieChildren()'s own tests give their one child. */
constexpr pid_t COUNTED_CHILD = 4242;

/** @brief A state read for zombieChildren()'s own tests: the letter @p letter. */
ProcessState stateLetter(char letter) {
  return ProcessState{ProcessState::Read::LETTER, letter, 0};
}

/**
 * @brief An observation of one child, COUNTED_CHILD, for zombieChildren()'s
 * own tests: the n-th list of children (from 0) names it while
 * @p listed(n), the n-th read of its state is @p state(n), and every look
 * answers @p look; @p lists and @p reads count the calls.
 */
ChildObservation oneChild(std::function<bool(int)> listed, std::function<ProcessState(int)> state,
                          int look, std::shared_ptr<int> lists, std::shared_ptr<int> reads) {
  ChildObservation observe;
  observe.children = [listed, lists] {
    return listed((*lists)++) ? std::vector<pid_t>{COUNTED_CHILD} : std::vector<pid_t>{};
  };
  observe.state = [state, reads](pid_t) { return state((*reads)++); };
  observe.look = [look](pid_t) { return look; };
  return observe;
}

/**
 * @test A child the list names throughout whose state cannot be read from
 * the start counts as "state unreadable", classified on its first
 * observation (one list, one read of its state): a failed read is not a
 * child gone.
 */
TEST(ZombieCount, StateUnreadableFromTheStartCounts) {
  auto lists = std::make_shared<int>(0);
  auto reads = std::make_shared<int>(0);
  const std::vector<CountedChild> COUNTED = zombieChildren(
      oneChild([](int) { return true; },
               [](int) { return ProcessState{ProcessState::Read::UNREADABLE, '\0', EACCES}; }, 0,
               lists, reads));
  ASSERT_EQ(COUNTED.size(), 1U) << "an unreadable state was taken for a child gone";
  EXPECT_EQ(COUNTED.front().pid, COUNTED_CHILD);
  EXPECT_NE(COUNTED.front().why.find("state unreadable"), std::string::npos) << COUNTED.front().why;
  EXPECT_EQ(*lists, 1) << "an unreadable state was given the grace";
  EXPECT_EQ(*reads, 1) << "an unreadable state was given the grace";
}

/**
 * @test A child in state X whose state cannot be read any more during the
 * grace, while the list still names it, counts once the grace has passed: an
 * unreadable state while it is listed is no evidence that it has gone.
 */
TEST(ZombieCount, StateUnreadableDuringTheGraceCounts) {
  auto lists = std::make_shared<int>(0);
  auto reads = std::make_shared<int>(0);
  const std::vector<CountedChild> COUNTED = zombieChildren(oneChild(
      [](int) { return true; },
      [](int n) {
        return n == 0 ? stateLetter('X') : ProcessState{ProcessState::Read::UNREADABLE, '\0', EIO};
      },
      ECHILD, lists, reads));
  ASSERT_EQ(COUNTED.size(), 1U) << "a state unreadable in the grace was taken for a child gone";
  EXPECT_EQ(COUNTED.front().pid, COUNTED_CHILD);
  EXPECT_GT(*reads, 1) << "the grace never read its state again";
}

/**
 * @test A child the kernel is releasing (state X) does not count once a
 * fresh list stops naming it within the grace, nor once its /proc entry has
 * gone.
 */
TEST(ZombieCount, ChildThatGoesWithinTheGraceDoesNotCount) {
  auto lists = std::make_shared<int>(0);
  auto reads = std::make_shared<int>(0);
  EXPECT_TRUE(zombieChildren(oneChild([](int n) { return n < 3; },
                                      [](int) { return stateLetter('X'); }, ECHILD, lists, reads))
                  .empty());
  EXPECT_EQ(*lists, 4) << "it left the count before the list stopped naming it";
  auto lists2 = std::make_shared<int>(0);
  auto reads2 = std::make_shared<int>(0);
  EXPECT_TRUE(zombieChildren(oneChild([](int) { return true; },
                                      [](int n) {
                                        return n < 3 ? stateLetter('X')
                                                     : ProcessState{ProcessState::Read::GONE, '\0',
                                                                    0};
                                      },
                                      ECHILD, lists2, reads2))
                  .empty());
}

/**
 * @test A child still in state X once the grace has passed counts, and so
 * does a zombie the look says is not this process's (ECHILD) that the list
 * still names then; the count waits the grace for each first.
 */
TEST(ZombieCount, ChildStillReleasingAfterTheGraceCounts) {
  for (const char LETTER : {'X', 'Z'}) {
    SCOPED_TRACE(std::string{"state "} + LETTER);
    auto lists = std::make_shared<int>(0);
    auto reads = std::make_shared<int>(0);
    const auto START = std::chrono::steady_clock::now();
    const std::vector<CountedChild> COUNTED = zombieChildren(
        oneChild([](int) { return true; }, [LETTER](int) { return stateLetter(LETTER); }, ECHILD,
                 lists, reads));
    ASSERT_EQ(COUNTED.size(), 1U);
    EXPECT_GE(std::chrono::steady_clock::now() - START, VANISH_WAIT);
    EXPECT_NE(COUNTED.front().why.find("still listed"), std::string::npos) << COUNTED.front().why;
  }
}

/**
 * @test A zombie that the look finds this process's counts, classified on
 * its first observation (one list, one read of its state), without the
 * grace.
 */
TEST(ZombieCount, ZombieThisProcessMayWaitForCountsAtOnce) {
  auto lists = std::make_shared<int>(0);
  auto reads = std::make_shared<int>(0);
  const std::vector<CountedChild> COUNTED = zombieChildren(
      oneChild([](int) { return true; }, [](int) { return stateLetter('Z'); }, 0, lists, reads));
  ASSERT_EQ(COUNTED.size(), 1U);
  EXPECT_EQ(*lists, 1) << "a zombie of this process's was given the grace";
  EXPECT_EQ(*reads, 1) << "a zombie of this process's was given the grace";
}

/**
 * @test A zombie that the look fails on with an errno other than ECHILD is
 * not known to be gone: it counts at once, the errno named.
 */
TEST(ZombieCount, LookFailingOtherwiseThanEchildCounts) {
  for (const int ERR : {EINTR, EINVAL}) {
    SCOPED_TRACE(std::system_category().message(ERR));
    auto lists = std::make_shared<int>(0);
    auto reads = std::make_shared<int>(0);
    const std::vector<CountedChild> COUNTED = zombieChildren(oneChild(
        [](int) { return true; }, [](int) { return stateLetter('Z'); }, ERR, lists, reads));
    ASSERT_EQ(COUNTED.size(), 1U);
    EXPECT_NE(COUNTED.front().why.find(std::system_category().message(ERR)), std::string::npos)
        << COUNTED.front().why;
    EXPECT_EQ(*lists, 1) << "a failed look was given the grace";
  }
}

/**
 * @test The naming of a counted child says "unavailable" for each field it
 * cannot read any more, and still gives why the child counted.
 */
TEST(ZombieCount, NamingSaysWhatItCannotReadIsUnavailable) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  // No process has the largest pid there is, so nothing of it can be read.
  const pid_t NONE = std::numeric_limits<pid_t>::max();
  const std::string NAMED = namedZombieChildren(dir, {{NONE, "state unreadable (a test's)"}});
  for (const char* FIELD :
       {"command line unavailable", "comm unavailable", "state unavailable", "parent unavailable",
        "group unavailable", "session unavailable", "(state unreadable (a test's))"}) {
    EXPECT_NE(NAMED.find(FIELD), std::string::npos) << FIELD << " in: " << NAMED;
  }
}

/* ----------------------------- Child Processes ----------------------------- */

namespace {

/** @brief A process table in a temporary directory, laid out as /proc is. */
class FakeProcTree {
public:
  FakeProcTree() {
    std::string pattern = (std::filesystem::temp_directory_path() / "vernier_proc_XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      root_ = pattern;
    }
  }
  ~FakeProcTree() {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }
  FakeProcTree(const FakeProcTree&) = delete;
  FakeProcTree& operator=(const FakeProcTree&) = delete;

  [[nodiscard]] const std::string& root() const { return root_; }

  /** @brief Process @p pid named @p comm with parent @p ppid (its stat line). */
  void process(pid_t pid, const std::string& comm, pid_t ppid) const {
    entry(std::to_string(pid), comm, ppid);
  }

  /** @brief A directory named @p name whose stat line names @p comm and @p ppid. */
  void entry(const std::string& name, const std::string& comm, pid_t ppid) const {
    const std::string DIR = root_ + "/" + name;
    std::filesystem::create_directories(DIR);
    std::ofstream(DIR + "/stat") << name << " (" << comm << ") S " << ppid << " " << name
                                 << " 0 0 -1 4194560 0 0 0 0 0 0 0 0 20 0 1 0\n";
  }

  /** @brief The kernel's children list of @p pid, holding @p text. */
  void childrenList(pid_t pid, const std::string& text) const {
    const std::string DIR = root_ + "/" + std::to_string(pid) + "/task/" + std::to_string(pid);
    std::filesystem::create_directories(DIR);
    std::ofstream(DIR + "/children") << text;
  }

private:
  std::string root_;
};

} // namespace

/** @test Where the kernel lists a process's children, the list is what counts. */
TEST(ReadinessChildProcesses, ReadsTheKernelsChildrenList) {
  FakeProcTree proc;
  ASSERT_FALSE(proc.root().empty());
  proc.process(10, "sudo", 1);
  proc.process(11, "bpftrace", 10);
  proc.process(12, "sh", 10); // listed by stat only: the kernel's list wins
  proc.childrenList(10, "11 ");
  EXPECT_EQ(childProcesses(10, proc.root()), (std::vector<pid_t>{11}));
  proc.childrenList(12, "");
  EXPECT_TRUE(childProcesses(12, proc.root()).empty());
}

/**
 * @test Without the kernel's list (CONFIG_PROC_CHILDREN off, as on the Jetson
 * AGX Thor), each process's parent is read from its stat line, whatever its
 * command name holds.
 */
TEST(ReadinessChildProcesses, WithoutTheListReadsEachParent) {
  FakeProcTree proc;
  ASSERT_FALSE(proc.root().empty());
  proc.process(10, "sudo", 1);
  proc.process(11, "a) (b 9 ", 10);
  proc.process(13, "tool name", 10);
  proc.process(14, "unrelated", 1);
  proc.process(15, "grandchild", 11);
  std::filesystem::create_directories(proc.root() + "/self");
  // Names that are not pids are skipped, even when their stat names the parent.
  proc.entry("99999999999999999999", "too large", 10);
  proc.entry("16x", "not a pid", 10);
  EXPECT_EQ(childProcesses(10, proc.root()), (std::vector<pid_t>{11, 13}));
  EXPECT_EQ(childProcesses(11, proc.root()), (std::vector<pid_t>{15}));
  EXPECT_TRUE(childProcesses(14, proc.root()).empty());
  EXPECT_TRUE(childProcesses(99, proc.root()).empty()) << "no such process";
}

/** @test The default root lists a child this process started. */
TEST(ReadinessChildProcesses, ListsALiveChildOfThisProcess) {
  const pid_t CHILD = ::fork();
  ASSERT_GE(CHILD, 0);
  if (CHILD == 0) {
    ::pause();
    ::_exit(0);
  }
  const std::vector<pid_t> CHILDREN = childProcesses(::getpid());
  ::kill(CHILD, SIGKILL);
  int status = 0;
  ::waitpid(CHILD, &status, 0);
  EXPECT_NE(std::find(CHILDREN.begin(), CHILDREN.end(), CHILD), CHILDREN.end());
}

namespace {

/** @brief A forked process holding children of its own, all paused. */
struct ParentOfPaused {
  pid_t pid = -1;
  std::vector<pid_t> children;
};

/** @brief Fork a process that starts @p count paused children and reports their pids. */
ParentOfPaused startParentOf(int count) {
  ParentOfPaused out;
  int fds[2] = {-1, -1};
  if (::pipe(fds) != 0) {
    return out;
  }
  out.pid = ::fork();
  if (out.pid == 0) {
    ::close(fds[0]);
    for (int i = 0; i < count; ++i) {
      const pid_t KID = ::fork();
      if (KID == 0) {
        ::pause();
        ::_exit(0);
      }
      (void)!::write(fds[1], &KID, sizeof(KID));
    }
    ::close(fds[1]);
    ::pause();
    ::_exit(0);
  }
  ::close(fds[1]);
  pid_t kid = -1;
  while (out.pid > 0 && static_cast<int>(out.children.size()) < count &&
         ::read(fds[0], &kid, sizeof(kid)) == static_cast<ssize_t>(sizeof(kid))) {
    out.children.push_back(kid);
  }
  ::close(fds[0]);
  return out;
}

/** @brief Kill @p parent's children and @p parent, and reap @p parent. */
void stopParentOf(const ParentOfPaused& parent) {
  for (const pid_t KID : parent.children) {
    if (KID > 0) {
      ::kill(KID, SIGKILL);
    }
  }
  if (parent.pid > 0) {
    ::kill(parent.pid, SIGKILL);
    int status = 0;
    ::waitpid(parent.pid, &status, 0);
  }
}

} // namespace

/**
 * @test tracerPid() names a process's only child, and the process itself when
 * it has none or more than one.
 */
TEST(ReadinessChildProcesses, TracerPidIsTheOnlyChild) {
  const ParentOfPaused ONE = startParentOf(1);
  const ParentOfPaused TWO = startParentOf(2);
  const pid_t ONLY = ONE.children.empty() ? -1 : ONE.children.front();
  const pid_t TRACER = vernier::bench::profiler_env::tracerPid(ONE.pid);
  const pid_t ALONE = vernier::bench::profiler_env::tracerPid(ONLY);
  const pid_t AMBIGUOUS = vernier::bench::profiler_env::tracerPid(TWO.pid);
  stopParentOf(ONE);
  stopParentOf(TWO);
  ASSERT_EQ(ONE.children.size(), 1U);
  ASSERT_EQ(TWO.children.size(), 2U);
  EXPECT_EQ(TRACER, ONLY);
  EXPECT_EQ(ALONE, ONLY) << "a process with no child is its own tracer";
  EXPECT_EQ(AMBIGUOUS, TWO.pid) << "two children: neither is the tracer";
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
