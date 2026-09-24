/**
 * @file ProfilerReadinessChecks_uTest.cpp
 * @brief Unit tests for the backends' own readiness checks.
 *
 * Each test asks the registry for one backend's decision about one request in
 * an explicit context whose PATH holds only fake tools (ReadinessFixtures.hpp),
 * and reads what the fakes were asked to do from their log. Nothing here
 * changes the process environment or needs privileges.
 */

#include "src/bench/inc/ProfilerBpftrace.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerOffCpu.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

using vernier::bench::BpftracePlan;
using vernier::bench::decideGperfAnalysis;
using vernier::bench::EnvReport;
using vernier::bench::GperfAnalysis;
using vernier::bench::GperfModes;
using vernier::bench::GperfPlan;
using vernier::bench::OffCpuPlan;
using vernier::bench::parseGperfModes;
using vernier::bench::parsePerfMode;
using vernier::bench::PerfMode;
using vernier::bench::PerfPlan;
using vernier::bench::PrivilegeRoute;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

namespace {

/** @brief A request for @p backend with @p scripts, as the doctor's selected row asks it. */
ReadinessRequest requestFor(const std::string& backend, std::vector<std::string> scripts = {}) {
  ReadinessRequest request;
  request.backend = backend;
  request.bpfScripts = std::move(scripts);
  request.scope = ReadinessScope::PREFLIGHT;
  return request;
}

/** @brief Pids of the fake bpftrace processes the log recorded. */
std::vector<pid_t> tracerPids(const FakeToolDir& dir) {
  std::vector<pid_t> pids;
  for (const std::string& line : dir.logLines("bpftrace ")) {
    const auto AT = line.rfind(" pid=");
    if (AT != std::string::npos && line.find("--version") == std::string::npos) {
      pids.push_back(static_cast<pid_t>(std::stol(line.substr(AT + 5))));
    }
  }
  return pids;
}

/** @brief True when no process @p pid exists any more. */
bool gone(pid_t pid) {
  errno = 0;
  return ::kill(pid, 0) != 0 && errno == ESRCH;
}

/** @brief Fake tools and a scripts directory holding one sched-tracepoint script. */
class BpfCheckTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    bpftrace_ = dir_.install("fake_bpftrace.sh", "bpftrace");
    dir_.makeDirectory("scripts");
    script_ = dir_.writeFile("scripts/probe_script.bt",
                             "tracepoint:sched:sched_switch /pid == {{PID}}/ { @c = count(); }\n");
  }

  void installSudoAndKill() {
    sudo_ = dir_.install("fake_sudo.sh", "sudo");
    kill_ = dir_.install("fake_kill.sh", "kill");
  }

  /** @brief The context: fakes on PATH, the scripts directory, and @p extra. */
  ReadinessContext ctx(std::map<std::string, std::string> extra = {},
                       uid_t euid = ::geteuid()) const {
    extra["PERF_BPF_SCRIPTS"] = dir_.path() + "/scripts";
    return dir_.context(std::move(extra), euid);
  }

  ReadinessResult check(const std::string& backend, const ReadinessContext& context) const {
    return ProfilerRegistry::instance().checkRequest(
        requestFor(backend, backend == "bpftrace" ? std::vector<std::string>{"probe_script"}
                                                  : std::vector<std::string>{}),
        context);
  }

  FakeToolDir dir_;
  std::string bpftrace_;
  std::string script_;
  std::string sudo_;
  std::string kill_;
};

} // namespace

/* ----------------------------- bpftrace ----------------------------- */

/** @test Without an opt-in the attach runs as the current user and sudo is never called. */
TEST_F(BpfCheckTest, BpftraceCurrentUserAttaches) {
  installSudoAndKill(); // present, and still unused
  const ReadinessResult R = check("bpftrace", ctx());
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  EXPECT_NE(R.report.message.find("probe_script attached for 1000 ms as the current user and "
                                  "stopped on SIGINT (probe with " +
                                  bpftrace_ + ")"),
            std::string::npos)
      << R.report.message;
  EXPECT_TRUE(dir_.logLines("sudo").empty()) << dir_.log();
  EXPECT_TRUE(dir_.logLines("kill").empty()) << dir_.log();
  EXPECT_EQ(dir_.logLines("bpftrace --version").size(), 1U) << dir_.log();
  EXPECT_EQ(dir_.logLines("bpftrace -q ").size(), 1U) << dir_.log();
}

/** @test A current user bpftrace refuses is denied, with the ways to get access. */
TEST_F(BpfCheckTest, BpftraceCurrentUserDenied) {
  installSudoAndKill();
  const ReadinessResult R = check("bpftrace", ctx({{"FAKE_BPFTRACE_MODE", "eperm"}}));
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  EXPECT_EQ(R.cause, ReadinessCause::DENIED);
  EXPECT_EQ(R.report.message,
            "denied: script 'probe_script' could not attach as the current user: ERROR: bpftrace "
            "currently only supports running as the root user.");
  EXPECT_EQ(R.report.hint, "Set BENCH_SUDO=1 with a scoped sudoers grant for " + bpftrace_ +
                               " and " + kill_ +
                               ", run with CAP_BPF and CAP_PERFMON, or run as root.");
  EXPECT_TRUE(dir_.logLines("sudo").empty()) << "no opt-in, no sudo\n" << dir_.log();
}

/** @test The sudo route runs the resolved tools, and the plan carries them. */
TEST_F(BpfCheckTest, BpftraceSudoRouteUsesResolvedTools) {
  installSudoAndKill();
  const ReadinessResult R = check("bpftrace", ctx({{"BENCH_SUDO", "1"}}));
  ASSERT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  EXPECT_NE(R.report.message.find("through sudo -n (BENCH_SUDO=1)"), std::string::npos);
  const auto PLAN = std::dynamic_pointer_cast<const BpftracePlan>(R.plan);
  ASSERT_NE(PLAN, nullptr);
  EXPECT_EQ(PLAN->route.privilege.route, PrivilegeRoute::SCOPED_SUDO);
  EXPECT_EQ(PLAN->route.bpftrace, bpftrace_);
  EXPECT_EQ(PLAN->route.sudo, sudo_);
  EXPECT_EQ(PLAN->route.kill, kill_);
  ASSERT_EQ(PLAN->scriptPaths.size(), 1U);
  EXPECT_EQ(PLAN->scriptPaths.front(), script_);
  // The attach and the stop go through sudo; the version check never does.
  const std::vector<std::string> SUDO = dir_.logLines("sudo ");
  ASSERT_EQ(SUDO.size(), 2U) << dir_.log();
  EXPECT_EQ(SUDO[0].rfind("sudo -n -- " + bpftrace_ + " -q ", 0), 0U) << SUDO[0];
  EXPECT_EQ(SUDO[1].rfind("sudo -n -- " + kill_ + " -2 ", 0), 0U) << SUDO[1];
  EXPECT_EQ(dir_.log().find("sudo -n -- " + bpftrace_ + " --version"), std::string::npos);
}

/** @test A grant that allows the version but not the attach is refused, never Ok. */
TEST_F(BpfCheckTest, BpftraceAttachRefusedByGrant) {
  installSudoAndKill();
  const ReadinessResult R = check("bpftrace", ctx({{"BENCH_SUDO", "1"}, {"FAKE_SUDO_DENY", "-q"}}));
  EXPECT_EQ(R.cause, ReadinessCause::DENIED);
  EXPECT_EQ(R.report.message, "denied: sudo -n refused " + bpftrace_ + " -q " + script_ +
                                  ": sudo: a password is required")
      << "the message names the script, not its temporary copy";
  EXPECT_EQ(R.report.hint.rfind("The grant must allow " + bpftrace_ +
                                    " with the run's script arguments and " + kill_ +
                                    " with -2, -15 and -9",
                                0),
            0U)
      << R.report.hint;
}

/** @test A refused SIGINT is a denied cleanup, and the probe tracer does not survive. */
TEST_F(BpfCheckTest, BpftraceStopSignalRefused) {
  installSudoAndKill();
  const ReadinessResult R =
      check("bpftrace", ctx({{"BENCH_SUDO", "1"}, {"FAKE_SUDO_DENY", "kill -2"}}));
  EXPECT_EQ(R.cause, ReadinessCause::DENIED);
  EXPECT_EQ(R.report.message.rfind("denied: cleanup: sudo -n refused " + kill_ + " -2 ", 0), 0U)
      << R.report.message;
  EXPECT_NE(R.report.hint.find("-2, -15 and -9"), std::string::npos);
  const std::vector<pid_t> TRACERS = tracerPids(dir_);
  ASSERT_EQ(TRACERS.size(), 1U) << dir_.log();
  EXPECT_TRUE(gone(TRACERS.front())) << "the probe tracer outlived the check";
}

/** @test Grants for signals the stop never needed are not demanded. */
TEST_F(BpfCheckTest, BpftraceUnusedSignalGrantsNotNeeded) {
  installSudoAndKill();
  const ReadinessResult R =
      check("bpftrace", ctx({{"BENCH_SUDO", "1"}, {"FAKE_SUDO_DENY", "kill -15|kill -9"}}));
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
}

/** @test Refusing invocations the check never makes does not reject the selected operation. */
TEST_F(BpfCheckTest, BpftraceUnrelatedInvocationsRefused) {
  installSudoAndKill();
  const ReadinessResult R = check(
      "bpftrace", ctx({{"BENCH_SUDO", "1"}, {"FAKE_SUDO_DENY", "--version|kill -0|sudo -l|-l "}}));
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  for (const std::string& line : dir_.logLines("sudo ")) {
    EXPECT_EQ(line.find("--version"), std::string::npos) << line;
    EXPECT_EQ(line.find("kill -0"), std::string::npos) << line;
  }
}

/** @test A tracer that ignores SIGINT is a caveat: the run's output may be incomplete. */
TEST_F(BpfCheckTest, BpftraceIgnoredInterruptIsCaveat) {
  installSudoAndKill();
  const ReadinessResult R =
      check("bpftrace", ctx({{"BENCH_SUDO", "1"}, {"FAKE_BPFTRACE_MODE", "ignore-int"}}));
  EXPECT_EQ(R.report.status, EnvReport::Status::Warning);
  EXPECT_EQ(R.cause, ReadinessCause::CAVEAT);
  EXPECT_EQ(R.report.message, "the probe tracer ignored SIGINT and stopped on SIGTERM; the run's "
                              "output may be incomplete");
  EXPECT_TRUE(R.collectionReady());
}

/** @test The sudo route needs sudo and kill on PATH. */
TEST_F(BpfCheckTest, BpftraceSudoRouteNeedsItsHelpers) {
  const ReadinessResult NO_SUDO = check("bpftrace", ctx({{"BENCH_SUDO", "1"}}));
  EXPECT_EQ(NO_SUDO.cause, ReadinessCause::MISSING_HELPER);
  EXPECT_EQ(NO_SUDO.report.message.rfind("missing helper: sudo is not on PATH; BENCH_SUDO=1 runs "
                                         "bpftrace through sudo -n",
                                         0),
            0U)
      << NO_SUDO.report.message;
  dir_.install("fake_sudo.sh", "sudo");
  const ReadinessResult NO_KILL = check("bpftrace", ctx({{"BENCH_SUDO", "1"}}));
  EXPECT_EQ(NO_KILL.cause, ReadinessCause::MISSING_HELPER);
  EXPECT_EQ(NO_KILL.report.message.rfind("missing helper: kill is not on PATH", 0), 0U)
      << NO_KILL.report.message;
  EXPECT_TRUE(dir_.logLines("bpftrace").empty()) << "nothing runs before the route is complete";
}

/** @test An invalid setting is a configuration error and launches nothing. */
TEST_F(BpfCheckTest, BpftraceInvalidSettingLaunchesNothing) {
  installSudoAndKill();
  for (const auto& [KEY, VALUE] : std::map<std::string, std::string>{
           {"BENCH_SUDO", "maybe"}, {"PERF_BPF_SUDO", "sometimes"}}) {
    const ReadinessResult R = check("bpftrace", ctx({{KEY, VALUE}}));
    EXPECT_EQ(R.cause, ReadinessCause::CONFIGURATION);
    EXPECT_EQ(R.report.message, "configuration: " + KEY + "='" + VALUE + "' is not a boolean");
    EXPECT_EQ(R.report.hint, "Use 1, true, yes or on to run the probe tool with sudo -n; 0, false, "
                             "no, off or an empty value to run it as the current user.");
  }
  EXPECT_EQ(dir_.log(), "") << "a configuration error must launch nothing";
}

/** @test The deprecated alias still selects the route, with a warning saying so. */
TEST_F(BpfCheckTest, BpftraceLegacyAliasWarns) {
  installSudoAndKill();
  const ReadinessResult ON = check("bpftrace", ctx({{"PERF_BPF_SUDO", "1"}}));
  EXPECT_EQ(ON.report.status, EnvReport::Status::Warning);
  EXPECT_NE(ON.report.message.find("through sudo -n (PERF_BPF_SUDO=1 (deprecated alias))"),
            std::string::npos)
      << ON.report.message;
  EXPECT_NE(ON.report.message.find("PERF_BPF_SUDO is deprecated; set BENCH_SUDO instead. "
                                   "PERF_BPF_SUDO=1 selects: the probe tool runs with sudo -n."),
            std::string::npos)
      << ON.report.message;
  EXPECT_FALSE(dir_.logLines("sudo ").empty());
}

/** @test Conflicting settings name the winner; equal ones are silent. */
TEST_F(BpfCheckTest, BpftraceConflictNamesWinner) {
  installSudoAndKill();
  const ReadinessResult CONFLICT =
      check("bpftrace", ctx({{"BENCH_SUDO", "1"}, {"PERF_BPF_SUDO", "0"}}));
  EXPECT_EQ(CONFLICT.report.status, EnvReport::Status::Warning);
  EXPECT_NE(CONFLICT.report.message.find(
                "BENCH_SUDO=1 and PERF_BPF_SUDO=0 disagree; BENCH_SUDO wins (the probe tool runs "
                "with sudo -n). PERF_BPF_SUDO is deprecated: remove it."),
            std::string::npos)
      << CONFLICT.report.message;
  const ReadinessResult EQUAL =
      check("bpftrace", ctx({{"BENCH_SUDO", "1"}, {"PERF_BPF_SUDO", "yes"}}));
  EXPECT_EQ(EQUAL.report.status, EnvReport::Status::Ok) << EQUAL.report.message;
}

/** @test Root never calls sudo, and an opt-in is noted as unneeded. */
TEST_F(BpfCheckTest, BpftraceRootNeverCallsSudo) {
  installSudoAndKill();
  const ReadinessResult R = check("bpftrace", ctx({{"BENCH_SUDO", "1"}}, 0));
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  EXPECT_NE(R.report.message.find(" as root and stopped on SIGINT"), std::string::npos);
  EXPECT_NE(R.report.message.find("; running as root; BENCH_SUDO not needed"), std::string::npos);
  EXPECT_TRUE(dir_.logLines("sudo").empty()) << dir_.log();
}

/** @test bpftrace's own attach errors keep their cause. */
TEST_F(BpfCheckTest, BpftraceAttachErrorsKeepTheirCause) {
  const ReadinessResult UNSUPPORTED =
      check("bpftrace", ctx({{"FAKE_BPFTRACE_MODE", "unsupported"}}));
  EXPECT_EQ(UNSUPPORTED.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(UNSUPPORTED.report.message, "unsupported: script 'probe_script': stdin:1:1-36: ERROR: "
                                        "tracepoint not found: syscalls:sys_enter_write");
  const ReadinessResult BROKEN = check("bpftrace", ctx({{"FAKE_BPFTRACE_MODE", "broken"}}));
  EXPECT_EQ(BROKEN.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(BROKEN.report.message, "unusable: script 'probe_script' did not stay attached: fake "
                                   "bpftrace: the program could not be loaded");
}

/** @test A missing, non-executable or broken bpftrace, or a missing script, is an error. */
TEST_F(BpfCheckTest, BpftraceToolAndScriptProblems) {
  const ReadinessResult BROKEN = check("bpftrace", ctx({{"FAKE_BPFTRACE_MODE", "version-fails"}}));
  EXPECT_EQ(BROKEN.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(
      BROKEN.report.message.rfind("unusable: " + bpftrace_ + " --version: exit status 127: ", 0),
      0U)
      << BROKEN.report.message;

  const ReadinessResult NO_SCRIPT =
      ProfilerRegistry::instance().checkRequest(requestFor("bpftrace", {"absent_script"}), ctx());
  EXPECT_EQ(NO_SCRIPT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(NO_SCRIPT.report.message, "missing: bpftrace script 'absent_script' not found at " +
                                          dir_.path() + "/scripts/absent_script.bt");

  dir_.install("fake_bpftrace.sh", "bpftrace", 0644);
  const ReadinessResult PLAIN = check("bpftrace", ctx());
  EXPECT_EQ(PLAIN.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(PLAIN.report.message, "unusable: " + bpftrace_ + " is not an executable file");

  FakeToolDir empty;
  const ReadinessResult MISSING = ProfilerRegistry::instance().checkRequest(
      requestFor("bpftrace", {"probe_script"}), empty.context());
  EXPECT_EQ(MISSING.cause, ReadinessCause::MISSING);
  EXPECT_EQ(MISSING.report.message, "missing: bpftrace not found on PATH");
}

/** @test An unreadable selected script is rejected before anything runs, and a run leaves no
 * folder. */
TEST_F(BpfCheckTest, BpftraceUnreadableScriptLaunchesNothing) {
  installSudoAndKill();
  ASSERT_EQ(::chmod(script_.c_str(), 0), 0);
  const struct Restore {
    std::string path;
    ~Restore() { (void)::chmod(path.c_str(), 0644); }
  } RESTORE{script_};
  if (::access(script_.c_str(), R_OK) == 0) {
    GTEST_SKIP() << "this user can read a file with mode 000 (root)";
  }
  for (const std::map<std::string, std::string>& extra :
       {std::map<std::string, std::string>{},
        std::map<std::string, std::string>{{"BENCH_SUDO", "1"}}}) {
    const ReadinessResult R = check("bpftrace", ctx(extra));
    EXPECT_EQ(R.cause, ReadinessCause::UNUSABLE);
    EXPECT_FALSE(R.collectionReady());
    EXPECT_EQ(R.report.message, "unusable: bpftrace script 'probe_script' at " + script_ +
                                    " cannot be read: Permission denied");
    EXPECT_EQ(R.report.hint, "Give this user read access to " + script_ +
                                 ", or select a script it can read with --bpf.");
  }
  // A run of the same request: a no-op profiler, no capture folder, and not
  // one bpftrace (or sudo) invocation, not even --version.
  vernier::bench::PerfConfig cfg;
  cfg.profileTool = "bpftrace";
  cfg.bpfScripts = {"probe_script"};
  cfg.artifactRoot = dir_.path() + "/captures";
  {
    StderrCapture quiet;
    auto profiler = ProfilerRegistry::instance().make("bpftrace", cfg, "Bpf.Unreadable", ctx());
    ASSERT_NE(profiler, nullptr);
    EXPECT_EQ(profiler->artifactDir(), "");
    profiler->beforeMeasure();
    profiler->afterMeasure(vernier::bench::Stats{});
    EXPECT_NE(quiet.text().find("cannot be read: Permission denied"), std::string::npos);
  }
  EXPECT_FALSE(std::filesystem::exists(cfg.artifactRoot)) << "a rejected request left a folder";
  EXPECT_TRUE(dir_.logLines("bpftrace").empty()) << dir_.log();
  EXPECT_TRUE(dir_.logLines("sudo").empty()) << dir_.log();
}

/** @test A probe copy that cannot be written is reported as the probe copy, not as the script. */
TEST_F(BpfCheckTest, BpftraceProbeCopyUnwritable) {
  const std::string ABSENT = dir_.path() + "/absent";
  const ReadinessResult R = check("bpftrace", ctx({{"TMPDIR", ABSENT}}));
  EXPECT_EQ(R.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(R.report.message, "unusable: cannot write the probe copy of script 'probe_script' in "
                              "a new directory under " +
                                  ABSENT);
  EXPECT_EQ(R.report.hint, "Point TMPDIR at a writable directory, or make /tmp writable.");
  EXPECT_TRUE(tracerPids(dir_).empty()) << "no tracer may start:\n" << dir_.log();
}

/* ----------------------------- offcpu ----------------------------- */

/** @test offcpu attaches as the current user by default, in the launch's own shape. */
TEST_F(BpfCheckTest, OffCpuCurrentUserAttaches) {
  installSudoAndKill();
  const ReadinessResult R = check("offcpu", ctx({{"PERF_BPF_SUDO", "1"}}));
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  EXPECT_NE(R.report.message.find("the off-CPU script attached for 1500 ms as the current user"),
            std::string::npos)
      << R.report.message;
  EXPECT_TRUE(dir_.logLines("sudo").empty()) << "PERF_BPF_SUDO does not apply to offcpu";
  ASSERT_EQ(dir_.logLines("bpftrace -e ").size(), 1U) << dir_.log();
  // The script text spans lines; the target pid is the argument after it.
  EXPECT_NE(dir_.log().find("exit(); }\n " + std::to_string(::getpid()) + " pid="),
            std::string::npos)
      << dir_.log();
  const auto PLAN = std::dynamic_pointer_cast<const OffCpuPlan>(R.plan);
  ASSERT_NE(PLAN, nullptr);
  EXPECT_EQ(PLAN->route.bpftrace, bpftrace_);
}

/** @test offcpu with BENCH_SUDO goes through sudo; a refusal there is denied. */
TEST_F(BpfCheckTest, OffCpuSudoRoute) {
  installSudoAndKill();
  const ReadinessResult OK = check("offcpu", ctx({{"BENCH_SUDO", "yes"}}));
  EXPECT_EQ(OK.report.status, EnvReport::Status::Ok) << OK.report.message;
  EXPECT_EQ(dir_.logLines("sudo -n -- " + bpftrace_ + " -e ").size(), 1U) << dir_.log();
  const ReadinessResult REFUSED =
      check("offcpu", ctx({{"BENCH_SUDO", "yes"}, {"FAKE_SUDO_DENY", "-e"}}));
  EXPECT_EQ(REFUSED.cause, ReadinessCause::DENIED);
  EXPECT_EQ(REFUSED.report.message.rfind("denied: sudo -n refused " + bpftrace_ + " -e ", 0), 0U)
      << REFUSED.report.message;
}

/** @test offcpu denied as the current user says how to get access. */
TEST_F(BpfCheckTest, OffCpuCurrentUserDenied) {
  installSudoAndKill();
  const ReadinessResult R = check("offcpu", ctx({{"FAKE_BPFTRACE_MODE", "eperm"}}));
  EXPECT_EQ(R.cause, ReadinessCause::DENIED);
  EXPECT_EQ(R.report.message.rfind("denied: the off-CPU script could not attach as the current "
                                   "user: ERROR:",
                                   0),
            0U)
      << R.report.message;
  EXPECT_EQ(R.report.hint.rfind("Set BENCH_SUDO=1 with a scoped sudoers grant", 0), 0U);
}

/* ----------------------------- perf ----------------------------- */

namespace {

/** @brief A fake perf on PATH, asked about one request. */
class PerfCheckTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    perf_ = dir_.install("fake_perf.sh", "perf");
  }

  ReadinessResult check(std::map<std::string, std::string> env, const std::string& args = "",
                        uid_t euid = 0) const {
    ReadinessRequest request;
    request.backend = "perf";
    request.profileArgs = args;
    request.scope = ReadinessScope::PREFLIGHT;
    return ProfilerRegistry::instance().checkRequest(request, dir_.context(std::move(env), euid));
  }

  FakeToolDir dir_;
  std::string perf_;
};

} // namespace

/** @test One parser decides the mode for the check and the launch. */
TEST(PerfModeTest, ParsesTheSelectedMode) {
  EXPECT_EQ(parsePerfMode(""), PerfMode::STAT);
  EXPECT_EQ(parsePerfMode("-a --per-thread"), PerfMode::STAT);
  EXPECT_EQ(parsePerfMode("record -g"), PerfMode::RECORD);
  EXPECT_EQ(parsePerfMode("  record"), PerfMode::RECORD);
  EXPECT_EQ(parsePerfMode("mem"), PerfMode::MEM);
  EXPECT_EQ(parsePerfMode("c2c"), PerfMode::C2C);
}

/** @test Counting access is probed with the resolved perf, which the plan keeps. */
TEST_F(PerfCheckTest, CountsWithTheResolvedTool) {
  const ReadinessResult R = check({});
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  EXPECT_EQ(R.report.message, "perf stat counted cpu-cycles,instructions,branches,branch-misses,"
                              "cache-misses on this process (probe with " +
                                  perf_ + ")");
  const auto PLAN = std::dynamic_pointer_cast<const PerfPlan>(R.plan);
  ASSERT_NE(PLAN, nullptr);
  EXPECT_EQ(PLAN->perf, perf_);
  EXPECT_EQ(PLAN->mode, PerfMode::STAT);
  EXPECT_EQ(dir_.logLines("perf " + perf_ +
                          " stat -x, -e cpu-cycles,instructions,branches,"
                          "branch-misses,cache-misses -p " +
                          std::to_string(::getpid()) + " --timeout 100")
                .size(),
            1U)
      << dir_.log();
}

/** @test A perf whose --version fails is unusable, and is never asked to count. */
TEST_F(PerfCheckTest, BrokenPerfIsNeverRun) {
  const ReadinessResult R = check({{"FAKE_PERF_MODE", "broken"}});
  EXPECT_EQ(R.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(R.report.message.rfind("unusable: " + perf_ +
                                       " --version: exit status 2: WARNING: "
                                       "perf not found for kernel 6.8.0-138",
                                   0),
            0U)
      << R.report.message;
  EXPECT_NE(R.report.hint.find("linux-tools-$(uname -r)"), std::string::npos);
  EXPECT_TRUE(dir_.logLines("perf " + perf_ + " stat").empty()) << dir_.log();
}

/** @test Refused counter access is denied, with the remedies and no elevation. */
TEST_F(PerfCheckTest, DeniedAccessNamesTheRemedy) {
  const ReadinessResult R = check({{"FAKE_PERF_MODE", "denied"}}, "", 1000);
  EXPECT_EQ(R.cause, ReadinessCause::DENIED);
  EXPECT_EQ(R.report.message, "denied: perf stat cannot open the counters as this user: Access to "
                              "performance monitoring and observability operations is limited.");
  EXPECT_EQ(R.report.hint, "Grant CAP_PERFMON to the benchmark, lower kernel.perf_event_paranoid "
                           "(sudo sysctl -w kernel.perf_event_paranoid=2), or run as root; vernier "
                           "does not elevate perf.");
}

/** @test An event the PMU lacks is a caveat, not a failure. */
TEST_F(PerfCheckTest, UnsupportedEventIsCaveat) {
  const ReadinessResult R = check({{"FAKE_PERF_MODE", "unsupported"}});
  EXPECT_EQ(R.cause, ReadinessCause::CAVEAT);
  EXPECT_EQ(R.report.message, "perf stat counts this process, but cache-misses <not supported> "
                              "here; those columns stay empty");
}

/** @test Modes beyond stat get the access probe and stay unverified past it. */
TEST_F(PerfCheckTest, OtherModesUnverifiedBeyondAccess) {
  const ReadinessResult RECORD = check({}, "record -g");
  EXPECT_EQ(RECORD.report.status, EnvReport::Status::Warning);
  EXPECT_EQ(RECORD.report.message,
            "unverified: perf stat counts this process; perf record itself is not probed before "
            "the run");
  const auto PLAN = std::dynamic_pointer_cast<const PerfPlan>(RECORD.plan);
  ASSERT_NE(PLAN, nullptr);
  EXPECT_EQ(PLAN->mode, PerfMode::RECORD);
  const ReadinessResult DENIED = check({{"FAKE_PERF_MODE", "denied"}}, "c2c", 1000);
  EXPECT_EQ(DENIED.cause, ReadinessCause::DENIED) << "known unavailability is not unverified";
}

/** @test A missing or non-executable perf is an error. */
TEST_F(PerfCheckTest, MissingOrNotExecutable) {
  dir_.install("fake_perf.sh", "perf", 0644);
  const ReadinessResult PLAIN = check({});
  EXPECT_EQ(PLAIN.report.message, "unusable: " + perf_ + " is not an executable file");
  FakeToolDir empty;
  ReadinessRequest request;
  request.backend = "perf";
  const ReadinessResult MISSING =
      ProfilerRegistry::instance().checkRequest(request, empty.context());
  EXPECT_EQ(MISSING.report.message, "missing: perf not found on PATH");
}

/** @test The launch runs the perf the plan names, not whatever PATH finds then. */
TEST_F(PerfCheckTest, LaunchRunsThePlannedPath) {
  FakeToolDir other; // the live PATH: no perf at all
  ASSERT_TRUE(other.ok());
  auto plan = std::make_shared<PerfPlan>();
  plan->perf = perf_;
  vernier::bench::PerfConfig cfg;
  cfg.profileTool = "perf";
  cfg.artifactRoot = other.path();
  {
    ScopedEnv path("PATH", other.path());
    ScopedEnv log("FAKE_LOG", dir_.logPath());
    vernier::bench::PerfStatProfiler profiler(cfg, "Perf.Launch", plan);
    profiler.beforeMeasure();
    profiler.afterMeasure(vernier::bench::Stats{});
  }
  EXPECT_EQ(dir_.logLines("perf " + perf_ +
                          " stat -e cpu-cycles,instructions,branches,"
                          "branch-misses,cache-misses -p " +
                          std::to_string(::getpid()))
                .size(),
            1U)
      << dir_.log();
}

/**
 * @test A perf and an artifact root whose paths hold shell punctuation run as
 * checked: the launch executes the checked perf and writes where it says.
 */
TEST_F(PerfCheckTest, LaunchKeepsPunctuationInPaths) {
  const std::string BIN_NAME = "tool's bin $HOME \"q\" `true`";
  const std::string BIN = dir_.makeDirectory(BIN_NAME);
  const std::string PERF = dir_.install("fake_perf.sh", BIN_NAME + "/perf");
  ReadinessRequest request;
  request.backend = "perf";
  request.scope = ReadinessScope::PREFLIGHT;
  const ReadinessResult R = ProfilerRegistry::instance().checkRequest(
      request, ReadinessContext(0, ::getpid(), {{"PATH", BIN}, {"FAKE_LOG", dir_.logPath()}}));
  ASSERT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  const auto CHECKED = std::dynamic_pointer_cast<const PerfPlan>(R.plan);
  ASSERT_NE(CHECKED, nullptr);
  ASSERT_EQ(CHECKED->perf, PERF);

  const std::string ROOT = dir_.makeDirectory("art's $PWD \"x\" `id`");
  const std::string PID = std::to_string(::getpid());
  for (const std::string& args : {std::string{}, std::string{"record -g"}}) {
    auto plan = std::make_shared<PerfPlan>(*CHECKED);
    plan->mode = parsePerfMode(args);
    vernier::bench::PerfConfig cfg;
    cfg.profileTool = "perf";
    cfg.profileArgs = args;
    cfg.artifactRoot = ROOT;
    ScopedEnv log("FAKE_LOG", dir_.logPath());
    StderrCapture err;
    vernier::bench::PerfStatProfiler profiler(cfg, args.empty() ? "Perf.Stat" : "Perf.Record",
                                              plan);
    profiler.beforeMeasure();
    profiler.afterMeasure(vernier::bench::Stats{});
    const std::string TEXT = err.text();
    EXPECT_EQ(TEXT.find("Syntax error"), std::string::npos) << TEXT;
    EXPECT_EQ(TEXT.find("not found"), std::string::npos) << TEXT;
  }
  EXPECT_EQ(dir_.logLines("perf " + PERF +
                          " stat -e cpu-cycles,instructions,branches,branch-misses,cache-misses "
                          "-p " +
                          PID + " pid=")
                .size(),
            1U)
      << dir_.log();
  EXPECT_EQ(dir_.logLines("perf " + PERF + " record -g -p " + PID + " -o " + ROOT +
                          "/Perf.Record.perf/perf.data pid=")
                .size(),
            1U)
      << dir_.log();
  EXPECT_TRUE(std::filesystem::exists(ROOT + "/Perf.Stat.perf/stat.txt"))
      << "the stat redirect did not reach the artifact folder";
  EXPECT_TRUE(std::filesystem::exists(ROOT + "/Perf.Record.perf/record.err.txt"));
}

/* ----------------------------- gperf ----------------------------- */

namespace {

/** @brief Analyzer fakes on PATH, asked about one gperf request. */
class GperfCheckTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    if (UB_HAS_GPERF_CPU == 0) {
      GTEST_SKIP() << "gperftools CPU profiling is not compiled in";
    }
  }

  /** @brief What the row says the build supports. */
  static std::string built() {
    return UB_HAS_GPERF_HEAP != 0 ? " (built: cpu, heap)" : " (built: cpu)";
  }

  ReadinessResult check(bool analyze, const std::string& args = "") const {
    ReadinessRequest request;
    request.backend = "gperf";
    request.profileArgs = args;
    request.analyze = analyze;
    request.scope = ReadinessScope::PREFLIGHT;
    return ProfilerRegistry::instance().checkRequest(request, dir_.context());
  }

  FakeToolDir dir_;
};

} // namespace

/** @test One parser decides the modes for the check and the profiler. */
TEST(GperfModeTest, ParsesTheSelectedModes) {
  EXPECT_TRUE(parseGperfModes("").cpu);
  EXPECT_FALSE(parseGperfModes("").heap);
  EXPECT_FALSE(parseGperfModes("heap").cpu);
  EXPECT_TRUE(parseGperfModes("heap").heap);
  EXPECT_TRUE(parseGperfModes("both").cpu && parseGperfModes("both").heap);
  EXPECT_TRUE(parseGperfModes("cpu,heap").cpu && parseGperfModes("cpu,heap").heap);
}

/** @test Without --profile-analyze a missing analyzer is only information. */
TEST_F(GperfCheckTest, AnalysisOffNeedsNoAnalyzer) {
  const ReadinessResult R = check(false);
  EXPECT_EQ(R.report.status, EnvReport::Status::Ok) << R.report.message;
  EXPECT_EQ(R.report.message, "gperftools profiles cpu" + built() +
                                  "; no analyzer on PATH (only --profile-analyze needs one)");
}

/** @test The same environment with --profile-analyze is an analysis error that still collects. */
TEST_F(GperfCheckTest, AnalysisOnWithoutAnalyzerIsAnalysisError) {
  const ReadinessResult R = check(true);
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  EXPECT_EQ(R.stage, vernier::bench::ReadinessStage::ANALYSIS);
  EXPECT_TRUE(R.collectionReady()) << "the capture still runs";
  EXPECT_EQ(R.report.message, "analysis: missing: --profile-analyze needs google-pprof or pprof, "
                              "and neither is on PATH; the cpu capture still runs and cpu.prof "
                              "is kept");
  EXPECT_EQ(R.report.hint, "Install an analyzer (google-pprof from gperftools, or Go's pprof), or "
                           "drop --profile-analyze.");
  const auto PLAN = std::dynamic_pointer_cast<const GperfPlan>(R.plan);
  ASSERT_NE(PLAN, nullptr);
  EXPECT_TRUE(PLAN->analyzer.empty());
}

/** @test The analyzer is the first of google-pprof and pprof found, whichever exists. */
TEST_F(GperfCheckTest, EitherAnalyzerSpellingIsFound) {
  const std::string PPROF = dir_.install("fake_pprof.sh", "pprof");
  const ReadinessResult ONLY_PPROF = check(true);
  EXPECT_EQ(ONLY_PPROF.report.status, EnvReport::Status::Ok) << ONLY_PPROF.report.message;
  EXPECT_EQ(ONLY_PPROF.report.message,
            "gperftools profiles cpu" + built() + "; --profile-analyze runs " + PPROF);
  EXPECT_EQ(std::dynamic_pointer_cast<const GperfPlan>(ONLY_PPROF.plan)->analyzer, PPROF);

  const std::string GOOGLE = dir_.install("fake_pprof.sh", "google-pprof");
  const ReadinessResult BOTH = check(true);
  EXPECT_EQ(std::dynamic_pointer_cast<const GperfPlan>(BOTH.plan)->analyzer, GOOGLE)
      << "google-pprof is tried first";

  FakeToolDir onlyGoogle;
  const std::string ALONE = onlyGoogle.install("fake_pprof.sh", "google-pprof");
  ReadinessRequest request;
  request.backend = "gperf";
  request.analyze = true;
  const ReadinessResult GOOGLE_ONLY =
      ProfilerRegistry::instance().checkRequest(request, onlyGoogle.context());
  EXPECT_EQ(std::dynamic_pointer_cast<const GperfPlan>(GOOGLE_ONLY.plan)->analyzer, ALONE);
}

/** @test A heap request without heap support is a collection error naming the option. */
TEST_F(GperfCheckTest, HeapWithoutSupportIsCollectionError) {
  if (UB_HAS_GPERF_HEAP != 0) {
    GTEST_SKIP() << "heap profiling is compiled in";
  }
  for (const char* ARGS : {"heap", "both"}) {
    const ReadinessResult R = check(false, ARGS);
    EXPECT_EQ(R.report.status, EnvReport::Status::Error) << ARGS;
    EXPECT_FALSE(R.collectionReady()) << ARGS;
    EXPECT_EQ(R.report.message.rfind("unsupported: heap profiling is not compiled in", 0), 0U);
    EXPECT_NE(R.report.hint.find("-DVERNIER_LINK_TCMALLOC=ON"), std::string::npos);
  }
}

/** @test A failing analyzer is reported with its status, and cpu.prof is kept. */
TEST_F(GperfCheckTest, FailingAnalyzerKeepsTheRawProfile) {
  const std::string PPROF = dir_.install("fake_pprof.sh", "pprof");
  auto plan = std::make_shared<GperfPlan>();
  plan->modes.cpu = true;
  plan->analyze = true;
  plan->analyzer = PPROF;
  plan->analysisReady = true; // its --help worked; reading this profile fails
  vernier::bench::PerfConfig cfg;
  cfg.profileTool = "gperf";
  cfg.profileAnalyze = true;
  cfg.artifactRoot = dir_.path();
  std::string err;
  {
    ScopedEnv mode("FAKE_PPROF_MODE", "fail-on-profile");
    ScopedEnv log("FAKE_LOG", dir_.logPath());
    vernier::bench::GperfProfiler profiler(cfg, "Gperf.Fails", plan);
    vernier::bench::test::StderrCapture capture;
    profiler.beforeMeasure();
    volatile double sink = 0;
    for (int i = 0; i < 2000000; ++i) {
      sink = sink + i * 0.5;
    }
    profiler.afterMeasure(vernier::bench::Stats{});
    err = capture.text();
  }
  const std::string CPU_PROF = dir_.path() + "/Gperf.Fails.gperf/cpu.prof";
  EXPECT_NE(
      err.find("[gperf] " + PPROF +
               " failed: exit status 1: fake pprof: cannot read profile; raw profile kept at " +
               CPU_PROF),
      std::string::npos)
      << err;
  std::error_code ec;
  EXPECT_TRUE(std::filesystem::exists(CPU_PROF, ec)) << "the raw capture is never removed";
  EXPECT_EQ(dir_.logLines("pprof " + PPROF + " --text --cum --lines ").size(), 1U) << dir_.log();
  EXPECT_TRUE(dir_.logLines("pprof " + PPROF + " --text --lines ").empty())
      << "the second view is skipped after a failure";
}

/** @test An analyzer that prints no report is said to, instead of leaving empty headers. */
TEST_F(GperfCheckTest, EmptyReportIsSaidSo) {
  const std::string PPROF = dir_.install("fake_pprof.sh", "pprof");
  auto plan = std::make_shared<GperfPlan>();
  plan->modes.cpu = true;
  plan->analyze = true;
  plan->analyzer = PPROF;
  plan->analysisReady = true;
  vernier::bench::PerfConfig cfg;
  cfg.profileTool = "gperf";
  cfg.profileAnalyze = true;
  cfg.artifactRoot = dir_.path();
  ScopedEnv mode("FAKE_PPROF_MODE", "empty");
  vernier::bench::GperfProfiler profiler(cfg, "Gperf.Empty", plan);
  profiler.beforeMeasure();
  testing::internal::CaptureStdout();
  profiler.afterMeasure(vernier::bench::Stats{});
  const std::string OUT = testing::internal::GetCapturedStdout();
  EXPECT_NE(OUT.find("(the analyzer printed no report; a very short run may hold no samples)"),
            std::string::npos)
      << OUT;
}

/** @test Through the registry, a broken analyzer makes the promised analysis an Error, alone. */
TEST_F(GperfCheckTest, BrokenAnalyzerRequestIsAnalysisError) {
  const std::string GOOGLE = dir_.install("fake_pprof.sh", "google-pprof");
  ReadinessRequest request;
  request.backend = "gperf";
  request.scope = ReadinessScope::PREFLIGHT;
  const ReadinessContext CTX = dir_.context({{"FAKE_PPROF_MODE", "fail"}});
  request.analyze = true;
  const ReadinessResult ON = ProfilerRegistry::instance().checkRequest(request, CTX);
  EXPECT_EQ(ON.report.status, EnvReport::Status::Error);
  EXPECT_EQ(ON.stage, vernier::bench::ReadinessStage::ANALYSIS);
  EXPECT_EQ(ON.cause, ReadinessCause::UNUSABLE);
  EXPECT_TRUE(ON.collectionReady());
  const auto PLAN = std::dynamic_pointer_cast<const GperfPlan>(ON.plan);
  ASSERT_NE(PLAN, nullptr);
  EXPECT_FALSE(PLAN->analysisReady);
  EXPECT_EQ(PLAN->analysisSkipped, GOOGLE + " does not run");
  request.analyze = false;
  const ReadinessResult OFF = ProfilerRegistry::instance().checkRequest(request, CTX);
  EXPECT_EQ(OFF.report.status, EnvReport::Status::Ok) << OFF.report.message;
}

/* ----------------------------- gperf analysis (every build) ----------------------------- */

namespace {

const GperfModes CPU{true, false};

/** @brief The analysis decision in @p dir's context with @p extra settings. */
GperfAnalysis analysisIn(const FakeToolDir& dir, bool analyze,
                         std::map<std::string, std::string> extra = {}) {
  return decideGperfAnalysis(CPU, analyze, dir.context(std::move(extra)));
}

} // namespace

/**
 * @test A promised analysis without an analyzer is an analysis-stage Error; without the promise
 * the same environment is fine
 *
 * Needs no gperftools, so the rule is pinned on every build, not only where
 * the gperf backend can run.
 */
TEST(GperfAnalysisTest, PromisedAnalysisWithoutAnalyzerIsAnalysisError) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const GperfAnalysis ON = analysisIn(dir, true);
  ASSERT_TRUE(ON.error.has_value()) << "a promised analysis with no analyzer must not be ready";
  EXPECT_EQ(ON.error->report.status, EnvReport::Status::Error);
  EXPECT_EQ(ON.error->stage, vernier::bench::ReadinessStage::ANALYSIS);
  EXPECT_EQ(ON.error->cause, ReadinessCause::MISSING);
  EXPECT_TRUE(ON.error->collectionReady()) << "the capture still runs";
  EXPECT_EQ(ON.error->report.message,
            "analysis: missing: --profile-analyze needs google-pprof or pprof, and neither is on "
            "PATH; the cpu capture still runs and cpu.prof is kept");
  EXPECT_TRUE(ON.analyzer.empty());

  const GperfAnalysis OFF = analysisIn(dir, false);
  EXPECT_FALSE(OFF.error.has_value()) << "without the promise a missing analyzer is only noted";
}

/** @test An analyzer that does not run makes a promised analysis an Error; it is not run otherwise.
 */
TEST(GperfAnalysisTest, BrokenAnalyzerIsAnalysisError) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string PPROF = dir.install("fake_pprof.sh", "pprof");
  const GperfAnalysis ON = analysisIn(dir, true, {{"FAKE_PPROF_MODE", "fail"}});
  ASSERT_TRUE(ON.error.has_value()) << "a broken analyzer must not be ready";
  EXPECT_EQ(ON.error->report.status, EnvReport::Status::Error);
  EXPECT_EQ(ON.error->stage, vernier::bench::ReadinessStage::ANALYSIS);
  EXPECT_EQ(ON.error->cause, ReadinessCause::UNUSABLE);
  EXPECT_TRUE(ON.error->collectionReady());
  EXPECT_EQ(ON.error->report.message,
            "analysis: unusable: --profile-analyze would run " + PPROF +
                ", which does not run: --help exit status 1: fake pprof: cannot read profile; the "
                "cpu capture still runs and cpu.prof is kept");
  EXPECT_EQ(ON.analyzer, PPROF);

  const std::size_t RUNS = dir.logLines("pprof ").size();
  const GperfAnalysis OFF = analysisIn(dir, false, {{"FAKE_PPROF_MODE", "fail"}});
  EXPECT_FALSE(OFF.error.has_value());
  EXPECT_EQ(OFF.analyzer, PPROF) << "the analyzer is still named";
  EXPECT_EQ(dir.logLines("pprof ").size(), RUNS) << "without the promise it is not run";
}

/** @test Either spelling is found, google-pprof first, and only the selected one is probed. */
TEST(GperfAnalysisTest, EitherSpellingOnlyTheSelectedRuns) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string PPROF = dir.install("fake_pprof.sh", "pprof");
  const GperfAnalysis ONLY_PPROF = analysisIn(dir, true);
  EXPECT_FALSE(ONLY_PPROF.error.has_value());
  EXPECT_EQ(ONLY_PPROF.analyzer, PPROF);
  EXPECT_EQ(dir.logLines("pprof " + PPROF + " --help").size(), 1U) << dir.log();

  const std::string GOOGLE = dir.install("fake_pprof.sh", "google-pprof");
  const GperfAnalysis BOTH = analysisIn(dir, true);
  EXPECT_FALSE(BOTH.error.has_value());
  EXPECT_EQ(BOTH.analyzer, GOOGLE);
  EXPECT_EQ(dir.logLines("pprof " + GOOGLE + " --help").size(), 1U) << dir.log();
  EXPECT_EQ(dir.logLines("pprof " + PPROF + " --help").size(), 1U) << "pprof was probed again";
}

/** @test A request without a CPU capture has nothing to analyze and needs no analyzer. */
TEST(GperfAnalysisTest, NoCpuCaptureNeedsNoAnalyzer) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const GperfAnalysis HEAP = decideGperfAnalysis(GperfModes{false, true}, true, dir.context());
  EXPECT_FALSE(HEAP.error.has_value());
}
