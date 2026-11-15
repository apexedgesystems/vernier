/**
 * @file ProfilerReadinessChecks_uTest.cpp
 * @brief Unit tests for the backends' own readiness checks.
 *
 * The perf and gperf checks, each the registry's decision about one request
 * in an explicit context whose PATH holds only fake tools
 * (ReadinessFixtures.hpp), read back from the fakes' log, with the launches
 * built from their plans; and the pieces of the capture window that the
 * bpftrace and offcpu backends share. Nothing here needs privileges.
 */

#include "src/bench/inc/ProfilerBpftrace.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using vernier::bench::CHILD_STATUS_REMEDY;
using vernier::bench::decideGperfAnalysis;
using vernier::bench::describeStatusLoss;
using vernier::bench::EnvReport;
using vernier::bench::GperfAnalysis;
using vernier::bench::GperfModes;
using vernier::bench::GperfPlan;
using vernier::bench::parseGperfModes;
using vernier::bench::parsePerfMode;
using vernier::bench::PerfMode;
using vernier::bench::PerfPlan;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::StatusLoss;
using vernier::bench::test::ChildReaping;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

/* ----------------------------- bpftrace ----------------------------- */

/**
 * @test The capture window's program is bound to the process, the arm thread
 * and the stopping thread, disarms once, and counts nothing
 */
TEST(BpfWindowTest, ProgramIsBoundToTheProcessAndItsTwoThreads) {
  const std::string PROGRAM =
      vernier::bench::bpftrace_tool::captureWindowProgram("offcpu", 42, 43, 44);
  for (const char* PART : {"tracepoint:sched:sched_switch {",
                           "if (pid == 42 && (args->prev_state == 1 || args->prev_state == 2)) {",
                           "if (tid == 43 && comm == \"vernier-arm\") {",
                           "if (@vernier_window == 0) {\n        @vernier_window = 1;\n"
                           "        printf(\"offcpu armed %d %d\\n\", pid, tid);",
                           "} else if (tid == 44 && comm == \"vernier-stop\") {",
                           "if (@vernier_window == 1) {\n        @vernier_window = 2;\n"
                           "        printf(\"offcpu disarmed %d %d\\n\", pid, tid);\n"
                           "        clear(@vernier_window);"}) {
    EXPECT_NE(PROGRAM.find(PART), std::string::npos) << PART << "\n" << PROGRAM;
  }
  for (const char* ABSENT : {"@vernier_recorded", "vernier-wait", "tid != ", "++"}) {
    EXPECT_EQ(PROGRAM.find(ABSENT), std::string::npos) << ABSENT << "\n" << PROGRAM;
  }
}

/** @test Acknowledgements are read from text and JSON lines, and nothing else is one */
TEST(BpfWindowTest, AcknowledgementsReadFromTextAndJson) {
  namespace tool = vernier::bench::bpftrace_tool;
  const auto ARMED = tool::parseArmAck("bpftrace armed 12 34", "bpftrace");
  ASSERT_TRUE(ARMED.has_value());
  EXPECT_EQ(ARMED->pid, 12);
  EXPECT_EQ(ARMED->tid, 34);
  const auto JSON =
      tool::parseArmAck(R"({"type": "printf", "data": "bpftrace armed 12 34\n"})", "bpftrace");
  ASSERT_TRUE(JSON.has_value());
  EXPECT_EQ(JSON->tid, 34);
  const auto DISARMED = tool::parseDisarmAck("bpftrace disarmed 12 12", "bpftrace");
  ASSERT_TRUE(DISARMED.has_value());
  EXPECT_EQ(DISARMED->tid, 12);
  const auto COUNTED = tool::parseDisarmAck("offcpu disarmed 12 13 5", "offcpu");
  ASSERT_TRUE(COUNTED.has_value()) << "what follows the ids is a backend's own";
  EXPECT_EQ(COUNTED->tid, 13);
  EXPECT_FALSE(tool::parseArmAck("offcpu armed 12 34", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseArmAck("xbpftrace armed 12 34", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseArmAck("bpftrace armed 12", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseArmAck("bpftrace armed a 34", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseArmAck("bpftrace disarmed 12 34", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseDisarmAck("bpftrace armed 12 34", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseDisarmAck("bpftrace disarmed 12", "bpftrace").has_value());
  EXPECT_FALSE(tool::parseArmAck("bpftrace armed 1234567890123456789 1", "bpftrace").has_value())
      << "more digits than an id holds";
}

/** @test The watch reads a line once it is complete, keeps the order, and says when the file is
 * gone */
TEST(BpfWindowTest, WatchReadsLinesAsTheyComplete) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string PATH = dir.writeFile("out.text", "\nbpftrace armed 7 ");
  vernier::bench::bpftrace_tool::WindowWatch watch(PATH, "bpftrace");
  EXPECT_EQ(watch.poll(), 0);
  EXPECT_FALSE(watch.armed().has_value()) << "a line not yet complete";
  {
    std::ofstream more(PATH, std::ios::app);
    more << "8\nbpftrace disarmed 7 7\n";
  }
  EXPECT_EQ(watch.poll(), 0);
  ASSERT_TRUE(watch.armed().has_value());
  EXPECT_EQ(watch.armed()->tid, 8);
  ASSERT_TRUE(watch.disarmed().has_value());
  EXPECT_EQ(watch.disarmed()->tid, 7);
  EXPECT_FALSE(watch.disarmedBeforeArmed());
  std::filesystem::remove(PATH);
  EXPECT_EQ(watch.poll(), ENOENT);

  const std::string REVERSED =
      dir.writeFile("reversed.text", "bpftrace disarmed 7 7\nbpftrace armed 7 8\n");
  vernier::bench::bpftrace_tool::WindowWatch backwards(REVERSED, "bpftrace");
  EXPECT_EQ(backwards.poll(), 0);
  EXPECT_TRUE(backwards.armed().has_value());
  EXPECT_TRUE(backwards.disarmedBeforeArmed()) << "a stop line ahead of the arm line";
}

/** @test The window's own lines: its acknowledgements and its maps; everything else is the script's
 */
TEST(BpfWindowTest, WindowLinesAreTheAcknowledgementsAndTheReservedMaps) {
  namespace tool = vernier::bench::bpftrace_tool;
  for (const char* LINE : {"bpftrace armed 1 2", "bpftrace disarmed 1 1", "@vernier_window: 2",
                           R"({"type": "printf", "data": "bpftrace armed 1 2\n"})",
                           R"({"type": "map", "data": {"@vernier_window": 2}})"}) {
    EXPECT_TRUE(tool::isWindowLine(LINE, "bpftrace")) << LINE;
  }
  for (const char* LINE : {"@c: 1", "@c: 0", "[0]   12 |@@@@|", "offcpu armed 1 2", "",
                           R"({"type": "printf", "data": "hello\n"})", "@vernier: 3"}) {
    EXPECT_FALSE(tool::isWindowLine(LINE, "bpftrace")) << LINE;
  }
}

/** @test The reserved names are read from a script's code and strings, not from its comments */
TEST(BpfWindowTest, ReservedNamesAreReadFromCodeNotComments) {
  namespace tool = vernier::bench::bpftrace_tool;
  EXPECT_EQ(tool::reservedNameIn("k:f { @vernier_window = 1; }", "bpftrace"),
            "the map @vernier_window");
  EXPECT_EQ(tool::reservedNameIn("k:f { @vernier_2[tid] = 1; }", "bpftrace"), "the map @vernier_2");
  EXPECT_EQ(tool::reservedNameIn("k:f { printf(\"x bpftrace armed %d\\n\", 1); }", "bpftrace"),
            "the text 'bpftrace armed'");
  EXPECT_EQ(tool::reservedNameIn("k:f { printf(\"bpftrace disarmed\"); }", "bpftrace"),
            "the text 'bpftrace disarmed'");
  EXPECT_EQ(tool::reservedNameIn("k:f { printf(\"offcpu armed\"); }", "offcpu"),
            "the text 'offcpu armed'");
  EXPECT_EQ(tool::reservedNameIn("// @vernier_window\n/* \"bpftrace armed\" */\n"
                                 "k:f { @vernier = 1; printf(\"armed \\\"q\\\"\"); }",
                                 "bpftrace"),
            "");
  EXPECT_EQ(tool::reservedNameIn("k:f { printf(\"@vernier_window\"); }", "bpftrace"), "")
      << "a string naming a map is not the map";
}

/** @test Iterator probes are found by their name or alias where probes are declared */
TEST(BpfWindowTest, IteratorProbesAreFound) {
  namespace tool = vernier::bench::bpftrace_tool;
  EXPECT_EQ(tool::iteratorProbeIn("iter:task { }"), "iter:task");
  EXPECT_EQ(tool::iteratorProbeIn("BEGIN { }\n\nit:task_file /ctx->task/ { }"), "it:task_file");
  EXPECT_EQ(tool::iteratorProbeIn("iter:task_vma,iter:task { }"), "iter:task_vma");
  EXPECT_EQ(tool::iteratorProbeIn("tracepoint:sched:sched_switch { $it = 1; }"), "");
  EXPECT_EQ(tool::iteratorProbeIn("// iter:task\nk:f { printf(\"iter:task\"); }"), "");
  EXPECT_EQ(tool::iteratorProbeIn("kprobe:iter_next { }"), "") << "a function named iter_...";
}

/** @test The waits' names: a thread named for a scope gets its name back after it */
TEST(BpfWindowTest, ThreadNameScopeRestoresTheName) {
  namespace tool = vernier::bench::bpftrace_tool;
  const auto NAME = [] {
    std::ifstream comm("/proc/thread-self/comm");
    std::string name;
    std::getline(comm, name);
    return name;
  };
  const std::string BEFORE = NAME();
  {
    const tool::ThreadNameScope SCOPE(tool::WAIT_THREAD);
    EXPECT_EQ(NAME(), "vernier-wait");
  }
  EXPECT_EQ(NAME(), BEFORE);
}

/** @test The arm thread names itself before its constructor returns, and is not the main thread */
TEST(BpfWindowTest, ArmThreadNamesItself) {
  const vernier::bench::bpftrace_tool::ArmThread ARM;
  ASSERT_GT(ARM.tid(), 0);
  EXPECT_NE(ARM.tid(), static_cast<long>(::getpid()));
  std::ifstream comm("/proc/self/task/" + std::to_string(ARM.tid()) + "/comm");
  std::string name;
  std::getline(comm, name);
  EXPECT_EQ(name, "vernier-arm");
}

/**
 * @test Only the host's PID namespace, the kernel's first (inode 0xEFFFFFFC),
 * is not foreign: any other is named by its link, or by its inode when the
 * link was not read; one that could not be read at all is not called foreign.
 */
TEST(BpfWindowTest, ForeignPidNamespaceOnlyOutsideTheHosts) {
  using vernier::bench::bpftrace_tool::foreignPidNamespace;
  using vernier::bench::bpftrace_tool::PidNamespaceId;
  EXPECT_EQ(foreignPidNamespace(PidNamespaceId{true, 0xEFFFFFFCUL, "pid:[4026531836]"}), "");
  EXPECT_EQ(foreignPidNamespace(PidNamespaceId{true, 4026532284UL, "pid:[4026532284]"}),
            "pid:[4026532284]");
  EXPECT_EQ(foreignPidNamespace(PidNamespaceId{true, 4026532284UL, ""}), "pid:[4026532284]");
  EXPECT_EQ(foreignPidNamespace(PidNamespaceId{false, 0, ""}), "");
}

/** @test This process's PID namespace is read from /proc/self/ns/pid: its inode and its link */
TEST(BpfWindowTest, PidNamespaceIsReadFromProc) {
  struct stat info{};
  ASSERT_EQ(::stat("/proc/self/ns/pid", &info), 0);
  char link[64] = {};
  ASSERT_GT(::readlink("/proc/self/ns/pid", link, sizeof(link) - 1), 0);
  const vernier::bench::bpftrace_tool::PidNamespaceId NS =
      vernier::bench::bpftrace_tool::readPidNamespace();
  EXPECT_TRUE(NS.read);
  EXPECT_EQ(NS.inode, static_cast<unsigned long>(info.st_ino));
  EXPECT_EQ(NS.link, std::string(link));
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

/**
 * @test A probe whose exit status this process could not read proves
 * nothing: in a process that ignores SIGCHLD or sets SA_NOCLDWAIT, perf's
 * --version shows neither a perf that runs nor one that fails, so the check
 * is not ready, says why and how to let the status through, and never asks
 * that perf to count, whatever its --version did.
 */
TEST_F(PerfCheckTest, ProbeWithoutItsStatusProvesNothing) {
  using Policy = ChildReaping::Policy;
  for (const Policy POLICY : {Policy::IGNORED, Policy::NO_CHILD_WAIT}) {
    const StatusLoss LOSS =
        POLICY == Policy::IGNORED ? StatusLoss::SIGCHLD_IGNORED : StatusLoss::NO_CHILD_WAIT;
    for (const std::string MODE : {"ok", "broken"}) {
      ReadinessResult r;
      {
        const ChildReaping REAPING(POLICY);
        ASSERT_TRUE(REAPING.ok());
        r = check({{"FAKE_PERF_MODE", MODE}});
      }
      EXPECT_EQ(r.cause, ReadinessCause::UNUSABLE) << MODE << ": " << r.report.message;
      EXPECT_EQ(r.report.message.rfind("unusable: " + perf_ +
                                           " --version: ended without an exit status this "
                                           "process could read: " +
                                           describeStatusLoss(LOSS),
                                       0),
                0U)
          << r.report.message;
      EXPECT_EQ(r.report.hint, CHILD_STATUS_REMEDY);
      EXPECT_TRUE(dir_.logLines("perf " + perf_ + " stat").empty()) << dir_.log();
    }
  }
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
  // The checked perf answers on --control, so each launch ends with its fifos.
  EXPECT_EQ(dir_.logLines("perf " + PERF +
                          " stat -e cpu-cycles,instructions,branches,branch-misses,cache-misses "
                          "-p " +
                          PID + " --control fifo:")
                .size(),
            1U)
      << dir_.log();
  EXPECT_EQ(dir_.logLines("perf " + PERF + " record -g -p " + PID + " -o " + ROOT +
                          "/Perf.Record.perf/perf.data --control fifo:")
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

namespace {

/** @brief The modes @p args selects; fails the test when it is refused. */
GperfModes gperfModes(const std::string& args) {
  GperfModes modes;
  const auto REFUSED = parseGperfModes(args, modes);
  EXPECT_FALSE(REFUSED.has_value()) << "'" << args << "': " << REFUSED->report.message;
  return modes;
}

} // namespace

/** @test Each word, and each combination of them, selects its modes; none selects cpu. */
TEST(GperfModeTest, ParsesTheSelectedModes) {
  for (const auto& [ARGS, CPU, HEAP] : std::vector<std::tuple<std::string, bool, bool>>{
           {"", true, false},
           {"  ", true, false},
           {"cpu", true, false},
           {"heap", false, true},
           {"both", true, true},
           {"cpu,heap", true, true},
           {"heap cpu", true, true},
           {"heap,heap", false, true},
           {"both, cpu", true, true},
       }) {
    const GperfModes MODES = gperfModes(ARGS);
    EXPECT_EQ(MODES.cpu, CPU) << "'" << ARGS << "'";
    EXPECT_EQ(MODES.heap, HEAP) << "'" << ARGS << "'";
  }
}

/**
 * @test A word that is not a mode is refused whole, with the configuration
 * error every backend gives such a word, and selects nothing; a mode's name
 * inside another word is not that mode.
 */
TEST(GperfModeTest, RefusesAWordThatIsNotAMode) {
  for (const auto& [ARGS, WORD] : std::vector<std::pair<std::string, std::string>>{
           {"nonsense", "nonsense"},
           {"cpuheap", "cpuheap"},
           {"xcpu", "xcpu"},
           {"heap-ish", "heap-ish"},
           {"CPU", "CPU"},
           {"cpu,nonsense", "nonsense"},
           {"both;", "both;"},
       }) {
    GperfModes modes;
    modes.cpu = modes.heap = true;
    const auto REFUSED = parseGperfModes(ARGS, modes);
    ASSERT_TRUE(REFUSED.has_value()) << "'" << ARGS << "' was accepted";
    EXPECT_EQ(REFUSED->report.status, EnvReport::Status::Error);
    EXPECT_EQ(REFUSED->cause, ReadinessCause::CONFIGURATION);
    EXPECT_FALSE(REFUSED->collectionReady());
    EXPECT_EQ(REFUSED->report.message,
              "configuration: '" + WORD +
                  "' is not a mode of gperf; its modes are cpu, heap, both");
    EXPECT_EQ(REFUSED->report.hint, "Use one of cpu, heap, both in --profile-args, or drop it.");
    EXPECT_FALSE(modes.cpu || modes.heap) << "'" << ARGS << "' left a mode selected";
  }
}

/**
 * @test The check refuses such a word first, in every build (with gperftools
 * or without), so a run creates nothing for it and fails, as the doctor says.
 */
TEST(GperfModeTest, CheckRefusesAWordBeforeAnythingElse) {
  FakeToolDir dir;
  ReadinessRequest request;
  request.backend = "gperf";
  request.profileArgs = "nonsense";
  request.scope = ReadinessScope::RUNTIME;
  const ReadinessResult R = ProfilerRegistry::instance().checkRequest(request, dir.context());
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  EXPECT_EQ(R.cause, ReadinessCause::CONFIGURATION);
  EXPECT_FALSE(R.collectionReady());
  EXPECT_EQ(R.report.message,
            "configuration: 'nonsense' is not a mode of gperf; its modes are cpu, heap, both");
  EXPECT_EQ(R.plan, nullptr) << "a refused request carries no plan to build a profiler from";
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
  ProfilerRegistry::instance().resetFailures();
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
  EXPECT_NE(err.find("[FAIL] Profiler 'gperf' (Gperf.Fails): analysis: unusable: " + PPROF +
                     " failed on the profile: exit status 1: fake pprof: cannot read profile; "
                     "raw profile kept at " +
                     CPU_PROF + "\n   Run it by hand to see why: " + PPROF +
                     " --text --cum --lines "),
            std::string::npos)
      << err;
  const std::vector<vernier::bench::ProfileFailure> FAILED =
      ProfilerRegistry::instance().failures();
  ProfilerRegistry::instance().resetFailures();
  ASSERT_EQ(FAILED.size(), 1U) << "the failed analysis is the run's failure";
  EXPECT_EQ(FAILED[0].backend, "gperf");
  EXPECT_EQ(FAILED[0].test, "Gperf.Fails");
  EXPECT_EQ(FAILED[0].result.stage, vernier::bench::ReadinessStage::ANALYSIS);
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

/**
 * @test An analyzer whose --help exit status this process could not read is
 * not known to run: a promised analysis is an analysis-stage Error that says
 * why, and the capture still runs.
 */
TEST(GperfAnalysisTest, AnalyzerWithoutItsStatusIsNotKnownToRun) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string PPROF = dir.install("fake_pprof.sh", "pprof");
  GperfAnalysis on;
  {
    const ChildReaping REAPING(ChildReaping::Policy::IGNORED);
    ASSERT_TRUE(REAPING.ok());
    on = analysisIn(dir, true);
  }
  ASSERT_TRUE(on.error.has_value()) << "an analyzer not seen to run must not be ready";
  EXPECT_EQ(on.error->stage, vernier::bench::ReadinessStage::ANALYSIS);
  EXPECT_EQ(on.error->cause, ReadinessCause::UNUSABLE);
  EXPECT_TRUE(on.error->collectionReady()) << "the capture still runs";
  EXPECT_EQ(on.error->report.message,
            "analysis: unusable: --profile-analyze would run " + PPROF +
                ", whose --help ended without an exit status this process could read: " +
                describeStatusLoss(StatusLoss::SIGCHLD_IGNORED) +
                "; the cpu capture still runs and cpu.prof is kept");
  EXPECT_EQ(on.error->report.hint, CHILD_STATUS_REMEDY);
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
