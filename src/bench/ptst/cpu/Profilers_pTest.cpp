/**
 * @file Profilers_pTest.cpp
 * @brief Profiler integration samples: each backend attached by the profiler
 *        guard and checked by what it wrote for the measurement
 *
 * HookAttachment shows that UB_PERF_GUARD attaches the profiler hooks whether
 * or not --profile names a profiler. PerfIntegration, GperfIntegration,
 * BpftraceIntegration and CallgrindIntegration run only under their own
 * --profile and build their case with the same guard, so the request attaches
 * that backend to the measurement. When the backend's readiness decision for
 * the request says it cannot collect here, the case skips with the decision's
 * words. Otherwise it measures, then reads what the backend wrote for this
 * process into the case's capture folder:
 *  - perf: stat.txt, perf stat's report on this process, ended normally
 *  - gperf: cpu.prof, a gperftools CPU profile with its end written
 *  - bpftrace: each selected script's report, whose capture window this
 *    process armed and disarmed
 *  - callgrind: nothing, in-process. valgrind writes the profile when the
 *    process exits, so the case runs only under valgrind and the profile is
 *    read after the run.
 * The CSV row of a profiled case names the backend and that folder
 * (profileTool, profileDir). RAPLIntegration builds a PerfCase without the
 * hooks, so --profile attaches no profiler to it.
 *
 * Usage:
 *   @code{.sh}
 *   # Without --profile only HookAttachment measures; the other cases skip
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Profilers.*"
 *
 *   # One backend, its capture folders under ./profiles
 *   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Profilers.GperfIntegration" --profile gperf --artifact-root profiles
 *
 *   # callgrind records a process only when valgrind runs it
 *   valgrind --tool=callgrind --instr-atstart=no \
 *       --callgrind-out-file=profiles/Profilers.CallgrindIntegration.callgrind/callgrind.out \
 *       ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST \
 *       --gtest_filter="Profilers.CallgrindIntegration" --profile callgrind \
 *       --artifact-root profiles
 *   @endcode
 *
 * perf needs access to this process's counters and bpftrace needs root or
 * BENCH_SUDO; `--profile-check --profile <name>` says what this machine allows.
 *
 * @see makePerfCaseWithProfiler
 * @see attachProfilerHooks
 * @see ProfilerRegistry::checkRequest
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/ProfilerBpftrace.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;
namespace fs = std::filesystem;

namespace {

/** @brief Get current configuration */
inline const ub::PerfConfig& config() { return ub::detail::getPerfConfig(); }

/**
 * @brief The readiness decision for this run's --profile request.
 *
 * The registry decides a request the same way whoever asks, so this is the
 * decision the guard's profiler was built from, asked again.
 */
ub::ReadinessResult profileDecision() {
  const ub::ReadinessContext CTX = ub::ReadinessContext::capture();
  return ub::ProfilerRegistry::instance().checkRequest(
      ub::readinessRequestFor(config(), ub::ReadinessScope::RUNTIME, CTX), CTX);
}

/**
 * @brief The capture folder of @p perf's case under @p suffix ("perf",
 * "gperf", ...): the rule the backends write by, resolveArtifactDir().
 */
fs::path captureFolder(const ub::PerfCase& perf, std::string_view suffix) {
  return ub::profiler_env::resolveArtifactDir(config().profileTool, config().artifactRoot,
                                              perf.testName(), suffix);
}

/** @brief The whole file at @p path; empty when it is missing or cannot be read. */
std::string readFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

/**
 * @brief True when perf stat's report @p report has a line for @p event,
 * named alone ("instructions") or with its PMU ("cpu_core/instructions/"),
 * with or without a modifier (":u" where perf counts user space only). A
 * line's comment, after '#', is not read.
 */
bool reportsEvent(const std::string& report, std::string_view event) {
  const std::string WITH_PMU = "/" + std::string(event) + "/";
  std::istringstream lines(report);
  for (std::string line; std::getline(lines, line);) {
    std::istringstream words(line.substr(0, line.find('#')));
    for (std::string word; words >> word;) {
      if (word.substr(0, word.find(':')) == event || word.find(WITH_PMU) != std::string::npos) {
        return true;
      }
    }
  }
  return false;
}

/** @brief What GperfIntegration reads of a gperftools CPU profile. */
struct CpuProfile {
  bool header = false; ///< The header words 0, 3, 0, the period, 0.
  bool end = false;    ///< The words 0, 1, 0 that end the samples.
  std::uintptr_t periodUs = 0;
  std::uintptr_t samples = 0;
};

/**
 * @brief Read @p bytes as gperftools' CPU profile: pointer-sized words, a
 * header, sample records (count, depth, that many addresses) and the record
 * that ends them, which ProfilerStop() writes before the process's mappings.
 */
CpuProfile readCpuProfile(const std::string& bytes) {
  CpuProfile profile;
  std::vector<std::uintptr_t> words(bytes.size() / sizeof(std::uintptr_t));
  if (words.size() < 5) {
    return profile;
  }
  std::memcpy(words.data(), bytes.data(), words.size() * sizeof(std::uintptr_t));
  if (words[0] != 0 || words[1] != 3 || words[2] != 0 || words[3] == 0 || words[4] != 0) {
    return profile;
  }
  profile.header = true;
  profile.periodUs = words[3];
  for (std::size_t i = 5; i + 2 < words.size() && words[i + 1] <= words.size();
       i += 2 + words[i + 1]) {
    if (words[i] == 0 && words[i + 1] == 1 && words[i + 2] == 0) {
      profile.end = true;
      break;
    }
    profile.samples += words[i];
  }
  return profile;
}

} // anonymous namespace

/**
 * @brief Profiler hook auto-attachment validation
 *
 * Validates that UB_PERF_GUARD auto-attaches profiler hooks without error
 * regardless of whether profiler is enabled.
 *
 * @test HookAttachment
 *
 * Validates:
 *  - UB_PERF_GUARD auto-attaches profiler hooks
 *  - No crash when profiler not enabled
 *  - Test executes normally after hook attachment
 *
 * Expected performance:
 *  - Always passes (hooks should be safe to attach)
 */
PERF_TEST(Profilers, HookAttachment) {
  UB_PERF_GUARD(perf); // Auto-attaches profiler hooks

  // This should not crash even if no profiler enabled

  const std::size_t DATA_SIZE = 1024;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "hook_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should execute normally with hooks attached";

  (void)sink;
}

/**
 * @brief perf attached by the guard, checked by its report on this process
 *
 * Runs under --profile perf, in perf's default stat mode. The guard attaches
 * perf to the case: perf stat counts this process during the measured
 * repeats and writes its report when afterMeasure() stops it.
 *
 * @test PerfIntegration
 *
 * Validates:
 *  - stat.txt in the case's capture folder is perf stat's report on this
 *    process (its process id), ended by the elapsed-time line
 *  - the report has a line for each event perf stat was asked to count
 *
 * Skips without --profile perf, under another perf mode, and, with the
 * readiness decision's words, where perf cannot count this process.
 */
PERF_TEST(Profilers, PerfIntegration) {
  if (config().profileTool != "perf") {
    GTEST_SKIP() << "Test requires --profile perf";
  }

  // The guard first: it builds perf's profiler from the request, and a
  // request that cannot collect is reported by the run whatever happens next.
  UB_PERF_GUARD(perf);
  const ub::ReadinessResult DECISION = profileDecision();
  if (!DECISION.collectionReady()) {
    GTEST_SKIP() << "perf cannot collect here: " << DECISION.report.message;
  }
  // The mode the decision chose.
  const auto PLAN = std::dynamic_pointer_cast<const ub::PerfPlan>(DECISION.plan);
  ASSERT_NE(PLAN, nullptr) << "perf's decision names no mode";
  if (PLAN->mode != ub::PerfMode::STAT) {
    GTEST_SKIP() << "This case reads perf stat's report; --profile-args '" << config().profileArgs
                 << "' selects another perf mode";
  }

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "perf_test");

  // What perf wrote for this measurement. The process id ties the report to
  // this run, and the elapsed-time line ends a report perf stat completed.
  const fs::path STAT = captureFolder(perf, "perf") / "stat.txt";
  const std::string REPORT = readFile(STAT);
  ASSERT_FALSE(REPORT.empty()) << "perf left no report at " << STAT;
  const std::string THIS_PROCESS =
      "Performance counter stats for process id '" + std::to_string(::getpid()) + "'";
  EXPECT_NE(REPORT.find(THIS_PROCESS), std::string::npos)
      << STAT << " is not perf stat's report on this process:\n"
      << REPORT;
  EXPECT_NE(REPORT.find("seconds time elapsed"), std::string::npos)
      << "perf stat did not end its report in " << STAT << ":\n"
      << REPORT;
  std::istringstream events(ub::PERF_STAT_EVENTS);
  for (std::string event; std::getline(events, event, ',');) {
    EXPECT_TRUE(reportsEvent(REPORT, event)) << STAT << " has no line for " << event;
  }

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with perf profiling";

  EXPECT_LT(result.stats.cv, 0.35)
      << "Profiling should not excessively impact measurement stability";

  (void)sink;
}

/**
 * @brief gperftools attached by the guard, checked by its CPU profile
 *
 * Runs under --profile gperf with CPU profiling (the default). The guard
 * attaches gperftools to the case: ProfilerStart() before the measured
 * repeats and ProfilerStop() after them write cpu.prof.
 *
 * @test GperfIntegration
 *
 * Validates:
 *  - cpu.prof in the case's capture folder, written by this run (an earlier
 *    one is removed first), is a gperftools CPU profile with its end written
 *
 * Skips without --profile gperf, when --profile-args asks for the heap
 * profile alone, and, with the readiness decision's words, where gperftools
 * cannot profile.
 */
PERF_TEST(Profilers, GperfIntegration) {
  if (config().profileTool != "gperf") {
    GTEST_SKIP() << "Test requires --profile gperf";
  }

  // The guard first: it builds gperftools' profiler from the request, and a
  // request that cannot collect is reported by the run whatever happens next.
  UB_PERF_GUARD(perf);
  const ub::ReadinessResult DECISION = profileDecision();
  if (!DECISION.collectionReady()) {
    GTEST_SKIP() << "gperf cannot collect here: " << DECISION.report.message;
  }
  // The modes the decision chose.
  const auto PLAN = std::dynamic_pointer_cast<const ub::GperfPlan>(DECISION.plan);
  ASSERT_NE(PLAN, nullptr) << "gperf's decision names no mode";
  if (!PLAN->modes.cpu) {
    GTEST_SKIP() << "This case reads the CPU profile; --profile-args '" << config().profileArgs
                 << "' asks for the heap profile alone";
  }
  // Only a profile this run writes can satisfy the check below.
  const fs::path PROFILE = captureFolder(perf, "gperf") / "cpu.prof";
  std::error_code removeError;
  fs::remove(PROFILE, removeError);

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "gperf_test");

  // What gperftools wrote for this measurement. A short measurement may hold
  // no sample, so the samples are reported, not required.
  const CpuProfile CPU = readCpuProfile(readFile(PROFILE));
  ASSERT_TRUE(CPU.header) << PROFILE << " is missing or is not a gperftools CPU profile";
  EXPECT_TRUE(CPU.end) << "the samples in " << PROFILE << " have no end: ProfilerStop() did not "
                       << "complete the profile";
  std::printf("[gperf] %s: %ju samples, one every %ju us\n", PROFILE.c_str(),
              static_cast<std::uintmax_t>(CPU.samples), static_cast<std::uintmax_t>(CPU.periodUs));

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with gperf profiling";

  EXPECT_LT(result.stats.cv, 0.35)
      << "Profiling should not excessively impact measurement stability";

  (void)sink;
}

/**
 * @brief bpftrace attached by the guard, checked by its scripts' reports
 *
 * Runs under --profile bpftrace with the scripts the request selects. The
 * guard attaches bpftrace to the case: one tracer per script, each with the
 * capture window the backend adds, which the tracer arms before the measured
 * repeats and disarms after them, printing a line each time.
 *
 * @test BpftraceIntegration
 *
 * Validates:
 *  - each selected script's report in the case's capture folder holds the
 *    capture window armed and disarmed by this process (its process id)
 *
 * The scripts' own maps are not required: this workload writes nothing for
 * the default scripts to count. Skips without --profile bpftrace and, with
 * the readiness decision's words, where bpftrace cannot trace this process.
 */
PERF_TEST(Profilers, BpftraceIntegration) {
  if (config().profileTool != "bpftrace") {
    GTEST_SKIP() << "Test requires --profile bpftrace";
  }

  // The guard first: it builds bpftrace's profiler from the request, and a
  // request that cannot collect is reported by the run whatever happens next.
  UB_PERF_GUARD(perf);
  const ub::ReadinessResult DECISION = profileDecision();
  if (!DECISION.collectionReady()) {
    GTEST_SKIP() << "bpftrace cannot collect here: " << DECISION.report.message;
  }
  // The scripts and the report format the decision chose.
  const auto PLAN = std::dynamic_pointer_cast<const ub::BpftracePlan>(DECISION.plan);
  ASSERT_TRUE(PLAN && !PLAN->scriptPaths.empty()) << "bpftrace's decision selects no script";

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "bpftrace_test");

  // What each tracer wrote for this measurement. bpftrace writes into
  // PERF_BPF_OUT when the run gives no --artifact-root.
  const fs::path FOLDER = (config().artifactRoot.empty() && !PLAN->outputDir.empty())
                              ? fs::path{PLAN->outputDir}
                              : captureFolder(perf, "bpf");
  for (const std::string& script : PLAN->scriptPaths) {
    const fs::path REPORT = FOLDER / (fs::path{script}.stem().string() + ".out." + PLAN->format);
    bool armed = false;
    bool disarmed = false;
    std::istringstream lines(readFile(REPORT));
    for (std::string line; std::getline(lines, line);) {
      const auto ARM = ub::bpftrace_tool::parseArmAck(line, "bpftrace");
      const auto DISARM = ub::bpftrace_tool::parseDisarmAck(line, "bpftrace");
      armed = armed || (ARM && ARM->pid == ::getpid());
      disarmed = disarmed || (DISARM && DISARM->pid == ::getpid());
    }
    EXPECT_TRUE(armed) << REPORT << " has no capture window armed by this process";
    EXPECT_TRUE(disarmed) << REPORT << " has no capture window disarmed by this process";
  }

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with bpftrace profiling";

  EXPECT_LT(result.stats.cv, 0.35)
      << "Profiling should not excessively impact measurement stability";

  (void)sink;
}

/**
 * @brief RAPL profiler integration smoke test
 *
 * Validates that RAPL (power measurement) profiler integration works when enabled.
 * Test skips if RAPL profiler not enabled.
 *
 * @test RAPLIntegration
 *
 * Validates:
 *  - RAPL profiler executes without crash
 *  - Test completes successfully
 *  - Measurements are valid
 *
 * Expected performance:
 *  - Pass when --profile rapl provided
 *  - Skip otherwise
 */
PERF_TEST(Profilers, RAPLIntegration) {
  if (config().profileTool != "rapl") {
    GTEST_SKIP() << "Test requires --profile rapl";
  }

  ub::PerfCase perf{"Profilers.RAPLIntegration", config()};

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "rapl_test");

  EXPECT_GT(result.callsPerSecond, 100.0) << "Test should complete with RAPL profiling";

  EXPECT_LT(result.stats.cv, 0.35) << "Power measurement should not excessively impact stability";

  (void)sink;
}

/**
 * @brief callgrind attached by the guard, in a process valgrind runs
 *
 * Runs under --profile callgrind while valgrind's callgrind runs the process.
 * The guard attaches the callgrind backend to the case: under a wrap started
 * by hand with --instr-atstart=no it switches instrumentation on for the
 * measured repeats and off after them; under bench run's wrap the recording
 * covers the whole process.
 *
 * @test CallgrindIntegration
 *
 * valgrind writes the profile when the process exits, after every case has
 * ended, so no check here can read it: the profile is read after the run,
 * where the backend says it goes. Under valgrind, execution is 20-50x slower;
 * the thresholds below are relaxed accordingly. Skips without --profile
 * callgrind, where valgrind is not running this process, and, with the
 * readiness decision's words, where callgrind cannot record.
 */
PERF_TEST(Profilers, CallgrindIntegration) {
  if (config().profileTool != "callgrind") {
    GTEST_SKIP() << "Test requires --profile callgrind";
  }

  // The guard first: it builds the callgrind backend from the request, and a
  // request that cannot collect is reported by the run whatever happens next.
  UB_PERF_GUARD(perf);
  const ub::ReadinessResult DECISION = profileDecision();
  if (!DECISION.collectionReady()) {
    GTEST_SKIP() << "callgrind cannot collect here: " << DECISION.report.message;
  }
  if (!ub::profiler_env::isRunningUnderValgrind()) {
    GTEST_SKIP() << "callgrind records only a process valgrind runs, and valgrind is not running "
                    "this one (the backend's notice above gives the command that does)";
  }

  const std::size_t DATA_SIZE = 4096;
  auto data = test::makeTestData(DATA_SIZE);

  perf.warmup([&] {
    volatile auto result = test::sumBytes(data.data(), data.size());
    (void)result;
  });

  volatile std::uint64_t sink = 0;
  auto result = perf.throughputLoop([&] { sink = sink + test::sumBytes(data.data(), data.size()); },
                                    "callgrind_test");

  // Callgrind runs under valgrind which adds 20-50x overhead; use relaxed threshold
  EXPECT_GT(result.callsPerSecond, 10.0) << "Test should complete with callgrind profiling";

  EXPECT_LT(result.stats.cv, 0.50)
      << "Valgrind simulation should produce reasonably stable measurements";

  (void)sink;
}