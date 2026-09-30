/**
 * @file ProfilerPerf.cpp
 * @brief Implementation of Linux perf profiler backend.
 */

#include "src/bench/inc/ProfilerPerf.hpp"

#include "src/bench/inc/ProfilerRegistry.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <csignal>
#include <filesystem>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace vernier {
namespace bench {

/* ----------------------------- Readiness ----------------------------- */

namespace {

bool startsWithTrim(const std::string& s, const char* prefix) {
  std::size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
    ++i;
  }
  const std::size_t PLEN = std::strlen(prefix);
  return (s.size() >= i + PLEN && s.compare(i, PLEN, prefix) == 0);
}

const char* modeName(PerfMode mode) {
  switch (mode) {
  case PerfMode::RECORD:
    return "record";
  case PerfMode::MEM:
    return "mem";
  case PerfMode::C2C:
    return "c2c";
  case PerfMode::STAT:
    break;
  }
  return "stat";
}

bool containsAny(const std::string& text, std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (text.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

/** @brief The first line of @p text containing @p needle, else its last lines. */
std::string lineWith(const std::string& text, const char* needle) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(needle) != std::string::npos) {
      return outputTail(line, 1);
    }
  }
  return outputTail(text);
}

/** @brief Events `perf stat -x,` marked `<not supported>`, comma-separated. */
std::string unsupportedEvents(const std::string& csv) {
  std::istringstream in(csv);
  std::string line;
  std::string events;
  while (std::getline(in, line)) {
    if (line.rfind("<not supported>", 0) != 0) {
      continue;
    }
    // <value>,<unit>,<event>,...
    const std::size_t FIRST = line.find(',');
    const std::size_t SECOND = FIRST == std::string::npos ? FIRST : line.find(',', FIRST + 1);
    if (SECOND == std::string::npos) {
      continue;
    }
    const std::size_t THIRD = line.find(',', SECOND + 1);
    events += (events.empty() ? "" : ", ") + line.substr(SECOND + 1, THIRD - SECOND - 1);
  }
  return events;
}

#ifdef __linux__
/** @brief True when this process holds CAP_PERFMON or CAP_SYS_ADMIN. */
bool hasCounterCapability() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("CapEff:", 0) == 0) {
      const std::uint64_t CAPS = std::strtoull(line.c_str() + 7, nullptr, 16);
      constexpr std::uint64_t SYS_ADMIN = 1ULL << 21;
      constexpr std::uint64_t PERFMON = 1ULL << 38;
      return (CAPS & (SYS_ADMIN | PERFMON)) != 0;
    }
  }
  return false;
}

/** @brief "; kernel.perf_event_paranoid=N limits ..." when it restricts this user, else "". */
std::string paranoidNote(const ReadinessContext& ctx) {
  if (ctx.euid() == 0 || hasCounterCapability()) {
    return {};
  }
  std::ifstream in("/proc/sys/kernel/perf_event_paranoid");
  int paranoid = -1;
  if (!(in >> paranoid) || paranoid < 2) {
    return {};
  }
  return "; kernel.perf_event_paranoid=" + std::to_string(paranoid) +
         " limits this user to user-space events";
}
#endif

} // namespace

PerfMode parsePerfMode(const std::string& profileArgs) {
  if (startsWithTrim(profileArgs, "record")) {
    return PerfMode::RECORD;
  }
  if (startsWithTrim(profileArgs, "mem")) {
    return PerfMode::MEM;
  }
  if (startsWithTrim(profileArgs, "c2c")) {
    return PerfMode::C2C;
  }
  return PerfMode::STAT;
}

ReadinessResult checkPerfRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
#ifdef __linux__
  auto plan = std::make_shared<PerfPlan>();
  plan->mode = parsePerfMode(request.profileArgs);
  const auto TOOL = resolveExecutable("perf", ctx);
  if (!TOOL) {
    return readinessResult(ReadinessCause::MISSING, "perf not found on PATH",
                           "Install linux-tools-$(uname -r), the perf that matches the running "
                           "kernel.");
  }
  if (!TOOL->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, TOOL->path + " is not an executable file",
                           "Reinstall linux-tools, or fix PATH so it finds a working perf.");
  }
  plan->perf = TOOL->path;

  // The executable runs: a perf wrapper without the kernel-matched build
  // fails here and is never launched.
  const ProbeResult VERSION = runBoundedProbe({plan->perf, "--version"}, 5000, ctx);
  if (!VERSION.succeeded()) {
    const std::string TAIL = outputTail(VERSION.output, 3);
    return readinessResult(ReadinessCause::UNUSABLE,
                           plan->perf + " --version: " + VERSION.describe() +
                               (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Install linux-tools matching the running kernel: apt install "
                           "linux-tools-$(uname -r).");
  }

  // Effective access, not the sysctl: count this process for 100 ms as this
  // user. Root, CAP_PERFMON and container policy all show up here.
  const std::string SELF = std::to_string(static_cast<long>(ctx.self()));
  const ProbeResult ACCESS = runBoundedProbe(
      {plan->perf, "stat", "-x,", "-e", PERF_STAT_EVENTS, "-p", SELF, "--timeout", "100"}, 5000,
      ctx);
  if (!ACCESS.succeeded()) {
    if (containsAny(ACCESS.output, {"Access to performance monitoring", "Permission denied",
                                    "Operation not permitted", "No permission"})) {
      return readinessResult(
          ReadinessCause::DENIED,
          "perf stat cannot open the counters as this user: " +
              lineWith(ACCESS.output, "Access to performance monitoring"),
          "Grant CAP_PERFMON to the benchmark, lower kernel.perf_event_paranoid (sudo sysctl -w "
          "kernel.perf_event_paranoid=2), or run as root; vernier does not elevate perf.");
    }
    const std::string TAIL = outputTail(ACCESS.output);
    return readinessResult(ReadinessCause::UNUSABLE,
                           "perf stat -p " + SELF + ": " + ACCESS.describe() +
                               (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Run the same perf stat by hand to see why.");
  }

  const std::string UNSUPPORTED = unsupportedEvents(ACCESS.output);
  const std::string NOTE = paranoidNote(ctx);
  ReadinessResult result;
  if (plan->mode != PerfMode::STAT) {
    result =
        readinessResult(ReadinessCause::UNVERIFIED,
                        std::string{"perf stat counts this process; perf "} + modeName(plan->mode) +
                            " itself is not probed before the run" + NOTE,
                        "");
  } else if (!UNSUPPORTED.empty()) {
    result = readinessResult(ReadinessCause::CAVEAT,
                             "perf stat counts this process, but " + UNSUPPORTED +
                                 " <not supported> here; those columns stay empty" + NOTE,
                             "");
  } else {
    result = readinessResult(ReadinessCause::READY,
                             std::string{"perf stat counted "} + PERF_STAT_EVENTS +
                                 " on this process (probe with " + plan->perf + ")" + NOTE,
                             "");
  }
  result.plan = std::move(plan);
  return result;
#else
  (void)request;
  (void)ctx;
  return readinessResult(ReadinessCause::UNSUPPORTED, "perf is Linux-only",
                         "Run on Linux or use a different profiler.");
#endif
}

/* ----------------------------- PerfStatProfiler ----------------------------- */

namespace {

#ifdef __linux__
/** @brief @p text as one POSIX sh word: single-quoted, each ' written as '\''. */
std::string shellQuote(const std::string& text) {
  std::string out = "'";
  for (const char CH : text) {
    if (CH == '\'') {
      out += "'\\''";
    } else {
      out += CH;
    }
  }
  out += "'";
  return out;
}
#endif

std::shared_ptr<const PerfPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const PerfPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "perf";
  return checkPerfRequest(request, CTX);
}

} // namespace

PerfStatProfiler::PerfStatProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  const ReadinessResult DECISION = decideNow(cfg_);
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    // A rejected request leaves no folder behind.
    std::fprintf(stderr, "[perf] not started: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[perf] %s\n", DECISION.report.hint.c_str());
    }
    return;
  }
#ifdef __linux__
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "perf");
#endif
}

PerfStatProfiler::PerfStatProfiler(const PerfConfig& cfg, std::string testName,
                                   std::shared_ptr<const PerfPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
#ifdef __linux__
  if (plan_) {
    artifactDir_ =
        profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "perf");
  }
#endif
}

namespace {

#ifdef __linux__
/** @brief How a process ended, from its wait status. */
std::string waitStatusText(int status) {
  if (WIFEXITED(status)) {
    return "exit status " + std::to_string(WEXITSTATUS(status));
  }
  if (WIFSIGNALED(status)) {
    return "signal " + std::to_string(WTERMSIG(status));
  }
  return "an unknown end";
}

/** @brief The whole text of @p path, or "" when it cannot be read. */
std::string readText(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

/** @brief The end of perf's own output in @p path, for a report. */
std::string perfSaid(const std::string& path) {
  const std::string TAIL = outputTail(readText(path));
  return TAIL.empty() ? std::string{"perf wrote nothing to "} + path : "perf: " + TAIL;
}
#endif

} // namespace

void PerfStatProfiler::fail(ReadinessCause cause, const std::string& detail,
                            ReadinessStage stage) const {
  ProfilerRegistry::instance().reportFailure("perf", testName_,
                                             readinessResult(cause, detail, "", stage));
}

void PerfStatProfiler::beforeMeasure() {
#ifdef __linux__
  if (!plan_) {
    return;
  }

  // The executable the check ran, by its absolute path, as one shell word
  // whatever it contains. --profile-args stays shell text on purpose.
  const std::string PERF = shellQuote(plan_->perf);
  pid_t targetPid = ::getpid();
  std::string cmd;
  std::string errPath;

  if (plan_->mode == PerfMode::MEM) {
    // perf mem -- memory-access profiling. Captures load/store latency
    // distribution; useful for finding L1/L2/LLC stalls.
    dataPath_ = artifactDir_ + "/perf.mem.data";
    errPath_ = artifactDir_ + "/mem.err.txt";
    cmd = PERF + " mem record -p " + std::to_string(targetPid) + " -o " + shellQuote(dataPath_);
    errPath = errPath_;
  } else if (plan_->mode == PerfMode::C2C) {
    // perf c2c -- cache-line contention profiling. Surfaces false sharing
    // by attributing HITM events to the source line of the contended write.
    dataPath_ = artifactDir_ + "/perf.c2c.data";
    errPath_ = artifactDir_ + "/c2c.err.txt";
    cmd = PERF + " c2c record -p " + std::to_string(targetPid) + " -o " + shellQuote(dataPath_);
    errPath = errPath_;
  } else if (plan_->mode == PerfMode::RECORD) {
    // perf record mode
    dataPath_ = artifactDir_ + "/perf.data";
    errPath_ = artifactDir_ + "/record.err.txt";
    // Build: perf record <args> -p PID
    cmd = PERF + " record ";
    if (!cfg_.profileArgs.empty()) {
      // strip leading "record"
      auto i = cfg_.profileArgs.find_first_not_of(" \t", 6);
      auto rest = (i == std::string::npos) ? std::string{} : cfg_.profileArgs.substr(i);
      cmd += rest + " ";
    }
    cmd += "-p " + std::to_string(targetPid) + " -o " + shellQuote(dataPath_);
    errPath = errPath_;
  } else {
    // perf stat mode (default); perf stat writes its counts to stderr.
    statPath_ = artifactDir_ + "/stat.txt";
    cmd = PERF + " stat -e " + PERF_STAT_EVENTS + " -p " + std::to_string(targetPid);
    if (!cfg_.profileArgs.empty()) {
      cmd += " " + cfg_.profileArgs;
    }
    errPath = statPath_;
  }

  // `exec` makes perf this process's own child, so the stop can wait for it.
  // The start grace covers perf attaching before the measured phase.
  HelperStopPolicy policy;
  policy.interruptWaitMs = PERF_WRITE_WAIT_MS;
  helper_ = OwnedHelper(policy);
  const HelperStart START =
      helper_.start({"/bin/sh", "-c", "exec " + cmd}, "", errPath, PERF_START_GRACE_MS);
  if (!START.started) {
    fail(ReadinessCause::UNUSABLE, "perf could not be started: " + START.errorTail,
         ReadinessStage::COLLECTION);
    return;
  }
  if (START.exitedEarly) {
    fail(ReadinessCause::UNUSABLE,
         "perf ended (" + waitStatusText(START.waitStatus) +
             ") before the measured phase: " + perfSaid(errPath),
         ReadinessStage::COLLECTION);
    return;
  }
  started_ = true;
#endif
}

void PerfStatProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (!started_) {
    return;
  }
  started_ = false;
  const std::string OUTPUT = statPath_.empty() ? errPath_ : statPath_;
  const HelperStopResult STOP = helper_.stop();
  if (!STOP.wasRunning) {
    fail(ReadinessCause::UNUSABLE,
         "perf ended (" + waitStatusText(STOP.waitStatus) +
             ") during the measured phase, so its output covers part of it at most: " +
             perfSaid(OUTPUT),
         ReadinessStage::COMPLETION);
    return;
  }
  if (STOP.stillAlive) {
    fail(ReadinessCause::UNUSABLE,
         "perf did not end after SIGINT, SIGTERM and SIGKILL; its output is not final",
         ReadinessStage::COMPLETION);
    return;
  }
  if (STOP.stoppedBy != SIGINT) {
    fail(ReadinessCause::UNUSABLE,
         std::string{"perf did not finish writing within "} +
             std::to_string(PERF_WRITE_WAIT_MS / 1000) + " s of SIGINT and was stopped by " +
             (STOP.stoppedBy == SIGTERM ? "SIGTERM" : "SIGKILL") +
             ", so its output may be incomplete",
         ReadinessStage::COMPLETION);
    return;
  }
  checkOutput();
#endif
}

void PerfStatProfiler::checkOutput() const {
#ifdef __linux__
  if (!statPath_.empty()) {
    // perf stat prints this header before its counts; an error message in
    // its place is not a count (a zero or <not supported> count is one).
    if (readText(statPath_).find("Performance counter stats") == std::string::npos) {
      fail(ReadinessCause::UNUSABLE, statPath_ + " holds no counts: " + perfSaid(statPath_),
           ReadinessStage::COMPLETION);
    }
    return;
  }
  std::error_code ec;
  const auto SIZE = std::filesystem::file_size(dataPath_, ec);
  if (ec || SIZE == 0) {
    fail(ReadinessCause::MISSING,
         dataPath_ + (ec ? " was not written: " : " is empty: ") + perfSaid(errPath_),
         ReadinessStage::COMPLETION);
    return;
  }
  // perf record, mem and c2c confirm the finished file on their stderr.
  if (readText(errPath_).find("Captured and wrote") == std::string::npos) {
    fail(ReadinessCause::UNUSABLE,
         dataPath_ + " was not confirmed written (no \"Captured and wrote\" in " + errPath_ +
             "): " + perfSaid(errPath_),
         ReadinessStage::COMPLETION);
  }
#endif
}

// Factory implementation
std::unique_ptr<Profiler> makePerfProfiler(const PerfConfig& cfg, const std::string& testName) {
  auto plan = readyPlan(decideNow(cfg));
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<PerfStatProfiler>(cfg, testName, std::move(plan));
}

namespace {

std::unique_ptr<Profiler> makePlannedPerfProfiler(const PerfConfig& cfg,
                                                  const std::string& testName,
                                                  const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<PerfStatProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("perf", ::vernier::bench::checkPerfRequest,
                                   ::vernier::bench::makePlannedPerfProfiler,
                                   "Install linux-tools-$(uname -r) or run outside Docker.")
