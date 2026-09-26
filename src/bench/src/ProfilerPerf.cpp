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
#include <array>
#include <csignal>
#include <filesystem>
#include <thread>
#include <chrono>
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

void PerfStatProfiler::beforeMeasure() {
#ifdef __linux__
  if (!plan_) {
    return;
  }

  // The executable the check ran, by its absolute path, as one shell word
  // whatever it contains. --profile-args stays shell text on purpose.
  const std::string PERF = shellQuote(plan_->perf);
  pid_t targetPid = ::getpid();

  if (plan_->mode == PerfMode::MEM) {
    // perf mem -- memory-access profiling. Captures load/store latency
    // distribution; useful for finding L1/L2/LLC stalls.
    dataPath_ = artifactDir_ + "/perf.mem.data";
    errPath_ = artifactDir_ + "/mem.err.txt";
    std::string cmd =
        PERF + " mem record -p " + std::to_string(targetPid) + " -o " + shellQuote(dataPath_);
    launchBackground(cmd, /*stdoutPath*/ "", errPath_);
  } else if (plan_->mode == PerfMode::C2C) {
    // perf c2c -- cache-line contention profiling. Surfaces false sharing
    // by attributing HITM events to the source line of the contended write.
    dataPath_ = artifactDir_ + "/perf.c2c.data";
    errPath_ = artifactDir_ + "/c2c.err.txt";
    std::string cmd =
        PERF + " c2c record -p " + std::to_string(targetPid) + " -o " + shellQuote(dataPath_);
    launchBackground(cmd, /*stdoutPath*/ "", errPath_);
  } else if (plan_->mode == PerfMode::RECORD) {
    // perf record mode
    dataPath_ = artifactDir_ + "/perf.data";
    errPath_ = artifactDir_ + "/record.err.txt";
    // Build: perf record <args> -p PID
    std::string cmd = PERF + " record ";
    if (!cfg_.profileArgs.empty()) {
      // strip leading "record"
      auto i = cfg_.profileArgs.find_first_not_of(" \t", 6);
      auto rest = (i == std::string::npos) ? std::string{} : cfg_.profileArgs.substr(i);
      cmd += rest + " ";
    }
    cmd += "-p " + std::to_string(targetPid) + " -o " + shellQuote(dataPath_);
    launchBackground(cmd, /*stdoutPath*/ "", errPath_);
  } else {
    // perf stat mode (default)
    statPath_ = artifactDir_ + "/stat.txt";
    std::string cmd = PERF + " stat -e " + PERF_STAT_EVENTS + " -p " + std::to_string(targetPid);
    if (!cfg_.profileArgs.empty()) {
      cmd += " " + cfg_.profileArgs;
    }
    // perf stat writes to stderr -> redirect to file
    launchBackground(cmd, /*stdoutPath*/ "", statPath_);
  }

  // Grace period to ensure perf attaches properly before measurement starts
  // Critical for short benchmarks where measurement might start before perf is ready
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
#endif
}

void PerfStatProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (childPid_ <= 0) {
    return;
  }

  // ============================================================================
  // Proper perf termination with data flush
  // ============================================================================

  // Step 1: Send SIGINT to allow perf to finalize data gracefully
  if (::kill(childPid_, 0) == 0) { // Check if process exists
    ::kill(childPid_, SIGINT);
  } else {
    childPid_ = -1;
    return; // Process already exited
  }

  // Step 2: Give perf initial time to start shutdown (CRITICAL for perf record)
  // Perf needs time to stop sampling, write buffers, and finalize the data file header
  // This is the most critical step - perf.data header is written during shutdown
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Step 3: Wait for perf to finish writing data (with timeout)
  constexpr int TIMEOUT_MS = 5000; // 5 second total timeout
  constexpr int POLL_INTERVAL_MS = 100;
  int elapsed = 1000; // Already waited 1000ms
  int status = 0;

  while (elapsed < TIMEOUT_MS) {
    pid_t result = ::waitpid(childPid_, &status, WNOHANG);

    if (result == childPid_) {
      // Process exited - give filesystem time to flush buffers
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      childPid_ = -1;

      // Verify perf.data was written (for record mode)
      if (!dataPath_.empty()) {
        if (std::filesystem::exists(dataPath_)) {
          auto fileSize = std::filesystem::file_size(dataPath_);
          if (fileSize > 0) {
            // Success: perf.data written and has data
            return;
          } else {
            std::fprintf(stderr,
                         "Warning: perf.data exists but is empty (size=%zu) - perf may not have "
                         "flushed data\n",
                         fileSize);
          }
        } else {
          std::fprintf(stderr,
                       "Warning: perf.data not found - perf may have terminated abnormally\n");
        }
      }
      return;
    } else if (result == -1) {
      // Error in waitpid (process may have been reaped)
      childPid_ = -1;
      return;
    }

    // Process still running, wait a bit more
    std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    elapsed += POLL_INTERVAL_MS;
  }

  // Step 3: Timeout reached - force kill
  std::fprintf(stderr, "Warning: perf did not exit after %dms - forcing termination\n", TIMEOUT_MS);

  if (::kill(childPid_, 0) == 0) {
    ::kill(childPid_, SIGKILL);
    ::waitpid(childPid_, &status, 0); // Block until killed
  }

  childPid_ = -1;
#endif
}

void PerfStatProfiler::launchBackground(const std::string& cmdCore, const std::string& stdoutPath,
                                        const std::string& stderrPath) {
#ifdef __linux__
  // popen() runs "<cmd> >STDOUT 2>STDERR & echo $!" with /bin/sh itself, so
  // the launch needs no sh on PATH, and $! is perf's own pid.
  std::string cmd = cmdCore;
  std::string redirs;
  if (!stdoutPath.empty()) {
    redirs += " >" + shellQuote(stdoutPath);
  }
  if (!stderrPath.empty()) {
    redirs += " 2>" + shellQuote(stderrPath);
  }
  std::string shellCmd = cmd + redirs + " & echo $!";

  FILE* pipe = ::popen(shellCmd.c_str(), "r");
  if (!pipe) {
    return;
  }

  std::array<char, 64> buf{};
  if (::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    childPid_ = static_cast<pid_t>(std::strtol(buf.data(), nullptr, 10));
  }
  ::pclose(pipe);
#else
  (void)cmdCore;
  (void)stdoutPath;
  (void)stderrPath;
#endif
}

bool PerfStatProfiler::killChild(int sig) noexcept {
#ifdef __linux__
  if (childPid_ <= 0) {
    return false;
  }
  if (::kill(childPid_, 0) != 0) {
    return false;
  }
  ::kill(childPid_, sig);
  return true;
#else
  (void)sig;
  return false;
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
