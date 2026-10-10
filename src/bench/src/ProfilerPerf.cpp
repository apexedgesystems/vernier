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
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <poll.h>
#include <sys/stat.h>
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

/** @brief How the wait for perf's answer on its control fifo ended. */
enum class AckWait : std::uint8_t { ACKED, ENDED, TIMED_OUT };

/**
 * @brief A private pair of fifos for perf's `--control`: perf reads commands
 * from the first and answers "ack" on the second, from its main loop once its
 * counters are on. Both are opened here read-write and non-blocking, so no
 * open waits for perf, and the "ping" written at open waits in the fifo until
 * perf reads it. The pair is removed on close(); perf keeps its own
 * descriptors.
 */
class ControlFifos {
public:
  ControlFifos() = default;
  ~ControlFifos() { close(); }
  ControlFifos(const ControlFifos&) = delete;
  ControlFifos& operator=(const ControlFifos&) = delete;
  ControlFifos(ControlFifos&&) = delete;
  ControlFifos& operator=(ControlFifos&&) = delete;

  /** @brief Make the pair and write "ping"; "" on success, else why not. */
  std::string openWithPing() {
    std::error_code ec;
    const std::filesystem::path BASE = std::filesystem::temp_directory_path(ec);
    if (ec) {
      return "no temporary directory: " + ec.message();
    }
    std::string pattern = (BASE / "vernier_perf_XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
      return "cannot create a folder in " + BASE.string() + ": " + std::strerror(errno);
    }
    dir_ = pattern;
    for (const char* name : {"/ctl", "/ack"}) {
      if (::mkfifo((dir_ + name).c_str(), 0600) != 0) {
        return "cannot create " + dir_ + name + ": " + std::strerror(errno);
      }
    }
    ctl_ = ::open((dir_ + "/ctl").c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    ack_ = ::open((dir_ + "/ack").c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (ctl_ < 0 || ack_ < 0) {
      return "cannot open the fifos in " + dir_ + ": " + std::strerror(errno);
    }
    constexpr char PING[] = "ping\n";
    if (::write(ctl_, PING, sizeof(PING) - 1) != static_cast<ssize_t>(sizeof(PING) - 1)) {
      return "cannot write to " + dir_ + "/ctl: " + std::strerror(errno);
    }
    return {};
  }

  /** @brief perf's `--control` value for the pair. */
  [[nodiscard]] std::string option() const { return "fifo:" + dir_ + "/ctl," + dir_ + "/ack"; }

  /** @brief True once perf's "ack" is in the answer fifo; reads without waiting. */
  bool acked() {
    std::array<char, 64> buf{};
    ssize_t n = 0;
    while ((n = ::read(ack_, buf.data(), buf.size())) > 0) {
      answer_.append(buf.data(), static_cast<std::size_t>(n));
    }
    return answer_.find("ack\n") != std::string::npos;
  }

  /**
   * @brief Wait up to @p ms for perf's "ack", asking @p alive between polls;
   * an answer perf gave before it ended still counts.
   */
  AckWait awaitAck(int ms, const std::function<bool()>& alive) {
    using Clock = std::chrono::steady_clock;
    const auto DEADLINE = Clock::now() + std::chrono::milliseconds(ms);
    while (!acked()) {
      if (!alive()) {
        return acked() ? AckWait::ACKED : AckWait::ENDED;
      }
      const auto LEFT =
          std::chrono::duration_cast<std::chrono::milliseconds>(DEADLINE - Clock::now()).count();
      if (LEFT <= 0) {
        return AckWait::TIMED_OUT;
      }
      pollfd answer{ack_, POLLIN, 0};
      (void)::poll(&answer, 1, static_cast<int>(std::min<std::int64_t>(LEFT, 20)));
    }
    return AckWait::ACKED;
  }

  /** @brief Close and remove the pair; perf's own descriptors stay open. */
  void close() {
    for (int* fd : {&ctl_, &ack_}) {
      if (*fd >= 0) {
        ::close(*fd);
        *fd = -1;
      }
    }
    if (!dir_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(dir_, ec);
      dir_.clear();
    }
  }

private:
  std::string dir_;
  std::string answer_;
  int ctl_ = -1;
  int ack_ = -1;
};
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
  // user. Root, CAP_PERFMON and container policy all show up here. The probe
  // also finds whether this perf answers a ping on a --control fifo, which
  // the launch waits for before the measured phase.
  const std::string SELF = std::to_string(static_cast<long>(ctx.self()));
  const std::vector<std::string> PROBE = {plan->perf, "stat", "-x,",       "-e", PERF_STAT_EVENTS,
                                          "-p",       SELF,   "--timeout", "100"};
  ControlFifos control;
  const std::string FIFO_PROBLEM = control.openWithPing();
  std::string noAnswer; // Why the launch cannot wait for perf's answer; "" when it can.
  ProbeResult ACCESS;
  if (FIFO_PROBLEM.empty()) {
    std::vector<std::string> withControl = PROBE;
    withControl.insert(withControl.end(), {"--control", control.option()});
    ACCESS = runBoundedProbe(withControl, 5000, ctx);
    if (!ACCESS.succeeded() &&
        ACCESS.output.find("unknown option `control'") != std::string::npos) {
      noAnswer = plan->perf + " does not take --control";
      ACCESS = runBoundedProbe(PROBE, 5000, ctx);
    } else if (ACCESS.succeeded() && !control.acked()) {
      noAnswer = plan->perf + " did not answer a ping on its --control fifo";
    }
  } else {
    noAnswer = "no --control fifo could be made (" + FIFO_PROBLEM + ")";
    ACCESS = runBoundedProbe(PROBE, 5000, ctx);
  }
  control.close();
  plan->answersPing = noAnswer.empty();
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
  const std::string FIXED_START = "the measured phase starts after a fixed " +
                                  std::to_string(PerfStatProfiler::PERF_START_GRACE_MS) + " ms";
  ReadinessResult result;
  if (plan->mode != PerfMode::STAT) {
    std::string message = std::string{"perf stat counts this process; perf "} +
                          modeName(plan->mode) + " itself is not probed before the run";
    if (plan->mode == PerfMode::MEM) {
      message += ", and " + FIXED_START;
    } else if (!noAnswer.empty()) {
      message += "; " + noAnswer + ", so " + FIXED_START;
    }
    result = readinessResult(ReadinessCause::UNVERIFIED, message + NOTE, "");
  } else if (!UNSUPPORTED.empty() || !noAnswer.empty()) {
    std::string caveats;
    if (!UNSUPPORTED.empty()) {
      caveats = UNSUPPORTED + " <not supported> here; those columns stay empty";
    }
    if (!noAnswer.empty()) {
      caveats += (caveats.empty() ? "" : "; and ") + noAnswer + ", so " + FIXED_START +
                 " instead of once perf is counting";
    }
    result = readinessResult(ReadinessCause::CAVEAT,
                             "perf stat counts this process, but " + caveats + NOTE, "");
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
using detail::shellQuote;
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

/**
 * @brief True when @p status is how perf ends once SIGINT has made it write
 * its report: by SIGINT itself, which perf raises again after writing, or
 * with exit status 130 (128 + SIGINT) or 0.
 */
bool endedAsAfterSigint(int status) {
  if (WIFSIGNALED(status)) {
    return WTERMSIG(status) == SIGINT;
  }
  return WIFEXITED(status) && (WEXITSTATUS(status) == 0 || WEXITSTATUS(status) == 128 + SIGINT);
}

/** @brief What perf stat's report holds. */
struct StatReport {
  int counted = 0;                      ///< Events reported with a count.
  std::vector<std::string> unavailable; ///< "<event> <not supported>" or "<not counted>".
  bool error = false;                   ///< A line of the report begins with "Error".
};

/** @brief True when @p word is a count as perf stat prints one: digits, commas and a point. */
bool isCount(const std::string& word) {
  return word.find_first_of("0123456789") != std::string::npos &&
         word.find_first_not_of("0123456789,.") == std::string::npos;
}

/**
 * @brief Read perf stat's report: after its "Performance counter stats"
 * heading, a line is an event's count ("66,055  cpu-cycles:u", a unit such as
 * "msec" between them), or its mark when perf could not count it
 * ("<not supported>", "<not counted>"). The heading, the elapsed, user and
 * system times, notes and comments after '#' are none of these. A line
 * anywhere that begins with "Error" is perf's error.
 */
StatReport readStatReport(const std::string& text) {
  StatReport report;
  bool heading = false;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream words(line.substr(0, line.find('#')));
    std::vector<std::string> tokens;
    for (std::string word; words >> word;) {
      tokens.push_back(word);
    }
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0].rfind("Error", 0) == 0) {
      report.error = true;
      continue;
    }
    if (line.find("Performance counter stats") != std::string::npos) {
      heading = true;
      continue;
    }
    std::string mark;
    std::size_t first = 1;
    if (tokens.size() > 1 && tokens[0] == "<not" &&
        (tokens[1] == "supported>" || tokens[1] == "counted>")) {
      mark = tokens[0] + " " + tokens[1];
      first = 2;
    } else if (!isCount(tokens[0])) {
      continue;
    }
    if (!heading || tokens.size() <= first || tokens[first] == "seconds") {
      continue;
    }
    // The event is the last word before the comment that is not a share of
    // the time it was counted, "(50.00%)".
    std::string event;
    for (std::size_t i = tokens.size(); i > first && event.empty(); --i) {
      if (tokens[i - 1].front() != '(') {
        event = tokens[i - 1];
      }
    }
    if (event.empty()) {
      continue;
    }
    if (mark.empty()) {
      ++report.counted;
    } else {
      report.unavailable.push_back(event + " " + mark);
    }
  }
  return report;
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

  // With a perf that answers on --control, the measured phase starts once
  // perf has answered the ping, which it does after its counters are on; a
  // perf that cannot answer, and perf mem, get a fixed grace instead.
  const bool HANDSHAKE = plan_->answersPing && plan_->mode != PerfMode::MEM;
  ControlFifos control;
  if (HANDSHAKE) {
    const std::string PROBLEM = control.openWithPing();
    if (!PROBLEM.empty()) {
      fail(ReadinessCause::UNUSABLE, "perf's --control fifos could not be made: " + PROBLEM,
           ReadinessStage::COLLECTION);
      return;
    }
    cmd += " --control " + shellQuote(control.option());
  }

  // `exec` makes perf this process's own child, so the stop can wait for it.
  HelperStopPolicy policy;
  policy.interruptWaitMs = PERF_WRITE_WAIT_MS;
  helper_ = OwnedHelper(policy);
  const HelperStart START = helper_.start({"/bin/sh", "-c", "exec " + cmd}, "", errPath,
                                          HANDSHAKE ? 0 : PERF_START_GRACE_MS);
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
  if (HANDSHAKE) {
    const AckWait WAIT = control.awaitAck(PERF_ACK_WAIT_MS, [this] { return helper_.running(); });
    control.close();
    if (WAIT == AckWait::ENDED) {
      const HelperStopResult GONE = helper_.stop();
      fail(ReadinessCause::UNUSABLE,
           "perf ended (" + waitStatusText(GONE.waitStatus) +
               ") before the measured phase: " + perfSaid(errPath),
           ReadinessStage::COLLECTION);
      return;
    }
    if (WAIT == AckWait::TIMED_OUT) {
      (void)helper_.stop();
      fail(ReadinessCause::UNUSABLE,
           "perf did not answer on its --control fifo within " +
               std::to_string(PERF_ACK_WAIT_MS / 1000) +
               " s, so it was not known to be counting; it was stopped, and this case is not "
               "profiled",
           ReadinessStage::COLLECTION);
      return;
    }
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
  if (!endedAsAfterSigint(STOP.waitStatus)) {
    fail(ReadinessCause::UNUSABLE,
         "perf ended (" + waitStatusText(STOP.waitStatus) +
             ") after SIGINT, not as a perf that has written its report ends (by SIGINT, or exit "
             "status 130 or 0), so its output does not count: " +
             perfSaid(OUTPUT),
         ReadinessStage::COMPLETION);
    return;
  }
  checkOutput();
#endif
}

void PerfStatProfiler::checkOutput() const {
#ifdef __linux__
  if (!statPath_.empty()) {
    // Usable means at least one event counted (a zero count is one). A
    // heading or an error message is no count, an event marked unavailable
    // is named as such, and an error perf adds after its counts fails them.
    const StatReport REPORT = readStatReport(readText(statPath_));
    if (REPORT.counted == 0 && REPORT.unavailable.empty()) {
      fail(ReadinessCause::UNUSABLE, statPath_ + " holds no counts: " + perfSaid(statPath_),
           ReadinessStage::COMPLETION);
    } else if (REPORT.counted == 0) {
      std::string events;
      for (const std::string& EVENT : REPORT.unavailable) {
        events += (events.empty() ? "" : ", ") + EVENT;
      }
      fail(ReadinessCause::UNSUPPORTED,
           statPath_ + " holds no count: perf counted none of its events here (" + events + ")",
           ReadinessStage::COMPLETION);
    } else if (REPORT.error) {
      fail(ReadinessCause::UNUSABLE,
           statPath_ +
               " holds an error from perf after its counts, which therefore do not "
               "count: " +
               perfSaid(statPath_),
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
