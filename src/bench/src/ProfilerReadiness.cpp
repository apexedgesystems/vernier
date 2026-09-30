/**
 * @file ProfilerReadiness.cpp
 * @brief Shared readiness mechanisms: context snapshot, report formatter,
 * privilege policy, executable resolution, bounded probes, owned helpers and
 * the result memo.
 */

#include "src/bench/inc/ProfilerReadiness.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <system_error>
#include <thread>
#include <utility>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- ReadinessPlan ----------------------------- */

ReadinessPlan::~ReadinessPlan() = default;

namespace {

using Clock = std::chrono::steady_clock;

/** @brief errno as text; thread-safe, unlike strerror(). */
std::string errnoText(int err) { return std::system_category().message(err); }

/** @brief Variables that inject a tool into a process; a probe runs without them. */
bool isInjectionVariable(const std::string& name) {
  static constexpr std::string_view EXACT[] = {"LD_PRELOAD", "MALLOC_CONF", "CUDA_INJECTION64_PATH",
                                               "ROCP_TOOL_LIB", "ROCPROFILER_LIBRARY"};
  for (const std::string_view E : EXACT) {
    if (name == E) {
      return true;
    }
  }
  static constexpr std::string_view PREFIXES[] = {"HEAPTRACK_", "VERNIER_EXTERNAL_WRAP"};
  for (const std::string_view P : PREFIXES) {
    if (name.compare(0, P.size(), P) == 0) {
      return true;
    }
  }
  return false;
}

std::string lowerCopy(std::string text) {
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

/** @brief A pipe whose ends close on exec; the child dup2()s what it keeps. */
bool makePipe(int fds[2]) {
#ifdef __linux__
  return ::pipe2(fds, O_CLOEXEC) == 0;
#else
  if (::pipe(fds) != 0) {
    return false;
  }
  (void)::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  (void)::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
  return true;
#endif
}

/** @brief Null-terminated pointer array over @p strings, which must outlive it. */
std::vector<char*> cStrings(std::vector<std::string>& strings) {
  std::vector<char*> out;
  out.reserve(strings.size() + 1);
  for (std::string& s : strings) {
    out.push_back(s.data());
  }
  out.push_back(nullptr);
  return out;
}

/** @brief write() all of @p len bytes; async-signal-safe (used between fork and exec). */
void writeAll(int fd, const void* data, std::size_t len) {
  const char* p = static_cast<const char*>(data);
  while (len > 0) {
    const ssize_t N = ::write(fd, p, len);
    if (N < 0) {
      if (errno == EINTR) {
        continue;
      }
      return;
    }
    p += N;
    len -= static_cast<std::size_t>(N);
  }
}

/**
 * @brief Read the child's start report from @p fd: EOF means the exec
 * succeeded (the close-on-exec end went away); two ints mean the step that
 * failed and its errno.
 */
bool readSpawnFailure(int fd, int& step, int& err) {
  int report[2] = {0, 0};
  std::size_t got = 0;
  while (got < sizeof(report)) {
    const ssize_t N = ::read(fd, reinterpret_cast<char*>(report) + got, sizeof(report) - got);
    if (N < 0 && errno == EINTR) {
      continue;
    }
    if (N <= 0) {
      break;
    }
    got += static_cast<std::size_t>(N);
  }
  if (got != sizeof(report)) {
    return false;
  }
  step = report[0];
  err = report[1];
  return true;
}

/** @brief Blocking reap of @p pid, retrying on EINTR. */
void reapBlocking(pid_t pid, int& status) {
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
}

/** @brief Where a probe's program stands. */
enum class LeaderState : std::uint8_t {
  RUNNING, ///< Still running.
  ENDED,   ///< Ended, not yet reaped: its pid, which is its group's id, cannot be reused.
  GONE     ///< Reaped elsewhere (ECHILD): nothing of it may be signalled.
};

/** @brief The state of @p child, observed without reaping it. */
LeaderState leaderState(pid_t child) {
  siginfo_t info{};
  int rc = 0;
  do {
    info.si_pid = 0;
    rc = ::waitid(P_PID, static_cast<id_t>(child), &info, WEXITED | WNOHANG | WNOWAIT);
  } while (rc < 0 && errno == EINTR);
  if (rc < 0) {
    return LeaderState::GONE;
  }
  return info.si_pid == child ? LeaderState::ENDED : LeaderState::RUNNING;
}

#ifdef __linux__
/** @brief True while a process of group @p group runs (a zombie has ended). */
bool groupHasRunningMember(pid_t group) {
  DIR* proc = ::opendir("/proc");
  if (proc == nullptr) {
    return false;
  }
  bool running = false;
  while (!running) {
    const dirent* entry = ::readdir(proc);
    if (entry == nullptr) {
      break;
    }
    if (entry->d_name[0] < '1' || entry->d_name[0] > '9') {
      continue;
    }
    char path[sizeof("/proc//stat") + sizeof(entry->d_name)];
    std::snprintf(path, sizeof(path), "/proc/%s/stat", entry->d_name);
    const int FD = ::open(path, O_RDONLY | O_CLOEXEC);
    if (FD < 0) {
      continue;
    }
    char text[512];
    const ssize_t N = ::read(FD, text, sizeof(text) - 1);
    ::close(FD);
    if (N <= 0) {
      continue;
    }
    text[N] = '\0';
    // "<pid> (<comm>) <state> <ppid> <pgrp> ...": comm may hold spaces and ')'.
    const char* commEnd = std::strrchr(text, ')');
    char state = 0;
    int parent = 0;
    int pgrp = 0;
    if (commEnd != nullptr && std::sscanf(commEnd + 1, " %c %d %d", &state, &parent, &pgrp) == 3 &&
        pgrp == static_cast<int>(group) && state != 'Z' && state != 'X') {
      running = true;
    }
  }
  ::closedir(proc);
  return running;
}
#endif

/** @brief Wait until no process of group @p group runs, or until @p until. */
void waitForGroup(pid_t group, Clock::time_point until) {
  while (true) {
    errno = 0;
    if (::kill(-group, 0) != 0 && errno == ESRCH) {
      return; // no member left at all
    }
#ifdef __linux__
    if (!groupHasRunningMember(group)) {
      return; // only zombies, awaiting their new parent
    }
#endif
    if (Clock::now() >= until) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

/** @brief The last @p maxBytes of a file. */
std::string readFileTail(const std::string& path, std::size_t maxBytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {};
  }
  in.seekg(0, std::ios::end);
  const std::streamoff SIZE = in.tellg();
  const std::streamoff START = SIZE > static_cast<std::streamoff>(maxBytes)
                                   ? SIZE - static_cast<std::streamoff>(maxBytes)
                                   : 0;
  in.seekg(START);
  std::string out(static_cast<std::size_t>(SIZE - START), '\0');
  in.read(out.data(), static_cast<std::streamsize>(out.size()));
  out.resize(static_cast<std::size_t>(in.gcount()));
  return out;
}

const char* signalName(int sig) {
  switch (sig) {
  case SIGINT:
    return "SIGINT";
  case SIGTERM:
    return "SIGTERM";
  case SIGKILL:
    return "SIGKILL";
  default:
    return "signal";
  }
}

} // namespace

/* ----------------------------- ReadinessContext ----------------------------- */

ReadinessContext::ReadinessContext(uid_t euid, pid_t self,
                                   std::map<std::string, std::string> environment)
    : euid_(euid), self_(self), environment_(std::move(environment)) {}

ReadinessContext ReadinessContext::capture() {
  std::map<std::string, std::string> env;
  for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    const char* eq = std::strchr(*entry, '=');
    if (eq == nullptr) {
      continue;
    }
    env.emplace(std::string(*entry, static_cast<std::size_t>(eq - *entry)), std::string(eq + 1));
  }
  return ReadinessContext(::geteuid(), ::getpid(), std::move(env));
}

std::optional<std::string> ReadinessContext::get(std::string_view name) const {
  const auto IT = environment_.find(std::string(name));
  if (IT == environment_.end()) {
    return std::nullopt;
  }
  return IT->second;
}

std::string ReadinessContext::fingerprint(const std::vector<std::string>& extraKeys) const {
  std::string out = "euid=" + std::to_string(euid_) + ";";
  const auto ADD = [&](const std::string& key) {
    out += key;
    const auto VALUE = get(key);
    if (VALUE) {
      out += "=" + std::to_string(VALUE->size()) + ":" + *VALUE;
    } else {
      out += "!";
    }
    out += ";";
  };
  for (const char* KEY :
       {"PATH", "BENCH_SUDO", "VERNIER_EXTERNAL_WRAP", "VERNIER_EXTERNAL_WRAP_DIR"}) {
    ADD(KEY);
  }
  for (const std::string& key : extraKeys) {
    ADD(key);
  }
  return out;
}

std::vector<std::string> ReadinessContext::probeEnvironment() const {
  std::vector<std::string> out;
  out.reserve(environment_.size() + 1);
  for (const auto& [KEY, VALUE] : environment_) {
    if (KEY == "LC_ALL" || isInjectionVariable(KEY)) {
      continue;
    }
    out.push_back(KEY + "=" + VALUE);
  }
  out.emplace_back("LC_ALL=C");
  return out;
}

/* ----------------------------- Results ----------------------------- */

ReadinessResult readinessResult(ReadinessCause cause, std::string detail, std::string remedy,
                                ReadinessStage stage) {
  EnvReport::Status status = EnvReport::Status::Error;
  const char* prefix = "";
  switch (cause) {
  case ReadinessCause::READY:
    status = EnvReport::Status::Ok;
    break;
  case ReadinessCause::CAVEAT:
    status = EnvReport::Status::Warning;
    break;
  case ReadinessCause::UNVERIFIED:
    status = EnvReport::Status::Warning;
    prefix = "unverified: ";
    break;
  case ReadinessCause::MISSING:
    prefix = "missing: ";
    break;
  case ReadinessCause::UNUSABLE:
    prefix = "unusable: ";
    break;
  case ReadinessCause::UNSUPPORTED:
    prefix = "unsupported: ";
    break;
  case ReadinessCause::DENIED:
    prefix = "denied: ";
    break;
  case ReadinessCause::CONFIGURATION:
    prefix = "configuration: ";
    break;
  case ReadinessCause::MISSING_HELPER:
    prefix = "missing helper: ";
    break;
  case ReadinessCause::INTERNAL:
    prefix = "internal: ";
    break;
  }
  ReadinessResult result;
  result.cause = cause;
  result.stage = stage;
  std::string message = prefix + std::move(detail);
  if (stage == ReadinessStage::ANALYSIS) {
    message = "analysis: " + message;
  }
  result.report = EnvReport{status, std::move(message), std::move(remedy)};
  return result;
}

ReadinessRequest readinessRequestFor(const PerfConfig& cfg, ReadinessScope scope,
                                     const ReadinessContext& ctx) {
  ReadinessRequest request;
  request.backend = ProfilerRegistry::canonicalName(cfg.profileTool);
  request.profileArgs = cfg.profileArgs;
  request.bpfScripts = cfg.bpfScripts;
  request.analyze = cfg.profileAnalyze;
  request.scope = scope;
  const auto WRAP = ctx.get("VERNIER_EXTERNAL_WRAP");
  const bool WRAPPED =
      WRAP && !request.backend.empty() && ProfilerRegistry::canonicalName(*WRAP) == request.backend;
  request.launch = WRAPPED ? LaunchContext::RUNNER_WRAPPED : LaunchContext::IN_PROCESS;
  return request;
}

/* ----------------------------- Privilege Policy ----------------------------- */

EnvBool parseEnvBool(const std::optional<std::string>& raw) {
  if (!raw) {
    return EnvBool::ABSENT;
  }
  const std::string VALUE = lowerCopy(*raw);
  if (VALUE.empty() || VALUE == "0" || VALUE == "false" || VALUE == "no" || VALUE == "off") {
    return EnvBool::FALSE_VALUE;
  }
  if (VALUE == "1" || VALUE == "true" || VALUE == "yes" || VALUE == "on") {
    return EnvBool::TRUE_VALUE;
  }
  return EnvBool::INVALID;
}

PrivilegeDecision decidePrivilege(const ReadinessContext& ctx, std::string_view legacyAlias) {
  static const std::string INVALID_REMEDY =
      "Use 1, true, yes or on to run the probe tool with sudo -n; 0, false, no, off or an empty "
      "value to run it as the current user.";
  const auto SELECTS = [](bool optIn) {
    return optIn ? std::string{"the probe tool runs with sudo -n"}
                 : std::string{"the probe tool runs as the current user"};
  };

  PrivilegeDecision decision;
  const std::optional<std::string> BENCH_RAW = ctx.get("BENCH_SUDO");
  const EnvBool BENCH = parseEnvBool(BENCH_RAW);
  const std::string ALIAS{legacyAlias};
  const std::optional<std::string> LEGACY_RAW = ALIAS.empty() ? std::nullopt : ctx.get(legacyAlias);
  const EnvBool LEGACY = parseEnvBool(LEGACY_RAW);

  if (BENCH == EnvBool::INVALID) {
    decision.route = PrivilegeRoute::CONFIG_ERROR;
    decision.error = "BENCH_SUDO='" + *BENCH_RAW + "' is not a boolean";
    decision.remedy = INVALID_REMEDY;
    return decision;
  }
  if (BENCH == EnvBool::ABSENT) {
    if (LEGACY == EnvBool::INVALID) {
      decision.route = PrivilegeRoute::CONFIG_ERROR;
      decision.error = ALIAS + "='" + *LEGACY_RAW + "' is not a boolean";
      decision.remedy = INVALID_REMEDY;
      return decision;
    }
    if (LEGACY == EnvBool::ABSENT) {
      decision.source = "no opt-in (BENCH_SUDO unset)";
    } else {
      decision.optIn = LEGACY == EnvBool::TRUE_VALUE;
      decision.source = ALIAS + "=" + *LEGACY_RAW + " (deprecated alias)";
      decision.warning = ALIAS + " is deprecated; set BENCH_SUDO instead. " + ALIAS + "=" +
                         *LEGACY_RAW + " selects: " + SELECTS(decision.optIn) + ".";
    }
  } else {
    decision.optIn = BENCH == EnvBool::TRUE_VALUE;
    decision.source = "BENCH_SUDO=" + *BENCH_RAW;
    if (LEGACY == EnvBool::TRUE_VALUE || LEGACY == EnvBool::FALSE_VALUE) {
      if ((LEGACY == EnvBool::TRUE_VALUE) != decision.optIn) {
        decision.warning = "BENCH_SUDO=" + *BENCH_RAW + " and " + ALIAS + "=" + *LEGACY_RAW +
                           " disagree; BENCH_SUDO wins (" + SELECTS(decision.optIn) + "). " +
                           ALIAS + " is deprecated: remove it.";
      }
    } else if (LEGACY == EnvBool::INVALID) {
      decision.warning = ALIAS + "='" + *LEGACY_RAW +
                         "' is not a boolean and is ignored; BENCH_SUDO=" + *BENCH_RAW + " wins.";
    }
  }
  if (ctx.euid() == 0) {
    decision.route = PrivilegeRoute::ALREADY_ROOT;
  } else {
    decision.route = decision.optIn ? PrivilegeRoute::SCOPED_SUDO : PrivilegeRoute::CURRENT_USER;
  }
  return decision;
}

/* ----------------------------- Executables ----------------------------- */

std::optional<ResolvedTool> resolveExecutable(std::string_view name, const ReadinessContext& ctx) {
  const std::string NAME{name};
  if (NAME.empty()) {
    return std::nullopt;
  }
  // 0: absent; 1: present but not an executable regular file; 2: executable.
  const auto CLASSIFY = [](const std::string& candidate) {
    struct stat st{};
    if (::stat(candidate.c_str(), &st) != 0) {
      return 0;
    }
    return (S_ISREG(st.st_mode) && ::access(candidate.c_str(), X_OK) == 0) ? 2 : 1;
  };
  const auto ABSOLUTE = [](const std::string& path) {
    std::error_code ec;
    const std::filesystem::path ABS = std::filesystem::absolute(path, ec);
    return ec ? path : ABS.lexically_normal().string();
  };

  if (NAME.find('/') != std::string::npos) {
    const int KIND = CLASSIFY(NAME);
    if (KIND == 0) {
      return std::nullopt;
    }
    return ResolvedTool{NAME, ABSOLUTE(NAME), KIND == 2};
  }

  // An unset PATH searches what execvp() would: /bin:/usr/bin.
  const std::string SEARCH = ctx.get("PATH").value_or("/bin:/usr/bin");
  std::optional<ResolvedTool> notExecutable;
  std::size_t start = 0;
  while (start <= SEARCH.size()) {
    std::size_t end = SEARCH.find(':', start);
    if (end == std::string::npos) {
      end = SEARCH.size();
    }
    std::string dir = SEARCH.substr(start, end - start);
    if (dir.empty()) {
      dir = "."; // POSIX: an empty PATH element is the current directory
    }
    const std::string CANDIDATE = dir + "/" + NAME;
    const int KIND = CLASSIFY(CANDIDATE);
    if (KIND == 2) {
      return ResolvedTool{NAME, ABSOLUTE(CANDIDATE), true};
    }
    if (KIND == 1 && !notExecutable) {
      notExecutable = ResolvedTool{NAME, ABSOLUTE(CANDIDATE), false};
    }
    start = end + 1;
  }
  return notExecutable;
}

/* ----------------------------- Bounded Probes ----------------------------- */

std::string ProbeResult::describe() const {
  if (!started) {
    return "could not be started: " + errnoText(spawnErrno);
  }
  if (timedOut) {
    return "did not finish within its bound and was killed";
  }
  if (exited) {
    return "exit status " + std::to_string(exitCode);
  }
  if (termSignal != 0) {
    return "ended by signal " + std::to_string(termSignal);
  }
  return "ended in an unknown state";
}

ProbeResult runBoundedProbe(const std::vector<std::string>& argvIn, int timeoutMs,
                            const ReadinessContext& ctx, ProbeStreams streams) {
  ProbeResult result;
  if (argvIn.empty() || argvIn.front().empty()) {
    result.spawnErrno = EINVAL;
    return result;
  }
  // Everything the child needs is built before fork(): after it, in a
  // threaded process, the child may only make async-signal-safe calls.
  std::vector<std::string> argvStore = argvIn;
  std::vector<std::string> envStore = ctx.probeEnvironment();
  const std::vector<char*> ARGV = cStrings(argvStore);
  const std::vector<char*> ENVP = cStrings(envStore);
  const bool SEPARATE = streams == ProbeStreams::SEPARATE;

  int output[2] = {-1, -1};
  int errors[2] = {-1, -1};
  int report[2] = {-1, -1};
  const auto CLOSE_ALL = [&] {
    for (int* fds : {output, errors, report}) {
      for (int i = 0; i < 2; ++i) {
        if (fds[i] >= 0) {
          ::close(fds[i]);
          fds[i] = -1;
        }
      }
    }
  };
  if (!makePipe(output) || (SEPARATE && !makePipe(errors)) || !makePipe(report)) {
    result.spawnErrno = errno;
    CLOSE_ALL();
    return result;
  }
  const int DEV_NULL = ::open("/dev/null", O_RDONLY | O_CLOEXEC);

  const pid_t CHILD = ::fork();
  if (CHILD < 0) {
    result.spawnErrno = errno;
    CLOSE_ALL();
    if (DEV_NULL >= 0) {
      ::close(DEV_NULL);
    }
    return result;
  }
  if (CHILD == 0) {
    (void)::setpgid(0, 0);
    if (DEV_NULL >= 0) {
      (void)::dup2(DEV_NULL, STDIN_FILENO);
    }
    (void)::dup2(output[1], STDOUT_FILENO);
    (void)::dup2(SEPARATE ? errors[1] : output[1], STDERR_FILENO);
    ::execve(ARGV[0], ARGV.data(), ENVP.data());
    const int FAILURE[2] = {3, errno};
    writeAll(report[1], FAILURE, sizeof(FAILURE));
    ::_exit(127);
  }
  (void)::setpgid(CHILD, CHILD); // both sides set it; whichever runs first wins the race
  ::close(output[1]);
  output[1] = -1;
  if (SEPARATE) {
    ::close(errors[1]);
    errors[1] = -1;
  }
  ::close(report[1]);
  report[1] = -1;
  if (DEV_NULL >= 0) {
    ::close(DEV_NULL);
  }

  int step = 0;
  int execErrno = 0;
  const bool SPAWN_FAILED = readSpawnFailure(report[0], step, execErrno);
  if (SPAWN_FAILED) {
    int status = 0;
    reapBlocking(CHILD, status);
    CLOSE_ALL();
    result.spawnErrno = execErrno;
    return result;
  }
  result.started = true;

  struct Stream {
    int fd;
    std::string* text;
    bool eof;
  };
  std::vector<Stream> open{{output[0], &result.output, false}};
  if (SEPARATE) {
    open.push_back({errors[0], &result.errorOutput, false});
  }
  char buf[4096];
  // Reads whatever is ready on the open streams, waiting at most waitMs.
  const auto PUMP = [&](int waitMs) {
    std::vector<pollfd> fds;
    for (const Stream& s : open) {
      fds.push_back(pollfd{s.eof ? -1 : s.fd, POLLIN, 0});
    }
    if (::poll(fds.data(), fds.size(), waitMs) <= 0) {
      return false;
    }
    bool gotData = false;
    for (std::size_t i = 0; i < fds.size(); ++i) {
      if (open[i].eof || fds[i].revents == 0) {
        continue;
      }
      const ssize_t N = ::read(open[i].fd, buf, sizeof(buf));
      if (N > 0) {
        std::string& text = *open[i].text;
        const std::size_t ROOM = PROBE_OUTPUT_LIMIT - std::min(PROBE_OUTPUT_LIMIT, text.size());
        text.append(buf, std::min(static_cast<std::size_t>(N), ROOM));
        gotData = true;
      } else if (N == 0 || (errno != EINTR && errno != EAGAIN)) {
        open[i].eof = true;
      }
    }
    return gotData;
  };
  const auto ALL_EOF = [&] {
    return std::all_of(open.begin(), open.end(), [](const Stream& s) { return s.eof; });
  };

  // Run until the program ends or the bound passes. The program is observed,
  // not reaped: while it stays unreaped its pid, which is its group's id,
  // cannot be reused, so the group signal below reaches only processes this
  // probe started.
  const auto DEADLINE = Clock::now() + std::chrono::milliseconds(std::max(timeoutMs, 0));
  LeaderState leader = LeaderState::RUNNING;
  while (true) {
    if (!ALL_EOF()) {
      const auto LEFT =
          std::chrono::duration_cast<std::chrono::milliseconds>(DEADLINE - Clock::now()).count();
      (void)PUMP(static_cast<int>(std::clamp<long long>(LEFT, 0, 50)));
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    leader = leaderState(CHILD);
    if (leader != LeaderState::RUNNING) {
      break;
    }
    if (Clock::now() >= DEADLINE) {
      result.timedOut = true;
      break;
    }
  }

  // Finish the group on every path: nothing the probe started may run on
  // into the measurement it precedes.
  if (leader != LeaderState::GONE) {
    (void)::kill(-CHILD, SIGKILL);
    (void)::kill(CHILD, SIGKILL); // the program itself, had it left its group
  }

  // What the pipes still hold. Processes the signal ended close them; one
  // that left the group may keep them open, so the drain has its own bound.
  const auto DRAIN_END = Clock::now() + std::chrono::milliseconds(PROBE_DRAIN_MS);
  while (!ALL_EOF()) {
    const auto LEFT =
        std::chrono::duration_cast<std::chrono::milliseconds>(DRAIN_END - Clock::now()).count();
    if (LEFT <= 0) {
      break;
    }
    (void)PUMP(static_cast<int>(std::min<long long>(LEFT, 50)));
  }

  // Reap the program, then wait for the rest of its group; both bounded.
  int status = 0;
  bool reaped = leader == LeaderState::GONE; // ECHILD: reaped elsewhere; the status reads as 0
  const auto REAP_END = Clock::now() + std::chrono::milliseconds(PROBE_REAP_MS);
  while (!reaped) {
    const pid_t W = ::waitpid(CHILD, &status, WNOHANG);
    if (W == CHILD || (W < 0 && errno != EINTR)) {
      reaped = true;
      break;
    }
    if (Clock::now() >= REAP_END) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (reaped && leader != LeaderState::GONE) {
    waitForGroup(CHILD, REAP_END);
  }
  CLOSE_ALL();
  if (reaped && !result.timedOut) {
    if (WIFEXITED(status)) {
      result.exited = true;
      result.exitCode = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      result.termSignal = WTERMSIG(status);
    }
  }
  return result;
}

std::string outputTail(const std::string& text, std::size_t maxLines) {
  constexpr std::size_t MAX_LINE = 300;
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start < text.size()) {
    std::size_t end = text.find('\n', start);
    if (end == std::string::npos) {
      end = text.size();
    }
    std::string line = text.substr(start, end - start);
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())) != 0) {
      line.pop_back();
    }
    std::size_t lead = 0;
    while (lead < line.size() && std::isspace(static_cast<unsigned char>(line[lead])) != 0) {
      ++lead;
    }
    line.erase(0, lead);
    if (!line.empty()) {
      if (line.size() > MAX_LINE) {
        line = line.substr(0, MAX_LINE) + "...";
      }
      lines.push_back(std::move(line));
    }
    start = end + 1;
  }
  const std::size_t FIRST = lines.size() > maxLines ? lines.size() - maxLines : 0;
  std::string out;
  for (std::size_t i = FIRST; i < lines.size(); ++i) {
    if (!out.empty()) {
      out += " | ";
    }
    out += lines[i];
  }
  return out;
}

/* ----------------------------- Probe Scratch ----------------------------- */

ProbeScratch::ProbeScratch(const ReadinessContext& ctx) {
  std::string base = ctx.get("TMPDIR").value_or("");
  if (base.empty()) {
    base = "/tmp";
  }
  std::string pattern = base + "/vernier_probe_XXXXXX";
  if (::mkdtemp(pattern.data()) != nullptr) {
    path_ = pattern;
  }
}

ProbeScratch::~ProbeScratch() {
  if (!path_.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
}

std::string ProbeScratch::write(const std::string& name, const std::string& text) const {
  if (path_.empty()) {
    return {};
  }
  const std::string TARGET = path_ + "/" + name;
  std::ofstream out(TARGET, std::ios::binary | std::ios::trunc);
  out << text;
  return out ? TARGET : std::string{};
}

/* ----------------------------- Owned Helpers ----------------------------- */

bool HelperStopResult::allDelivered() const noexcept {
  return std::all_of(deliveries.begin(), deliveries.end(),
                     [](const StopDelivery& d) { return d.delivered; });
}

OwnedHelper::OwnedHelper(HelperStopPolicy policy) : policy_(std::move(policy)) {}

OwnedHelper::~OwnedHelper() {
  if (child_ > 0 && !reaped_) {
    (void)stop();
  }
}

OwnedHelper::OwnedHelper(OwnedHelper&& other) noexcept
    : policy_(std::move(other.policy_)), child_(other.child_), reaped_(other.reaped_),
      waitStatus_(other.waitStatus_) {
  other.child_ = -1;
  other.reaped_ = true;
}

OwnedHelper& OwnedHelper::operator=(OwnedHelper&& other) noexcept {
  if (this != &other) {
    if (child_ > 0 && !reaped_) {
      (void)stop();
    }
    policy_ = std::move(other.policy_);
    child_ = other.child_;
    reaped_ = other.reaped_;
    waitStatus_ = other.waitStatus_;
    other.child_ = -1;
    other.reaped_ = true;
  }
  return *this;
}

namespace {

/** @brief Reap @p child if it has ended; true once it is gone. */
bool tryReap(pid_t child, bool& reaped, int& waitStatus) {
  if (child <= 0 || reaped) {
    return true;
  }
  int status = 0;
  pid_t w = 0;
  do {
    w = ::waitpid(child, &status, WNOHANG);
  } while (w < 0 && errno == EINTR);
  if (w == child) {
    reaped = true;
    waitStatus = status;
  } else if (w < 0) {
    reaped = true; // not our child any more (ECHILD): never signal it again
  }
  return reaped;
}

/** @brief Poll for the child's end for up to @p ms. */
bool waitGone(pid_t child, bool& reaped, int& waitStatus, int ms) {
  const auto DEADLINE = Clock::now() + std::chrono::milliseconds(std::max(ms, 0));
  while (!tryReap(child, reaped, waitStatus)) {
    if (Clock::now() >= DEADLINE) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

} // namespace

bool OwnedHelper::running() { return child_ > 0 && !tryReap(child_, reaped_, waitStatus_); }

HelperStart OwnedHelper::start(const std::vector<std::string>& argvIn,
                               const std::string& stdoutPath, const std::string& stderrPath,
                               int graceMs, const ReadinessContext* probeEnv) {
  HelperStart outcome;
  if (child_ > 0 && !reaped_) {
    (void)stop();
  }
  child_ = -1;
  reaped_ = true;
  waitStatus_ = 0;
  if (argvIn.empty() || argvIn.front().empty()) {
    outcome.spawnErrno = EINVAL;
    return outcome;
  }
  std::vector<std::string> argvStore = argvIn;
  std::vector<std::string> envStore;
  if (probeEnv != nullptr) {
    envStore = probeEnv->probeEnvironment();
  }
  const std::vector<char*> ARGV = cStrings(argvStore);
  const std::vector<char*> ENVP = cStrings(envStore);
  const std::string OUT_PATH = stdoutPath.empty() ? std::string{"/dev/null"} : stdoutPath;
  const std::string ERR_PATH = stderrPath.empty() ? std::string{"/dev/null"} : stderrPath;
  const bool USE_ENVP = probeEnv != nullptr;

  int report[2] = {-1, -1};
  if (!makePipe(report)) {
    outcome.spawnErrno = errno;
    return outcome;
  }
  const pid_t CHILD = ::fork();
  if (CHILD < 0) {
    outcome.spawnErrno = errno;
    ::close(report[0]);
    ::close(report[1]);
    return outcome;
  }
  if (CHILD == 0) {
    // Capture files are opened here, by the invoking user, before any
    // elevation: they stay the user's whatever the helper runs as.
    const int IN = ::open("/dev/null", O_RDONLY);
    if (IN >= 0) {
      (void)::dup2(IN, STDIN_FILENO);
      if (IN != STDIN_FILENO) {
        ::close(IN);
      }
    }
    const int OUT = ::open(OUT_PATH.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (OUT < 0) {
      const int FAILURE[2] = {1, errno};
      writeAll(report[1], FAILURE, sizeof(FAILURE));
      ::_exit(126);
    }
    (void)::dup2(OUT, STDOUT_FILENO);
    if (OUT != STDOUT_FILENO) {
      ::close(OUT);
    }
    const int ERR = ::open(ERR_PATH.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ERR < 0) {
      const int FAILURE[2] = {2, errno};
      writeAll(report[1], FAILURE, sizeof(FAILURE));
      ::_exit(126);
    }
    (void)::dup2(ERR, STDERR_FILENO);
    if (ERR != STDERR_FILENO) {
      ::close(ERR);
    }
    if (USE_ENVP) {
      ::execve(ARGV[0], ARGV.data(), ENVP.data());
    } else {
      ::execv(ARGV[0], ARGV.data());
    }
    const int FAILURE[2] = {3, errno};
    writeAll(report[1], FAILURE, sizeof(FAILURE));
    ::_exit(127);
  }
  ::close(report[1]);
  int step = 0;
  int err = 0;
  const bool FAILED = readSpawnFailure(report[0], step, err);
  ::close(report[0]);
  if (FAILED) {
    int status = 0;
    reapBlocking(CHILD, status);
    outcome.spawnErrno = err;
    const std::string& WHAT = step == 1 ? OUT_PATH : (step == 2 ? ERR_PATH : argvIn.front());
    outcome.errorTail =
        (step == 3 ? "could not execute " : "could not open ") + WHAT + ": " + errnoText(err);
    return outcome;
  }
  child_ = CHILD;
  reaped_ = false;
  outcome.started = true;
  if (waitGone(child_, reaped_, waitStatus_, graceMs)) {
    outcome.exitedEarly = true;
    outcome.waitStatus = waitStatus_;
    if (!stderrPath.empty()) {
      outcome.errorTail = readFileTail(stderrPath, 4096);
    }
  }
  return outcome;
}

HelperStopResult OwnedHelper::stop() {
  HelperStopResult result;
  if (child_ <= 0) {
    return result;
  }
  if (tryReap(child_, reaped_, waitStatus_)) {
    result.reaped = true;
    result.waitStatus = waitStatus_;
    return result;
  }
  result.wasRunning = true;

  const struct {
    int sig;
    int waitMs;
  } STEPS[] = {{SIGINT, policy_.interruptWaitMs},
               {SIGTERM, policy_.terminateWaitMs},
               {SIGKILL, policy_.killWaitMs}};
  for (const auto& step : STEPS) {
    // Re-derived before every signal: a reaped pid may already belong to
    // another process, so it is never signalled.
    if (tryReap(child_, reaped_, waitStatus_)) {
      break;
    }
    StopDelivery delivery;
    delivery.signal = step.sig;
    delivery.target = child_;
    if (policy_.route == PrivilegeRoute::SCOPED_SUDO) {
#ifdef __linux__
      // sudo may keep a monitor process between itself and the tool: the
      // tool is then the monitor's only child.
      char path[96];
      std::snprintf(path, sizeof(path), "/proc/%d/task/%d/children", static_cast<int>(child_),
                    static_cast<int>(child_));
      std::ifstream children(path);
      std::vector<long> pids;
      long p = 0;
      while (children >> p) {
        pids.push_back(p);
      }
      if (pids.size() == 1 && pids.front() > 0) {
        delivery.target = static_cast<pid_t>(pids.front());
      }
#endif
      const std::vector<std::string> ARGV = {policy_.sudoPath,
                                             "-n",
                                             "--",
                                             policy_.killPath,
                                             "-" + std::to_string(step.sig),
                                             std::to_string(delivery.target)};
      for (const std::string& part : ARGV) {
        delivery.command += (delivery.command.empty() ? "" : " ") + part;
      }
      const ReadinessContext CTX = policy_.context ? *policy_.context : ReadinessContext::capture();
      const ProbeResult PROBE = runBoundedProbe(ARGV, 5000, CTX);
      delivery.delivered = PROBE.succeeded();
      if (!delivery.delivered) {
        const std::string TAIL = outputTail(PROBE.output);
        delivery.detail = TAIL.empty() ? PROBE.describe() : TAIL;
      }
    } else {
      delivery.command = std::string{"kill("} + std::to_string(delivery.target) + ", " +
                         signalName(step.sig) + ")";
      delivery.delivered = ::kill(delivery.target, step.sig) == 0;
      if (!delivery.delivered) {
        delivery.detail = errnoText(errno);
      }
    }
    result.deliveries.push_back(delivery);
    if (delivery.delivered && waitGone(child_, reaped_, waitStatus_, step.waitMs)) {
      result.stoppedBy = step.sig;
      break;
    }
  }
  (void)tryReap(child_, reaped_, waitStatus_);
  result.reaped = reaped_;
  result.waitStatus = waitStatus_;
  result.stillAlive = !reaped_;
  return result;
}

/* ----------------------------- Memo ----------------------------- */

struct ReadinessMemo::State {
  mutable std::mutex mutex;
  std::map<std::string, std::shared_future<ReadinessResult>> results;
  std::set<std::string> notices;
};

ReadinessMemo::ReadinessMemo() : state_(std::make_unique<State>()) {}

ReadinessMemo::~ReadinessMemo() = default;

ReadinessResult ReadinessMemo::getOrCompute(const std::string& key,
                                            const std::function<ReadinessResult()>& compute) {
  std::promise<ReadinessResult> promise;
  std::shared_future<ReadinessResult> future;
  bool owner = false;
  {
    const std::lock_guard<std::mutex> LOCK(state_->mutex);
    const auto IT = state_->results.find(key);
    if (IT != state_->results.end()) {
      future = IT->second;
    } else {
      future = promise.get_future().share();
      state_->results.emplace(key, future);
      owner = true;
    }
  }
  if (!owner) {
    return future.get(); // waits outside the lock
  }
  ReadinessResult result;
  try {
    result = compute();
  } catch (const std::exception& e) {
    result = readinessResult(ReadinessCause::INTERNAL,
                             std::string{"the readiness check failed: "} + e.what(), "");
  } catch (...) {
    result = readinessResult(ReadinessCause::INTERNAL, "the readiness check failed", "");
  }
  promise.set_value(result);
  return result;
}

bool ReadinessMemo::claimNotice(const std::string& key) {
  const std::lock_guard<std::mutex> LOCK(state_->mutex);
  return state_->notices.insert(key).second;
}

void ReadinessMemo::reset() {
  const std::lock_guard<std::mutex> LOCK(state_->mutex);
  state_->results.clear();
  state_->notices.clear();
}

std::size_t ReadinessMemo::size() const {
  const std::lock_guard<std::mutex> LOCK(state_->mutex);
  return state_->results.size();
}

} // namespace bench
} // namespace vernier
