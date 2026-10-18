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
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
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
#include <sstream>
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

/** @brief A connected pair of stream sockets whose ends close on exec. */
bool makeSocketPair(int fds[2]) {
#ifdef SOCK_CLOEXEC
  return ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0;
#else
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
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

/**
 * @brief Give the signals a helper is stopped with, SIGINT and SIGTERM, their
 * default action, unblocked, whatever this process inherited: a benchmark
 * started as a background job of a script ignores SIGINT, and a helper would
 * keep the ignore across exec. Async-signal-safe (used between fork and exec).
 */
void defaultStopSignals() {
  struct sigaction action{};
  action.sa_handler = SIG_DFL;
  (void)::sigemptyset(&action.sa_mask);
  sigset_t stopSignals;
  (void)::sigemptyset(&stopSignals);
  for (const int SIGNAL : {SIGINT, SIGTERM}) {
    (void)::sigaction(SIGNAL, &action, nullptr);
    (void)::sigaddset(&stopSignals, SIGNAL);
  }
  (void)::sigprocmask(SIG_UNBLOCK, &stopSignals, nullptr);
}

/**
 * @brief Where a forked keeper stops closing descriptors outside Linux, where
 * it has no list of them: the hard limit on open descriptors (at least 256;
 * 1 << 20, the system's default ceiling, where there is none), not the soft
 * one, which may have been lowered after a descriptor above it was opened.
 * On Linux the keeper closes a range or every listed descriptor, or reports
 * that it could not.
 */
int descriptorBound() {
  constexpr rlim_t CEILING = rlim_t{1} << 20;
  struct rlimit limit{};
  if (::getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_max != RLIM_INFINITY) {
    return static_cast<int>(std::clamp<rlim_t>(limit.rlim_max, 256, INT_MAX));
  }
  return static_cast<int>(CEILING);
}

/**
 * @brief 0 when a keeper forked now may close every descriptor it inherits:
 * close_range works here (asked over an empty range, which closes nothing),
 * or /proc/self/fd can be opened to be listed. Otherwise the errno the
 * probe's setup fails with before anything starts: EMFILE or ENFILE when no
 * descriptor was left to open the list with, ENOSYS when the system offers
 * neither. Only an early refusal: the keeper's own report decides whether it
 * did (readKeeperReport). Not called between fork and exec.
 */
int keeperIsolation() {
#ifdef __linux__
#ifdef SYS_close_range
  if (::syscall(SYS_close_range, ~0U, ~0U, 0U) == 0) {
    return 0;
  }
#endif
  const int LIST = ::open("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (LIST >= 0) {
    ::close(LIST);
    return 0;
  }
  return errno == EMFILE || errno == ENFILE ? errno : ENOSYS;
#else
  return 0;
#endif
}

#ifdef __linux__
/**
 * @brief Close every descriptor from @p lowest up that /proc/self/fd lists,
 * whatever its number. 0 once none is left; otherwise the errno of what
 * failed: the list could not be opened or read, or a listed descriptor
 * would not close. The list is read with getdents64, which allocates
 * nothing, and again from its start after a read that closed any.
 * Async-signal-safe.
 */
int closeListedDescriptors(int lowest) {
  const int LIST = ::open("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (LIST < 0) {
    return errno;
  }
  // A record: inode (8 bytes), offset (8), its length (2), type (1), name.
  constexpr std::size_t LENGTH_AT = 16;
  constexpr std::size_t NAME_AT = 19;
  alignas(8) char records[4096];
  int failure = 0;
  while (failure == 0) {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 30))
    const ssize_t N = ::getdents64(LIST, records, sizeof(records));
#else
    const ssize_t N = ::syscall(SYS_getdents64, LIST, records, sizeof(records));
#endif
    if (N < 0 && errno == EINTR) {
      continue;
    }
    if (N <= 0) {
      failure = N == 0 ? 0 : errno;
      break;
    }
    bool closedAny = false;
    for (std::size_t at = 0; failure == 0 && at + NAME_AT < static_cast<std::size_t>(N);) {
      unsigned short length = 0;
      std::memcpy(&length, records + at + LENGTH_AT, sizeof(length));
      if (length == 0) {
        break;
      }
      int fd = 0;
      bool number = records[at + NAME_AT] != '\0';
      for (const char* c = records + at + NAME_AT; number && *c != '\0'; ++c) {
        number = *c >= '0' && *c <= '9' && fd <= (INT_MAX - 9) / 10;
        if (number) {
          fd = fd * 10 + (*c - '0');
        }
      }
      if (number && fd >= lowest && fd != LIST) {
        // Linux releases the descriptor whatever close() returns, but for
        // EBADF: a listed descriptor that is still there would not close.
        if (::close(fd) == 0 || errno != EBADF) {
          closedAny = true;
        } else {
          failure = EBADF;
        }
      }
      at += length;
    }
    if (failure == 0 && closedAny && ::lseek(LIST, 0, SEEK_SET) != 0) {
      failure = errno;
    }
  }
  (void)::close(LIST);
  return failure;
}
#endif

/**
 * @brief Close every descriptor from @p lowest up, whatever its number:
 * close_range where the system has it, else each one /proc/self/fd lists
 * (closeListedDescriptors). 0 when every one is closed; otherwise the errno
 * of what failed, and some may be left open. Where there is no /proc,
 * outside Linux, each one below @p bound. Async-signal-safe.
 */
int closeDescriptorsFrom(int lowest, int bound) {
#ifdef __linux__
  (void)bound;
#ifdef SYS_close_range
  if (::syscall(SYS_close_range, static_cast<unsigned>(lowest), ~0U, 0U) == 0) {
    return 0;
  }
#endif
  return closeListedDescriptors(lowest);
#else
  for (int fd = lowest; fd < bound; ++fd) {
    (void)::close(fd);
  }
  return 0;
#endif
}

/** @brief What a keeper reports first on its end of the anchor: isolated, or not, and why. */
enum KeeperReport : int {
  KEEPER_ISOLATED = 0,     ///< It holds nothing of this process's but the anchor.
  KEEPER_NOT_ISOLATED = 1, ///< It could not let go of everything it inherits; errno follows.
};

/**
 * @brief A probe group's keeper, in the child of fork(): every signal it can
 * block blocked, a process group of its own, and nothing open of this
 * process's: its copy of this process's end of the anchor closed first, by
 * itself, so that it sees end-of-file once this process's copy closes,
 * whatever the descriptors' numbers; stdin its own end of the anchor; stdout
 * and stderr /dev/null, or closed where /dev/null cannot be opened or put
 * there; every other descriptor from 3 up closed (closeDescriptorsFrom), the
 * /dev/null it opened too. Then it reports on the anchor, in the
 * shape of a program's start report, whether it holds nothing else
 * (KEEPER_ISOLATED) or could not let go of everything (KEEPER_NOT_ISOLATED
 * and the errno of the first step that failed: its end of the anchor not
 * put on stdin, reported from where that end is; stdout or stderr neither
 * replaced nor shown closed; a descriptor from 3 up not closed), waits for
 * the anchor's end-of-file and exits. A stream is shown closed when its
 * close succeeds or a query of it then fails with EBADF, so a standard
 * stream this process had closed counts as closed; one whose state the
 * query cannot read does not. One whose number the anchor's end or
 * /dev/null took counts as replaced. Async-signal-safe.
 */
[[noreturn]] void runGroupKeeper(int own, int peer, int bound) {
  sigset_t all;
  (void)::sigfillset(&all);
  (void)::sigprocmask(SIG_SETMASK, &all, nullptr);
  (void)::setpgid(0, 0);
  // What of it stays open is closed below, with stdout and stderr or from 3 up.
  (void)::close(peer);
  int anchor = STDIN_FILENO;
  int failure = 0;
  if (own != STDIN_FILENO) {
    if (::dup2(own, STDIN_FILENO) == STDIN_FILENO) {
      (void)::close(own);
    } else {
      // stdin is still this process's: the report goes out where the anchor is.
      failure = errno;
      anchor = own;
    }
  }
  if (failure == 0) {
    // A stream that cannot be given /dev/null is closed. It is let go once
    // the close succeeds or a query of it fails with EBADF; still open, it
    // is this process's and the failure is the replacement's; where the
    // query fails otherwise, whether it is open is unknown and the failure
    // is the query's.
    const int NUL = ::open("/dev/null", O_WRONLY);
    const int NO_NUL = errno;
    for (const int STREAM : {STDOUT_FILENO, STDERR_FILENO}) {
      if (NUL >= 0 && ::dup2(NUL, STREAM) == STREAM) {
        continue;
      }
      const int WHY = NUL >= 0 ? errno : NO_NUL;
      if (::close(STREAM) == 0) {
        continue;
      }
      const int FLAGS = ::fcntl(STREAM, F_GETFD);
      if (FLAGS == -1 && errno == EBADF) {
        continue;
      }
      if (failure == 0) {
        failure = FLAGS == -1 ? errno : WHY;
      }
    }
    // Its number free again for the list below; left open, it is closed there.
    if (NUL > STDERR_FILENO) {
      (void)::close(NUL);
    }
    const int NOT_CLOSED = closeDescriptorsFrom(3, bound);
    if (failure == 0) {
      failure = NOT_CLOSED;
    }
  }
  const int REPORT[2] = {failure == 0 ? KEEPER_ISOLATED : KEEPER_NOT_ISOLATED, failure};
  const char* next = reinterpret_cast<const char*>(REPORT);
  std::size_t left = sizeof(REPORT);
  while (left > 0) {
    const ssize_t N = ::send(anchor, next, left, MSG_NOSIGNAL);
    if (N < 0 && errno == EINTR) {
      continue;
    }
    if (N <= 0) {
      break;
    }
    next += N;
    left -= static_cast<std::size_t>(N);
  }
  char byte = 0;
  while (::read(anchor, &byte, 1) < 0 && errno == EINTR) {
  }
  ::_exit(0);
}

/** @brief How long a probe waits for its keeper's report. */
constexpr int KEEPER_REPORT_MS = 5000;

/**
 * @brief The keeper's report on @p fd, this process's end of the anchor: 0
 * once the keeper holds nothing of this process's (KEEPER_ISOLATED);
 * otherwise the errno the probe's setup fails with: the one the keeper
 * reports (KEEPER_NOT_ISOLATED), ESRCH when it ended without a report,
 * ETIMEDOUT when none came within KEEPER_REPORT_MS.
 */
int readKeeperReport(int fd) {
  int report[2] = {KEEPER_NOT_ISOLATED, 0};
  std::size_t got = 0;
  const auto UNTIL = Clock::now() + std::chrono::milliseconds(KEEPER_REPORT_MS);
  while (got < sizeof(report)) {
    const auto LEFT =
        std::chrono::duration_cast<std::chrono::milliseconds>(UNTIL - Clock::now()).count();
    if (LEFT <= 0) {
      return ETIMEDOUT;
    }
    pollfd ready{fd, POLLIN, 0};
    const int READY = ::poll(&ready, 1, static_cast<int>(LEFT));
    if (READY < 0 && errno == EINTR) {
      continue;
    }
    if (READY < 0) {
      return errno;
    }
    if (READY == 0) {
      return ETIMEDOUT;
    }
    const ssize_t N = ::read(fd, reinterpret_cast<char*>(report) + got, sizeof(report) - got);
    if (N < 0 && errno == EINTR) {
      continue;
    }
    if (N < 0) {
      return errno;
    }
    if (N == 0) {
      return ESRCH;
    }
    got += static_cast<std::size_t>(N);
  }
  if (report[0] == KEEPER_ISOLATED) {
    return 0;
  }
  return report[1] != 0 ? report[1] : EIO;
}

/**
 * @brief In a child between fork and exec: make @p fd its standard stream
 * @p stream for the program it runs. A descriptor whose number already is
 * that stream's (where this process had the stream closed) keeps it, its
 * close-on-exec cleared: dup2() onto itself leaves the flag, and the exec
 * would close the stream. Any other is duplicated onto it.
 * Async-signal-safe.
 */
void becomeStream(int fd, int stream) {
  if (fd != stream) {
    (void)::dup2(fd, stream);
    return;
  }
  const int FLAGS = ::fcntl(fd, F_GETFD);
  if (FLAGS != -1) {
    (void)::fcntl(fd, F_SETFD, FLAGS & ~FD_CLOEXEC);
  }
}

/**
 * @brief In a child between fork and exec, before it sets any of the
 * program's standard streams: @p fd, which the child still needs once a
 * stream is set (the start report's end; a stream's source), moved above
 * the streams when its number is one of theirs that @p becomes (a mask,
 * bit n for stream n) does not name: duplicated from 3 up, close-on-exec,
 * the old number closed. So no stream set before it can overwrite it,
 * whatever order the descriptors were made in; one that already is a stream
 * it becomes stays (becomeStream()). Returns the descriptor to use, or -1
 * and errno when none from 3 up is free. Async-signal-safe.
 */
int clearOfStreams(int fd, unsigned becomes) {
  if (fd < 0 || fd > STDERR_FILENO || (becomes & (1U << fd)) != 0) {
    return fd;
  }
  const int MOVED = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
  if (MOVED >= 0) {
    (void)::close(fd);
  }
  return MOVED;
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
  } else if (stage == ReadinessStage::COMPLETION) {
    message = "completion: " + message;
  }
  result.report = EnvReport{status, std::move(message), std::move(remedy)};
  return result;
}

std::string ReadinessRequest::identity() const {
  // Every field is length-prefixed, so the key reads back as exactly one
  // sequence of fields: no value can imitate a separator, and the number of
  // scripts follows from the number of fields.
  std::string key;
  const auto FIELD = [&key](const std::string& value) {
    key += std::to_string(value.size());
    key += ':';
    key += value;
    key += ';';
  };
  FIELD(backend);
  FIELD(profileArgs);
  for (const std::string& script : bpfScripts) {
    FIELD(script);
  }
  FIELD(analyze ? "1" : "0");
  FIELD(std::to_string(static_cast<int>(scope)));
  FIELD(std::to_string(static_cast<int>(launch)));
  return key;
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

/* ----------------------------- Child Exit Statuses ----------------------------- */

StatusLoss currentStatusLoss() {
  struct sigaction current{};
  if (::sigaction(SIGCHLD, nullptr, &current) == 0) {
    if ((current.sa_flags & SA_SIGINFO) == 0 && current.sa_handler == SIG_IGN) {
      return StatusLoss::SIGCHLD_IGNORED;
    }
    if ((current.sa_flags & SA_NOCLDWAIT) != 0) {
      return StatusLoss::NO_CHILD_WAIT;
    }
  }
  return StatusLoss::ANOTHER_WAIT;
}

std::string describeStatusLoss(StatusLoss loss) {
  switch (loss) {
  case StatusLoss::SIGCHLD_IGNORED:
    return "this process ignores SIGCHLD, so the system discards its children's exit statuses";
  case StatusLoss::NO_CHILD_WAIT:
    return "this process sets SA_NOCLDWAIT for SIGCHLD, so the system discards its children's "
           "exit statuses";
  case StatusLoss::ANOTHER_WAIT:
    return "another wait in this process reaped it first";
  case StatusLoss::NONE:
    break;
  }
  return "";
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
  if (!statusObtained) {
    return "ended without an exit status this process could read: " +
           (statusLoss == StatusLoss::NONE ? std::string{"it was not reaped within the bound"}
                                           : describeStatusLoss(statusLoss));
  }
  return "ended in an unknown state";
}

std::optional<ReadinessResult> unobservedProbe(const std::string& what, const ProbeResult& probe,
                                               ReadinessStage stage) {
  if (!probe.endUnobserved()) {
    return std::nullopt;
  }
  const std::string TAIL = outputTail(probe.output.empty() ? probe.errorOutput : probe.output);
  return readinessResult(ReadinessCause::UNUSABLE,
                         what + ": " + probe.describe() + (TAIL.empty() ? "" : ": " + TAIL),
                         CHILD_STATUS_REMEDY, stage);
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
  const int DESCRIPTOR_BOUND = descriptorBound();
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

  // A keeper that could not close what it inherits would hold this
  // process's descriptors (a lock, a pipe's write end) while the probe runs:
  // where neither close_range nor the /proc list can work, nothing is
  // started. Where they can, the keeper's report below decides.
  if (const int ISOLATION = keeperIsolation(); ISOLATION != 0) {
    result.spawnErrno = ISOLATION;
    return result;
  }

  // The group's keeper, started first and ended last: the program runs in
  // the keeper's process group, and until the keeper is reaped, here, after
  // the group has ended, the group's id cannot belong to another process. So
  // the group is signalled and waited for even when the program has been
  // reaped elsewhere (this process ignores SIGCHLD or sets SA_NOCLDWAIT, and
  // the system reaps the program as it exits), and never by an id that may
  // have passed to another process. It is forked before the probe's pipes
  // exist, so it holds none of them. The anchor is a pair of connected
  // sockets: the keeper reports on its end whether it holds nothing of this
  // process's, and sees end-of-file once this process closes its own.
  constexpr int KEEPER_END = 0;
  constexpr int OWN_END = 1;
  int anchor[2] = {-1, -1};
  if (!makeSocketPair(anchor)) {
    result.spawnErrno = errno;
    return result;
  }
  const pid_t KEEPER = ::fork();
  if (KEEPER < 0) {
    result.spawnErrno = errno;
    ::close(anchor[KEEPER_END]);
    ::close(anchor[OWN_END]);
    return result;
  }
  if (KEEPER == 0) {
    runGroupKeeper(anchor[KEEPER_END], anchor[OWN_END], DESCRIPTOR_BOUND);
  }
  (void)::setpgid(KEEPER, KEEPER); // both sides set it; whichever runs first wins the race
  ::close(anchor[KEEPER_END]);
  // Ends the keeper on every path, then reaps it within PROBE_REAP_MS. A
  // keeper still running that the group's end has not signalled (a setup
  // that failed before the program ran) is killed while this process still
  // holds its end of the anchor: it cannot have ended by itself, so its pid
  // is still its own. Then the anchor closes, and a keeper reaped elsewhere
  // is gone.
  bool keeperSignalled = false;
  const auto END_KEEPER = [&] {
    if (!keeperSignalled && leaderState(KEEPER) == LeaderState::RUNNING) {
      (void)::kill(KEEPER, SIGKILL);
      keeperSignalled = true;
    }
    if (anchor[OWN_END] >= 0) {
      ::close(anchor[OWN_END]);
      anchor[OWN_END] = -1;
    }
    const auto UNTIL = Clock::now() + std::chrono::milliseconds(PROBE_REAP_MS);
    int keeperStatus = 0;
    while (true) {
      const pid_t W = ::waitpid(KEEPER, &keeperStatus, WNOHANG);
      if (W == KEEPER || (W < 0 && errno != EINTR) || Clock::now() >= UNTIL) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  };

  // The keeper's own report decides: a program starts only once the keeper
  // holds nothing of this process's; one that could not let go of everything
  // it inherits, its standard streams among them, or said nothing in time,
  // fails the setup and is ended here.
  if (const int NOT_ISOLATED = readKeeperReport(anchor[OWN_END]); NOT_ISOLATED != 0) {
    result.spawnErrno = NOT_ISOLATED;
    END_KEEPER();
    return result;
  }

  if (!makePipe(output) || (SEPARATE && !makePipe(errors)) || !makePipe(report)) {
    result.spawnErrno = errno;
    CLOSE_ALL();
    END_KEEPER();
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
    END_KEEPER();
    return result;
  }
  if (CHILD == 0) {
    // Into the keeper's group, or not run at all: outside it, nothing the
    // program starts could be ended with it.
    if (::setpgid(0, KEEPER) != 0) {
      const int FAILURE[2] = {2, errno};
      writeAll(report[1], FAILURE, sizeof(FAILURE));
      ::_exit(127);
    }
    defaultStopSignals();
    // What the child needs once a stream is set goes above the streams
    // first, but where it already is a stream it becomes.
    const int REPORT = clearOfStreams(report[1], 0U);
    if (REPORT < 0) {
      const int FAILURE[2] = {3, errno};
      writeAll(report[1], FAILURE, sizeof(FAILURE));
      ::_exit(127);
    }
    const int IN = clearOfStreams(DEV_NULL, 1U << STDIN_FILENO);
    const int OUT =
        clearOfStreams(output[1], (1U << STDOUT_FILENO) | (SEPARATE ? 0U : 1U << STDERR_FILENO));
    const int ERR = SEPARATE ? clearOfStreams(errors[1], 1U << STDERR_FILENO) : OUT;
    if (OUT < 0 || ERR < 0 || (DEV_NULL >= 0 && IN < 0)) {
      const int FAILURE[2] = {3, errno};
      writeAll(REPORT, FAILURE, sizeof(FAILURE));
      ::_exit(127);
    }
    if (IN >= 0) {
      becomeStream(IN, STDIN_FILENO);
    }
    becomeStream(OUT, STDOUT_FILENO);
    becomeStream(ERR, STDERR_FILENO);
    ::execve(ARGV[0], ARGV.data(), ENVP.data());
    const int FAILURE[2] = {3, errno};
    writeAll(REPORT, FAILURE, sizeof(FAILURE));
    ::_exit(127);
  }
  (void)::setpgid(CHILD, KEEPER); // both sides set it; whichever runs first wins the race
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
    END_KEEPER();
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
  // not reaped, so that its pid stays its own while it is signalled below
  // (unless the system reaped it already: then it is gone).
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
  // into the measurement it precedes. The unreaped keeper keeps the group's
  // id this call's; a keeper reaped elsewhere (another wait, against the
  // documented ownership of children) is gone, and nothing is signalled by
  // its id.
  if (leaderState(KEEPER) != LeaderState::GONE) {
    (void)::kill(-KEEPER, SIGKILL);
    keeperSignalled = true;
  }
  if (leader != LeaderState::GONE) {
    (void)::kill(CHILD, SIGKILL); // the program itself, had it left the group
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

  // Reap the program, wait for the rest of its group, then reap the keeper;
  // each bounded. A program reaped elsewhere (ECHILD) is gone without a
  // status: its end is unobserved, never read as an exit.
  int status = 0;
  bool reaped = leader == LeaderState::GONE;
  bool lost = reaped;
  const auto REAP_END = Clock::now() + std::chrono::milliseconds(PROBE_REAP_MS);
  while (!reaped) {
    const pid_t W = ::waitpid(CHILD, &status, WNOHANG);
    if (W == CHILD) {
      reaped = true;
      break;
    }
    if (W < 0 && errno != EINTR) {
      reaped = true;
      lost = true;
      break;
    }
    if (Clock::now() >= REAP_END) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  waitForGroup(KEEPER, REAP_END);
  END_KEEPER();
  CLOSE_ALL();
  result.statusObtained = reaped && !lost;
  if (lost) {
    result.statusLoss = currentStatusLoss();
  }
  if (result.statusObtained && !result.timedOut) {
    if (WIFEXITED(status)) {
      result.exited = true;
      result.exitCode = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      result.termSignal = WTERMSIG(status);
    }
  }
  return result;
}

std::string detail::shellQuote(const std::string& word) {
  static constexpr const char* SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "0123456789_@%+=:,./-";
  if (!word.empty() && word.find_first_not_of(SAFE) == std::string::npos) {
    return word;
  }
  std::string quoted = "'";
  for (const char C : word) {
    if (C == '\'') {
      quoted += "'\\''";
    } else {
      quoted += C;
    }
  }
  quoted += "'";
  return quoted;
}

std::string detail::escapePercent(const std::string& path) {
  std::string out;
  for (const char CH : path) {
    out += CH;
    if (CH == '%') {
      out += '%';
    }
  }
  return out;
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
      waitStatus_(other.waitStatus_), loss_(other.loss_) {
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
    loss_ = other.loss_;
    other.child_ = -1;
    other.reaped_ = true;
  }
  return *this;
}

namespace {

/**
 * @brief Reap @p child if it has ended; true once it is gone. A child reaped
 * elsewhere (ECHILD) is gone too, without a status: @p waitStatus stays
 * HELPER_STATUS_UNAVAILABLE, and @p loss says why.
 */
bool tryReap(pid_t child, bool& reaped, int& waitStatus, StatusLoss& loss) {
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
    waitStatus = HELPER_STATUS_UNAVAILABLE;
    loss = currentStatusLoss();
  }
  return reaped;
}

/** @brief Poll for the child's end for up to @p ms. */
bool waitGone(pid_t child, bool& reaped, int& waitStatus, StatusLoss& loss, int ms) {
  const auto DEADLINE = Clock::now() + std::chrono::milliseconds(std::max(ms, 0));
  while (!tryReap(child, reaped, waitStatus, loss)) {
    if (Clock::now() >= DEADLINE) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

/**
 * @brief The command name of process @p pid as the kernel keeps it (its
 * executable's file name, at most 15 characters), or "" when unreadable.
 */
std::string processName(pid_t pid) {
  std::ifstream in("/proc/" + std::to_string(pid) + "/comm");
  std::string name;
  std::getline(in, name);
  return name;
}

/**
 * @brief The process that runs the tool sudo was asked to run, from
 * @p started, the process this helper started with @p sudoPath.
 *
 * sudo either stays the tool's parent (it forks the tool, sometimes with a
 * monitor process of its own between them, which is also named sudo) or
 * executes the tool in its own process. The command name tells them apart:
 * while a process is still sudo and has exactly one child, the tool is
 * below it; a process that is not sudo is the tool, and its own child is
 * never taken for it. An unreadable name, or a sudo with no child or
 * several, ends the walk there.
 */
pid_t toolUnderSudo(pid_t started, const std::string& sudoPath) {
  const std::string SUDO_NAME = std::filesystem::path(sudoPath).filename().string().substr(0, 15);
  pid_t target = started;
  for (int depth = 0; depth < 3 && processName(target) == SUDO_NAME; ++depth) {
    const std::vector<pid_t> CHILDREN = childProcesses(target);
    if (CHILDREN.size() != 1) {
      break;
    }
    target = CHILDREN.front();
  }
  return target;
}

} // namespace

bool OwnedHelper::running() { return child_ > 0 && !tryReap(child_, reaped_, waitStatus_, loss_); }

HelperStart OwnedHelper::start(const std::vector<std::string>& argvIn,
                               const std::string& stdoutPath, const std::string& stderrPath,
                               int graceMs, const ReadinessContext* probeEnv) {
  HelperStart outcome;
  if (child_ > 0 && !reaped_) {
    (void)stop();
  }
  child_ = -1;
  reaped_ = true;
  waitStatus_ = HELPER_STATUS_UNAVAILABLE;
  loss_ = StatusLoss::NONE;
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
    // stop() relies on SIGINT and SIGTERM: the helper takes them at their
    // default action whatever this process inherited.
    defaultStopSignals();
    // The start report's end goes above the streams before any is set.
    const int REPORT = clearOfStreams(report[1], 0U);
    if (REPORT < 0) {
      const int FAILURE[2] = {3, errno};
      writeAll(report[1], FAILURE, sizeof(FAILURE));
      ::_exit(127);
    }
    // Capture files are opened here, by the invoking user, before any
    // elevation: they stay the user's whatever the helper runs as.
    const int IN = ::open("/dev/null", O_RDONLY);
    if (IN >= 0) {
      becomeStream(IN, STDIN_FILENO);
      if (IN != STDIN_FILENO) {
        ::close(IN);
      }
    }
    const int OUT = ::open(OUT_PATH.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (OUT < 0) {
      const int FAILURE[2] = {1, errno};
      writeAll(REPORT, FAILURE, sizeof(FAILURE));
      ::_exit(126);
    }
    becomeStream(OUT, STDOUT_FILENO);
    if (OUT != STDOUT_FILENO) {
      ::close(OUT);
    }
    const int ERR = ::open(ERR_PATH.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ERR < 0) {
      const int FAILURE[2] = {2, errno};
      writeAll(REPORT, FAILURE, sizeof(FAILURE));
      ::_exit(126);
    }
    becomeStream(ERR, STDERR_FILENO);
    if (ERR != STDERR_FILENO) {
      ::close(ERR);
    }
    if (USE_ENVP) {
      ::execve(ARGV[0], ARGV.data(), ENVP.data());
    } else {
      ::execv(ARGV[0], ARGV.data());
    }
    const int FAILURE[2] = {3, errno};
    writeAll(REPORT, FAILURE, sizeof(FAILURE));
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
  if (waitGone(child_, reaped_, waitStatus_, loss_, graceMs)) {
    outcome.exitedEarly = true;
    outcome.waitStatus = waitStatus_;
    outcome.statusObtained = waitStatus_ != HELPER_STATUS_UNAVAILABLE;
    outcome.statusLoss = loss_;
    if (!stderrPath.empty()) {
      outcome.errorTail = readFileTail(stderrPath, 4096);
    }
  }
  return outcome;
}

std::vector<pid_t> childProcesses(pid_t parent, const std::string& procRoot) {
  std::vector<pid_t> children;
  const std::string PID = std::to_string(parent);
  std::ifstream list(procRoot + "/" + PID + "/task/" + PID + "/children");
  if (list) {
    long pid = 0;
    while (list >> pid) {
      if (pid > 0) {
        children.push_back(static_cast<pid_t>(pid));
      }
    }
    std::sort(children.begin(), children.end());
    return children;
  }
  // No children list on this kernel: read each process's parent instead. The
  // walk throws nothing (stop() runs from a destructor): an unreadable
  // directory ends it and a name that is not a pid is skipped.
  std::error_code ec;
  std::filesystem::directory_iterator entry(procRoot, ec);
  for (; !ec && entry != std::filesystem::directory_iterator(); entry.increment(ec)) {
    const std::string NAME = entry->path().filename().string();
    const char* const END = NAME.data() + NAME.size();
    pid_t pid = 0;
    const auto [STOP, ERR] = std::from_chars(NAME.data(), END, pid);
    if (NAME.empty() || ERR != std::errc{} || STOP != END || pid <= 0) {
      continue;
    }
    std::ifstream stat(entry->path() / "stat");
    std::string line;
    if (!std::getline(stat, line)) {
      continue;
    }
    const std::size_t CLOSE = line.rfind(')');
    if (CLOSE == std::string::npos) {
      continue;
    }
    std::istringstream fields(line.substr(CLOSE + 1));
    std::string state;
    long ppid = 0;
    if (fields >> state >> ppid && ppid == static_cast<long>(parent)) {
      children.push_back(pid);
    }
  }
  std::sort(children.begin(), children.end());
  return children;
}

HelperStopResult OwnedHelper::stop() {
  HelperStopResult result;
  if (child_ <= 0) {
    return result;
  }
  if (tryReap(child_, reaped_, waitStatus_, loss_)) {
    result.reaped = true;
    result.waitStatus = waitStatus_;
    result.statusObtained = waitStatus_ != HELPER_STATUS_UNAVAILABLE;
    result.statusLoss = loss_;
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
    if (tryReap(child_, reaped_, waitStatus_, loss_)) {
      break;
    }
    StopDelivery delivery;
    delivery.signal = step.sig;
    delivery.target = child_;
    if (policy_.route == PrivilegeRoute::SCOPED_SUDO) {
#ifdef __linux__
      delivery.target = toolUnderSudo(child_, policy_.sudoPath);
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
      if (PROBE.endUnobserved()) {
        // Neither delivered nor refused, as far as this process can see: the
        // stop waits for the helper as after a delivery, so a signal that did
        // arrive keeps its chance to end it (a tracer's flush on SIGINT).
        delivery.statusObtained = false;
        delivery.statusLoss = PROBE.statusLoss;
        delivery.detail = PROBE.describe();
      } else if (!delivery.delivered) {
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
    if ((delivery.delivered || !delivery.statusObtained) &&
        waitGone(child_, reaped_, waitStatus_, loss_, step.waitMs)) {
      result.stoppedBy = step.sig;
      break;
    }
  }
  (void)tryReap(child_, reaped_, waitStatus_, loss_);
  result.reaped = reaped_;
  result.waitStatus = waitStatus_;
  result.statusObtained = waitStatus_ != HELPER_STATUS_UNAVAILABLE;
  result.statusLoss = loss_;
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
