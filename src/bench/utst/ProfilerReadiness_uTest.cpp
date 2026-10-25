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
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using vernier::bench::childProcesses;
using vernier::bench::decidePrivilege;
using vernier::bench::describeStatusLoss;
using vernier::bench::EnvBool;
using vernier::bench::EnvReport;
using vernier::bench::HELPER_STATUS_UNAVAILABLE;
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
using vernier::bench::StatusLoss;
using vernier::bench::test::ChildReaping;
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

/**
 * @brief While it lives, this process ignores SIGINT and this thread blocks
 * SIGTERM, as a benchmark started as a background job of a script, or by a
 * caller that blocked the signal, inherits them; restored after.
 */
class StopSignalsInherited {
public:
  StopSignalsInherited() {
    struct sigaction ignore{};
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    ::sigaction(SIGINT, &ignore, &savedInterrupt_);
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGTERM);
    ::pthread_sigmask(SIG_BLOCK, &block, &savedMask_);
  }
  ~StopSignalsInherited() {
    ::sigaction(SIGINT, &savedInterrupt_, nullptr);
    ::pthread_sigmask(SIG_SETMASK, &savedMask_, nullptr);
  }
  StopSignalsInherited(const StopSignalsInherited&) = delete;
  StopSignalsInherited& operator=(const StopSignalsInherited&) = delete;

private:
  struct sigaction savedInterrupt_{};
  sigset_t savedMask_{};
};

/**
 * @brief The hexadecimal mask after @p key ("SigIgn:") in /proc/<pid>/status
 *        text @p status; all bits set when the line is missing.
 */
unsigned long long statusMask(const std::string& status, const std::string& key) {
  const std::size_t AT = status.find(key);
  return AT == std::string::npos ? ~0ULL : std::stoull(status.substr(AT + key.size()), nullptr, 16);
}

/** @brief The bit for signal @p sig in a /proc status mask. */
constexpr unsigned long long bitOf(int sig) { return 1ULL << (sig - 1); }

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

/** @brief The whole of /proc/<pid>/@p file; none when it cannot be opened. */
std::optional<std::string> procFile(pid_t pid, const char* file) {
  std::ifstream in("/proc/" + std::to_string(pid) + "/" + file, std::ios::binary);
  if (!in.is_open()) {
    return std::nullopt;
  }
  return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/** @brief A process's state as /proc gives it: its letter, or why there is none. */
struct ProcessState {
  enum class Read : std::uint8_t {
    LETTER,    ///< The letter is the state ('R', 'S', 'Z', 'X', ...).
    GONE,      ///< No such process any more: its /proc entry has gone.
    UNREADABLE ///< Not known: its stat could not be read, or held no state.
  };
  Read read = Read::UNREADABLE;
  char letter = '\0';
  /// With UNREADABLE: the errno of the failed open or read; 0 for a stat without a state.
  int err = 0;
};

/**
 * @brief The state of process @p pid from /proc/<pid>/stat: GONE when its
 * entry has gone (ENOENT, or ESRCH while it is read), UNREADABLE when the
 * file cannot be opened or read otherwise or holds no state after the
 * command name, the state's letter otherwise.
 */
ProcessState readProcessState(pid_t pid) {
  const std::string PATH = "/proc/" + std::to_string(pid) + "/stat";
  const int FD = ::open(PATH.c_str(), O_RDONLY | O_CLOEXEC);
  if (FD < 0) {
    const int ERR = errno;
    return ERR == ENOENT || ERR == ESRCH ? ProcessState{ProcessState::Read::GONE, '\0', 0}
                                         : ProcessState{ProcessState::Read::UNREADABLE, '\0', ERR};
  }
  char buf[512];
  const ssize_t N = ::read(FD, buf, sizeof(buf));
  const int ERR = errno;
  ::close(FD);
  if (N < 0) {
    return ERR == ENOENT || ERR == ESRCH ? ProcessState{ProcessState::Read::GONE, '\0', 0}
                                         : ProcessState{ProcessState::Read::UNREADABLE, '\0', ERR};
  }
  // "<pid> (<comm>) <state> ...": comm may hold spaces and ')'.
  const std::string TEXT(buf, static_cast<std::size_t>(N));
  const std::size_t CLOSE = TEXT.rfind(')');
  if (CLOSE == std::string::npos || CLOSE + 2 >= TEXT.size() || TEXT[CLOSE + 1] != ' ') {
    return ProcessState{ProcessState::Read::UNREADABLE, '\0', 0};
  }
  return ProcessState{ProcessState::Read::LETTER, TEXT[CLOSE + 2], 0};
}

/**
 * @brief Whether @p pid is still this process's to wait for, by a look that
 * does not reap it (waitid WEXITED | WNOHANG | WNOWAIT): 0 when it is,
 * otherwise the look's errno, ECHILD when it is not this process's child
 * any more and another errno when the look itself failed.
 */
int lookWithoutReaping(pid_t pid) {
  siginfo_t info{};
  return ::waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0 ? 0
                                                                                          : errno;
}

/**
 * @brief Where zombieChildren() reads what it counts: this process's children
 * (its main thread's /proc list), each one's state and the look, the real
 * readers by default; the count's own tests give their own.
 */
struct ChildObservation {
  std::function<std::vector<pid_t>()> children = [] { return childProcesses(::getpid()); };
  std::function<ProcessState(pid_t)> state = readProcessState;
  std::function<int(pid_t)> look = lookWithoutReaping;
};

/** @brief A child zombieChildren() counts, and why it counts. */
struct CountedChild {
  pid_t pid = -1;
  std::string why;
};

/** @brief The longest zombieChildren() gives a child the kernel is releasing to leave. */
constexpr std::chrono::milliseconds VANISH_WAIT{200};

/**
 * @brief The children of this process left to reap, or whose end cannot be
 * known, each with why it counts. A zombie (state Z) that the look, which
 * does not reap it, finds this process's counts at once. A child that ended
 * while SIGCHLD was ignored or SA_NOCLDWAIT set is reaped by the kernel
 * itself, which lists it among this process's children for a moment while
 * it releases it: in state X (dead), or not this process's to the look any
 * more (ECHILD). Such a child is vanishing: it is given up to VANISH_WAIT,
 * polled, and leaves the count only on evidence that it has gone (its pid
 * missing from a fresh list of this process's children, or its /proc entry
 * gone); still there at the end, it counts. A child whose state cannot be
 * read, or a zombie the look fails on with another errno, is not known to
 * have gone: it counts, with the reason. A child whose /proc entry has gone
 * does not count.
 */
std::vector<CountedChild> zombieChildren(const ChildObservation& observe = {}) {
  std::vector<CountedChild> counted;
  std::vector<pid_t> vanishing;
  for (const pid_t CHILD : observe.children()) {
    const ProcessState STATE = observe.state(CHILD);
    if (STATE.read == ProcessState::Read::GONE) {
      continue;
    }
    if (STATE.read == ProcessState::Read::UNREADABLE) {
      counted.push_back({CHILD, "state unreadable (" +
                                    (STATE.err != 0 ? std::system_category().message(STATE.err)
                                                    : std::string{"no state in its stat"}) +
                                    ")"});
      continue;
    }
    if (STATE.letter == 'X') {
      vanishing.push_back(CHILD);
      continue;
    }
    if (STATE.letter != 'Z') {
      continue; // it runs
    }
    const int LOOK = observe.look(CHILD);
    if (LOOK == 0) {
      counted.push_back({CHILD, "a zombie this process may still wait for"});
    } else if (LOOK == ECHILD) {
      vanishing.push_back(CHILD);
    } else {
      counted.push_back({CHILD, "state Z, and the look that would tell whether it is this "
                                "process's failed: " +
                                    std::system_category().message(LOOK)});
    }
  }
  const auto UNTIL = std::chrono::steady_clock::now() + VANISH_WAIT;
  while (!vanishing.empty()) {
    const std::vector<pid_t> LISTED = observe.children();
    vanishing.erase(std::remove_if(vanishing.begin(), vanishing.end(),
                                   [&](pid_t pid) {
                                     return std::find(LISTED.begin(), LISTED.end(), pid) ==
                                                LISTED.end() ||
                                            observe.state(pid).read == ProcessState::Read::GONE;
                                   }),
                    vanishing.end());
    if (vanishing.empty() || std::chrono::steady_clock::now() >= UNTIL) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  for (const pid_t PID : vanishing) {
    counted.push_back({PID, "still listed after " + std::to_string(VANISH_WAIT.count()) +
                                " ms of release (state X, or not this process's to the look)"});
  }
  return counted;
}

/**
 * @brief Each child in @p counted (the set a zombie check counted), named for
 * the check's failure message: its pid and why it counted, then what /proc
 * still gives of it now (its command line, which a zombie keeps none of,
 * command name, state, parent, process group and session; "unavailable"
 * for what cannot be read any more), and which of a probe's processes it is,
 * against the pids the fake helper logged in @p dir: the program ("helper
 * orphan ... pid=N"), the child the program left ("helper child pid=N"), or
 * the keeper, a fork of this binary leading a group of its own; otherwise
 * none of them.
 */
std::string namedZombieChildren(const FakeToolDir& dir, const std::vector<CountedChild>& counted) {
  const auto LOGGED = [&dir](const std::string& prefix) {
    std::vector<pid_t> pids;
    for (const std::string& line : dir.logLines(prefix)) {
      const std::size_t AT = line.rfind("pid=");
      if (AT != std::string::npos) {
        pids.push_back(static_cast<pid_t>(std::stol(line.substr(AT + 4))));
      }
    }
    return pids;
  };
  // "<pid> (<comm>) <state> <ppid> <pgrp> <session> ...": comm may hold spaces and ')'.
  const auto COMM = [](const std::string& stat) {
    const std::size_t OPEN = stat.find('(');
    const std::size_t CLOSE = stat.rfind(')');
    return OPEN == std::string::npos || CLOSE == std::string::npos || CLOSE < OPEN
               ? std::string{}
               : stat.substr(OPEN + 1, CLOSE - OPEN - 1);
  };
  const std::vector<pid_t> PROGRAMS = LOGGED("helper orphan ");
  const std::vector<pid_t> LEFT = LOGGED("helper child pid=");
  const std::string SELF = COMM(procFile(::getpid(), "stat").value_or(""));
  std::string named;
  for (const CountedChild& CHILD : counted) {
    const std::optional<std::string> STAT = procFile(CHILD.pid, "stat");
    std::optional<std::string> command = procFile(CHILD.pid, "cmdline");
    if (command) {
      std::replace(command->begin(), command->end(), '\0', ' ');
      while (!command->empty() && command->back() == ' ') {
        command->pop_back();
      }
    }
    char state = '\0';
    int parent = 0;
    int group = 0;
    int session = 0;
    const std::size_t CLOSE = STAT ? STAT->rfind(')') : std::string::npos;
    const bool FIELDS =
        CLOSE != std::string::npos && std::sscanf(STAT->c_str() + CLOSE + 1, " %c %d %d %d", &state,
                                                  &parent, &group, &session) == 4;
    const std::string UNAVAILABLE = "unavailable";
    const std::string NAME = STAT ? COMM(*STAT) : std::string{};
    const std::string ROLE =
        std::find(PROGRAMS.begin(), PROGRAMS.end(), CHILD.pid) != PROGRAMS.end()
            ? "the program (logged \"helper orphan ... pid=" + std::to_string(CHILD.pid) + "\")"
        : std::find(LEFT.begin(), LEFT.end(), CHILD.pid) != LEFT.end()
            ? "the child the program left (logged \"helper child pid=" + std::to_string(CHILD.pid) +
                  "\")"
        : !FIELDS ? std::string{"not a logged pid; whether it is the keeper is unavailable"}
        : NAME == SELF && group == static_cast<int>(CHILD.pid)
            ? std::string{"the keeper: a fork of this binary leading its own group"}
            : std::string{"none of the probe's processes"};
    named += (named.empty() ? "" : "; ") + std::string{"pid "} + std::to_string(CHILD.pid) + " (" +
             CHILD.why + "), command line " +
             (!command           ? UNAVAILABLE
              : command->empty() ? std::string{"none"}
                                 : "\"" + *command + "\"") +
             ", comm " + (NAME.empty() ? UNAVAILABLE : NAME) + ", state " +
             (FIELDS ? std::string(1, state) : UNAVAILABLE) + ", parent " +
             (FIELDS ? std::to_string(parent) : UNAVAILABLE) + ", group " +
             (FIELDS ? std::to_string(group) : UNAVAILABLE) + ", session " +
             (FIELDS ? std::to_string(session) : UNAVAILABLE) + ": " + ROLE;
  }
  return named.empty() ? "none" : named;
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

/**
 * @brief What a failed zombie check after @p out's probe in @p dir says
 * besides its count: each child of @p counted, the set the check counted,
 * named (namedZombieChildren()), how long the probe took and the check's
 * @p iteration.
 */
std::string zombieReport(const FakeToolDir& dir, const OrphanOutcome& out,
                         const std::string& iteration, const std::vector<CountedChild>& counted) {
  return "zombie children: " + namedZombieChildren(dir, counted) + "; the probe took " +
         std::to_string(out.elapsed.count()) + " ms; iteration: " + iteration;
}

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
  const auto& [backend, profileArgs, bpfScripts, analyze, scope, launch] = request;
  EXPECT_TRUE(backend.empty());
  EXPECT_TRUE(profileArgs.empty());
  EXPECT_EQ(bpfScripts.size(), 1U);
  EXPECT_FALSE(analyze);
  EXPECT_EQ(scope, ReadinessScope::DEFAULT_INVENTORY);
  EXPECT_EQ(launch, LaunchContext::IN_PROCESS);
}

/** @test A change to any one field is a different identity, and no value imitates a separator. */
TEST(ReadinessRequestIdentityTest, EveryFieldChangesTheKey) {
  ReadinessRequest base;
  base.backend = "b";
  base.profileArgs = "m";
  base.bpfScripts = {"x"};
  std::vector<ReadinessRequest> variants(6, base);
  variants[0].backend = "c";
  variants[1].profileArgs = "n";
  variants[2].bpfScripts = {"y"};
  variants[3].analyze = true;
  variants[4].scope = ReadinessScope::RUNTIME;
  variants[5].launch = LaunchContext::RUNNER_WRAPPED;
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

/**
 * @test A probe reaped elsewhere, as an application that ignores SIGCHLD or
 * sets SA_NOCLDWAIT reaps it, leaves no exit status to read: whether it
 * exited 0 or 42, it neither exited nor succeeded as far as its result says,
 * its end is unobserved and the result says why, and its output is still
 * read. Reaped here, with SIGCHLD at its default action, each status is read.
 */
TEST(ReadinessProbes, StatusReapedElsewhereIsNeitherSuccessNorFailure) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  using Policy = ChildReaping::Policy;
  for (const Policy POLICY : {Policy::IGNORED, Policy::NO_CHILD_WAIT, Policy::DEFAULT}) {
    const std::string NAME = POLICY == Policy::IGNORED         ? "SIGCHLD ignored"
                             : POLICY == Policy::NO_CHILD_WAIT ? "SA_NOCLDWAIT"
                                                               : "the default action";
    const StatusLoss LOSS = POLICY == Policy::IGNORED         ? StatusLoss::SIGCHLD_IGNORED
                            : POLICY == Policy::NO_CHILD_WAIT ? StatusLoss::NO_CHILD_WAIT
                                                              : StatusLoss::NONE;
    ProbeResult zero;
    ProbeResult failed;
    {
      const ChildReaping REAPING(POLICY);
      ASSERT_TRUE(REAPING.ok());
      zero = runBoundedProbe({HELPER, "exit", "0"}, 5000, CTX);
      failed = runBoundedProbe({HELPER, "exit", "42"}, 5000, CTX);
    }
    for (const ProbeResult* R : {&zero, &failed}) {
      EXPECT_TRUE(R->started) << NAME;
      EXPECT_FALSE(R->timedOut) << NAME;
      EXPECT_EQ(R->statusLoss, LOSS) << NAME;
    }
    EXPECT_NE(failed.output.find("args: exit 42"), std::string::npos)
        << NAME << ": " << failed.output;
    if (POLICY == Policy::DEFAULT) {
      EXPECT_TRUE(zero.statusObtained);
      EXPECT_TRUE(zero.succeeded()) << zero.describe();
      EXPECT_TRUE(failed.statusObtained);
      EXPECT_TRUE(failed.exited);
      EXPECT_EQ(failed.exitCode, 42);
      EXPECT_FALSE(failed.succeeded());
      EXPECT_FALSE(failed.endUnobserved());
      EXPECT_EQ(failed.describe(), "exit status 42");
      continue;
    }
    for (const ProbeResult* R : {&zero, &failed}) {
      EXPECT_FALSE(R->statusObtained) << NAME;
      EXPECT_FALSE(R->exited) << NAME << ": an exit status not read was read as an exit";
      EXPECT_EQ(R->exitCode, -1) << NAME;
      EXPECT_EQ(R->termSignal, 0) << NAME;
      EXPECT_FALSE(R->succeeded()) << NAME << ": an exit status not read passes for a success";
      EXPECT_TRUE(R->endUnobserved()) << NAME;
      EXPECT_EQ(R->describe(),
                "ended without an exit status this process could read: " + describeStatusLoss(LOSS))
          << NAME;
    }
  }
}

/**
 * @test A probe's program, a stop's `sudo -n kill` among them, takes SIGINT
 * and SIGTERM at their default action whatever this process inherited.
 */
TEST(ReadinessProbes, StopSignalsAtTheirDefaultWhateverThisProcessInherited) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  ProbeResult masks;
  {
    const StopSignalsInherited INHERITED;
    masks = runBoundedProbe({HELPER, "masks"}, 5000, CTX);
  }
  ASSERT_TRUE(masks.succeeded()) << masks.describe() << "\n" << masks.output;
  EXPECT_EQ(statusMask(masks.output, "SigIgn:") & bitOf(SIGINT), 0ULL) << masks.output;
  EXPECT_EQ(statusMask(masks.output, "SigBlk:") & bitOf(SIGTERM), 0ULL) << masks.output;
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

/**
 * @test A child the program leaves behind ends with the probe, whether it
 * exits, fails or times out, and the call leaves no child of its own
 * unreaped; a check that finds one names each zombie child.
 */
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
    const std::size_t ZOMBIES_BEFORE = zombieChildren().size();
    const OrphanOutcome OUT = runOrphanProbe(dir, HELPER, row.kind, row.status, row.timeoutMs);
    ASSERT_TRUE(OUT.returned) << "the probe did not return";
    EXPECT_EQ(OUT.result.succeeded(), row.succeeded) << OUT.result.describe();
    EXPECT_EQ(OUT.result.timedOut, row.timedOut);
    EXPECT_LT(OUT.elapsed, std::chrono::milliseconds(row.timeoutMs + 3000));
    ASSERT_GT(OUT.child, 0) << dir.log();
    EXPECT_TRUE(OUT.childEnded) << "child " << OUT.child << " outlived the probe";
    const std::vector<CountedChild> COUNTED = zombieChildren();
    EXPECT_LE(COUNTED.size(), ZOMBIES_BEFORE)
        << "the probe left a child unreaped; "
        << zombieReport(dir, OUT, std::string{"orphan "} + row.kind + " " + row.status, COUNTED);
  }
}

/**
 * @test A child the program leaves in the probe's group ends with the probe
 * whatever reaps the program. With SIGCHLD ignored, or with SA_NOCLDWAIT, the
 * system reaps the program as it exits, so the program's pid cannot name the
 * group any more; the group is still ended, and the call leaves no child of
 * its own unreaped; a check that finds one names each zombie child, the
 * probe's time and the policy. A child the kernel is still releasing after it
 * reaped it (the keeper in state X) is not one: the count gives it time to
 * leave (zombieChildren()). Here the program starts a 30 s sleep and exits at
 * once.
 */
TEST(ReadinessProbes, LeftChildEndsWithTheProbeWhateverReapsTheProgram) {
  using Policy = ChildReaping::Policy;
  for (const Policy POLICY : {Policy::DEFAULT, Policy::IGNORED, Policy::NO_CHILD_WAIT}) {
    const std::string NAME = POLICY == Policy::IGNORED         ? "SIGCHLD ignored"
                             : POLICY == Policy::NO_CHILD_WAIT ? "SA_NOCLDWAIT"
                                                               : "the default action";
    SCOPED_TRACE(NAME);
    FakeToolDir dir;
    ASSERT_TRUE(dir.ok());
    const std::string HELPER = dir.install("fake_helper.sh", "helper");
    const std::size_t ZOMBIES_BEFORE = zombieChildren().size();
    OrphanOutcome out;
    {
      const ChildReaping REAPING(POLICY);
      ASSERT_TRUE(REAPING.ok());
      out = runOrphanProbe(dir, HELPER, "quiet", "0", 5000);
    }
    ASSERT_TRUE(out.returned) << "the probe did not return";
    EXPECT_LT(out.elapsed, std::chrono::milliseconds(3000));
    ASSERT_GT(out.child, 0) << dir.log();
    EXPECT_TRUE(out.childEnded) << "child " << out.child << " outlived the probe";
    EXPECT_EQ(out.result.endUnobserved(), POLICY != Policy::DEFAULT) << out.result.describe();
    EXPECT_EQ(out.result.succeeded(), POLICY == Policy::DEFAULT) << out.result.describe();
    const std::vector<CountedChild> COUNTED = zombieChildren();
    EXPECT_LE(COUNTED.size(), ZOMBIES_BEFORE)
        << "the probe left a child unreaped; " << zombieReport(dir, out, NAME, COUNTED);
  }
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
 * the start counts, at once, as "state unreadable": a failed read is not a
 * child gone.
 */
TEST(ZombieCount, StateUnreadableFromTheStartCounts) {
  auto lists = std::make_shared<int>(0);
  auto reads = std::make_shared<int>(0);
  const auto START = std::chrono::steady_clock::now();
  const std::vector<CountedChild> COUNTED = zombieChildren(
      oneChild([](int) { return true; },
               [](int) { return ProcessState{ProcessState::Read::UNREADABLE, '\0', EACCES}; }, 0,
               lists, reads));
  ASSERT_EQ(COUNTED.size(), 1U) << "an unreadable state was taken for a child gone";
  EXPECT_EQ(COUNTED.front().pid, COUNTED_CHILD);
  EXPECT_NE(COUNTED.front().why.find("state unreadable"), std::string::npos) << COUNTED.front().why;
  EXPECT_LT(std::chrono::steady_clock::now() - START, VANISH_WAIT);
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
 * @test A zombie that the look finds this process's counts at once, without
 * the grace.
 */
TEST(ZombieCount, ZombieThisProcessMayWaitForCountsAtOnce) {
  auto lists = std::make_shared<int>(0);
  auto reads = std::make_shared<int>(0);
  const auto START = std::chrono::steady_clock::now();
  const std::vector<CountedChild> COUNTED = zombieChildren(
      oneChild([](int) { return true; }, [](int) { return stateLetter('Z'); }, 0, lists, reads));
  ASSERT_EQ(COUNTED.size(), 1U);
  EXPECT_LT(std::chrono::steady_clock::now() - START, VANISH_WAIT);
  EXPECT_EQ(*lists, 1) << "a zombie of this process's was given the grace";
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

#ifdef __linux__
namespace {

/** @brief The duplicates answerHeldCalls() has failed with EINTR in this process. */
std::atomic<int> duplicatesInterrupted{0};

/** @brief The kill(2) calls answerHeldCalls() has held in this process (signal 0 aside). */
std::atomic<int> killsSeen{0};

/** @brief Those of them aimed at the keeper answerHeldCalls() ended, by its pid or group. */
std::atomic<int> killsOnLost{0};

/** @brief The keeper answerHeldCalls() ended (Denied::keeperLoss), once it is gone; or 0. */
std::atomic<pid_t> keeperEnded{0};

/**
 * @brief The counted kills by what they were aimed at: a group (a negative
 * target), a group's leader (a probe's keeper), or any other process (the
 * probe's program, the helper's child).
 */
std::atomic<int> killsAtGroup{0};
std::atomic<int> killsAtLeader{0};
std::atomic<int> killsAtChild{0};

/** @brief What deny()'s listener does with each kill(2) it holds (Denied::kills). */
enum class KillWatch : std::uint8_t {
  NONE,        ///< kill(2) is not held.
  REFUSED,     ///< Each counted and refused (EPERM): none is expected.
  LOST_REFUSED ///< Each counted; one aimed at the keeper the listener ended, by its pid or its
               ///< group, also counted apart and refused (EPERM); the others carried out.
};

/** @brief What deny()'s listener does with each execve(2) it holds (Denied::keeperLoss). */
enum class KeeperLoss : std::uint8_t {
  NONE,         ///< Nothing of this: execve(2) is held only by execHeld.
  THEN_REFUSED, ///< The caller's keeper ended and awaited, then the exec refused (ENOENT).
  THEN_HELD     ///< The caller's keeper ended and awaited, then the exec left held.
};

/** @brief The calls deny() makes fail, and how. */
struct Denied {
  bool closeRange = true;      ///< close_range(2): ENOSYS, as on a kernel before 5.9.
  int close = -1;              ///< close(2) of this descriptor: EBADF, so no process can close it.
  bool directories = false;    ///< Opening a directory: ENOENT, so /proc/self/fd cannot be listed.
  bool directoryReads = false; ///< Reading a directory (getdents64): EPERM; it still opens.
  int replace = -1;            ///< dup2(2) and dup3(2) onto this descriptor: EPERM.
  bool nullOpen = false;       ///< An open for writing and nothing else, as the keeper opens
                               ///< /dev/null: EACCES.
  int query = -1;              ///< fcntl(2) F_GETFD of this descriptor: EPERM.
  int flagSet = -1;            ///< fcntl(2) F_SETFD of this descriptor: EPERM.
  int replaceErrno = EPERM;    ///< What the duplicates onto `replace` fail with.
  int interruptOnce = -1;      ///< dup2(2) and dup3(2) onto this descriptor held for
                               ///< answerHeldCalls(): a process's first fails with EINTR.
  bool reportWrite = false;    ///< Every write(2) of 8 bytes, a start report's: EINTR.
  bool blockingWait = false;   ///< Every wait4(2) that blocks (no options): EINTR.
  bool noHangWait = false;     ///< Every wait4(2) that does not block (WNOHANG): EINTR.
  bool stateLook = false;      ///< Every waitid(2): EINTR.
  bool execHeld = false;       ///< Every execve(2) held, for a listener that never answers.
  bool joinHeld = false;       ///< A process's join of another's group (setpgid(0, g), g not
                               ///< 0) held, for a listener that never answers; a process's
                               ///< join of another to a group not that one's own (setpgid(p,
                               ///< g), p not 0, g not p): EPERM.
  bool stopAction = false;     ///< sigaction(2) of SIGINT: EPERM.
  bool stopUnblock = false;    ///< sigprocmask(2) SIG_UNBLOCK: EPERM.
  int readOpen = -1;           ///< An open with exactly these flags: EACCES.

  /// Every kill(2) held for the listener, which counts and refuses it as KillWatch says.
  KillWatch kills = KillWatch::NONE;
  /// Every execve(2) held: the caller's group leader, the probe's keeper, ended first.
  KeeperLoss keeperLoss = KeeperLoss::NONE;
  /// Every poll(2) and ppoll(2) held until this process has no child left, 3 s at most.
  bool reportHeld = false;
  /// Every execve(2): ENOENT, so that a start's child reports its exec failing.
  bool execRefused = false;
};

#if defined(SECCOMP_RET_USER_NOTIF) && defined(SECCOMP_USER_NOTIF_FLAG_CONTINUE)
/**
 * @brief Answers the calls deny() holds on @p listener as @p denied asks, for
 * as long as this process lives; a call it does not answer stays held (a
 * held exec or join). A duplicate onto `interruptOnce`: a process's first
 * fails with EINTR and its later ones are carried out; a group's leader (the
 * probe's keeper) goes on at once. A kill (`kills`): counted and refused as
 * KillWatch says; signal 0, which signals nothing, carried out uncounted. An
 * exec (`keeperLoss`): the caller's group leader, the probe's keeper, when
 * it is not this process's, is ended (tgkill(2), which is not held) and
 * awaited until it is gone, 2 s at most, then the exec refused (ENOENT) or
 * left held. A poll (`reportHeld`): carried out once this process has no
 * child left, 3 s at most.
 */
void answerHeldCalls(int listener, Denied denied) {
  std::vector<std::uint32_t> interrupted;
  for (;;) {
    seccomp_notif request{};
    if (::ioctl(listener, SECCOMP_IOCTL_NOTIF_RECV, &request) != 0) {
      if (errno == EINTR || errno == ENOENT) {
        continue;
      }
      return;
    }
    seccomp_notif_resp response{};
    response.id = request.id;
    response.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
    const auto CALL = static_cast<long>(request.data.nr);
    const auto PID = static_cast<pid_t>(request.pid);
    bool duplicate = CALL == SYS_dup3;
    bool polling = CALL == SYS_ppoll;
#ifdef SYS_dup2
    duplicate = duplicate || CALL == SYS_dup2;
#endif
#ifdef SYS_poll
    polling = polling || CALL == SYS_poll;
#endif
    if (CALL == SYS_kill && denied.kills != KillWatch::NONE) {
      const auto TARGET =
          static_cast<pid_t>(static_cast<std::int32_t>(request.data.args[0] & 0xffffffffU));
      const pid_t LOST = keeperEnded.load();
      const bool AT_LOST = LOST > 0 && (TARGET == LOST || TARGET == -LOST);
      if (request.data.args[1] != 0) {
        killsSeen.fetch_add(1);
        if (TARGET < 0) {
          killsAtGroup.fetch_add(1);
        } else if (::getpgid(TARGET) == TARGET) {
          killsAtLeader.fetch_add(1);
        } else {
          killsAtChild.fetch_add(1);
        }
        if (AT_LOST) {
          killsOnLost.fetch_add(1);
        }
        if (denied.kills == KillWatch::REFUSED || AT_LOST) {
          response.flags = 0;
          response.error = -EPERM;
        }
      }
    } else if (CALL == SYS_execve && denied.keeperLoss != KeeperLoss::NONE) {
      const pid_t KEEPER = ::getpgid(PID);
      if (KEEPER > 0 && KEEPER != ::getpgrp()) {
        (void)::syscall(SYS_tgkill, KEEPER, KEEPER, SIGKILL);
        const auto UNTIL = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < UNTIL &&
               !(::syscall(SYS_tgkill, KEEPER, KEEPER, 0) != 0 && errno == ESRCH)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        keeperEnded.store(KEEPER);
      }
      if (denied.keeperLoss == KeeperLoss::THEN_HELD) {
        continue;
      }
      response.flags = 0;
      response.error = -ENOENT;
    } else if (polling && denied.reportHeld) {
      const auto UNTIL = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!childProcesses(::getpid()).empty() && std::chrono::steady_clock::now() < UNTIL) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    } else if (duplicate && denied.interruptOnce >= 0) {
      if (::getpgid(PID) != PID &&
          std::find(interrupted.begin(), interrupted.end(), request.pid) == interrupted.end()) {
        interrupted.push_back(request.pid);
        duplicatesInterrupted.fetch_add(1);
        response.flags = 0;
        response.error = -EINTR;
      }
    } else {
      continue;
    }
    (void)::ioctl(listener, SECCOMP_IOCTL_NOTIF_SEND, &response);
  }
}
#endif

/**
 * @brief In a process of its own (a death test's): from now on the calls
 * @p denied names fail, for this process and every process it starts.
 * Seccomp, no privilege needed; false when the filter cannot be installed.
 */
bool deny(const Denied& denied) {
#ifdef SYS_close_range
  constexpr unsigned WORD = __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ ? 4U : 0U;
  constexpr unsigned NUMBER = offsetof(seccomp_data, nr);
  std::vector<sock_filter> filter = {BPF_STMT(BPF_LD | BPF_W | BPF_ABS, NUMBER)};
  // The call fails with @p error.
  const auto DENY_CALL = [&filter](unsigned call, int error) {
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, call, 0, 1));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | static_cast<__u32>(error)));
  };
  // The call fails with @p error when the low word of its argument @p n
  // equals @p value (@p test BPF_JEQ) or has a bit of it set (BPF_JSET);
  // anything else goes on to the checks below with the call's number loaded
  // again.
  const auto DENY_WHEN = [&filter](unsigned call, unsigned n, unsigned test, unsigned value,
                                   int error) {
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, call, 0, 4));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                              static_cast<__u32>(offsetof(seccomp_data, args) + 8U * n + WORD)));
    filter.push_back(BPF_JUMP(BPF_JMP | test | BPF_K, value, 0, 1));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | static_cast<__u32>(error)));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, NUMBER));
  };
  if (denied.closeRange) {
    DENY_CALL(SYS_close_range, ENOSYS);
  }
  // An open's flags: openat(2)'s third argument, open(2)'s, where it exists, second.
  if (denied.directories) {
    DENY_WHEN(SYS_openat, 2, BPF_JSET, static_cast<unsigned>(O_DIRECTORY), ENOENT);
#ifdef SYS_open
    DENY_WHEN(SYS_open, 1, BPF_JSET, static_cast<unsigned>(O_DIRECTORY), ENOENT);
#endif
  }
  if (denied.directoryReads) {
    DENY_CALL(SYS_getdents64, EPERM);
  }
  if (denied.close >= 0) {
    DENY_WHEN(SYS_close, 0, BPF_JEQ, static_cast<unsigned>(denied.close), EBADF);
  }
  // The new descriptor: dup2(2)'s second argument, and dup3(2)'s.
  if (denied.replace >= 0) {
#ifdef SYS_dup2
    DENY_WHEN(SYS_dup2, 1, BPF_JEQ, static_cast<unsigned>(denied.replace), denied.replaceErrno);
#endif
    DENY_WHEN(SYS_dup3, 1, BPF_JEQ, static_cast<unsigned>(denied.replace), denied.replaceErrno);
  }
  // write(2)'s count is its third argument; wait4(2)'s options its third.
  if (denied.reportWrite) {
    DENY_WHEN(SYS_write, 2, BPF_JEQ, 8U, EINTR);
  }
  if (denied.blockingWait) {
    DENY_WHEN(SYS_wait4, 2, BPF_JEQ, 0U, EINTR);
  }
  if (denied.noHangWait) {
    DENY_WHEN(SYS_wait4, 2, BPF_JSET, static_cast<unsigned>(WNOHANG), EINTR);
  }
  if (denied.stateLook) {
    DENY_CALL(SYS_waitid, EINTR);
  }
  if (denied.execRefused) {
    DENY_CALL(SYS_execve, ENOENT);
  }
  // rt_sigaction(2)'s signal and rt_sigprocmask(2)'s how are their first argument.
  if (denied.stopAction) {
    DENY_WHEN(SYS_rt_sigaction, 0, BPF_JEQ, static_cast<unsigned>(SIGINT), EPERM);
  }
  if (denied.stopUnblock) {
    DENY_WHEN(SYS_rt_sigprocmask, 0, BPF_JEQ, static_cast<unsigned>(SIG_UNBLOCK), EPERM);
  }
  if (denied.readOpen >= 0) {
    DENY_WHEN(SYS_openat, 2, BPF_JEQ, static_cast<unsigned>(denied.readOpen), EACCES);
#ifdef SYS_open
    DENY_WHEN(SYS_open, 1, BPF_JEQ, static_cast<unsigned>(denied.readOpen), EACCES);
#endif
  }
  if (denied.nullOpen) {
    DENY_WHEN(SYS_openat, 2, BPF_JEQ, static_cast<unsigned>(O_WRONLY), EACCES);
#ifdef SYS_open
    DENY_WHEN(SYS_open, 1, BPF_JEQ, static_cast<unsigned>(O_WRONLY), EACCES);
#endif
  }
  // fcntl(2) of one descriptor (its first argument) with one command (its second).
  const auto DENY_FCNTL = [&filter](int fd, unsigned command, int error) {
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_fcntl, 0, 6));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                              static_cast<__u32>(offsetof(seccomp_data, args) + WORD)));
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, static_cast<unsigned>(fd), 0, 3));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                              static_cast<__u32>(offsetof(seccomp_data, args) + 8U + WORD)));
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, command, 0, 1));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | static_cast<__u32>(error)));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, NUMBER));
  };
  if (denied.query >= 0) {
    DENY_FCNTL(denied.query, static_cast<unsigned>(F_GETFD), EPERM);
  }
  if (denied.flagSet >= 0) {
    DENY_FCNTL(denied.flagSet, static_cast<unsigned>(F_SETFD), EPERM);
  }
#ifdef SECCOMP_RET_USER_NOTIF
  // A call held for the listener, whatever its arguments.
  const auto HOLD_CALL = [&filter](unsigned call) {
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, call, 0, 1));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF));
  };
  if (denied.execHeld || denied.keeperLoss != KeeperLoss::NONE) {
    HOLD_CALL(SYS_execve);
  }
  if (denied.kills != KillWatch::NONE) {
    HOLD_CALL(SYS_kill);
  }
  if (denied.reportHeld) {
    HOLD_CALL(SYS_ppoll);
#ifdef SYS_poll
    HOLD_CALL(SYS_poll);
#endif
  }
  // setpgid(2)'s pid and group are its first and second arguments.
  if (denied.joinHeld) {
    constexpr auto PID_ARG = static_cast<__u32>(offsetof(seccomp_data, args) + WORD);
    constexpr auto GROUP_ARG = static_cast<__u32>(offsetof(seccomp_data, args) + 8U + WORD);
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_setpgid, 0, 9));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, PID_ARG));
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 3));
    // Its own join: of a group of its own (0) allowed, of another's held.
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, GROUP_ARG));
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 5, 0));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF));
    // Another's join: to that one's own group (pid and group alike) allowed.
    filter.push_back(BPF_STMT(BPF_MISC | BPF_TAX, 0));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, GROUP_ARG));
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_X, 0, 1, 0));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | static_cast<__u32>(EPERM)));
    filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, NUMBER));
  }
  // dup3(2) and, where it exists, dup2(2) onto `interruptOnce` wait for the
  // listener's answer.
  if (denied.interruptOnce >= 0) {
    std::vector<unsigned> duplicates = {SYS_dup3};
#ifdef SYS_dup2
    duplicates.push_back(SYS_dup2);
#endif
    for (const unsigned CALL : duplicates) {
      filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, CALL, 0, 4));
      filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                                static_cast<__u32>(offsetof(seccomp_data, args) + 8U + WORD)));
      filter.push_back(
          BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, static_cast<unsigned>(denied.interruptOnce), 0, 1));
      filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF));
      filter.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, NUMBER));
    }
  }
#endif
  filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
  const sock_fprog PROGRAM{static_cast<unsigned short>(filter.size()), filter.data()};
  if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
    return false;
  }
  const bool ANSWERED = denied.interruptOnce >= 0 || denied.kills != KillWatch::NONE ||
                        denied.keeperLoss != KeeperLoss::NONE || denied.reportHeld;
  if (!denied.execHeld && !denied.joinHeld && !ANSWERED) {
    return ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &PROGRAM) == 0;
  }
#ifdef SECCOMP_RET_USER_NOTIF
  // The listener stays open until this process exits: a held exec or join
  // waits for an answer that never comes, until then or until it is killed;
  // the other held calls are answered by answerHeldCalls(), on a thread of
  // this process.
  const long LISTENER =
      ::syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_NEW_LISTENER, &PROGRAM);
  if (LISTENER < 0) {
    return false;
  }
  if (ANSWERED) {
#ifdef SECCOMP_USER_NOTIF_FLAG_CONTINUE
    std::thread(answerHeldCalls, static_cast<int>(LISTENER), denied).detach();
#else
    return false;
#endif
  }
  return true;
#else
  return false;
#endif
#else
  (void)denied;
  return false;
#endif
}

/** @brief Why the keeper tests cannot run here, or empty when they can. */
std::string keeperTestsUnavailable(rlim_t softLimit) {
#ifndef SYS_close_range
  return "this system's headers have no close_range(2)";
#endif
  struct rlimit limit{};
  if (::getrlimit(RLIMIT_NOFILE, &limit) != 0 ||
      (limit.rlim_max != RLIM_INFINITY && limit.rlim_max < softLimit)) {
    return "the hard limit on open descriptors is below " + std::to_string(softLimit);
  }
  std::ifstream actions("/proc/sys/kernel/seccomp/actions_avail");
  std::string text;
  std::getline(actions, text);
  if (text.find("errno") == std::string::npos) {
    return "seccomp cannot make a call fail here";
  }
  return "";
}

/** @brief Raise or lower this process's soft limit on open descriptors to @p softLimit. */
bool softDescriptorLimit(rlim_t softLimit) {
  struct rlimit limit{};
  if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    return false;
  }
  limit.rlim_cur = softLimit;
  return ::setrlimit(RLIMIT_NOFILE, &limit) == 0;
}

/**
 * @brief The body of a death test: descriptors run out under a soft limit of
 * @p softLimit but for the two highest, which a probe's anchor then takes
 * (this process's end the higher), so that the probe's next pipe cannot be
 * made and its setup fails (EMFILE) after its keeper started. With
 * @p withoutRange close_range is unavailable (deny()), and with
 * @p unclosableAnchor this process's end of the anchor cannot be closed
 * either, so the keeper never sees end-of-file and cannot close everything
 * it inherits: the setup fails with the keeper's report (EBADF). Prints what
 * the call came to and exits 0 when it returned with that failure and left
 * no child of this process behind, running or unreaped.
 */
[[noreturn]] void setupFailureEndsTheKeeper(rlim_t softLimit, bool withoutRange,
                                            bool unclosableAnchor) {
  const ReadinessContext CTX = ReadinessContext::capture();
  if (!softDescriptorLimit(softLimit)) {
    std::fprintf(stderr, "the soft limit could not be set\n");
    std::_Exit(4);
  }
  std::vector<int> held;
  held.reserve(static_cast<std::size_t>(softLimit));
  for (int fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC); fd >= 0;
       fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC)) {
    held.push_back(fd);
  }
  for (int i = 0; i < 2 && !held.empty(); ++i) {
    ::close(held.back());
    held.pop_back();
  }
  const int ANCHOR_OWN_END = static_cast<int>(softLimit) - 1;
  if (withoutRange && !deny({.close = unclosableAnchor ? ANCHOR_OWN_END : -1})) {
    std::fprintf(stderr, "close_range could not be made unavailable\n");
    std::_Exit(3);
  }
  const auto START = std::chrono::steady_clock::now();
  const ProbeResult RESULT = runBoundedProbe({"/bin/true"}, 5000, CTX);
  const auto ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - START);
  for (const int FD : held) {
    ::close(FD);
  }
  // Children of this process the call left: never reaped, so their pids are
  // still theirs to be signalled by.
  const std::vector<pid_t> LEFT = childProcesses(::getpid());
  for (const pid_t PID : LEFT) {
    (void)::kill(PID, SIGKILL);
    int status = 0;
    (void)::waitpid(PID, &status, 0);
  }
  std::fprintf(stderr, "started=%d errno=%d returned after %lld ms, children left: %zu\n",
               RESULT.started ? 1 : 0, RESULT.spawnErrno, static_cast<long long>(ELAPSED.count()),
               LEFT.size());
  const bool FAILED_AS_EXPECTED =
      !RESULT.started && RESULT.spawnErrno == (unclosableAnchor ? EBADF : EMFILE);
  std::fprintf(stderr, "%s\n",
               !LEFT.empty()         ? "the call left its keeper"
               : !FAILED_AS_EXPECTED ? "the setup did not fail as expected"
                                     : "the keeper ended with the call");
  std::_Exit(LEFT.empty() && FAILED_AS_EXPECTED ? 0 : 1);
}

/**
 * @brief The body of a death test: this process holds a descriptor at
 * @p planted, inherited by whatever it starts, and close_range is
 * unavailable when @p withoutRange. With @p closedStreams (bit 0 stdin, 1
 * stdout, 2 stderr) this process has those standard streams closed, so that
 * the probe's anchor and the keeper's /dev/null may take their numbers; with
 * @p noNull the keeper cannot open /dev/null (deny()). A probe runs a program
 * that lists, once their count settles, the descriptors its process group's
 * leader (the probe's keeper) holds. Prints the list and exits 0 when it is
 * the keeper's stdin, its end of the anchor (a socket), and /dev/null on
 * stdout and stderr (nothing there with @p noNull), nothing else.
 */
[[noreturn]] void keeperHoldsNothingOfThisProcess(int planted, bool withoutRange,
                                                  int closedStreams = 0, bool noNull = false) {
  const ReadinessContext CTX = ReadinessContext::capture();
  // This process's stderr where closing the standard streams leaves it, for what it prints.
  const int OUT = ::fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 100);
  const int NUL = ::open("/dev/null", O_RDONLY);
  if (OUT < 0 || !softDescriptorLimit(static_cast<rlim_t>(planted) + 10) || NUL < 0 ||
      ::dup2(NUL, planted) != planted) {
    std::fprintf(stderr, "the descriptor could not be planted\n");
    std::_Exit(4);
  }
  ::close(NUL);
  for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
    if ((closedStreams & (1 << stream)) != 0) {
      ::close(stream);
    }
  }
  if ((withoutRange || noNull) && !deny({.closeRange = withoutRange, .nullOpen = noNull})) {
    ::dprintf(OUT, "the calls could not be denied\n");
    std::_Exit(3);
  }
  const std::string LIST =
      "PATH=/usr/bin:/bin; read -r s < /proc/$$/stat; set -- $s; k=$5; i=0\n"
      "while :; do\n"
      "  n=0; for f in /proc/$k/fd/*; do n=$((n + 1)); done\n"
      "  [ \"$n\" -le 3 ] || [ \"$i\" -ge 100 ] && break\n"
      "  i=$((i + 1)); sleep 0.02\n"
      "done\n"
      "for f in /proc/$k/fd/*; do printf '%s=%s ' \"${f##*/}\" \"$(readlink \"$f\")\"; done\n";
  const ProbeResult R = runBoundedProbe({"/bin/sh", "-c", LIST}, 10000, CTX);
  ::dprintf(OUT, "the keeper's descriptors: %s\n", R.output.c_str());
  const std::size_t ANCHOR_END = R.output.find("] ");
  const std::string STREAMS = noNull ? "" : "1=/dev/null 2=/dev/null ";
  const bool NOTHING_ELSE = R.succeeded() && R.output.rfind("0=socket:[", 0) == 0 &&
                            ANCHOR_END != std::string::npos &&
                            R.output.substr(ANCHOR_END) == "] " + STREAMS;
  ::dprintf(OUT, NOTHING_ELSE ? "the keeper holds nothing of this process\n"
                              : "the keeper holds more than its own\n");
  std::_Exit(NOTHING_ELSE ? 0 : 1);
}

/**
 * @brief The body of a death test: with each combination of this process's
 * standard streams closed (none, each, and all three; restored after each),
 * a probe runs a program that writes "out" to its stdout, "err" to its
 * stderr, and exits 7 once both writes succeeded, its output merged and
 * separate. Prints what each run read and exits 0 when every run read both
 * lines and the program's status, 7.
 */
[[noreturn]] void programHasItsStreams() {
  const ReadinessContext CTX = ReadinessContext::capture();
  // This process's standard streams, kept above them to restore them and to print with.
  int saved[3] = {-1, -1, -1};
  for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
    saved[stream] = ::fcntl(stream, F_DUPFD_CLOEXEC, 100);
  }
  const int OUT = saved[STDERR_FILENO];
  if (saved[STDIN_FILENO] < 0 || saved[STDOUT_FILENO] < 0 || OUT < 0) {
    std::fprintf(stderr, "the standard streams could not be kept\n");
    std::_Exit(4);
  }
  bool every = true;
  for (int closed = 0; closed < 8; ++closed) {
    for (const ProbeStreams STREAMS : {ProbeStreams::MERGED, ProbeStreams::SEPARATE}) {
      for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
        if ((closed & (1 << stream)) != 0) {
          ::close(stream);
        }
      }
      const ProbeResult R = runBoundedProbe({"/bin/sh", "-c", "echo out; echo err >&2 && exit 7"},
                                            5000, CTX, STREAMS);
      for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
        (void)::dup2(saved[stream], stream);
      }
      const bool SEPARATE = STREAMS == ProbeStreams::SEPARATE;
      const bool BOTH =
          SEPARATE ? R.output == "out\n" && R.errorOutput == "err\n" : R.output == "out\nerr\n";
      const bool AS_IT_SAID = R.started && R.exited && R.exitCode == 7 && BOTH;
      ::dprintf(OUT,
                "closed %d (bit 0 stdin, 1 stdout, 2 stderr), %s: %s; output [%s], errors [%s]\n",
                closed, SEPARATE ? "separate" : "merged", R.describe().c_str(), R.output.c_str(),
                R.errorOutput.c_str());
      every = every && AS_IT_SAID;
    }
  }
  ::dprintf(OUT, every ? "the program had both its streams every time\n"
                       : "the program lost a stream\n");
  std::_Exit(every ? 0 : 1);
}

/**
 * @brief The body of a death test: with each combination of this process's
 * standard streams closed (none, each, and all three; restored after each),
 * an owned helper and a probe, its output merged and separate, are started
 * on a program that does not exist. Prints how each start came out and
 * exits 0 when none started and every one failed with ENOENT, the exec's
 * own errno: where the streams are closed, the descriptor a start reports
 * on may have taken a stream's number.
 */
[[noreturn]] void programThatCannotRunIsReported() {
  const ReadinessContext CTX = ReadinessContext::capture();
  // This process's standard streams, kept above them to restore them and to print with.
  int saved[3] = {-1, -1, -1};
  for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
    saved[stream] = ::fcntl(stream, F_DUPFD_CLOEXEC, 100);
  }
  const int OUT = saved[STDERR_FILENO];
  if (saved[STDIN_FILENO] < 0 || saved[STDOUT_FILENO] < 0 || OUT < 0) {
    std::fprintf(stderr, "the standard streams could not be kept\n");
    std::_Exit(4);
  }
  const std::string ABSENT = "/nonexistent/vernier-program";
  bool every = true;
  for (int closed = 0; closed < 8; ++closed) {
    for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
      if ((closed & (1 << stream)) != 0) {
        ::close(stream);
      }
    }
    OwnedHelper helper(quickPolicy());
    const HelperStart HELPER = helper.start({ABSENT}, "", "", 200, nullptr);
    (void)helper.stop();
    const ProbeResult MERGED = runBoundedProbe({ABSENT}, 2000, CTX);
    const ProbeResult SEPARATE = runBoundedProbe({ABSENT}, 2000, CTX, ProbeStreams::SEPARATE);
    for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
      (void)::dup2(saved[stream], stream);
    }
    const bool REPORTED = !HELPER.started && HELPER.spawnErrno == ENOENT && !MERGED.started &&
                          MERGED.spawnErrno == ENOENT && !SEPARATE.started &&
                          SEPARATE.spawnErrno == ENOENT;
    ::dprintf(OUT,
              "closed %d (bit 0 stdin, 1 stdout, 2 stderr): the helper started=%d errno=%d; the "
              "probe started=%d errno=%d, separate started=%d errno=%d\n",
              closed, HELPER.started ? 1 : 0, HELPER.spawnErrno, MERGED.started ? 1 : 0,
              MERGED.spawnErrno, SEPARATE.started ? 1 : 0, SEPARATE.spawnErrno);
    every = every && REPORTED;
  }
  ::dprintf(OUT, every ? "every start reported the program that cannot run\n"
                       : "a start took a program that cannot run for one that ran\n");
  std::_Exit(every ? 0 : 1);
}

/** @brief How the stream control keeps a program's stream from being set. */
enum class StreamDenial : std::uint8_t {
  FLAGS_UNREADABLE, ///< This process's streams closed; F_GETFD of descriptor 2 refused.
  FLAGS_UNSETTABLE, ///< This process's streams closed; F_SETFD of descriptor 2 refused.
  UNDUPLICABLE      ///< Nothing duplicated onto stdout (dup2, dup3).
};

/**
 * @brief The body of a death test: with @p which denied, an owned helper and
 * a probe, its output merged and separate, start a program that writes
 * "out" to its stdout and "err" to its stderr and exits 7 once both writes
 * succeeded. Where a start cannot set a stream (with this process's streams
 * closed, the merged output's pipe and the helper's error file take
 * descriptor 2, whose close-on-exec cannot then be read or cleared; or
 * nothing can be put on stdout), it must fail with EPERM, the helper with
 * "could not set up /bin/sh"; where it starts, both lines must be read and
 * the status be 7. Prints how each start came out and exits 0 when every one
 * did so, with no child of this process left.
 */
[[noreturn]] void streamThatCannotBeSetFailsTheStart(StreamDenial which) {
  const ReadinessContext CTX = ReadinessContext::capture();
  // This process's stderr, kept where closing the standard streams leaves it, for what it prints.
  const int OUT = ::fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 100);
  if (OUT < 0) {
    std::fprintf(stderr, "stderr could not be kept\n");
    std::_Exit(4);
  }
  Denied denied{.closeRange = false};
  if (which == StreamDenial::UNDUPLICABLE) {
    denied.replace = STDOUT_FILENO;
  } else {
    for (int stream = STDIN_FILENO; stream <= STDERR_FILENO; ++stream) {
      ::close(stream);
    }
    (which == StreamDenial::FLAGS_UNREADABLE ? denied.query : denied.flagSet) = STDERR_FILENO;
  }
  if (!deny(denied)) {
    ::dprintf(OUT, "the calls could not be denied\n");
    std::_Exit(3);
  }
  const std::vector<std::string> PROGRAM = {"/bin/sh", "-c", "echo out; echo err >&2 && exit 7"};
  OwnedHelper helper(quickPolicy());
  const HelperStart HELPER = helper.start(PROGRAM, "", "", 200, nullptr);
  (void)helper.stop();
  const ProbeResult MERGED = runBoundedProbe(PROGRAM, 5000, CTX);
  const ProbeResult SEPARATE = runBoundedProbe(PROGRAM, 5000, CTX, ProbeStreams::SEPARATE);
  const std::vector<pid_t> LEFT = childProcesses(::getpid());
  for (const pid_t PID : LEFT) {
    (void)::kill(PID, SIGKILL);
    int status = 0;
    (void)::waitpid(PID, &status, 0);
  }
  // A start that sets every stream runs the program whole; one that cannot fails with EPERM.
  const auto PROBE_AS_DUE = [](const ProbeResult& r, bool refused, bool separate) {
    if (refused) {
      return !r.started && r.spawnErrno == EPERM;
    }
    const bool BOTH =
        separate ? r.output == "out\n" && r.errorOutput == "err\n" : r.output == "out\nerr\n";
    return r.started && r.exited && r.exitCode == 7 && BOTH;
  };
  // The separate output's pipe is moved above the streams, so no stream's flag is read there.
  const bool SEPARATE_REFUSED = which == StreamDenial::UNDUPLICABLE;
  const bool AS_DUE =
      LEFT.empty() && !HELPER.started && HELPER.spawnErrno == EPERM &&
      HELPER.errorTail == "could not set up /bin/sh: " + std::system_category().message(EPERM) &&
      PROBE_AS_DUE(MERGED, true, false) && PROBE_AS_DUE(SEPARATE, SEPARATE_REFUSED, true);
  ::dprintf(OUT,
            "the helper started=%d errno=%d %s; merged: %s, output [%s]; separate: %s, output "
            "[%s], errors [%s]; children left: %zu\n",
            HELPER.started ? 1 : 0, HELPER.spawnErrno, HELPER.errorTail.c_str(),
            MERGED.describe().c_str(), MERGED.output.c_str(), SEPARATE.describe().c_str(),
            SEPARATE.output.c_str(), SEPARATE.errorOutput.c_str(), LEFT.size());
  ::dprintf(OUT, AS_DUE ? "a stream that could not be set failed the start\n"
                        : "a start went on without a stream it could not set\n");
  std::_Exit(AS_DUE ? 0 : 1);
}

/** @brief What the start control keeps from going on, for every process (deny()). */
enum class StartHazard : std::uint8_t {
  DUPLICATE_INTERRUPTED,      ///< Every duplicate onto stdout interrupted (EINTR).
  REPORT_WRITE_INTERRUPTED,   ///< Every write of a start report interrupted; no program to run.
  EXEC_HELD,                  ///< Every exec held, never answered.
  JOIN_HELD,                  ///< The program's join of the keeper's group held, never answered,
                              ///< and the parent's join of it refused: the probe alone.
  REAP_INTERRUPTED,           ///< Every blocking wait interrupted; no program to run.
  LOOK_INTERRUPTED,           ///< Every waitid interrupted; the keeper refuses (stdout kept).
  DUPLICATE_INTERRUPTED_ONCE, ///< A program's first duplicate onto stdout interrupted.
  NOTHING_DENIED,             ///< Nothing: the starts as they ordinarily go.
  LIVE_ORDINARY,              ///< Nothing; a program that runs past its bound and leaves a
                              ///< child in its group: the probe alone.
  LOOK_INTERRUPTED_LIVE,      ///< Every waitid interrupted; that program: the probe alone.
  LOOK_AND_WAIT_INTERRUPTED,  ///< Every waitid and every wait that does not block
                              ///< interrupted; that program; every kill counted: the probe
                              ///< alone.
  HELPER_LOST_FIRST,          ///< SIGCHLD ignored; a program that cannot run; the start report's
                              ///< read held until no child is left; every kill refused: the
                              ///< helper alone.
  KEEPER_LOST_REPORTED,       ///< SIGCHLD ignored; the keeper ended at the program's exec, then
                              ///< the exec refused; a kill at it refused: the probe alone.
  KEEPER_LOST_UNREPORTED,     ///< As KEEPER_LOST_REPORTED, the exec left held: the probe alone.
  UNREAD_EXEC_HELD,           ///< Every waitid and every wait that does not block interrupted;
                              ///< every exec held; every kill counted.
  UNREAD_EXEC_REFUSED         ///< The same, every exec refused (ENOENT).
};

/**
 * @brief In a child of the death test's process, with @p which denied there
 * alone: an owned helper and a probe are started, and each must end within
 * its bound, its children reaped (the reaps here do not block). Where every
 * duplicate onto stdout is interrupted, both fail with EINTR at once; where
 * every start report's write is interrupted and the program cannot run, the
 * report is lost and both end at once as a program that exited with 127;
 * where every exec is held, both fail with ETIMEDOUT after the start
 * report's bound, 5 s; where the program's join of the keeper's group is held
 * and the parent's join of it refused, the probe alone runs, fails with
 * ETIMEDOUT, and its program, outside the group, is killed itself; where
 * every blocking wait is interrupted and the program cannot run, both fail
 * with ENOENT at once; where every waitid is interrupted, a probe whose
 * keeper refuses (EACCES) ends at once. Where a program's first duplicate
 * onto stdout is interrupted, once in each start, the duplicate is tried
 * again and both complete as they do with nothing denied: the program exits
 * 7, its output and errors captured apart. A program that runs past its
 * bound and leaves a child in its group (the fake helper's "orphan quiet
 * stay", bounded at 300 ms) is ended, that child with it, and no process of
 * either is left running: with nothing denied and with every waitid
 * interrupted, it is killed at its bound and its status read (the wait that
 * does not block decides where the look cannot); with every waitid and
 * every wait that does not block interrupted, its group is ended, one kill
 * aimed at the group and none at the program's pid, whose state cannot be
 * read, and it is reported unreaped, never as reaped by another wait. With
 * SIGCHLD ignored (the system reaps each child as it ends) and every kill
 * held and counted: the owned helper's child that cannot execute its
 * program, reaped before its report is read (that read held until no child
 * is left), fails the start with ENOENT and is sent no signal; the probe
 * whose keeper is ended, and reaped by the system, at its program's exec
 * fails with ENOENT when that exec is then refused, sending no signal, or
 * with ETIMEDOUT after the start report's bound when it is left held,
 * killing its program, still its own, once and nothing by the keeper's pid
 * or group. With every look and every wait that does not block interrupted
 * and every kill counted by what it is aimed at, no child is signalled, its
 * state unread: the exec held, both fail with ETIMEDOUT, the probe's group
 * (the keeper shown running by their link) killed once and the helper's held
 * child left running; the exec refused, both fail with ENOENT and only the
 * probe's keeper is killed, at its end. Prints how each ended and exits 0
 * when every one did so.
 */
[[noreturn]] void startsUnderInterruptions(StartHazard which) {
  const bool LOST = which == StartHazard::HELPER_LOST_FIRST ||
                    which == StartHazard::KEEPER_LOST_REPORTED ||
                    which == StartHazard::KEEPER_LOST_UNREPORTED;
  // The system reaps each child of this process as it ends.
  std::unique_ptr<ChildReaping> reaping;
  if (LOST) {
    reaping = std::make_unique<ChildReaping>(ChildReaping::Policy::IGNORED);
    if (!reaping->ok()) {
      std::fprintf(stderr, "SIGCHLD could not be ignored\n");
      std::_Exit(3);
    }
  }
  const bool LIVE = which == StartHazard::LIVE_ORDINARY ||
                    which == StartHazard::LOOK_INTERRUPTED_LIVE ||
                    which == StartHazard::LOOK_AND_WAIT_INTERRUPTED;
  std::unique_ptr<FakeToolDir> dir;
  if (LIVE) {
    dir = std::make_unique<FakeToolDir>();
    if (!dir->ok()) {
      std::fprintf(stderr, "no fake tool directory\n");
      std::_Exit(3);
    }
  }
  const ReadinessContext CTX = LIVE ? dir->context() : ReadinessContext::capture();
  Denied denied{.closeRange = false};
  const std::vector<std::string> ABSENT = {"/nonexistent/vernier-program"};
  std::vector<std::string> program = {"/bin/sh", "-c", "echo out; echo err >&2 && exit 7"};
  int timeoutMs = 5000;
  if (LIVE) {
    program = {dir->install("fake_helper.sh", "helper"), "orphan", "quiet", "stay"};
    timeoutMs = 300;
  }
  switch (which) {
  case StartHazard::DUPLICATE_INTERRUPTED:
    denied.replace = STDOUT_FILENO;
    denied.replaceErrno = EINTR;
    break;
  case StartHazard::REPORT_WRITE_INTERRUPTED:
    denied.reportWrite = true;
    program = ABSENT;
    break;
  case StartHazard::EXEC_HELD:
    denied.execHeld = true;
    break;
  case StartHazard::JOIN_HELD:
    denied.joinHeld = true;
    break;
  case StartHazard::REAP_INTERRUPTED:
    denied.blockingWait = true;
    program = ABSENT;
    break;
  case StartHazard::LOOK_INTERRUPTED:
    denied = {.closeRange = false, .close = STDOUT_FILENO, .nullOpen = true, .stateLook = true};
    break;
  case StartHazard::DUPLICATE_INTERRUPTED_ONCE:
    denied.interruptOnce = STDOUT_FILENO;
    break;
  case StartHazard::NOTHING_DENIED:
  case StartHazard::LIVE_ORDINARY:
    break;
  case StartHazard::LOOK_INTERRUPTED_LIVE:
    denied.stateLook = true;
    break;
  case StartHazard::LOOK_AND_WAIT_INTERRUPTED:
    denied.stateLook = true;
    denied.noHangWait = true;
    denied.kills = KillWatch::LOST_REFUSED;
    break;
  case StartHazard::HELPER_LOST_FIRST:
    denied.kills = KillWatch::REFUSED;
    denied.reportHeld = true;
    program = ABSENT;
    break;
  case StartHazard::KEEPER_LOST_REPORTED:
    denied.kills = KillWatch::LOST_REFUSED;
    denied.keeperLoss = KeeperLoss::THEN_REFUSED;
    break;
  case StartHazard::KEEPER_LOST_UNREPORTED:
    denied.kills = KillWatch::LOST_REFUSED;
    denied.keeperLoss = KeeperLoss::THEN_HELD;
    break;
  case StartHazard::UNREAD_EXEC_HELD:
  case StartHazard::UNREAD_EXEC_REFUSED:
    denied.stateLook = true;
    denied.noHangWait = true;
    denied.kills = KillWatch::LOST_REFUSED;
    denied.execHeld = which == StartHazard::UNREAD_EXEC_HELD;
    denied.execRefused = which == StartHazard::UNREAD_EXEC_REFUSED;
    break;
  }
  if (!deny(denied)) {
    std::fprintf(stderr, "the calls could not be denied\n");
    std::_Exit(3);
  }
  using Clock = std::chrono::steady_clock;
  const auto MS = [](Clock::time_point since) {
    return static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - since).count());
  };
  auto since = Clock::now();
  HelperStart helper{};
  long long helperMs = 0;
  int helperInterrupted = 0;
  const bool HELPER_RUNS =
      which != StartHazard::LOOK_INTERRUPTED && which != StartHazard::JOIN_HELD && !LIVE &&
      which != StartHazard::KEEPER_LOST_REPORTED && which != StartHazard::KEEPER_LOST_UNREPORTED;
  if (HELPER_RUNS) {
    OwnedHelper owned(quickPolicy());
    helper = owned.start(program, "", "", 200, nullptr);
    helperInterrupted = duplicatesInterrupted.load();
    (void)owned.stop();
    helperMs = MS(since);
  }
  since = Clock::now();
  const ProbeResult PROBE = which == StartHazard::HELPER_LOST_FIRST
                                ? ProbeResult{}
                                : runBoundedProbe(program, timeoutMs, CTX, ProbeStreams::SEPARATE);
  const long long PROBE_MS = MS(since);
  const int PROBE_INTERRUPTED = duplicatesInterrupted.load() - helperInterrupted;
  const std::vector<pid_t> LEFT = childProcesses(::getpid());
  const auto HELPER_EXITED = [&helper](int code) {
    return helper.started && helper.exitedEarly && helper.statusObtained &&
           WIFEXITED(helper.waitStatus) && WEXITSTATUS(helper.waitStatus) == code;
  };
  const bool HELPER_127 = HELPER_EXITED(127);
  const bool BOTH_COMPLETED = HELPER_EXITED(7) && helperMs < 2000 && PROBE.started &&
                              PROBE.exited && PROBE.exitCode == 7 && PROBE.output == "out\n" &&
                              PROBE.errorOutput == "err\n" && PROBE_MS < 2000;
  // The live program's child, and whether any process of the call still runs.
  pid_t left = -1;
  if (LIVE) {
    const std::vector<std::string> LOGGED = dir->logLines("helper child pid=");
    if (!LOGGED.empty()) {
      left = static_cast<pid_t>(std::stol(LOGGED.front().substr(LOGGED.front().find('=') + 1)));
    }
  }
  const bool LEFT_ENDED = left > 0 && ended(left);
  const std::size_t RUNNING_LEFT = static_cast<std::size_t>(
      std::count_if(LEFT.begin(), LEFT.end(), [](pid_t pid) { return !ended(pid); }));
  const bool LIVE_KILLED = PROBE.started && PROBE.timedOut && PROBE.statusObtained &&
                           PROBE_MS < 2500 && LEFT_ENDED && LEFT.empty();
  bool asDue = LEFT.empty();
  switch (which) {
  case StartHazard::DUPLICATE_INTERRUPTED:
    asDue = asDue && !helper.started && helper.spawnErrno == EINTR && helperMs < 2000 &&
            !PROBE.started && PROBE.spawnErrno == EINTR && PROBE_MS < 2000;
    break;
  case StartHazard::REPORT_WRITE_INTERRUPTED:
    asDue = asDue && HELPER_127 && helperMs < 2000 && PROBE.started && PROBE.exited &&
            PROBE.exitCode == 127 && PROBE_MS < 2000;
    break;
  case StartHazard::EXEC_HELD:
    asDue = asDue && !helper.started && helper.spawnErrno == ETIMEDOUT && helperMs >= 4500 &&
            helperMs < 9000 && !PROBE.started && PROBE.spawnErrno == ETIMEDOUT &&
            PROBE_MS >= 4500 && PROBE_MS < 9000;
    break;
  case StartHazard::JOIN_HELD:
    asDue = asDue && !PROBE.started && PROBE.spawnErrno == ETIMEDOUT && PROBE_MS >= 4500 &&
            PROBE_MS < 9000;
    break;
  case StartHazard::REAP_INTERRUPTED:
    asDue = asDue && !helper.started && helper.spawnErrno == ENOENT && helperMs < 2000 &&
            !PROBE.started && PROBE.spawnErrno == ENOENT && PROBE_MS < 2000;
    break;
  case StartHazard::LOOK_INTERRUPTED:
    asDue = asDue && !PROBE.started && PROBE.spawnErrno == EACCES && PROBE_MS < 2000;
    break;
  case StartHazard::DUPLICATE_INTERRUPTED_ONCE:
    asDue = asDue && BOTH_COMPLETED && helperInterrupted == 1 && PROBE_INTERRUPTED == 1;
    break;
  case StartHazard::NOTHING_DENIED:
    asDue = asDue && BOTH_COMPLETED;
    break;
  case StartHazard::LIVE_ORDINARY:
  case StartHazard::LOOK_INTERRUPTED_LIVE:
    asDue = LIVE_KILLED;
    break;
  case StartHazard::LOOK_AND_WAIT_INTERRUPTED:
    // The group is the call's (the keeper shown running by its end of their
    // link) and is killed once; the program, whose state cannot be read, is
    // never signalled by its pid, nor the keeper by its own.
    asDue = PROBE.started && !PROBE.timedOut && !PROBE.statusObtained &&
            PROBE.statusLoss == StatusLoss::NONE && PROBE_MS < 3500 && LEFT_ENDED &&
            RUNNING_LEFT == 0 && killsAtGroup.load() == 1 && killsAtChild.load() == 0 &&
            killsAtLeader.load() == 0;
    break;
  case StartHazard::HELPER_LOST_FIRST:
    asDue = asDue && !helper.started && helper.spawnErrno == ENOENT &&
            helper.errorTail.rfind("could not execute ", 0) == 0 && helperMs < 3000 &&
            killsSeen.load() == 0;
    break;
  case StartHazard::KEEPER_LOST_REPORTED:
    asDue = asDue && keeperEnded.load() > 0 && !PROBE.started && PROBE.spawnErrno == ENOENT &&
            PROBE_MS < 3000 && killsSeen.load() == 0;
    break;
  case StartHazard::KEEPER_LOST_UNREPORTED:
    asDue = asDue && keeperEnded.load() > 0 && !PROBE.started && PROBE.spawnErrno == ETIMEDOUT &&
            PROBE_MS >= 4500 && PROBE_MS < 9000 && killsOnLost.load() == 0 && killsSeen.load() == 1;
    break;
  case StartHazard::UNREAD_EXEC_HELD:
    // No child's state can be read: none is signalled. The probe's group,
    // its id the call's while the keeper's end of their link shows it
    // running, is killed once, the probe's program in it; the helper's held
    // child, in no group of the call's, is left running to this fixture's
    // parent. What was killed stays unreaped (no wait can be read).
    asDue = !helper.started && helper.spawnErrno == ETIMEDOUT &&
            helper.statusLoss == StatusLoss::NONE && helperMs >= 4500 && helperMs < 9000 &&
            !PROBE.started && PROBE.spawnErrno == ETIMEDOUT &&
            PROBE.statusLoss == StatusLoss::NONE && PROBE_MS >= 4500 && PROBE_MS < 9500 &&
            killsAtChild.load() == 0 && killsAtGroup.load() == 1 && killsAtLeader.load() == 0 &&
            RUNNING_LEFT == 1;
    break;
  case StartHazard::UNREAD_EXEC_REFUSED:
    // The children reported their exec failing and ended by themselves:
    // none is signalled; no group is, the program never having run; the
    // keeper, running, is killed at its end. Nothing is left running.
    asDue = !helper.started && helper.spawnErrno == ENOENT &&
            helper.statusLoss == StatusLoss::NONE && helperMs < 3000 && !PROBE.started &&
            PROBE.spawnErrno == ENOENT && PROBE.statusLoss == StatusLoss::NONE && PROBE_MS < 3500 &&
            killsAtChild.load() == 0 && killsAtGroup.load() == 0 && killsAtLeader.load() == 1 &&
            RUNNING_LEFT == 0;
    break;
  }
  std::fprintf(stderr,
               "the helper started=%d errno=%d early=%d %s after %lld ms, %d interrupted; the "
               "probe %s, output [%s] errors [%s] after %lld ms, %d interrupted; children left: "
               "%zu, %zu running; the program's child %d ended=%d; kills %d, at the lost keeper "
               "%d (%d), at a group %d, at a leader %d, at a child %d\n",
               helper.started ? 1 : 0, helper.spawnErrno, helper.exitedEarly ? 1 : 0,
               helper.errorTail.c_str(), helperMs, helperInterrupted, PROBE.describe().c_str(),
               PROBE.output.c_str(), PROBE.errorOutput.c_str(), PROBE_MS, PROBE_INTERRUPTED,
               LEFT.size(), RUNNING_LEFT, static_cast<int>(left), LEFT_ENDED ? 1 : 0,
               killsSeen.load(), killsOnLost.load(), static_cast<int>(keeperEnded.load()),
               killsAtGroup.load(), killsAtLeader.load(), killsAtChild.load());
  dir.reset();
  std::_Exit(asDue ? 0 : 1);
}

/**
 * @brief The body of a death test that supervises startsUnderInterruptions():
 * it runs in a child of this process, which this process ends after 30 s;
 * whatever that child left (a start's child that never ended) comes here,
 * this process being a subreaper, and is ended and reaped before this one
 * exits. Exits 0 when the child exited 0 within the 30 s.
 */
[[noreturn]] void startEndsUnderInterruptions(StartHazard which) {
  using Clock = std::chrono::steady_clock;
  if (::prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
    std::fprintf(stderr, "this process could not be made a subreaper\n");
    std::_Exit(4);
  }
  const pid_t RUNNER = ::fork();
  if (RUNNER < 0) {
    std::fprintf(stderr, "the starts could not be run\n");
    std::_Exit(4);
  }
  if (RUNNER == 0) {
    startsUnderInterruptions(which);
  }
  int status = 0;
  bool ended = false;
  for (const auto UNTIL = Clock::now() + std::chrono::seconds(30); Clock::now() < UNTIL;) {
    if (::waitpid(RUNNER, &status, WNOHANG) == RUNNER) {
      ended = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!ended) {
    (void)::kill(RUNNER, SIGKILL);
  }
  for (const auto UNTIL = Clock::now() + std::chrono::seconds(5); Clock::now() < UNTIL;) {
    const std::vector<pid_t> LEFT = childProcesses(::getpid());
    if (LEFT.empty()) {
      break;
    }
    for (const pid_t PID : LEFT) {
      (void)::kill(PID, SIGKILL);
    }
    int reaped = 0;
    while (::waitpid(-1, &reaped, WNOHANG) > 0) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const bool AS_DUE = ended && WIFEXITED(status) && WEXITSTATUS(status) == 0;
  std::fprintf(stderr, "%s\n",
               AS_DUE  ? "the start ended within its bound"
               : ended ? "the start did not end as it should"
                       : "the start did not end within 30 s: ended with what it left");
  std::_Exit(AS_DUE ? 0 : 1);
}

/** @brief What the setup control keeps a start from setting, for every process (deny()). */
enum class StartSetup : std::uint8_t {
  STOP_ACTION_REFUSED,  ///< SIGINT's default action (sigaction refused).
  STOP_UNBLOCK_REFUSED, ///< The stop signals unblocked (sigprocmask SIG_UNBLOCK refused).
  PROBE_STDIN_UNOPENED, ///< The probe's /dev/null for its program's stdin (its open refused).
  HELPER_STDIN_UNOPENED ///< The owned helper's /dev/null for its stdin (its open refused).
};

/**
 * @brief The body of a death test: with @p which refused, the start fails
 * with that errno before its program runs, and leaves no child: the owned
 * helper and the probe where the stop signals' default action or their
 * unblocking is refused (EPERM, the helper's "could not set up /bin/sh");
 * the probe where its /dev/null cannot be opened, and the helper where its
 * own cannot (EACCES, "could not open /dev/null"). Prints how each start
 * ended and exits 0 when every one did so.
 */
[[noreturn]] void startRefusesWhatItCannotSet(StartSetup which) {
  const ReadinessContext CTX = ReadinessContext::capture();
  Denied denied{.closeRange = false};
  bool helperRuns = true;
  bool probeRuns = true;
  int expected = EPERM;
  switch (which) {
  case StartSetup::STOP_ACTION_REFUSED:
    denied.stopAction = true;
    break;
  case StartSetup::STOP_UNBLOCK_REFUSED:
    denied.stopUnblock = true;
    break;
  case StartSetup::PROBE_STDIN_UNOPENED:
    denied.readOpen = O_RDONLY | O_CLOEXEC;
    helperRuns = false;
    expected = EACCES;
    break;
  case StartSetup::HELPER_STDIN_UNOPENED:
    denied.readOpen = O_RDONLY;
    probeRuns = false;
    expected = EACCES;
    break;
  }
  if (!deny(denied)) {
    std::fprintf(stderr, "the calls could not be denied\n");
    std::_Exit(3);
  }
  const std::vector<std::string> PROGRAM = {"/bin/sh", "-c", "exit 7"};
  HelperStart helper{};
  if (helperRuns) {
    OwnedHelper owned(quickPolicy());
    helper = owned.start(PROGRAM, "", "", 200, nullptr);
    (void)owned.stop();
  }
  ProbeResult probe{};
  if (probeRuns) {
    probe = runBoundedProbe(PROGRAM, 5000, CTX);
  }
  int status = 0;
  errno = 0;
  const bool NO_CHILD = ::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD;
  // The helper names the step that failed and its errno.
  const std::string HELPER_TEXT =
      (which == StartSetup::HELPER_STDIN_UNOPENED ? "could not open /dev/null: "
                                                  : "could not set up /bin/sh: ") +
      std::system_category().message(expected);
  const bool HELPER_AS_DUE = !helperRuns || (!helper.started && helper.spawnErrno == expected &&
                                             helper.errorTail == HELPER_TEXT);
  const bool PROBE_AS_DUE = !probeRuns || (!probe.started && probe.spawnErrno == expected);
  std::fprintf(stderr, "the helper started=%d errno=%d %s; the probe %s; no child: %d\n",
               helper.started ? 1 : 0, helper.spawnErrno, helper.errorTail.c_str(),
               probe.describe().c_str(), NO_CHILD ? 1 : 0);
  const bool AS_DUE = NO_CHILD && HELPER_AS_DUE && PROBE_AS_DUE;
  std::fprintf(stderr, AS_DUE ? "the start refused what it could not set\n"
                              : "a start went on without what it could not set\n");
  std::_Exit(AS_DUE ? 0 : 1);
}

/** @brief Whether this kernel can hold a call for a seccomp listener (user_notif). */
bool callsCanBeHeld() {
#ifdef SECCOMP_RET_USER_NOTIF
  std::ifstream actions("/proc/sys/kernel/seccomp/actions_avail");
  std::string text;
  std::getline(actions, text);
  return text.find("user_notif") != std::string::npos;
#else
  return false;
#endif
}

/** @brief Whether this kernel can let a held call go on (Linux 5.5 and later). */
bool callsCanGoOn() {
#ifdef SECCOMP_USER_NOTIF_FLAG_CONTINUE
  utsname name{};
  if (!callsCanBeHeld() || ::uname(&name) != 0) {
    return false;
  }
  char* rest = nullptr;
  const long MAJOR = std::strtol(name.release, &rest, 10);
  const long MINOR = *rest == '.' ? std::strtol(rest + 1, nullptr, 10) : 0;
  return MAJOR > 5 || (MAJOR == 5 && MINOR >= 5);
#else
  return false;
#endif
}

/** @brief Where the lock control holds its lock, what fails there, and the limits. */
enum class LockCase : std::uint8_t {
  NATIVE,               ///< At 65590; close_range.
  LISTED,               ///< At 65590; no close_range; /proc/self/fd can be listed.
  UNLISTED,             ///< At 65590; neither: no close_range, and no directory can be opened.
  LISTED_HARD_LOWERED,  ///< At 65590; no close_range, the list, and the hard limit lowered too.
  UNREADABLE,           ///< At 65590; no close_range, /proc/self/fd opens but cannot be read;
                        ///< the hard limit lowered too.
  STDIN,                ///< On stdin.
  STDIN_UNREPLACEABLE,  ///< On stdin, onto which nothing can be duplicated (dup2, dup3).
  STDOUT,               ///< On stdout.
  STDOUT_UNREPLACEABLE, ///< On stdout, onto which nothing can be duplicated (by the keeper, or
                        ///< by the program's start).
  STDOUT_UNRELEASABLE,  ///< On stdout, onto which nothing can be duplicated and which cannot
                        ///< be closed.
  STDERR,               ///< On stderr.
  STDERR_UNREPLACEABLE, ///< On stderr, onto which nothing can be duplicated (likewise).
  STDERR_UNRELEASABLE,  ///< On stderr, onto which nothing can be duplicated and which cannot
                        ///< be closed.
  STDOUT_UNKNOWN,       ///< On stdout; the keeper's /dev/null not opened, stdout neither closed
                        ///< nor its flags read; duplicating onto it allowed.
  STDERR_UNKNOWN,       ///< On stderr, likewise.
  NO_NULL,              ///< On stdout; /dev/null cannot be opened for writing.
  NO_NULL_UNCLOSABLE    ///< On stdout; /dev/null cannot be opened for writing, nor stdout closed.
};

/**
 * @brief The body of a death test: this process holds a lock (flock) through a
 * close-on-exec descriptor: at 65590, under a soft limit on descriptors then
 * lowered to 64 (and the hard limit too, with LISTED_HARD_LOWERED and
 * UNREADABLE), close_range unavailable but in NATIVE; or on one of its
 * standard streams, with close_range, and dup2 and dup3 onto it, close of it
 * or the keeper's open of /dev/null failing as @p which says (deny()). A
 * probe's program waits until its keeper's descriptor of that number is no
 * longer the lock (3 s at the most), says it is ready, and once another
 * thread here has released this process's descriptor (closed it), which
 * releases the lock as far as this process goes, tries the lock. Prints how
 * the call came out and exits 0 when the program got the lock, or, where
 * the keeper cannot let go of the lock, when the call refused the setup with
 * the errno the case expects (ENOSYS, EPERM, EACCES) and the lock was free
 * once this process released it; in every case with no child of this
 * process left behind.
 */
[[noreturn]] void keeperReleasesTheCallersLock(LockCase which) {
  constexpr int PLANTED = 65590;
  int slot = PLANTED;
  Denied denied;
  int refusal = 0;
  switch (which) {
  case LockCase::NATIVE:
    denied.closeRange = false;
    break;
  case LockCase::LISTED:
  case LockCase::LISTED_HARD_LOWERED:
    break;
  case LockCase::UNLISTED:
    denied.directories = true;
    refusal = ENOSYS;
    break;
  case LockCase::UNREADABLE:
    denied.directoryReads = true;
    refusal = EPERM;
    break;
  case LockCase::STDIN:
  case LockCase::STDOUT:
  case LockCase::STDERR:
    slot = which == LockCase::STDIN    ? STDIN_FILENO
           : which == LockCase::STDOUT ? STDOUT_FILENO
                                       : STDERR_FILENO;
    denied.closeRange = false;
    break;
  case LockCase::STDIN_UNREPLACEABLE:
    slot = STDIN_FILENO;
    denied = {.closeRange = false, .replace = slot};
    refusal = EPERM;
    break;
  case LockCase::STDOUT_UNREPLACEABLE:
  case LockCase::STDERR_UNREPLACEABLE:
    // The keeper closes the stream it cannot replace; then the program's
    // start cannot set the same stream either, and fails with that errno.
    slot = which == LockCase::STDOUT_UNREPLACEABLE ? STDOUT_FILENO : STDERR_FILENO;
    denied = {.closeRange = false, .replace = slot};
    refusal = EPERM;
    break;
  case LockCase::STDOUT_UNRELEASABLE:
  case LockCase::STDERR_UNRELEASABLE:
    slot = which == LockCase::STDOUT_UNRELEASABLE ? STDOUT_FILENO : STDERR_FILENO;
    denied = {.closeRange = false, .close = slot, .replace = slot};
    refusal = EPERM;
    break;
  case LockCase::STDOUT_UNKNOWN:
  case LockCase::STDERR_UNKNOWN:
    // The keeper alone is refused: duplicating onto the stream stays
    // allowed, so the program's own start could set it, and only the
    // keeper's refusal (the query's errno) keeps the program from running.
    slot = which == LockCase::STDOUT_UNKNOWN ? STDOUT_FILENO : STDERR_FILENO;
    denied = {.closeRange = false, .close = slot, .nullOpen = true, .query = slot};
    refusal = EPERM;
    break;
  case LockCase::NO_NULL:
    slot = STDOUT_FILENO;
    denied = {.closeRange = false, .nullOpen = true};
    break;
  case LockCase::NO_NULL_UNCLOSABLE:
    slot = STDOUT_FILENO;
    denied = {.closeRange = false, .close = slot, .nullOpen = true};
    refusal = EACCES;
    break;
  }
  const bool DENIES = denied.closeRange || denied.close >= 0 || denied.directories ||
                      denied.directoryReads || denied.replace >= 0 || denied.nullOpen ||
                      denied.query >= 0;
  const ReadinessContext CTX = ReadinessContext::capture();
  // This process's stderr, where the lock on stderr leaves it, for what it prints.
  const int OUT =
      slot == STDERR_FILENO ? ::fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 100) : STDERR_FILENO;
  char directory[] = "/tmp/keeper_lock_XXXXXX";
  if (OUT < 0 || ::mkdtemp(directory) == nullptr || !softDescriptorLimit(PLANTED + 10)) {
    std::fprintf(stderr, "the control could not be set up\n");
    std::_Exit(4);
  }
  const std::string FOLDER = directory;
  const std::string LOCK = FOLDER + "/lock";
  const std::string READY = FOLDER + "/ready";
  const std::string RELEASED = FOLDER + "/released";
  const std::string RESULT = FOLDER + "/result";
  const int FD = ::open(LOCK.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (FD < 0 || ::flock(FD, LOCK_EX | LOCK_NB) != 0 || ::dup3(FD, slot, O_CLOEXEC) != slot) {
    ::dprintf(OUT, "the lock could not be planted\n");
    std::_Exit(4);
  }
  ::close(FD);
  if (slot == PLANTED) {
    struct rlimit limit{};
    (void)::getrlimit(RLIMIT_NOFILE, &limit);
    limit.rlim_cur = 64;
    if (which == LockCase::LISTED_HARD_LOWERED || which == LockCase::UNREADABLE) {
      limit.rlim_max = 64;
    }
    if (::setrlimit(RLIMIT_NOFILE, &limit) != 0) {
      ::dprintf(OUT, "the limits could not be lowered\n");
      std::_Exit(4);
    }
  }
  if (DENIES && !deny(denied)) {
    ::dprintf(OUT, "the calls could not be denied\n");
    std::_Exit(3);
  }
  const std::string HELD = "/proc/$k/fd/" + std::to_string(slot);
  const std::string SCRIPT = "PATH=/usr/bin:/bin; read -r s < /proc/$$/stat; set -- $s; k=$5\n"
                             "i=0; while [ \"$(readlink " +
                             HELD + ")\" = '" + LOCK +
                             "' ] && [ $i -lt 300 ]; do sleep 0.01; i=$((i + 1)); done\n"
                             ": > '" +
                             READY +
                             "'\n"
                             "i=0; while [ ! -e '" +
                             RELEASED +
                             "' ] && [ $i -lt 300 ]; do sleep 0.01; i=$((i + 1)); done\n"
                             "flock -n '" +
                             LOCK + "' true; e=$?; echo \"lock_acquire_status=$e\" > '" + RESULT +
                             "'; exit $e\n";
  std::atomic<bool> returned{false};
  std::thread release([&] {
    const auto UNTIL = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (::access(READY.c_str(), F_OK) != 0 && !returned.load() &&
           std::chrono::steady_clock::now() < UNTIL) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Where close(2) of it fails, as for the keeper, close_range(2) releases it.
    if (::close(slot) != 0 && errno == EBADF) {
      (void)::syscall(SYS_close_range, static_cast<unsigned>(slot), static_cast<unsigned>(slot),
                      0U);
    }
    const int MARK = ::open(RELEASED.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
    if (MARK >= 0) {
      ::close(MARK);
    }
  });
  const ProbeResult R = runBoundedProbe({"/bin/sh", "-c", SCRIPT}, 10000, CTX);
  returned = true;
  release.join();
  const std::vector<pid_t> LEFT = childProcesses(::getpid());
  for (const pid_t PID : LEFT) {
    (void)::kill(PID, SIGKILL);
    int status = 0;
    (void)::waitpid(PID, &status, 0);
  }
  // Once this process released its descriptor, nothing else holds the lock.
  const int AGAIN = ::open(LOCK.c_str(), O_RDWR | O_CLOEXEC);
  const bool FREE = AGAIN >= 0 && ::flock(AGAIN, LOCK_EX | LOCK_NB) == 0;
  if (AGAIN >= 0) {
    ::close(AGAIN);
  }
  std::string result;
  std::getline(std::ifstream(RESULT), result);
  for (const std::string& path : {LOCK, READY, RELEASED, RESULT}) {
    (void)::unlink(path.c_str());
  }
  (void)::rmdir(FOLDER.c_str());
  ::dprintf(
      OUT, "started=%d errno=%d %s; the program: %s; children left: %zu; the lock free after: %d\n",
      R.started ? 1 : 0, R.spawnErrno, R.describe().c_str(), result.c_str(), LEFT.size(),
      FREE ? 1 : 0);
  const char* wrong = nullptr;
  if (!LEFT.empty()) {
    wrong = "a child of this process was left behind";
  } else if (refusal != 0) {
    if (R.started) {
      wrong = "the probe started where its keeper could not let go of what it inherits";
    } else if (R.spawnErrno != refusal) {
      wrong = "the setup failed for another reason";
    } else if (!FREE) {
      wrong = "the lock was held once this process released it";
    }
  } else if (!R.succeeded() || result != "lock_acquire_status=0") {
    wrong = "the program did not get the lock the caller released";
  }
  ::dprintf(OUT, "%s\n",
            wrong == nullptr ? "the caller's lock was the caller's to release" : wrong);
  std::_Exit(wrong == nullptr ? 0 : 1);
}

} // namespace

/**
 * @test The keeper of a probe whose setup fails after the keeper started (its
 * first pipe cannot be made: descriptors ran out) is ended and reaped before
 * the call returns, wherever the descriptors lie and whether or not the system
 * has close_range: with descriptors below 64 and above 65535, with close_range
 * and without it, and with this process's end of the anchor that no process
 * can close, so that the keeper never sees end-of-file (and reports that it
 * could not close everything). Each case in a process of its own.
 */
TEST(ReadinessProbes, KeeperEndsWhenTheProbeCannotBeSetUp) {
  constexpr rlim_t LOW = 64;
  constexpr rlim_t HIGH = 65540;
  const std::string UNAVAILABLE = keeperTestsUnavailable(HIGH);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(setupFailureEndsTheKeeper(LOW, false, false), ::testing::ExitedWithCode(0),
              "the keeper ended with the call")
      << "descriptors below 64, close_range";
  EXPECT_EXIT(setupFailureEndsTheKeeper(LOW, true, false), ::testing::ExitedWithCode(0),
              "the keeper ended with the call")
      << "descriptors below 64, no close_range";
  EXPECT_EXIT(setupFailureEndsTheKeeper(HIGH, false, false), ::testing::ExitedWithCode(0),
              "the keeper ended with the call")
      << "descriptors above 65535, close_range";
  EXPECT_EXIT(setupFailureEndsTheKeeper(HIGH, true, false), ::testing::ExitedWithCode(0),
              "the keeper ended with the call")
      << "descriptors above 65535, no close_range";
  EXPECT_EXIT(setupFailureEndsTheKeeper(HIGH, true, true), ::testing::ExitedWithCode(0),
              "the keeper ended with the call")
      << "descriptors above 65535, no close_range, the anchor unclosable";
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A lock this process releases is released for the probe too, whatever
 * the keeper can use to close what it inherits: the lock's descriptor lies at
 * 65590, opened before the soft limit was lowered to 64. With close_range, and
 * without it but with /proc/self/fd to list (also with the hard limit lowered
 * to 64, which only the list reaches past), the keeper drops it and the
 * program gets the lock; with neither, the setup is refused (ENOSYS) before
 * anything starts; with the list opening but not readable (and the hard limit
 * lowered), the setup fails with the keeper's EPERM and no keeper is left;
 * refused, the lock is free once this process releases it. Each case in a
 * process of its own.
 */
TEST(ReadinessProbes, KeeperNeverHoldsALockTheCallerReleased) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(65600);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  if (::access("/usr/bin/flock", X_OK) != 0) {
    GTEST_SKIP() << "flock(1) is not installed";
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(keeperReleasesTheCallersLock(LockCase::NATIVE), ::testing::ExitedWithCode(0),
              "the caller's lock was the caller's to release")
      << "close_range";
  EXPECT_EXIT(keeperReleasesTheCallersLock(LockCase::LISTED), ::testing::ExitedWithCode(0),
              "the caller's lock was the caller's to release")
      << "no close_range, the list";
  EXPECT_EXIT(keeperReleasesTheCallersLock(LockCase::UNLISTED), ::testing::ExitedWithCode(0),
              "the caller's lock was the caller's to release")
      << "no close_range, no list";
  EXPECT_EXIT(keeperReleasesTheCallersLock(LockCase::LISTED_HARD_LOWERED),
              ::testing::ExitedWithCode(0), "the caller's lock was the caller's to release")
      << "no close_range, the list, the hard limit lowered";
  EXPECT_EXIT(keeperReleasesTheCallersLock(LockCase::UNREADABLE), ::testing::ExitedWithCode(0),
              "the caller's lock was the caller's to release")
      << "no close_range, the list opens but cannot be read, the hard limit lowered";
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test The keeper lets go of this process's standard streams, or the setup
 * fails before the program starts: a lock this process holds through its
 * stdin, stdout or stderr (close-on-exec) is released for the probe once
 * this process releases its copy. Natively the keeper puts its end of the
 * anchor on stdin and /dev/null on stdout and stderr; where /dev/null cannot
 * be opened, the keeper closes them and the program gets the lock. Where
 * nothing can be put on stdout or stderr (dup2 and dup3 onto it fail, for
 * every process), the keeper closes it, and the program's start, which
 * cannot set that stream either, fails. Where its end of the anchor cannot
 * be put on stdin, or a stream can be neither replaced nor closed, or
 * whether it is closed cannot be read (its F_GETFD refused), the setup fails
 * before the program starts; where only the keeper is refused (its
 * /dev/null, its close and its query of the stream), duplicating onto the
 * stream staying allowed, it is the keeper's refusal, with the query's
 * errno, that keeps the program from running. A setup or start that fails
 * does so with the errno of the step that failed (EPERM, EACCES), no child
 * is left, and the lock is free once this process releases its copy. Each
 * case in a process of its own.
 */
TEST(ReadinessProbes, KeeperLetsGoOfTheStandardStreams) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(65600);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  if (::access("/usr/bin/flock", X_OK) != 0) {
    GTEST_SKIP() << "flock(1) is not installed";
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const struct {
    LockCase which;
    const char* name;
  } CASES[] = {
      {LockCase::STDIN, "stdin"},
      {LockCase::STDIN_UNREPLACEABLE, "stdin, the anchor not put there: refused (EPERM)"},
      {LockCase::STDOUT, "stdout"},
      {LockCase::STDOUT_UNREPLACEABLE,
       "stdout, nothing put there: closed, then the start refused (EPERM)"},
      {LockCase::STDOUT_UNRELEASABLE, "stdout, neither replaced nor closed: refused (EPERM)"},
      {LockCase::STDERR, "stderr"},
      {LockCase::STDERR_UNREPLACEABLE,
       "stderr, nothing put there: closed, then the start refused (EPERM)"},
      {LockCase::STDERR_UNRELEASABLE, "stderr, neither replaced nor closed: refused (EPERM)"},
      {LockCase::STDOUT_UNKNOWN,
       "stdout, the keeper's /dev/null not opened, not closed, unreadable: refused (EPERM)"},
      {LockCase::STDERR_UNKNOWN,
       "stderr, the keeper's /dev/null not opened, not closed, unreadable: refused (EPERM)"},
      {LockCase::NO_NULL, "stdout, /dev/null not opened: closed"},
      {LockCase::NO_NULL_UNCLOSABLE, "stdout, /dev/null not opened, not closed: refused (EACCES)"},
  };
  for (const auto& c : CASES) {
    EXPECT_EXIT(keeperReleasesTheCallersLock(c.which), ::testing::ExitedWithCode(0),
                "the caller's lock was the caller's to release")
        << c.name;
  }
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test The keeper holds nothing of the benchmark's while the probe runs: a
 * descriptor of this process's above 65535 is not among its descriptors,
 * with close_range and without it; where this process has standard streams
 * closed, so that the anchor and /dev/null may take their numbers, the
 * keeper still has its end of the anchor on stdin and /dev/null on stdout
 * and stderr; where /dev/null cannot be opened, it holds its end of the
 * anchor alone. Each case in a process of its own.
 */
TEST(ReadinessProbes, KeeperHoldsNothingOfThisProcess) {
  constexpr int PLANTED = 65590;
  const std::string UNAVAILABLE = keeperTestsUnavailable(PLANTED + 10);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  constexpr int STDIN = 1 << STDIN_FILENO;
  constexpr int STDOUT = 1 << STDOUT_FILENO;
  constexpr int STDERR = 1 << STDERR_FILENO;
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(keeperHoldsNothingOfThisProcess(PLANTED, false), ::testing::ExitedWithCode(0),
              "the keeper holds nothing of this process")
      << "close_range";
  EXPECT_EXIT(keeperHoldsNothingOfThisProcess(PLANTED, true), ::testing::ExitedWithCode(0),
              "the keeper holds nothing of this process")
      << "no close_range";
  const struct {
    int closed;
    const char* name;
  } CLOSED[] = {{STDIN, "stdin closed"},
                {STDOUT, "stdout closed"},
                {STDERR, "stderr closed"},
                {STDIN | STDOUT | STDERR, "the three closed"}};
  for (const auto& c : CLOSED) {
    EXPECT_EXIT(keeperHoldsNothingOfThisProcess(PLANTED, false, c.closed),
                ::testing::ExitedWithCode(0), "the keeper holds nothing of this process")
        << c.name;
  }
  EXPECT_EXIT(keeperHoldsNothingOfThisProcess(PLANTED, false, 0, true),
              ::testing::ExitedWithCode(0), "the keeper holds nothing of this process")
      << "/dev/null not opened";
  GTEST_FLAG_SET(death_test_style, SAVED);
}
/**
 * @test The program has its stdout and its stderr whatever of this process's
 * standard streams are closed: with none, each and all three closed, merged
 * and separate, both lines it writes are read and its status is its own.
 * Where this process's streams are closed, a pipe the probe makes may take
 * the number of the stream it becomes. In a process of its own.
 */
TEST(ReadinessProbes, ProgramHasItsStreamsWhateverThisProcessClosed) {
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(programHasItsStreams(), ::testing::ExitedWithCode(0),
              "the program had both its streams every time");
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A start reports a program that cannot run whatever of this
 * process's standard streams are closed: with none, each and all three
 * closed, an owned helper and a probe (merged and separate) started on a
 * program that does not exist report it unstarted, with ENOENT. Where this
 * process's streams are closed, the descriptor a start reports on may have
 * taken a stream's number; the child moves it above them before it sets
 * any. In a process of its own.
 */
TEST(ReadinessProbes, ProgramThatCannotRunIsReportedWhateverThisProcessClosed) {
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(programThatCannotRunIsReported(), ::testing::ExitedWithCode(0),
              "every start reported the program that cannot run");
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A stream a start cannot set fails the start, with the errno of the
 * step that failed, before the program runs: with this process's streams
 * closed and descriptor 2's F_GETFD or F_SETFD refused, the owned helper's
 * error file and the merged probe output's pipe, which already are
 * descriptor 2, cannot be kept open across the exec; with nothing
 * duplicated onto stdout, no start can set it. The helper and the probe
 * fail with EPERM there and leave no child, the helper with "could not set
 * up /bin/sh"; the separate probe, whose pipe is moved above the streams,
 * runs whole where its streams can be set. Each case in a process of its
 * own.
 */
TEST(ReadinessProbes, StreamThatCannotBeSetFailsTheStart) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(64);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(streamThatCannotBeSetFailsTheStart(StreamDenial::FLAGS_UNREADABLE),
              ::testing::ExitedWithCode(0), "a stream that could not be set failed the start")
      << "descriptor 2's F_GETFD refused";
  EXPECT_EXIT(streamThatCannotBeSetFailsTheStart(StreamDenial::FLAGS_UNSETTABLE),
              ::testing::ExitedWithCode(0), "a stream that could not be set failed the start")
      << "descriptor 2's F_SETFD refused";
  EXPECT_EXIT(streamThatCannotBeSetFailsTheStart(StreamDenial::UNDUPLICABLE),
              ::testing::ExitedWithCode(0), "a stream that could not be set failed the start")
      << "nothing duplicated onto stdout";
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A start ends within its bound whatever keeps its calls from going
 * on, its children reaped: with every duplicate onto stdout interrupted
 * (EINTR), the owned helper and the probe fail with EINTR at once, the
 * duplicate tried a few times; with every start report's write interrupted
 * and a program that cannot run, the report is lost and both end at once as
 * a program that exited with 127; with every exec held, both fail with
 * ETIMEDOUT after the start report's bound (5 s), the held child ended; with
 * the program's join of the keeper's group held and the parent's join of it
 * refused, the probe fails with ETIMEDOUT and its program, never in the
 * group, is ended itself; with every blocking wait interrupted and a program
 * that cannot run, both fail with ENOENT at once, the child reaped without
 * blocking; with every waitid interrupted, a probe whose keeper refuses ends
 * at once. With a program's first duplicate onto stdout interrupted, once in
 * each start, both complete as they do with nothing denied: exit 7, the
 * program's output and errors captured apart. Each case in a process of its
 * own, which its parent ends after 30 s with whatever it left.
 */
TEST(ReadinessProbes, StartEndsWithinItsBound) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(64);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const struct {
    StartHazard which;
    const char* name;
  } CASES[] = {
      {StartHazard::DUPLICATE_INTERRUPTED, "every duplicate onto stdout interrupted"},
      {StartHazard::REPORT_WRITE_INTERRUPTED, "every start report's write interrupted"},
      {StartHazard::EXEC_HELD, "every exec held"},
      {StartHazard::JOIN_HELD, "the program's join held, the parent's refused"},
      {StartHazard::REAP_INTERRUPTED, "every blocking wait interrupted"},
      {StartHazard::LOOK_INTERRUPTED, "every waitid interrupted"},
      {StartHazard::DUPLICATE_INTERRUPTED_ONCE,
       "a program's first duplicate onto stdout interrupted"},
      {StartHazard::NOTHING_DENIED, "nothing denied"},
  };
  for (const auto& c : CASES) {
    if ((c.which == StartHazard::EXEC_HELD || c.which == StartHazard::JOIN_HELD) &&
        !callsCanBeHeld()) {
      std::fprintf(stderr, "this kernel cannot hold a call for a listener: %s not run\n", c.name);
      continue;
    }
    if (c.which == StartHazard::DUPLICATE_INTERRUPTED_ONCE && !callsCanGoOn()) {
      std::fprintf(stderr, "this kernel cannot let a held call go on: %s not run\n", c.name);
      continue;
    }
    EXPECT_EXIT(startEndsUnderInterruptions(c.which), ::testing::ExitedWithCode(0),
                "the start ended within its bound")
        << c.name;
  }
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A program whose state cannot be read is not taken for one reaped
 * elsewhere, and still ends with the probe. A program that runs past its
 * bound and leaves a child in its group (the fake helper's "orphan quiet
 * stay", bounded at 300 ms) is killed at its bound and its status read with
 * nothing denied and with every waitid interrupted, where a wait that does
 * not block says that it runs and reaps it once it is killed. With every
 * waitid and every wait that does not block interrupted, its group is ended
 * while the keeper's end of their link shows the keeper running, with every
 * kill counted by what it is aimed at: one at the group, none at the
 * program's pid, whose state cannot be read, none at the keeper's own; and
 * the result says that it was not reaped within the bound, never that
 * another wait reaped it. No process of the program or of its child is left
 * running. Each case in a process of its own, which its parent ends after
 * 30 s with whatever it left.
 */
TEST(ReadinessProbes, ProgramWhoseStateCannotBeReadStillEnds) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(64);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const struct {
    StartHazard which;
    const char* name;
  } CASES[] = {
      {StartHazard::LIVE_ORDINARY, "nothing denied"},
      {StartHazard::LOOK_INTERRUPTED_LIVE, "every waitid interrupted"},
      {StartHazard::LOOK_AND_WAIT_INTERRUPTED,
       "every waitid and every wait that does not block interrupted"},
  };
  for (const auto& c : CASES) {
    if (c.which == StartHazard::LOOK_AND_WAIT_INTERRUPTED && !callsCanGoOn()) {
      std::fprintf(stderr, "this kernel cannot let a held call go on: %s not run\n", c.name);
      continue;
    }
    EXPECT_EXIT(startEndsUnderInterruptions(c.which), ::testing::ExitedWithCode(0),
                "the start ended within its bound")
        << c.name;
  }
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A failed start signals nothing it may have lost, and still ends what
 * is its own. With SIGCHLD ignored, so that the system reaps each child as it
 * ends, and every kill(2) held, counted and, when aimed at what was lost,
 * refused: the owned helper's child that cannot execute its program, reaped
 * before its report is read, fails the start with ENOENT and is sent no
 * signal; the probe's keeper, ended and reaped at the program's exec, leaves
 * a start that fails with ENOENT (the exec then refused) and sends no signal,
 * or with ETIMEDOUT after 5 s (the exec left held), its program, still its
 * own, killed once and nothing sent to the keeper's pid or group. With
 * SIGCHLD at its default action and every look and every wait that does not
 * block interrupted, no child's state can be read and no child is signalled:
 * the exec held, both starts fail with ETIMEDOUT, the probe's group, its id
 * the call's while the keeper is shown running, killed once with the
 * program in it, the helper's held child left running; the exec refused,
 * both fail with ENOENT, the children ending by themselves, the keeper
 * killed at its end. Neither result names a loss. Each case in a process of
 * its own, which its parent ends after 30 s with whatever it left.
 */
TEST(ReadinessProbes, FailedStartSignalsOnlyWhatItOwns) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(64);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  if (!callsCanGoOn()) {
    GTEST_SKIP() << "this kernel cannot let a held call go on";
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const struct {
    StartHazard which;
    const char* name;
  } CASES[] = {
      {StartHazard::HELPER_LOST_FIRST, "the helper's child reaped before its report is read"},
      {StartHazard::KEEPER_LOST_REPORTED, "the keeper reaped, then the exec refused"},
      {StartHazard::KEEPER_LOST_UNREPORTED, "the keeper reaped, the exec left held"},
      {StartHazard::UNREAD_EXEC_HELD, "no state readable, the exec held"},
      {StartHazard::UNREAD_EXEC_REFUSED, "no state readable, the exec refused"},
  };
  for (const auto& c : CASES) {
    EXPECT_EXIT(startEndsUnderInterruptions(c.which), ::testing::ExitedWithCode(0),
                "the start ended within its bound")
        << c.name;
  }
  GTEST_FLAG_SET(death_test_style, SAVED);
}

/**
 * @test A start that cannot set what its program needs fails with that
 * errno before the program runs, and leaves no child: with SIGINT's default
 * action or the stop signals' unblocking refused, the owned helper and the
 * probe fail with EPERM, the helper with "could not set up /bin/sh"; with
 * the probe's /dev/null for its program's stdin unopenable, the probe fails
 * with EACCES, and with the helper's own, the helper fails with EACCES,
 * "could not open /dev/null". Each case in a process of its own.
 */
TEST(ReadinessProbes, StartRefusesWhatItCannotSet) {
  const std::string UNAVAILABLE = keeperTestsUnavailable(64);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }
  const std::string SAVED = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const struct {
    StartSetup which;
    const char* name;
  } CASES[] = {
      {StartSetup::STOP_ACTION_REFUSED, "SIGINT's default action refused"},
      {StartSetup::STOP_UNBLOCK_REFUSED, "the stop signals' unblocking refused"},
      {StartSetup::PROBE_STDIN_UNOPENED, "the probe's /dev/null for stdin unopenable"},
      {StartSetup::HELPER_STDIN_UNOPENED, "the helper's /dev/null for stdin unopenable"},
  };
  for (const auto& c : CASES) {
    EXPECT_EXIT(startRefusesWhatItCannotSet(c.which), ::testing::ExitedWithCode(0),
                "the start refused what it could not set")
        << c.name;
  }
  GTEST_FLAG_SET(death_test_style, SAVED);
}
#endif

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

/**
 * @test A helper takes the stop signals at their default action whatever this
 * process inherited: started while this process ignores SIGINT and blocks
 * SIGTERM, it neither ignores nor blocks them, and it stops on SIGINT.
 */
TEST(ReadinessOwnedHelper, StopsOnInterruptThatThisProcessIgnores) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  OwnedHelper helper(quickPolicy());
  HelperStart start;
  {
    const StopSignalsInherited INHERITED;
    start = helper.start({HELPER, "run"}, "", dir.path() + "/err.txt", 200, &CTX);
  }
  ASSERT_TRUE(start.running()) << start.errorTail;
  std::ifstream statusFile("/proc/" + std::to_string(helper.pid()) + "/status");
  const std::string STATUS((std::istreambuf_iterator<char>(statusFile)),
                           std::istreambuf_iterator<char>());
  EXPECT_EQ(statusMask(STATUS, "SigIgn:") & bitOf(SIGINT), 0ULL) << STATUS;
  EXPECT_EQ(statusMask(STATUS, "SigBlk:") & bitOf(SIGTERM), 0ULL) << STATUS;
  const HelperStopResult STOP = helper.stop();
  EXPECT_TRUE(STOP.reaped);
  EXPECT_EQ(STOP.stoppedBy, SIGINT);
  EXPECT_TRUE(WIFSIGNALED(STOP.waitStatus) && WTERMSIG(STOP.waitStatus) == SIGINT)
      << "wait status " << STOP.waitStatus;
}

/**
 * @test A helper reaped elsewhere, as an application that ignores SIGCHLD or
 * sets SA_NOCLDWAIT reaps it, leaves no exit status to read: its stop, or its
 * start when it ended within the grace (here with status 0), says the status
 * was not read and why, holds HELPER_STATUS_UNAVAILABLE, does not pass for a
 * clean exit, and once the helper is gone it is never signalled again. Reaped
 * here, with SIGCHLD at its default action, its status is read.
 */
TEST(ReadinessOwnedHelper, StatusReapedElsewhereIsNotACleanExit) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string HELPER = dir.install("fake_helper.sh", "helper");
  const ReadinessContext CTX = dir.context();
  using Policy = ChildReaping::Policy;
  for (const Policy POLICY : {Policy::IGNORED, Policy::NO_CHILD_WAIT, Policy::DEFAULT}) {
    const std::string NAME = POLICY == Policy::IGNORED         ? "SIGCHLD ignored"
                             : POLICY == Policy::NO_CHILD_WAIT ? "SA_NOCLDWAIT"
                                                               : "the default action";
    const StatusLoss LOSS = POLICY == Policy::IGNORED         ? StatusLoss::SIGCHLD_IGNORED
                            : POLICY == Policy::NO_CHILD_WAIT ? StatusLoss::NO_CHILD_WAIT
                                                              : StatusLoss::NONE;
    OwnedHelper helper(quickPolicy());
    OwnedHelper early(quickPolicy());
    HelperStart start;
    HelperStopResult stop;
    HelperStart ended;
    HelperStopResult afterEnd;
    {
      const ChildReaping REAPING(POLICY);
      ASSERT_TRUE(REAPING.ok());
      start = helper.start({HELPER, "run"}, "", dir.path() + "/err.txt", 200, &CTX);
      ASSERT_TRUE(start.running()) << NAME << ": " << start.errorTail;
      stop = helper.stop();
      ended = early.start({HELPER, "exit", "0"}, "", dir.path() + "/early.txt", 2000, &CTX);
      afterEnd = early.stop();
    }
    EXPECT_TRUE(stop.wasRunning) << NAME;
    EXPECT_TRUE(stop.reaped) << NAME;
    EXPECT_FALSE(stop.stillAlive) << NAME;
    EXPECT_EQ(stop.stoppedBy, SIGINT) << NAME;
    ASSERT_EQ(stop.deliveries.size(), 1U) << NAME << ": a helper gone is never signalled again";
    ASSERT_TRUE(ended.exitedEarly) << NAME;
    EXPECT_FALSE(afterEnd.wasRunning) << NAME;
    EXPECT_TRUE(afterEnd.deliveries.empty()) << NAME << ": a helper gone is never signalled";
    EXPECT_EQ(stop.statusLoss, LOSS) << NAME;
    EXPECT_EQ(ended.statusLoss, LOSS) << NAME;
    EXPECT_EQ(afterEnd.statusLoss, LOSS) << NAME;
    if (POLICY == Policy::DEFAULT) {
      // Reaped here: each status is read, the SIGINT that ended the one and
      // the other's exit with status 0.
      EXPECT_TRUE(stop.statusObtained) << NAME;
      EXPECT_TRUE(WIFSIGNALED(stop.waitStatus) && WTERMSIG(stop.waitStatus) == SIGINT)
          << NAME << ": wait status " << stop.waitStatus;
      for (const auto& [OBTAINED, STATUS] :
           {std::pair{ended.statusObtained, ended.waitStatus},
            std::pair{afterEnd.statusObtained, afterEnd.waitStatus}}) {
        EXPECT_TRUE(OBTAINED) << NAME;
        EXPECT_TRUE(WIFEXITED(STATUS) && WEXITSTATUS(STATUS) == 0)
            << NAME << ": wait status " << STATUS;
      }
      continue;
    }
    for (const auto& [OBTAINED, STATUS] :
         {std::pair{stop.statusObtained, stop.waitStatus},
          std::pair{ended.statusObtained, ended.waitStatus},
          std::pair{afterEnd.statusObtained, afterEnd.waitStatus}}) {
      EXPECT_FALSE(OBTAINED) << NAME;
      EXPECT_EQ(STATUS, HELPER_STATUS_UNAVAILABLE) << NAME;
      EXPECT_FALSE(WIFEXITED(STATUS) && WEXITSTATUS(STATUS) == 0)
          << NAME << ": an exit status not read passes for a clean exit";
    }
  }
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

/**
 * @test On the sudo route, a `sudo -n kill` reaped elsewhere (this process
 * ignores SIGCHLD) shows neither a delivery nor a refusal: the stop records
 * it as not known, and still waits for the helper as after a delivery, so a
 * SIGINT that did arrive gets its flush. With SIGCHLD at its default action
 * the delivery is proven. A refused SIGINT under the same loss is not known
 * either: the stop waits its bounded time, then goes on to SIGTERM.
 */
TEST(ReadinessOwnedHelper, SudoDeliveryNotSeenIsWaitedOn) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
  const std::string KILL = dir.install("fake_kill.sh", "kill");
  const std::string FLUSH =
      "trap 'sleep 0.2; echo flushed > \"$0\"; exit 0' INT; while :; do sleep 0.05; done";
  using Policy = ChildReaping::Policy;
  for (const Policy POLICY : {Policy::IGNORED, Policy::DEFAULT}) {
    for (const bool REFUSE_INT : {false, true}) {
      if (REFUSE_INT && POLICY == Policy::DEFAULT) {
        continue; // RefusedDeliveryIsReported holds the refusal seen
      }
      const std::string NAME =
          std::string(POLICY == Policy::IGNORED ? "SIGCHLD ignored" : "the default action") +
          (REFUSE_INT ? ", SIGINT refused" : "");
      const std::string FLUSHED = dir.path() + "/flushed" + (REFUSE_INT ? "-refused" : "") +
                                  (POLICY == Policy::IGNORED ? "-ignored" : "-default");
      auto ctx = std::make_shared<const ReadinessContext>(
          REFUSE_INT ? dir.context({{"FAKE_SUDO_DENY", "kill -2"}}) : dir.context());
      HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
      policy.interruptWaitMs = 3000;
      policy.terminateWaitMs = 3000;
      policy.sudoPath = SUDO;
      policy.killPath = KILL;
      policy.context = ctx;
      OwnedHelper helper(policy);
      // Started with SIGCHLD at its default action, so the helper's own
      // shell reaps its children as it always does.
      ASSERT_TRUE(
          helper.start({SUDO, "-n", "--", "/bin/sh", "-c", FLUSH, FLUSHED}, "", "", 300, ctx.get())
              .running())
          << NAME;
      HelperStopResult stop;
      std::chrono::steady_clock::duration took{};
      {
        const ChildReaping REAPING(POLICY);
        ASSERT_TRUE(REAPING.ok());
        const auto START = std::chrono::steady_clock::now();
        stop = helper.stop();
        took = std::chrono::steady_clock::now() - START;
      }
      EXPECT_TRUE(stop.wasRunning) << NAME;
      EXPECT_TRUE(stop.reaped) << NAME;
      EXPECT_FALSE(stop.stillAlive) << NAME;
      if (REFUSE_INT) {
        ASSERT_EQ(stop.deliveries.size(), 2U) << NAME;
        EXPECT_EQ(stop.stoppedBy, SIGTERM) << NAME;
        EXPECT_GE(took, std::chrono::milliseconds(policy.interruptWaitMs))
            << NAME << ": the stop did not wait after a SIGINT whose refusal it could not see";
        EXPECT_FALSE(std::filesystem::exists(FLUSHED)) << NAME;
      } else {
        ASSERT_EQ(stop.deliveries.size(), 1U) << NAME;
        EXPECT_EQ(stop.stoppedBy, SIGINT) << NAME;
        std::ifstream in(FLUSHED);
        std::string line;
        std::getline(in, line);
        EXPECT_EQ(line, "flushed") << NAME << ": the helper did not get its flush on SIGINT";
      }
      for (const auto& D : stop.deliveries) {
        if (POLICY == Policy::DEFAULT) {
          EXPECT_TRUE(D.statusObtained) << NAME;
          EXPECT_TRUE(D.delivered) << NAME;
          continue;
        }
        EXPECT_FALSE(D.statusObtained) << NAME;
        EXPECT_FALSE(D.delivered) << NAME << ": a delivery not seen passes for one proven";
        EXPECT_EQ(D.statusLoss, StatusLoss::SIGCHLD_IGNORED) << NAME;
        EXPECT_EQ(D.detail, "ended without an exit status this process could read: " +
                                describeStatusLoss(StatusLoss::SIGCHLD_IGNORED))
            << NAME;
      }
      EXPECT_EQ(stop.allDelivered(), POLICY == Policy::DEFAULT) << NAME;
    }
  }
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

/**
 * @test sudo that executed the tool in its own process: the tool is the one
 * signalled, not the one child the tool has of its own.
 */
TEST(ReadinessOwnedHelper, SudoRouteSignalsAToolSudoExecutedNotItsChild) {
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
  const pid_t TOOL = helper.pid();
  ASSERT_EQ(dir.logLines("helper child pid=").size(), 1U) << dir.log();
  const HelperStopResult STOP = helper.stop();
  killLoggedChildren(dir);
  EXPECT_TRUE(STOP.reaped);
  EXPECT_EQ(STOP.stoppedBy, SIGINT);
  ASSERT_FALSE(STOP.deliveries.empty());
  for (const auto& delivery : STOP.deliveries) {
    EXPECT_EQ(delivery.target, TOOL) << delivery.command;
  }
}

/**
 * @test sudo that stays the tool's parent, directly or through a monitor of
 * its own (also named sudo): the tool below is the one signalled, and the
 * stop ends it.
 */
TEST(ReadinessOwnedHelper, SudoRouteSignalsTheToolBelowSudo) {
  for (const char* stays : {"child", "monitor"}) {
    FakeToolDir dir;
    ASSERT_TRUE(dir.ok());
    const std::string SUDO = dir.install("fake_sudo.sh", "sudo");
    const std::string KILL = dir.install("fake_kill.sh", "kill");
    const std::string HELPER = dir.install("fake_helper.sh", "helper");
    auto ctx = std::make_shared<const ReadinessContext>(dir.context({{"FAKE_SUDO_STAYS", stays}}));
    HelperStopPolicy policy = quickPolicy(PrivilegeRoute::SCOPED_SUDO);
    policy.sudoPath = SUDO;
    policy.killPath = KILL;
    policy.context = ctx;
    OwnedHelper helper(policy);
    ASSERT_TRUE(helper.start({SUDO, "-n", "--", HELPER, "run"}, "", "", 400, ctx.get()).running())
        << stays;
    const std::vector<std::string> STARTED = dir.logLines("helper run pid=");
    ASSERT_EQ(STARTED.size(), 1U) << stays << ": " << dir.log();
    const pid_t TOOL =
        static_cast<pid_t>(std::stoi(STARTED.front().substr(STARTED.front().find('=') + 1)));
    ASSERT_NE(TOOL, helper.pid()) << stays;
    const HelperStopResult STOP = helper.stop();
    const bool TOOL_ENDED = ::kill(TOOL, 0) != 0;
    if (!TOOL_ENDED) {
      (void)::kill(TOOL, SIGKILL); // this test started it, through the fake
    }
    EXPECT_TRUE(STOP.reaped) << stays;
    EXPECT_EQ(STOP.stoppedBy, SIGINT) << stays;
    ASSERT_FALSE(STOP.deliveries.empty()) << stays;
    for (const auto& delivery : STOP.deliveries) {
      EXPECT_EQ(delivery.target, TOOL) << stays << ": " << delivery.command;
    }
    EXPECT_TRUE(TOOL_ENDED) << stays << ": the tool outlived the stop";
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
