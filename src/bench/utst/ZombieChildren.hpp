#ifndef VERNIER_ZOMBIECHILDREN_HPP
#define VERNIER_ZOMBIECHILDREN_HPP
/**
 * @file ZombieChildren.hpp
 * @brief Test helper: the children of this process left to reap.
 *
 * zombieChildren() counts them, each with why it counts, and
 * namedZombieChildren() names each for a failure message. The bounded-probe
 * tests count before and after a probe; the count's own tests (ZombieCount)
 * give it observations of their own. With them, contextWith(): a context
 * holding exactly the environment a test gives it, which the unit and the
 * component readiness tests both decide in.
 */

#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace vernier {
namespace bench {
namespace test {

/** @brief A context with exactly @p env and effective uid @p euid. */
inline ReadinessContext contextWith(std::map<std::string, std::string> env, uid_t euid = 1000) {
  return ReadinessContext(euid, ::getpid(), std::move(env));
}

/** @brief The whole of /proc/<pid>/@p file; none when it cannot be opened. */
inline std::optional<std::string> procFile(pid_t pid, const char* file) {
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
inline ProcessState readProcessState(pid_t pid) {
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
inline int lookWithoutReaping(pid_t pid) {
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
inline constexpr std::chrono::milliseconds VANISH_WAIT{200};

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
inline std::vector<CountedChild> zombieChildren(const ChildObservation& observe = {}) {
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
inline std::string namedZombieChildren(const FakeToolDir& dir,
                                       const std::vector<CountedChild>& counted) {
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

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_ZOMBIECHILDREN_HPP
