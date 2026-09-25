/**
 * @file 07_CallgrindProfiler_Check.hpp
 * @brief The process plumbing of demo 07's instruction-count check
 *
 * CallgrindProfiler.InstructionCounts runs the demo binary under callgrind as
 * a child process and reads what each run left: how it ended, its log (did
 * its test start and pass, or did valgrind give up before it ran) and the
 * program total in its callgrind output. These are the helpers it does that
 * with. The demo file keeps what the check asserts and when it skips, and the
 * CountCalls fixture it counts, so the example stays short enough to copy.
 *
 * Private to 07_CallgrindProfiler_Demo.cpp.
 */

#ifndef VERNIER_DEMO_07_CALLGRIND_CHECK_HPP
#define VERNIER_DEMO_07_CALLGRIND_CHECK_HPP

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace callgrind_check {

namespace fs = std::filesystem;

/* ----------------------------- Child Runs ----------------------------- */

/// The instructions a callgrind output file counted for the whole program (its
/// "totals:" line), or 0 when the file has none.
inline std::uint64_t programTotal(const fs::path& profile) {
  std::ifstream in(profile);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("totals:", 0) == 0) {
      return std::stoull(line.substr(7));
    }
  }
  return 0;
}

/// How a child process ended.
struct ChildExit {
  enum class How { NotStarted, Exited, Signaled };
  How how = How::NotStarted;
  /// The exit status, the signal number, or the error that kept it from
  /// being run or waited for.
  int code = 0;
};

/// Runs @p args with @p extraVariable ("NAME=value") added to this process's
/// environment, replacing any variable of that name, and with stdout and stderr
/// in @p log. Returns how the program ended.
inline ChildExit runLogged(const std::vector<std::string>& args, const std::string& extraVariable,
                           const fs::path& log) {
  std::vector<char*> argv;
  for (const std::string& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  const std::string NAME = extraVariable.substr(0, extraVariable.find('=') + 1);
  std::vector<char*> envp;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    if (std::strncmp(*entry, NAME.c_str(), NAME.size()) != 0) {
      envp.push_back(*entry);
    }
  }
  envp.push_back(const_cast<char*>(extraVariable.c_str()));
  envp.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

  pid_t pid = 0;
  const int SPAWNED = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  if (SPAWNED != 0) {
    return {ChildExit::How::NotStarted, SPAWNED};
  }
  int status = 0;
  pid_t waited = 0;
  do {
    waited = ::waitpid(pid, &status, 0);
  } while (waited == -1 && errno == EINTR);
  if (waited != pid) {
    return {ChildExit::How::NotStarted, errno};
  }
  if (WIFSIGNALED(status)) {
    return {ChildExit::How::Signaled, WTERMSIG(status)};
  }
  return {ChildExit::How::Exited, WEXITSTATUS(status)};
}

/// True when the child ran and exited with status 0.
inline bool exitedCleanly(const ChildExit& end) {
  return end.how == ChildExit::How::Exited && end.code == 0;
}

/// How the child ended, in words, for a failure message.
inline std::string describe(const ChildExit& end) {
  switch (end.how) {
  case ChildExit::How::NotStarted:
    return std::string("could not be run: ") + std::strerror(end.code);
  case ChildExit::How::Signaled:
    return "was killed by signal " + std::to_string(end.code) + " (" + ::strsignal(end.code) + ")";
  case ChildExit::How::Exited:
    break;
  }
  return "exited with status " + std::to_string(end.code);
}

/// A whole file as text; empty when it cannot be read.
inline std::string readText(const fs::path& file) {
  std::ifstream in(file);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// The last @p count lines of a log, for a failure message.
inline std::string lastLines(const std::string& text, std::size_t count = 12) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  std::string out;
  for (std::size_t i = lines.size() > count ? lines.size() - count : 0; i < lines.size(); ++i) {
    out += lines[i] + "\n";
  }
  return out;
}

/* ----------------------------- Reading a Run ----------------------------- */

/// True when GoogleTest printed its opening banner: the program under valgrind
/// reached its tests.
inline bool testsStarted(const std::string& log) {
  return log.find("[==========]") != std::string::npos;
}

/// True when the run's one selected test passed, as a counting run's does.
inline bool oneTestPassed(const std::string& log) {
  return log.find("[  PASSED  ] 1 test.") != std::string::npos;
}

/// valgrind's own two lines, as it printed them, when its debug-information
/// reader gave up before the program ran, as a valgrind older than the
/// compiler that wrote a file does: the reader's line ("Valgrind: debuginfo
/// reader: Possibly corrupted debuginfo file.") and the next one, "Valgrind:
/// I can't recover.  Giving up.  Sorry.". Empty for any other outcome. A skip
/// quotes these lines, so what it rests on is valgrind's text, not the check's.
inline std::string debugInfoGiveUp(const std::string& log) {
  const std::size_t GAVE_UP = log.find("Valgrind: I can't recover.  Giving up.");
  if (GAVE_UP == std::string::npos) {
    return "";
  }
  // The line that gives up, and the reader's line just before it.
  const std::size_t LINE_START = log.rfind('\n', GAVE_UP);
  if (LINE_START == std::string::npos || LINE_START == 0) {
    return "";
  }
  const std::size_t READER_BREAK = log.rfind('\n', LINE_START - 1);
  const std::size_t FROM = READER_BREAK == std::string::npos ? 0 : READER_BREAK + 1;
  if (log.substr(FROM, LINE_START - FROM).find("Valgrind: debuginfo reader: ") ==
      std::string::npos) {
    return "";
  }
  const std::size_t TO = log.find('\n', GAVE_UP);
  return log.substr(FROM, (TO == std::string::npos ? log.size() : TO) - FROM);
}

/// valgrind expands '%' in output file names; "%%" is a literal one.
inline std::string escapePercent(const std::string& path) {
  std::string out;
  for (const char CH : path) {
    out += CH;
    if (CH == '%') {
      out += '%';
    }
  }
  return out;
}

} // namespace callgrind_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_07_CALLGRIND_CHECK_HPP
