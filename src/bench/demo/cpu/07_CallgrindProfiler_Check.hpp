/**
 * @file 07_CallgrindProfiler_Check.hpp
 * @brief The process plumbing of demo 07's instruction-count check
 *
 * CallgrindProfiler.InstructionCounts runs the demo binary under callgrind as
 * a child process and reads what each run left: its log and the program total
 * in its callgrind output. These are the helpers it does that with. The demo
 * file keeps what the check asserts, and the CountCalls fixture it counts, so
 * the example stays short enough to copy.
 *
 * Private to 07_CallgrindProfiler_Demo.cpp.
 */

#ifndef VERNIER_DEMO_07_CALLGRIND_CHECK_HPP
#define VERNIER_DEMO_07_CALLGRIND_CHECK_HPP

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <filesystem>
#include <fstream>
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

/// Runs @p args with @p extraVariable ("NAME=value") added to this process's
/// environment, replacing any variable of that name, and with stdout and stderr
/// in @p log. Returns the exit status, or -1 when the program could not be
/// started or did not exit normally.
inline int runLogged(const std::vector<std::string>& args, const std::string& extraVariable,
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
    return -1;
  }
  int status = 0;
  if (::waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
    return -1;
  }
  return WEXITSTATUS(status);
}

/// The last lines of a log, for a failure message.
inline std::string tail(const fs::path& log) {
  std::ifstream in(log);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  std::string out;
  for (std::size_t i = lines.size() > 8 ? lines.size() - 8 : 0; i < lines.size(); ++i) {
    out += lines[i] + "\n";
  }
  return out;
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
