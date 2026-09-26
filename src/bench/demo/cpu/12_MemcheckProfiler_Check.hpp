#ifndef VERNIER_DEMO_12_MEMCHECK_CHECK_HPP
#define VERNIER_DEMO_12_MEMCHECK_CHECK_HPP
/**
 * @file 12_MemcheckProfiler_Check.hpp
 * @brief The process plumbing and log reading of demo 12's memcheck check
 *
 * Memcheck.FindsTheOffByOne runs the demo binary under memcheck as a child and
 * reads what each run left: how valgrind ended, what the program printed (did
 * its test start and pass, or did valgrind give up before it ran) and
 * memcheck's log: the error summary and, from the error list valgrind prints
 * with --show-error-list=yes, each reported error with its count, its address
 * and its stacks. These are the helpers it does that with. The demo file keeps
 * what the check asserts and when it skips, so the example stays short enough
 * to copy.
 *
 * Private to 12_MemcheckProfiler_Demo.cpp.
 */

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

extern char** environ;

namespace vernier {
namespace bench {
namespace demo {
namespace memcheck_check {

namespace fs = std::filesystem;

/* ----------------------------- Child Runs ----------------------------- */

/// How a child process ended.
struct ChildExit {
  enum class How { NotStarted, Exited, Signaled };
  How how = How::NotStarted;
  /// The exit status, the signal number, or the error that kept it from
  /// being run or waited for.
  int code = 0;
};

/// What a run under memcheck left.
struct MemcheckRun {
  ChildExit end;      ///< How valgrind ended
  std::string output; ///< What the program wrote to stdout and stderr
  std::string log;    ///< memcheck's log, the file --log-file names
};

/// Runs @p args with this process's environment, and with stdout and stderr
/// in @p outputFile. Returns how the program ended.
inline ChildExit runLogged(const std::vector<std::string>& args, const fs::path& outputFile) {
  std::vector<char*> argv;
  for (const std::string& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, outputFile.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

  pid_t pid = 0;
  const int SPAWNED = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
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

/// True when the child ran and exited with status @p code.
inline bool exitedWith(const ChildExit& end, int code) {
  return end.how == ChildExit::How::Exited && end.code == code;
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

/// The last @p count lines of a text, for a failure message.
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

/// One report that is not the program's, suppressed in the check's runs.
/// gperftools' profiler library, where it is linked into the benchmark, probes
/// memory at start-up through libunwind, which writes the probed bytes to a
/// pipe to see whether they can be read; when those bytes are uninitialised
/// memcheck reports the write's argument, before main runs. Whether it does
/// depends on what the stack holds, so the report comes and goes between
/// runs. Seen on x86-64 with libunwind 1.6; the bytes are never used.
inline constexpr const char* START_UP_PROBE_SUPPRESSION = "{\n"
                                                          "   gperftools start-up memory probe\n"
                                                          "   Memcheck:Param\n"
                                                          "   write(buf)\n"
                                                          "   ...\n"
                                                          "   obj:*libunwind*\n"
                                                          "}\n";

/// Writes START_UP_PROBE_SUPPRESSION into @p dir and returns the file's path.
inline fs::path writeSuppressions(const fs::path& dir) {
  const fs::path FILE = dir / "start_up_probe.supp";
  std::ofstream out(FILE);
  out << START_UP_PROBE_SUPPRESSION;
  return FILE;
}

/**
 * @brief Run @p self under memcheck on the tests @p testFilter selects, the
 *        way `bench run --profile memcheck` wraps it (leak check on, an exit
 *        code for errors, the log in a file), plus valgrind's error list and
 *        the start-up probe suppression.
 * @param extraArgs  More arguments for the program, after the filter.
 * @param dir        Where the run's files go: program.txt, memcheck.log and
 *                   the suppression file.
 * @param errorExitCode  What valgrind exits with when it reported an error.
 */
inline MemcheckRun runUnderMemcheck(const std::string& self, const std::string& testFilter,
                                    const std::vector<std::string>& extraArgs, const fs::path& dir,
                                    int errorExitCode) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path LOG = dir / "memcheck.log";
  const fs::path OUTPUT = dir / "program.txt";
  std::vector<std::string> args = {"valgrind",
                                   "--tool=memcheck",
                                   "--leak-check=full",
                                   "--show-error-list=yes",
                                   "--error-exitcode=" + std::to_string(errorExitCode),
                                   "--log-file=" + escapePercent(LOG.string()),
                                   "--suppressions=" + writeSuppressions(dir).string(),
                                   self,
                                   "--gtest_filter=" + testFilter,
                                   "--gtest_print_time=0"};
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());

  MemcheckRun run;
  run.end = runLogged(args, OUTPUT);
  run.output = readText(OUTPUT);
  run.log = readText(LOG);
  return run;
}

/* ----------------------------- Reading a Run ----------------------------- */

/// True when GoogleTest printed its opening banner: the program under valgrind
/// reached its tests.
inline bool testsStarted(const std::string& output) {
  return output.find("[==========]") != std::string::npos;
}

/// True when the run's one selected test passed.
inline bool oneTestPassed(const std::string& output) {
  return output.find("[  PASSED  ] 1 test.") != std::string::npos;
}

/// valgrind's own two lines, as it printed them, when its debug-information
/// reader gave up before the program ran, as a valgrind older than the
/// compiler that wrote a file does: the reader's line ("Valgrind: debuginfo
/// reader: Possibly corrupted debuginfo file.") and the next one, "Valgrind:
/// I can't recover.  Giving up.  Sorry.". Empty for any other outcome. A skip
/// quotes these lines, so what it rests on is valgrind's text, not the check's.
inline std::string debugInfoGiveUp(const std::string& text) {
  const std::size_t GAVE_UP = text.find("Valgrind: I can't recover.  Giving up.");
  if (GAVE_UP == std::string::npos) {
    return "";
  }
  const std::size_t LINE_START = text.rfind('\n', GAVE_UP);
  if (LINE_START == std::string::npos || LINE_START == 0) {
    return "";
  }
  const std::size_t READER_BREAK = text.rfind('\n', LINE_START - 1);
  const std::size_t FROM = READER_BREAK == std::string::npos ? 0 : READER_BREAK + 1;
  if (text.substr(FROM, LINE_START - FROM).find("Valgrind: debuginfo reader: ") ==
      std::string::npos) {
    return "";
  }
  const std::size_t TO = text.find('\n', GAVE_UP);
  return text.substr(FROM, (TO == std::string::npos ? text.size() : TO) - FROM);
}

/* ----------------------------- Reading memcheck's Log ----------------------------- */

/// The totals of memcheck's ERROR SUMMARY line, or -1 each when the log has
/// none.
struct ErrorSummary {
  long errors = -1;
  long contexts = -1;
};

/// One entry of the error list memcheck prints with --show-error-list=yes:
/// "N errors in context i of M:", the error's kind on the next line, then its
/// address and stacks.
struct ReportedError {
  long count = 0;   ///< How many times this error occurred
  std::string kind; ///< Its first line, for instance "Invalid write of size 1"
  std::string text; ///< The whole entry, without valgrind's "==pid==" prefix
};

/// A log line without its "==pid== " prefix; the line itself when it has none.
inline std::string payload(const std::string& line) {
  if (line.size() > 2 && line[0] == '=' && line[1] == '=') {
    const std::size_t END = line.find("== ", 2);
    if (END != std::string::npos) {
      return line.substr(END + 3);
    }
    if (line.compare(line.size() - 2, 2, "==") == 0) {
      return "";
    }
  }
  return line;
}

/// The log's ERROR SUMMARY totals (the last such line, which valgrind prints
/// again after the error list).
inline ErrorSummary errorSummary(const std::string& log) {
  ErrorSummary summary;
  std::istringstream in(log);
  std::string line;
  while (std::getline(in, line)) {
    long errors = 0;
    long contexts = 0;
    if (std::sscanf(payload(line).c_str(), "ERROR SUMMARY: %ld errors from %ld contexts", &errors,
                    &contexts) == 2) {
      summary.errors = errors;
      summary.contexts = contexts;
    }
  }
  return summary;
}

/// Every entry of the error list, in the order valgrind printed them.
inline std::vector<ReportedError> errorList(const std::string& log) {
  std::vector<ReportedError> errors;
  std::istringstream in(log);
  std::string line;
  bool inEntry = false;
  while (std::getline(in, line)) {
    const std::string TEXT = payload(line);
    long count = 0;
    long index = 0;
    long total = 0;
    if (std::sscanf(TEXT.c_str(), "%ld errors in context %ld of %ld:", &count, &index, &total) ==
        3) {
      errors.push_back({count, "", ""});
      inEntry = true;
      continue;
    }
    if (!inEntry) {
      continue;
    }
    if (TEXT.empty()) {
      inEntry = false;
      continue;
    }
    if (errors.back().kind.empty()) {
      errors.back().kind = TEXT;
    }
    errors.back().text += TEXT + "\n";
  }
  return errors;
}

/// Every entry whose kind is @p kind. One error can be filed under several
/// entries: memcheck tells contexts apart by their stacks, and a compiler that
/// unrolls a loop of calls gives each call its own call site.
inline std::vector<const ReportedError*> errorsOfKind(const std::vector<ReportedError>& errors,
                                                      const std::string& kind) {
  std::vector<const ReportedError*> found;
  for (const ReportedError& error : errors) {
    if (error.kind == kind) {
      found.push_back(&error);
    }
  }
  return found;
}

/// The counts of @p entries added up: how often the error occurred in all.
inline long occurrences(const std::vector<const ReportedError*>& entries) {
  long total = 0;
  for (const ReportedError* entry : entries) {
    total += entry->count;
  }
  return total;
}

/// A number valgrind printed with thousands separators, read from @p text at
/// the position after @p marker; -1 when the marker is absent.
inline long numberAfter(const std::string& text, const std::string& marker) {
  const std::size_t AT = text.find(marker);
  if (AT == std::string::npos) {
    return -1;
  }
  long value = 0;
  bool any = false;
  for (std::size_t i = AT + marker.size(); i < text.size(); ++i) {
    const char CH = text[i];
    if (CH >= '0' && CH <= '9') {
      value = value * 10 + (CH - '0');
      any = true;
    } else if (CH != ',') {
      break;
    }
  }
  return any ? value : -1;
}

/// How far past the end of a heap block the error's address is, from
/// "Address 0x... is N bytes after a block of size M alloc'd"; -1 when the
/// entry describes no such address.
inline long bytesAfterBlock(const ReportedError& error) { return numberAfter(error.text, " is "); }

/// The size of the heap block the error's address is described against; -1
/// when the entry describes no block.
inline long blockSize(const ReportedError& error) {
  return numberAfter(error.text, "bytes after a block of size ");
}

/// How many lines of the entry name @p function: the frames of its stacks.
inline long framesNaming(const ReportedError& error, const std::string& function) {
  long frames = 0;
  std::istringstream in(error.text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(function) != std::string::npos) {
      ++frames;
    }
  }
  return frames;
}

} // namespace memcheck_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_12_MEMCHECK_CHECK_HPP
