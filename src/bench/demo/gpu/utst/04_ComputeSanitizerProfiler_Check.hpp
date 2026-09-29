#ifndef VERNIER_DEMO_04_COMPUTESANITIZER_CHECK_HPP
#define VERNIER_DEMO_04_COMPUTESANITIZER_CHECK_HPP
/**
 * @file 04_ComputeSanitizerProfiler_Check.hpp
 * @brief The tool runs and report reading of demo 04's check
 *
 * The compute-sanitizer checks run the demo binary under the tool as a
 * child and read the report each run left: every invalid access memcheck
 * reported, with its kind, its frame, the thread and block that made it, the
 * address and the allocation it lies past, and the error summary. Starting
 * the child, reading its files and GoogleTest's lines are walkthrough 15's
 * check support (12_MemcheckProfiler_Check.hpp), used as it is;
 * 04_ComputeSanitizerProfiler_uTest.cpp keeps what the checks assert and
 * when they skip.
 *
 * Test support for 04_ComputeSanitizerProfiler_uTest.cpp; not part of the demo.
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"

#include <cstddef>
#include <cstdio>

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace sanitizer_check {

namespace fs = std::filesystem;
namespace vg = vernier::bench::demo::memcheck_check;

/* ----------------------------- Tool Runs ----------------------------- */

/// What a run under compute-sanitizer left.
struct SanitizerRun {
  vg::ChildExit end;  ///< How the tool ended
  std::string output; ///< What the program wrote to stdout and stderr
  std::string log;    ///< The tool's report, the file --log-file names
};

/**
 * @brief Run @p demo under compute-sanitizer's memcheck on the test
 *        @p testFilter selects, the way `bench run --profile
 *        compute-sanitizer` wraps it (the report in a file), plus an exit
 *        code for errors.
 * @param extraArgs      More arguments for the program, after the filter.
 * @param dir            Where the run's files go: program.txt and sanitizer.log.
 * @param errorExitCode  What the tool exits with when it reported an error.
 */
inline SanitizerRun runUnderSanitizer(const std::string& demo, const std::string& testFilter,
                                      const std::vector<std::string>& extraArgs,
                                      const fs::path& dir, int errorExitCode) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path LOG = dir / "sanitizer.log";
  const fs::path OUTPUT = dir / "program.txt";
  // The tool expands %p, %q{VAR} and %% in the log's name; "%%" is a literal.
  std::vector<std::string> args = {"compute-sanitizer",
                                   "--tool=memcheck",
                                   "--error-exitcode",
                                   std::to_string(errorExitCode),
                                   "--log-file",
                                   vg::escapePercent(LOG.string()),
                                   demo,
                                   "--gtest_filter=" + testFilter,
                                   "--gtest_print_time=0"};
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());

  SanitizerRun run;
  run.end = vg::runLogged(args, OUTPUT);
  run.output = vg::readText(OUTPUT);
  run.log = vg::readText(LOG);
  return run;
}

/* ----------------------------- Reading the Report ----------------------------- */

/// One invalid access memcheck reported, from the lines of its entry:
/// "Invalid __global__ read of size 4 bytes", then "at <function>+0x<offset>
/// in <file>:<line>", "by thread (x,y,z) in block (x,y,z)", "Access to 0x...
/// is out of bounds" and "and is N bytes after the nearest allocation at
/// 0x... of size M bytes", as the tool prints them without its prefix.
struct InvalidAccess {
  std::string kind;    ///< The first line, for instance "Invalid __global__ read of size 4 bytes"
  std::string frame;   ///< The "at ..." line
  std::string thread;  ///< The "by thread ..." line
  std::string address; ///< The "Access to ..." line
  std::string allocation; ///< The "and is ..." line
  std::string text;       ///< The whole entry, one line per line
};

/// A report line without the tool's "========= " prefix; the line itself when
/// it has none. The prefix is followed by one space; the entry's own lines
/// are indented beyond it.
inline std::string payload(const std::string& line) {
  constexpr const char* PREFIX = "=========";
  if (line.rfind(PREFIX, 0) == 0) {
    std::size_t from = std::string(PREFIX).size();
    if (from < line.size() && line[from] == ' ') {
      ++from;
    }
    return line.substr(from);
  }
  return line;
}

/// @p text with the spaces at its start removed.
inline std::string trimmedStart(const std::string& text) {
  const std::size_t FROM = text.find_first_not_of(' ');
  return FROM == std::string::npos ? "" : text.substr(FROM);
}

/// Every invalid access in @p log, in the order the tool printed them. An
/// entry starts with a line "Invalid ..." and ends at the next empty payload
/// line or the next unindented one.
inline std::vector<InvalidAccess> invalidAccesses(const std::string& log) {
  std::vector<InvalidAccess> accesses;
  bool inEntry = false;
  std::istringstream in(log);
  std::string line;
  while (std::getline(in, line)) {
    const std::string TEXT = payload(line);
    if (TEXT.rfind("Invalid ", 0) == 0) {
      accesses.push_back({TEXT, "", "", "", "", TEXT + "\n"});
      inEntry = true;
      continue;
    }
    if (!inEntry) {
      continue;
    }
    if (TEXT.empty() || TEXT[0] != ' ') {
      inEntry = false;
      continue;
    }
    InvalidAccess& entry = accesses.back();
    const std::string LINE = trimmedStart(TEXT);
    entry.text += LINE + "\n";
    if (LINE.rfind("at ", 0) == 0 && entry.frame.empty()) {
      entry.frame = LINE;
    } else if (LINE.rfind("by thread ", 0) == 0 && entry.thread.empty()) {
      entry.thread = LINE;
    } else if (LINE.rfind("Access to ", 0) == 0 && entry.address.empty()) {
      entry.address = LINE;
    } else if (LINE.rfind("and is ", 0) == 0 && entry.allocation.empty()) {
      entry.allocation = LINE;
    }
  }
  return accesses;
}

/// The total of the report's first "ERROR SUMMARY: N errors" line, or -1
/// when the log has none. A second such line, when the print limit was
/// reached, counts what was not printed and is not the total.
inline long errorSummary(const std::string& log) {
  std::istringstream in(log);
  std::string line;
  while (std::getline(in, line)) {
    long errors = 0;
    if (std::sscanf(payload(line).c_str(), "ERROR SUMMARY: %ld error", &errors) == 1) {
      return errors;
    }
  }
  return -1;
}

/// How far past the end of an allocation the access lies, from "and is N
/// bytes after the nearest allocation ..."; -1 when the entry says no such
/// thing.
inline long bytesAfterAllocation(const InvalidAccess& access) {
  return vg::numberAfter(access.allocation, "and is ");
}

/// The size of the allocation the access lies past, from "... of size M
/// bytes", read with or without thousands separators (compute-sanitizer
/// 2025.3 prints them, 2025.4 does not); -1 when the entry names none.
inline long allocationSize(const InvalidAccess& access) {
  return vg::numberAfter(access.allocation, "of size ");
}

/// True when @p frame, an "at <function>+0x<offset> in <where>" line, names
/// a function containing @p function at @p location ("<file>:<line>"). The
/// file may carry a directory (the tool strips paths by default; a run with
/// --strip-paths no keeps them); another file or line is not accepted. A
/// frame without " in ", as a build without device line information prints,
/// is not at any location.
inline bool frameAt(const std::string& frame, const std::string& function,
                    const std::string& location) {
  if (frame.rfind("at ", 0) != 0) {
    return false;
  }
  const std::size_t IN_AT = frame.rfind(" in ");
  if (IN_AT == std::string::npos) {
    return false;
  }
  const std::string NAME = frame.substr(3, IN_AT - 3);
  const std::string WHERE = frame.substr(IN_AT + 4);
  const std::string IN_DIRECTORY = "/" + location;
  const bool AT_LOCATION =
      WHERE == location ||
      (WHERE.size() > IN_DIRECTORY.size() &&
       WHERE.compare(WHERE.size() - IN_DIRECTORY.size(), IN_DIRECTORY.size(), IN_DIRECTORY) == 0);
  return NAME.find(function) != std::string::npos && AT_LOCATION;
}

/// The 1-based number of the line of @p source that contains @p statement;
/// 0 when no line or more than one line does.
inline std::size_t lineOf(const std::string& source, const std::string& statement) {
  std::istringstream in(source);
  std::string line;
  std::size_t number = 0;
  std::size_t found = 0;
  std::size_t matches = 0;
  while (std::getline(in, line)) {
    ++number;
    if (line.find(statement) != std::string::npos) {
      found = number;
      ++matches;
    }
  }
  return matches == 1 ? found : 0;
}

} // namespace sanitizer_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_04_COMPUTESANITIZER_CHECK_HPP
