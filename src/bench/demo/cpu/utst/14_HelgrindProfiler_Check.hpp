#ifndef VERNIER_DEMO_14_HELGRIND_CHECK_HPP
#define VERNIER_DEMO_14_HELGRIND_CHECK_HPP
/**
 * @file 14_HelgrindProfiler_Check.hpp
 * @brief The helgrind runs and report reading of demo 14's check
 *
 * Helgrind's checks run the demo binary under helgrind as a child and read
 * the report each run left: every race helgrind reported, with the access
 * that raced, the locks it held and where it was, and the earlier access it
 * conflicts with. Starting the child, reading its files and GoogleTest's
 * lines, valgrind's own give-up and no-symbols lines and the error summary
 * are walkthrough 15's check support (12_MemcheckProfiler_Check.hpp), used as
 * it is; 14_HelgrindProfiler_uTest.cpp keeps what the checks assert and when
 * they skip.
 *
 * Test support for 14_HelgrindProfiler_uTest.cpp; not part of the demo.
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
namespace helgrind_check {

namespace fs = std::filesystem;
namespace vg = vernier::bench::demo::memcheck_check;

/* ----------------------------- Helgrind Runs ----------------------------- */

/// What a run under helgrind left.
struct HelgrindRun {
  vg::ChildExit end;  ///< How valgrind ended
  std::string output; ///< What the program wrote to stdout and stderr
  std::string log;    ///< helgrind's log, the file --log-file names
};

/**
 * @brief Run @p demo under helgrind on the test @p testFilter selects, the
 *        way `bench run --profile helgrind` wraps it (the log in a file),
 *        plus valgrind's error list and an exit code for errors.
 * @param extraArgs      More arguments for the program, after the filter.
 * @param dir            Where the run's files go: program.txt and helgrind.log.
 * @param errorExitCode  What valgrind exits with when helgrind reported an
 *                       error.
 */
inline HelgrindRun runUnderHelgrind(const std::string& demo, const std::string& testFilter,
                                    const std::vector<std::string>& extraArgs, const fs::path& dir,
                                    int errorExitCode) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path LOG = dir / "helgrind.log";
  const fs::path OUTPUT = dir / "program.txt";
  std::vector<std::string> args = {"valgrind",
                                   "--tool=helgrind",
                                   "--show-error-list=yes",
                                   "--error-exitcode=" + std::to_string(errorExitCode),
                                   "--log-file=" + vg::escapePercent(LOG.string()),
                                   demo,
                                   "--gtest_filter=" + testFilter,
                                   "--gtest_print_time=0"};
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());

  HelgrindRun run;
  run.end = vg::runLogged(args, OUTPUT);
  run.output = vg::readText(OUTPUT);
  run.log = vg::readText(LOG);
  return run;
}

/* ----------------------------- Reading helgrind's Report ----------------------------- */

/// One access in a race report: the one that raced, or the earlier one it
/// conflicts with.
struct RaceAccess {
  std::string kind;      ///< "read" or "write"
  long size = -1;        ///< Bytes accessed
  std::string locksHeld; ///< What follows "Locks held: ", as helgrind printed it
  std::string frame;     ///< Its stack's first frame, "at 0x...: <function> (<where>)"
};

/// One race helgrind reported: "Possible data race during <kind> of size
/// <n> at <address> by thread #<t>", and the access it conflicts with when
/// helgrind showed one.
struct RaceReport {
  RaceAccess race;
  RaceAccess conflict; ///< Empty kind when the report shows no earlier access
};

/// @p text with the spaces at its start removed.
inline std::string trimmedStart(const std::string& text) {
  const std::size_t FROM = text.find_first_not_of(' ');
  return FROM == std::string::npos ? "" : text.substr(FROM);
}

/// Reads "<kind> of size <n>" from @p text, which starts with the kind.
inline void readKindAndSize(const std::string& text, RaceAccess& access) {
  char kind[16] = {};
  long size = -1;
  if (std::sscanf(text.c_str(), "%15s of size %ld", kind, &size) == 2) {
    access.kind = kind;
    access.size = size;
  }
}

/// Every race report in @p log, in the order helgrind printed them. The
/// error list at the end of a log with --show-error-list=yes repeats each
/// context once, so a report can be read twice.
inline std::vector<RaceReport> raceReports(const std::string& log) {
  constexpr const char* RACE = "Possible data race during ";
  constexpr const char* CONFLICT = "This conflicts with a previous ";
  constexpr const char* LOCKS = "Locks held: ";
  std::vector<RaceReport> reports;
  RaceAccess* current = nullptr;
  std::istringstream in(log);
  std::string line;
  while (std::getline(in, line)) {
    const std::string TEXT = trimmedStart(vg::payload(line));
    if (TEXT.rfind(RACE, 0) == 0) {
      reports.emplace_back();
      current = &reports.back().race;
      readKindAndSize(TEXT.substr(std::string(RACE).size()), *current);
    } else if (TEXT.rfind(CONFLICT, 0) == 0 && !reports.empty()) {
      current = &reports.back().conflict;
      readKindAndSize(TEXT.substr(std::string(CONFLICT).size()), *current);
    } else if (current == nullptr) {
      continue;
    } else if (TEXT.rfind(LOCKS, 0) == 0) {
      current->locksHeld = TEXT.substr(std::string(LOCKS).size());
    } else if (TEXT.rfind("at 0x", 0) == 0 && current->frame.empty()) {
      current->frame = TEXT;
    } else if (TEXT.rfind("----", 0) == 0 || TEXT.rfind("ERROR SUMMARY", 0) == 0) {
      current = nullptr;
    }
  }
  return reports;
}

/// True when @p frame is in a function whose name contains @p function, at
/// @p location ("<file>:<line>"): "at 0x...: ...function... (location)". The
/// file may carry the directory its debug information records, as clang's
/// does ("(src/.../<file>:<line>)"); another file or line is not accepted.
inline bool frameAt(const std::string& frame, const std::string& function,
                    const std::string& location) {
  const std::size_t NAME_AT = frame.find(": ");
  const std::size_t WHERE_AT = frame.rfind(" (");
  if (NAME_AT == std::string::npos || WHERE_AT == std::string::npos || WHERE_AT < NAME_AT ||
      frame.back() != ')') {
    return false;
  }
  const std::string NAME = frame.substr(NAME_AT + 2, WHERE_AT - NAME_AT - 2);
  const std::string WHERE = frame.substr(WHERE_AT + 2, frame.size() - WHERE_AT - 3);
  const std::string IN_DIRECTORY = "/" + location;
  const bool AT_LOCATION =
      WHERE == location ||
      (WHERE.size() > IN_DIRECTORY.size() &&
       WHERE.compare(WHERE.size() - IN_DIRECTORY.size(), IN_DIRECTORY.size(), IN_DIRECTORY) == 0);
  return NAME.find(function) != std::string::npos && AT_LOCATION;
}

/// True when @p frame is in @p binary and unnamed, as valgrind prints a frame
/// of a file whose symbols it could not read ("at 0x...: ??? (in <binary>)").
inline bool frameUnnamedIn(const std::string& frame, const std::string& binary) {
  return frame.find(": ??? (in " + binary + ")") != std::string::npos;
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

} // namespace helgrind_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_14_HELGRIND_CHECK_HPP
