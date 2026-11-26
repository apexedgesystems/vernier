#ifndef VERNIER_DEMO_04_COMPUTESANITIZER_CHECK_HPP
#define VERNIER_DEMO_04_COMPUTESANITIZER_CHECK_HPP
/**
 * @file 04_ComputeSanitizerProfiler_Check.hpp
 * @brief The tool runs and report reading of demo 04's check
 *
 * The compute-sanitizer checks run the demo binary through `bench run
 * --profile compute-sanitizer` as a child, as walkthrough 17's steps do:
 * bench run builds the wrap, reads the report's summary and fails the run
 * for errors, and the checks read the report it left for what bench run
 * does not read: every invalid access memcheck reported, with its kind, its
 * frame, the thread and block that made it, and the address and the
 * allocation it lies past. Starting
 * the child, reading its files and GoogleTest's lines are walkthrough 15's
 * check support (12_MemcheckProfiler_Check.hpp), used as it is;
 * 04_ComputeSanitizerProfiler_uTest.cpp keeps what the checks assert and
 * when they skip. Nothing here needs CUDA, so
 * 04_ComputeSanitizerProfiler_Report_uTest.cpp tests the report reading in
 * every build.
 *
 * Test support for those two programs; not part of the demo.
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

/* ----------------------------- Sanitizer Builds ----------------------------- */

/// True in a build with the address sanitizer. In such a build the GPU demo
/// does not start (its run prints "Your application is linked against
/// incompatible ASan runtimes"), the CUDA runtime reports "out of memory" to
/// cudaGetDeviceCount, and compute-sanitizer's injected libraries make the
/// sanitizer abort a program before its tests start ("AddressSanitizer:
/// alloc-dealloc-mismatch"), so the checks that run the demo or the tool
/// skip there.
#if defined(__SANITIZE_ADDRESS__)
inline constexpr bool BUILT_WITH_ASAN = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
inline constexpr bool BUILT_WITH_ASAN = true;
#else
inline constexpr bool BUILT_WITH_ASAN = false;
#endif
#else
inline constexpr bool BUILT_WITH_ASAN = false;
#endif

/// True in a build with the thread sanitizer. In such a build the GPU demo,
/// run plainly, overflows its stack in __tls_get_addr ("ThreadSanitizer:
/// stack-overflow") before its first test ends, and run under the tool it
/// exits 66 on a race the sanitizer reports in pthread_mutex_destroy at its
/// end, so the checks that run the demo skip there; the tool itself runs a
/// program of that build.
#if defined(__SANITIZE_THREAD__)
inline constexpr bool BUILT_WITH_TSAN = true;
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
inline constexpr bool BUILT_WITH_TSAN = true;
#else
inline constexpr bool BUILT_WITH_TSAN = false;
#endif
#else
inline constexpr bool BUILT_WITH_TSAN = false;
#endif

/// The skip message of a check that runs the demo binary, in a build with
/// the address or the thread sanitizer; empty in any other build.
inline const char* reasonNotToRunTheDemo() {
  if (BUILT_WITH_ASAN) {
    return "this binary is built with the address sanitizer, in which the GPU demo does not start "
           "(its run prints `Your application is linked against incompatible ASan runtimes`) and "
           "the CUDA runtime reports `out of memory`; run this test in a build without it";
  }
  if (BUILT_WITH_TSAN) {
    return "this binary is built with the thread sanitizer, in which the GPU demo run plainly "
           "overflows its stack in __tls_get_addr (`ThreadSanitizer: stack-overflow`) and run "
           "under the tool exits 66 on a race the sanitizer reports in pthread_mutex_destroy at "
           "its end; run this test in a build without it";
  }
  return "";
}

/* ----------------------------- Tool Runs ----------------------------- */

/// What a run through bench run left.
struct BenchRun {
  vg::ChildExit end;  ///< How bench run ended
  std::string output; ///< What bench run and the program wrote to stdout and stderr
  fs::path report;    ///< Where bench run's wrap writes the tool's report
  std::string log;    ///< The report's text; empty when the run left none
};

/**
 * @brief Run @p demo through `bench run <demo> --profile compute-sanitizer`
 *        on the test @p testFilter selects, as walkthrough 17's steps do:
 *        bench run starts the tool around the program with an exit status
 *        for findings and the report in the run's folder, reads the report's
 *        summary, and fails the run, naming the report, when it counts errors.
 * @param bench     The bench CLI.
 * @param extraArgs More arguments for the program, after the filter.
 * @param dir       Where the run's files go: run.txt and the profile folder out/.
 */
inline BenchRun runThroughBench(const std::string& bench, const std::string& demo,
                                const std::string& testFilter,
                                const std::vector<std::string>& extraArgs, const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path OUT = dir / "out";
  const fs::path OUTPUT = dir / "run.txt";
  std::vector<std::string> args = {bench,
                                   "run",
                                   demo,
                                   "--profile",
                                   "compute-sanitizer",
                                   "--profile-output-dir",
                                   OUT.string(),
                                   "--",
                                   "--gtest_filter=" + testFilter,
                                   "--gtest_print_time=0"};
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());

  BenchRun run;
  run.end = vg::runLogged(args, OUTPUT);
  run.output = vg::readText(OUTPUT);
  run.report = OUT / (fs::path(demo).filename().string() + ".compute-sanitizer") / "sanitizer.log";
  run.log = vg::readText(run.report);
  return run;
}

/// The count of bench run's verdict line, "compute-sanitizer reported N
/// error(s) in the benchmark; the report is ...": what bench run read from
/// the report's summary. -1 when bench run printed no such line.
inline long benchReportedErrors(const std::string& output) {
  static const std::string VERDICT = "compute-sanitizer reported ";
  const std::size_t AT = output.find(VERDICT);
  if (AT == std::string::npos) {
    return -1;
  }
  long errors = -1;
  return std::sscanf(output.c_str() + AT + VERDICT.size(), "%ld error", &errors) == 1 ? errors : -1;
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

/// The unguarded statement, as the unguarded source has it: the report must
/// name its line. The check finds that line in the source, so an edit that
/// moves the statement moves the expectation.
inline constexpr const char* UNGUARDED_STATEMENT = "y[I] = a * x[I] + y[I];";

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
