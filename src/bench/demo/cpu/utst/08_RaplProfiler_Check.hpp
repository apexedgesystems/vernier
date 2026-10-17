#ifndef VERNIER_DEMO_08_RAPL_CHECK_HPP
#define VERNIER_DEMO_08_RAPL_CHECK_HPP
/**
 * @file 08_RaplProfiler_Check.hpp
 * @brief Running demo 08 under --profile rapl and reading what the run left
 *
 * Rapl.EnergyPerCall runs the demo binary as a child under --profile rapl, as
 * the walkthrough's command does, and reads two things: whether the run said
 * that RAPL cannot measure here (the decision's notice with the cause denied,
 * missing or unsupported, on which the check skips once it has held the
 * refusal to leaving nothing behind), and what each case's capture folder
 * holds. Starting the child and reading GoogleTest's lines are walkthrough
 * 15's check support (12_MemcheckProfiler_Check.hpp), used as it is.
 *
 * The interface:
 *  - runProfiled(): the demo under `--profile rapl`, captures in a folder of
 *    the run's own
 *  - captureFolder(): where a test's files are
 *  - refusalNotice(): the run's notice when RAPL cannot measure here
 *  - linesStarting(): how many lines of a run's output start with a text
 *
 * Test support for 08_RaplProfiler_uTest.cpp; not part of the demo.
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cstddef>

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace rapl_check {

namespace fs = std::filesystem;
namespace vg = vernier::bench::demo::memcheck_check;

/* ----------------------------- Profiled Runs ----------------------------- */

/// What a run under --profile rapl left.
struct ProfiledRun {
  vg::ChildExit end;  ///< How the demo ended
  std::string output; ///< What it wrote to stdout and stderr
  fs::path captures;  ///< The --profile-output-dir it was given
};

/**
 * @brief Run @p demo's tests under `--profile rapl`, as the walkthrough's
 *        command does, with @p extraArgs (the windows' sizes) after it.
 * @param dir  Where the run's files go: program.txt, what the demo printed,
 *             and captures/, the capture folders.
 */
inline ProfiledRun runProfiled(const std::string& demo, const std::vector<std::string>& extraArgs,
                               const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  ProfiledRun run;
  run.captures = dir / "captures";
  std::vector<std::string> args = {demo,
                                   "--profile",
                                   "rapl",
                                   "--profile-output-dir",
                                   run.captures.string(),
                                   "--gtest_filter=*",
                                   "--gtest_print_time=0"};
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());

  const fs::path OUTPUT = dir / "program.txt";
  run.end = vg::runLogged(args, OUTPUT);
  run.output = vg::readText(OUTPUT);
  return run;
}

/// The capture folder of @p testName in @p run: captures/<Suite.Case>.rapl.
inline fs::path captureFolder(const ProfiledRun& run, const std::string& testName) {
  return run.captures / profiler_env::artifactDirName(testName, "rapl");
}

/* ----------------------------- Reading a Run ----------------------------- */

/// True when @p text starts with @p prefix.
inline bool startsWith(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

/**
 * @brief The lines where the run says RAPL cannot measure here, as it printed
 *        them: the decision's notice ("[FAIL] Profiler 'rapl': denied: ...",
 *        "missing: ..." or "unsupported: ..."), printed once, with the
 *        indented lines under it; empty for any other outcome.
 *
 * Those three causes are this machine's: no access, no msr device, not an
 * Intel CPU. Any other cause (unusable, internal) is not these lines, and a
 * check fails on it.
 */
inline std::string refusalNotice(const std::string& output) {
  const std::string NOTICE = "[FAIL] Profiler 'rapl': ";
  std::vector<std::string> lines;
  std::istringstream in(output);
  for (std::string line; std::getline(in, line);) {
    lines.push_back(line);
  }
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (!startsWith(lines[i], NOTICE)) {
      continue;
    }
    const std::string CAUSE = lines[i].substr(NOTICE.size());
    if (!startsWith(CAUSE, "denied: ") && !startsWith(CAUSE, "missing: ") &&
        !startsWith(CAUSE, "unsupported: ")) {
      continue;
    }
    std::string found = lines[i] + "\n";
    for (std::size_t j = i + 1; j < lines.size() && startsWith(lines[j], "   "); ++j) {
      found += lines[j] + "\n";
    }
    return found;
  }
  return {};
}

/// How many lines of @p output start with @p prefix.
inline int linesStarting(const std::string& output, const std::string& prefix) {
  int count = 0;
  std::istringstream in(output);
  for (std::string line; std::getline(in, line);) {
    count += startsWith(line, prefix) ? 1 : 0;
  }
  return count;
}

} // namespace rapl_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_08_RAPL_CHECK_HPP
