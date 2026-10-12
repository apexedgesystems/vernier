#ifndef VERNIER_DEMO_09_BPFTRACE_CHECK_HPP
#define VERNIER_DEMO_09_BPFTRACE_CHECK_HPP
/**
 * @file 09_BpftraceProfiler_Check.hpp
 * @brief Running a demo binary under a bpftrace-based backend and reading
 *        what the run left
 *
 * A check that traces its demo runs the demo binary as a child, as the
 * walkthrough's command does, and reads three things: whether the run said it
 * could not trace here for want of a privilege or of a kernel probe (then its
 * notice quotes sudo's or bpftrace's own line, and the check skips on it),
 * the capture window's two lines, and the histogram a script left in the
 * test's capture folder. Whether this machine can trace at all is decided
 * before anything runs, with the bpftrace support's own
 * bpftrace_tool::foreignPidNamespace(): in a PID namespace other than the
 * host's the bundled scripts are not supported. Starting the child and
 * reading its files and GoogleTest's lines are walkthrough 15's check support
 * (12_MemcheckProfiler_Check.hpp), used as it is.
 *
 * The interface, for any demo whose check traces it:
 *  - runTraced(): the demo on one test, under `--profile <backend> ...`
 *  - captureFolder(): where that test's files are
 *  - cannotTraceHere(): the run's notice when it could not trace here
 *  - backendLines(): what a backend printed about its tracers
 *  - acknowledgement(), linesWith(): the capture window's lines in a report
 *  - boundThread(): the thread a run copy's capture window names
 *  - filledInPid(): the pid a run's copy of a script was given
 *  - histogramTotal(): the sum of a bpftrace histogram's counts
 *
 * Test support for 09_BpftraceProfiler_uTest.cpp; not part of the demo.
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cstddef>
#include <cstdlib>

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace bpftrace_check {

namespace fs = std::filesystem;
namespace vg = vernier::bench::demo::memcheck_check;

/* ----------------------------- Traced Runs ----------------------------- */

/// What a traced run left.
struct TracedRun {
  vg::ChildExit end;  ///< How the demo ended
  std::string output; ///< What it wrote to stdout and stderr
  fs::path captures;  ///< The --profile-output-dir it was given
};

/**
 * @brief Run @p demo on the test @p testName under a profiler, as the
 *        walkthrough's command does, with BENCH_SUDO=1: the backend runs its
 *        tracer, and the kill that stops it, through sudo -n (as root, directly).
 * @param profileArgs  The request: the backend's name, which follows
 *                     --profile, and its options, for instance
 *                     {"bpftrace", "--bpf", "write_latency"}.
 * @param extraArgs    More arguments for the demo, for instance its cycles
 *                     and repeats.
 * @param dir          Where the run's files go: program.txt, what the demo
 *                     printed, and captures/, the capture folders.
 *
 * The bpftrace backend's own settings are fixed for the run: the bundled
 * scripts (an empty PERF_BPF_SCRIPTS) and text reports (PERF_BPF_FMT=text),
 * whatever this process inherited. The rest of this process's environment is
 * the demo's.
 */
inline TracedRun runTraced(const std::string& demo, const std::string& testName,
                           const std::vector<std::string>& profileArgs,
                           const std::vector<std::string>& extraArgs, const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  TracedRun run;
  run.captures = dir / "captures";
  std::vector<std::string> args = {"env", "BENCH_SUDO=1", "PERF_BPF_SCRIPTS=", "PERF_BPF_FMT=text",
                                   demo,  "--profile"};
  args.insert(args.end(), profileArgs.begin(), profileArgs.end());
  args.push_back("--gtest_filter=" + testName);
  args.push_back("--gtest_print_time=0");
  args.push_back("--profile-output-dir");
  args.push_back(run.captures.string());
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());

  const fs::path OUTPUT = dir / "program.txt";
  run.end = vg::runLogged(args, OUTPUT);
  run.output = vg::readText(OUTPUT);
  return run;
}

/// The capture folder a traced run wrote for @p testName, named with
/// @p suffix as the backend names it ("bpf" for the bpftrace backend):
/// captures/<Suite.Case>.<suffix>.
inline fs::path captureFolder(const TracedRun& run, const std::string& testName,
                              const std::string& suffix) {
  return run.captures / profiler_env::artifactDirName(testName, suffix);
}

/* ----------------------------- Reading a Run ----------------------------- */

/// @p text split into lines.
inline std::vector<std::string> linesOfText(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

/// True when @p text starts with @p prefix.
inline bool startsWith(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

/**
 * @brief The lines where a run says @p backend could not trace here, for want
 *        of a privilege (the cause "denied") or of a kernel probe
 *        ("unsupported"), as the run printed them; empty for any other
 *        outcome.
 *
 * Two places say so: the decision's notice, printed once, when the first test
 * asks for the profiler ("[FAIL] Profiler '<backend>': denied: ...", with its
 * remedy on the lines under it), and a tracer's failed start
 * ("[<backend>] ...: denied: ...").
 * Either quotes sudo's or the tool's own line at its end, so a check that
 * skips on these lines rests on that text, not on the check's. Every other
 * cause, a script not found for one, is not these lines, and a check fails on
 * it.
 */
inline std::string cannotTraceHere(const std::string& output, const std::string& backend) {
  const std::string NOTICE = "[FAIL] Profiler '" + backend + "': ";
  const std::string OWN = "[" + backend + "] ";
  const std::vector<std::string> LINES = linesOfText(output);
  std::string found;
  for (std::size_t i = 0; i < LINES.size(); ++i) {
    const std::string& LINE = LINES[i];
    if (startsWith(LINE, NOTICE)) {
      const std::string CAUSE = LINE.substr(NOTICE.size());
      if (!startsWith(CAUSE, "denied: ") && !startsWith(CAUSE, "unsupported: ")) {
        continue;
      }
      found += LINE + "\n";
      for (std::size_t j = i + 1; j < LINES.size() && startsWith(LINES[j], "   "); ++j) {
        found += LINES[j] + "\n";
      }
    } else if (startsWith(LINE, OWN) && (LINE.find(": denied: ") != std::string::npos ||
                                         LINE.find(": unsupported: ") != std::string::npos)) {
      found += LINE + "\n";
    }
  }
  return found;
}

/// Every line @p backend printed about its tracers ("[<backend>] ..."). A
/// run whose tracers started, stopped and flushed their output prints none.
inline std::string backendLines(const std::string& output, const std::string& backend) {
  std::string found;
  for (const std::string& line : linesOfText(output)) {
    if (startsWith(line, "[" + backend + "] ")) {
      found += line + "\n";
    }
  }
  return found;
}

/**
 * @brief The numbers after "<label> <word> " in the first line of @p report
 *        that holds it, for instance {pid, tid} for "bpftrace armed 812 815"
 *        and "bpftrace disarmed 812 812": the capture window's
 *        acknowledgements, which a backend's tracer prints before the
 *        measured repeats start and after they end. Empty when no line holds
 *        it.
 */
inline std::vector<long> acknowledgement(const std::string& report, const std::string& label,
                                         const std::string& word) {
  const std::string PREFIX = label + " " + word + " ";
  for (const std::string& line : linesOfText(report)) {
    const std::size_t AT = line.find(PREFIX);
    if (AT == std::string::npos) {
      continue;
    }
    std::vector<long> numbers;
    std::istringstream rest(line.substr(AT + PREFIX.size()));
    long number = 0;
    while (rest >> number) {
      numbers.push_back(number);
    }
    return numbers;
  }
  return {};
}

/// How many lines of @p report hold @p text.
inline int linesWith(const std::string& report, const std::string& text) {
  int count = 0;
  for (const std::string& line : linesOfText(report)) {
    count += line.find(text) != std::string::npos ? 1 : 0;
  }
  return count;
}

/// The thread a run's copy of a script binds to the thread name @p name in
/// its capture window ("tid == <id> && comm == \"<name>\""); -1 when none.
inline long boundThread(const std::string& runCopy, const std::string& name) {
  const std::string KEY = "tid == ";
  const std::string AFTER = " && comm == \"" + name + "\"";
  const std::size_t AT = runCopy.find(AFTER);
  if (AT == std::string::npos) {
    return -1;
  }
  const std::size_t START = runCopy.rfind(KEY, AT);
  if (START == std::string::npos) {
    return -1;
  }
  char* end = nullptr;
  const long TID = std::strtol(runCopy.c_str() + START + KEY.size(), &end, 10);
  return end == runCopy.c_str() + AT ? TID : -1;
}

/// The pid a run's copy of a script (<name>.tmp.bt) was given in place of
/// {{PID}}: the number after its first "pid == "; -1 when it has none.
inline long filledInPid(const std::string& runCopy) {
  const std::string KEY = "pid == ";
  const std::size_t AT = runCopy.find(KEY);
  if (AT == std::string::npos) {
    return -1;
  }
  char* end = nullptr;
  const long PID = std::strtol(runCopy.c_str() + AT + KEY.size(), &end, 10);
  return end == runCopy.c_str() + AT + KEY.size() ? -1 : PID;
}

/// @p text without the spaces at its end.
inline std::string trimmedEnd(const std::string& text) {
  const std::size_t LAST = text.find_last_not_of(" \t\r");
  return LAST == std::string::npos ? "" : text.substr(0, LAST + 1);
}

/**
 * @brief The sum of the counts of the bpftrace histogram @p map in
 *        @p report, the text bpftrace printed at its exit; -1 when the report
 *        holds no such map.
 *
 * bpftrace prints a histogram as its name and a colon (bpftrace 0.14 adds a
 * space), then one line per bucket, the bucket's range in brackets and its
 * count, then an empty line:
 * @code
 * @write_latency_us:
 * [0]              1023352 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
 * [1]                  403 |                                                    |
 * [2, 4)                 4 |                                                    |
 * @endcode
 * It prints no map that recorded nothing (the rig's bpftrace 0.23.2 wrote only
 * empty lines for a script whose probe never fired), so an absent map is a
 * script that saw no event.
 */
inline long histogramTotal(const std::string& report, const std::string& map) {
  bool found = false;
  long total = 0;
  for (const std::string& line : linesOfText(report)) {
    if (!found) {
      found = trimmedEnd(line) == map + ":";
      continue;
    }
    if (line.empty() || (line[0] != '[' && line[0] != '(')) {
      break;
    }
    const std::size_t CLOSE = line.find_first_of("])");
    if (CLOSE == std::string::npos) {
      break;
    }
    total += std::strtol(line.c_str() + CLOSE + 1, nullptr, 10);
  }
  return found ? total : -1;
}

} // namespace bpftrace_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_09_BPFTRACE_CHECK_HPP
