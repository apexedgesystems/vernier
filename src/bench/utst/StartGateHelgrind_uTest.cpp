/**
 * @file StartGateHelgrind_uTest.cpp
 * @brief contentionRun's start gate under helgrind: the gate's own flags are
 *        not reported, and a race between the workers still is.
 *
 * Notes:
 *  - StartGate spins on two atomic flags, which helgrind cannot see as
 *    synchronisation; the gate asks libbench to mark them unchecked while it
 *    exists (PerfUtils.hpp, HelgrindRequests.cpp). The tests run this
 *    binary under helgrind as a child, on two probe cases that start four
 *    threads through contentionRun: workers that add to a counter under a
 *    mutex, and workers that add to it without one.
 *  - The helgrind runs skip in a build configured with the address or the
 *    thread sanitizer, where valgrind is not installed, and where valgrind
 *    stops reading this binary's debug information before the program runs,
 *    quoting valgrind; the locked run also skips where libbench says it does
 *    not mark the gate (built without helgrind.h, or with NVALGRIND). Any
 *    other way of not reaching a probe fails, with what the run printed.
 *  - Tests are independent of execution order.
 */

#include "src/bench/inc/PerfUtils.hpp"

#include "src/bench/inc/HelgrindRequests.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::profiler_env::isOnPath;
using vernier::bench::profiler_env::isRunningUnderValgrind;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// Threads each probe starts, and calls each thread makes.
constexpr int PROBE_THREADS = 4;
constexpr int PROBE_CYCLES = 2;

/// The probe cases, as the helgrind runs select them.
constexpr const char* LOCKED_PROBE = "StartGateProbe.LockedWorkers";
constexpr const char* RACY_PROBE = "StartGateProbe.RacyWorkers";

/// What valgrind exits with when helgrind reported an error: not 1, which is
/// also what a failed test exits with.
constexpr int HELGRIND_ERROR_EXIT = 99;

/// The sanitizer this build is configured with (-DSANITIZER), when it is one
/// valgrind cannot run: the address or the thread sanitizer.
#ifdef START_GATE_SANITIZER
constexpr const char* SANITIZER = START_GATE_SANITIZER;
#else
constexpr const char* SANITIZER = "";
#endif

} // namespace

/* ----------------------------- Probes ----------------------------- */

namespace {

/// The probes' harness settings: PROBE_THREADS threads of PROBE_CYCLES calls,
/// one repeat.
PerfConfig probeConfig() {
  PerfConfig cfg;
  cfg.threads = PROBE_THREADS;
  cfg.cycles = PROBE_CYCLES;
  cfg.repeats = 1;
  return cfg;
}

/// Adds one to @p counter without a lock: the racy probe's race. The counter
/// is eight bytes wide, which neither of the gate's flags is.
[[gnu::noinline]] void addUnlocked(std::uint64_t& counter) { ++counter; }

} // namespace

/** @test The locked probe: workers add under a mutex, and every call is counted */
TEST(StartGateProbe, LockedWorkers) {
  PerfCase perf{LOCKED_PROBE, probeConfig()};
  std::mutex counterMutex;
  std::uint64_t counter = 0;

  perf.contentionRun(
      [&] {
        std::lock_guard<std::mutex> lock(counterMutex);
        ++counter;
      },
      "locked");

  EXPECT_EQ(counter, static_cast<std::uint64_t>(PROBE_THREADS * PROBE_CYCLES));
}

/** @test The racy probe: workers add without a lock; runs only under valgrind */
TEST(StartGateProbe, RacyWorkers) {
  if (!isRunningUnderValgrind()) {
    GTEST_SKIP() << "this probe races on purpose, for helgrind to find, so it runs only under "
                    "valgrind";
  }
  PerfCase perf{RACY_PROBE, probeConfig()};
  std::uint64_t counter = 0;

  perf.contentionRun([&] { addUnlocked(counter); }, "racy");
}

/* ----------------------------- Child Runs ----------------------------- */

namespace {

/// What a helgrind run of one probe left.
struct HelgrindRun {
  int status = -1;    ///< valgrind's exit status; -1 when it was killed or not run
  std::string output; ///< What the run wrote to stdout and stderr
  std::string log;    ///< helgrind's log, the file --log-file names
};

/// @p text as one single-quoted shell word, whatever characters it holds.
std::string shellQuoted(const std::string& text) {
  std::string quoted = "'";
  for (const char C : text) {
    quoted += (C == '\'') ? std::string("'\\''") : std::string(1, C);
  }
  return quoted + "'";
}

/// valgrind expands '%' in output file names; "%%" is a literal one.
std::string escapePercent(const std::string& path) {
  std::string out;
  for (const char C : path) {
    out += C;
    if (C == '%') {
      out += '%';
    }
  }
  return out;
}

/// Path of this binary, which the helgrind runs execute.
std::string selfPath() {
  std::array<char, 4096> buf{};
  const ssize_t LEN = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  return LEN > 0 ? std::string(buf.data(), static_cast<std::size_t>(LEN)) : std::string{};
}

/// A whole file as text; empty when it cannot be read.
std::string readText(const std::string& file) {
  std::ifstream in(file);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// A temporary directory that is removed with everything in it.
class ScratchDir {
public:
  ScratchDir() {
    std::error_code ec;
    const std::filesystem::path TMP = std::filesystem::temp_directory_path(ec);
    if (ec) {
      return;
    }
    std::string pattern = (TMP / "vernier-start-gate-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    }
  }
  ~ScratchDir() {
    std::error_code ec;
    if (!path_.empty()) {
      std::filesystem::remove_all(path_, ec);
    }
  }
  ScratchDir(const ScratchDir&) = delete;
  ScratchDir& operator=(const ScratchDir&) = delete;

  /** @return The directory's path; empty if it could not be created. */
  const std::string& path() const noexcept { return path_; }

private:
  std::string path_;
};

/// Runs this binary's probe @p probe under helgrind, with an exit code for
/// errors, the log in @p dir/helgrind.log and the run's output in
/// @p dir/program.txt.
HelgrindRun runUnderHelgrind(const std::string& probe, const std::string& dir) {
  const std::string LOG = dir + "/helgrind.log";
  const std::string OUTPUT = dir + "/program.txt";
  const std::string COMMAND =
      "valgrind --tool=helgrind --error-exitcode=" + std::to_string(HELGRIND_ERROR_EXIT) + " " +
      shellQuoted("--log-file=" + escapePercent(LOG)) + " " + shellQuoted(selfPath()) +
      " --gtest_filter=" + probe + " --gtest_print_time=0 > " + shellQuoted(OUTPUT) + " 2>&1";
  HelgrindRun run;
  const int RAW = std::system(COMMAND.c_str());
  if (RAW != -1 && WIFEXITED(RAW)) {
    run.status = WEXITSTATUS(RAW);
  }
  run.output = readText(OUTPUT);
  run.log = readText(LOG);
  return run;
}

/// Why helgrind cannot run this binary here, decided before anything runs:
/// a sanitizer build, or no valgrind. Empty when it can.
std::string reasonHelgrindCannotRun() {
  if (SANITIZER[0] != '\0') {
    return std::string("this binary is built with the ") +
           (std::string(SANITIZER) == "asan" ? "address" : "thread") +
           " sanitizer (SANITIZER=" + SANITIZER +
           "), which valgrind cannot run as an ordinary program";
  }
  if (!isOnPath("valgrind")) {
    return "valgrind is not installed; this test runs a probe under helgrind";
  }
  return "";
}

/// True when GoogleTest reported the test @p name as passed, and the run
/// passed exactly that one test.
bool probePassed(const std::string& output, const std::string& name) {
  return output.find("[       OK ] " + name + "\n") != std::string::npos &&
         output.find("[  PASSED  ] 1 test.") != std::string::npos;
}

/// valgrind's own words when it stopped reading this binary's debug
/// information before the program ran: its reader's line and the one after
/// it ("Valgrind: I can't recover.  Giving up.  Sorry."), or an assertion
/// failing in its debug-information reader. Empty for any other outcome, so a
/// skip rests on valgrind's text.
std::string debugInfoGiveUp(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0 && lines[i].find("Valgrind: I can't recover.  Giving up.") != std::string::npos &&
        lines[i - 1].find("Valgrind: debuginfo reader: ") != std::string::npos) {
      return lines[i - 1] + "\n" + lines[i];
    }
    if (lines[i].find("valgrind: m_debuginfo/") != std::string::npos &&
        lines[i].find(": Assertion '") != std::string::npos &&
        lines[i].find("' failed.") != std::string::npos) {
      return lines[i];
    }
  }
  return "";
}

/// The totals of helgrind's ERROR SUMMARY line, or -1 each when the log has
/// none.
struct ErrorSummary {
  long errors = -1;
  long contexts = -1;
};

ErrorSummary errorSummary(const std::string& log) {
  ErrorSummary summary;
  constexpr const char* MARKER = "ERROR SUMMARY: ";
  const std::size_t AT = log.rfind(MARKER);
  if (AT != std::string::npos) {
    long errors = 0;
    long contexts = 0;
    if (std::sscanf(log.c_str() + AT, "ERROR SUMMARY: %ld errors from %ld contexts", &errors,
                    &contexts) == 2) {
      summary.errors = errors;
      summary.contexts = contexts;
    }
  }
  return summary;
}

/// True when helgrind reported a race on a read or a write of @p size bytes.
bool reportsRaceOfSize(const std::string& log, std::size_t size) {
  const std::string SIZE = " of size " + std::to_string(size) + " at ";
  return log.find("Possible data race during read" + SIZE) != std::string::npos ||
         log.find("Possible data race during write" + SIZE) != std::string::npos;
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test Under helgrind the locked probe reports no error: the gate's flags are not reported */
TEST(StartGateHelgrindTest, GateReportsNothing) {
  const std::string CANNOT_RUN = reasonHelgrindCannotRun();
  if (!CANNOT_RUN.empty()) {
    GTEST_SKIP() << CANNOT_RUN;
  }
  // Whether the gate marks its flags is libbench's build-time choice, so ask it
  if (!vernier::bench::helgrind::requestsBuiltIn()) {
    GTEST_SKIP() << "this libbench was built without valgrind's helgrind.h, or with NVALGRIND, "
                    "so StartGate does not mark its flags for helgrind";
  }
  const ScratchDir SCRATCH;
  ASSERT_FALSE(SCRATCH.path().empty()) << "could not create a scratch directory";

  const HelgrindRun RUN = runUnderHelgrind(LOCKED_PROBE, SCRATCH.path());
  if (RUN.output.find("[==========]") == std::string::npos) {
    const std::string GAVE_UP = debugInfoGiveUp(RUN.log + RUN.output);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind stopped reading this binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    FAIL() << "the locked probe did not start under helgrind (exit status " << RUN.status
           << "). The run printed:\n"
           << RUN.log << RUN.output;
  }

  const ErrorSummary SUMMARY = errorSummary(RUN.log);
  std::printf("[StartGateHelgrindTest.GateReportsNothing]  %ld errors from %ld contexts\n",
              SUMMARY.errors, SUMMARY.contexts);
  EXPECT_TRUE(probePassed(RUN.output, LOCKED_PROBE)) << RUN.output;
  EXPECT_EQ(SUMMARY.errors, 0) << "helgrind reported errors for workers that add under a "
                                  "mutex:\n"
                               << RUN.log;
  EXPECT_EQ(SUMMARY.contexts, 0);
  EXPECT_EQ(RUN.status, 0) << "valgrind exited with " << RUN.status << " for a run without errors";
}

/** @test Under helgrind the racy probe's race is reported: the gate hides nothing but its flags */
TEST(StartGateHelgrindTest, WorkerRaceIsStillReported) {
  const std::string CANNOT_RUN = reasonHelgrindCannotRun();
  if (!CANNOT_RUN.empty()) {
    GTEST_SKIP() << CANNOT_RUN;
  }
  const ScratchDir SCRATCH;
  ASSERT_FALSE(SCRATCH.path().empty()) << "could not create a scratch directory";

  const HelgrindRun RUN = runUnderHelgrind(RACY_PROBE, SCRATCH.path());
  if (RUN.output.find("[==========]") == std::string::npos) {
    const std::string GAVE_UP = debugInfoGiveUp(RUN.log + RUN.output);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind stopped reading this binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    FAIL() << "the racy probe did not start under helgrind (exit status " << RUN.status
           << "). The run printed:\n"
           << RUN.log << RUN.output;
  }

  const ErrorSummary SUMMARY = errorSummary(RUN.log);
  std::printf("[StartGateHelgrindTest.WorkerRaceIsStillReported]  %ld errors from %ld contexts\n",
              SUMMARY.errors, SUMMARY.contexts);
  EXPECT_TRUE(probePassed(RUN.output, RACY_PROBE)) << RUN.output;
  EXPECT_TRUE(reportsRaceOfSize(RUN.log, sizeof(std::uint64_t)))
      << "helgrind reported no race on the workers' eight-byte counter:\n"
      << RUN.log;
  EXPECT_EQ(RUN.status, HELGRIND_ERROR_EXIT)
      << "valgrind exited with " << RUN.status << ", not the --error-exitcode "
      << HELGRIND_ERROR_EXIT << " it was given for a run with errors";
}
