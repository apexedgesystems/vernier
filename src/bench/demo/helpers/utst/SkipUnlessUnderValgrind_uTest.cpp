/**
 * @file SkipUnlessUnderValgrind_uTest.cpp
 * @brief Unit tests for the demo helper that skips a case unless it runs
 *        under valgrind.
 *
 * Notes:
 *  - The helper decides from the process it runs in, so the tests run this
 *    binary as a child on a probe case that uses the helper, once plainly and
 *    once under valgrind, and read what GoogleTest printed for the probe.
 *  - The valgrind run skips in a build with a sanitizer (which valgrind does
 *    not run as an ordinary binary), where valgrind is not installed, and where
 *    valgrind gives up reading this binary before the program runs (a
 *    valgrind older than the compiler that built it), quoting valgrind's own
 *    lines; any other way of not reaching the probe fails.
 *  - Tests are platform-agnostic and independent of execution order.
 */

#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"

#include "src/bench/inc/ProfilerEnv.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include <gtest/gtest.h>

using vernier::bench::demo::BUILT_WITH_A_SANITIZER;
using vernier::bench::demo::reasonToSkipUnlessUnderValgrind;
using vernier::bench::demo::SANITIZER_UNDER_VALGRIND_REASON;
using vernier::bench::demo::SKIP_UNLESS_UNDER_VALGRIND_REASON;
using vernier::bench::profiler_env::isOnPath;

/* ----------------------------- Constants ----------------------------- */

/// The probe's name, as the child runs select it and GoogleTest prints it.
constexpr const char* PROBE_TEST = "SkipUnlessUnderValgrindProbe.RunsOnlyUnderValgrind";

/// What the probe prints once the helper has let it run.
constexpr const char* PROBE_RAN = "probe: the helper let this case run";

/* ----------------------------- Child Runs ----------------------------- */

namespace {

/// What a child run left: its exit status (-1 when it was killed or could not
/// be run) and everything it wrote to stdout and stderr.
struct ChildRun {
  int status = -1;
  std::string output;
};

/// @p text as one single-quoted shell word, whatever characters it holds.
std::string shellQuoted(const std::string& text) {
  std::string quoted = "'";
  for (const char C : text) {
    quoted += (C == '\'') ? std::string("'\\''") : std::string(1, C);
  }
  return quoted + "'";
}

/// Path of this binary, which the child runs execute.
std::string selfPath() {
  std::array<char, 4096> buf{};
  const ssize_t LEN = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  return LEN > 0 ? std::string(buf.data(), static_cast<std::size_t>(LEN)) : std::string{};
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
    std::string pattern = (TMP / "vernier-demo-helpers-XXXXXX").string();
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

/// Runs the shell command @p command with its stdout and stderr in @p log and
/// returns what it left.
ChildRun runLogged(const std::string& command, const std::string& log) {
  ChildRun run;
  const int RAW = std::system((command + " > " + shellQuoted(log) + " 2>&1").c_str());
  if (RAW != -1 && WIFEXITED(RAW)) {
    run.status = WEXITSTATUS(RAW);
  }
  std::ifstream in(log);
  run.output.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return run;
}

/// The probe, selected, with its timings off so the output is the same from
/// run to run.
std::string probeArguments() {
  return " --gtest_filter=" + std::string(PROBE_TEST) + " --gtest_print_time=0";
}

/// valgrind's own two lines when it gives up reading debug information before
/// the program runs: the reader's line ("Valgrind: debuginfo reader: ...") and
/// the next one, "Valgrind: I can't recover.  Giving up.  Sorry.", as it
/// printed them. Empty for any other outcome, so a skip rests on valgrind's
/// text and never on this file's.
std::string debugInfoGiveUp(const std::string& text) {
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

} // namespace

/* ----------------------------- Probe ----------------------------- */

/** @test The probe the child runs select: skips unless under valgrind, then reports that it ran */
TEST(SkipUnlessUnderValgrindProbe, RunsOnlyUnderValgrind) {
  DEMO_SKIP_UNLESS_UNDER_VALGRIND();
  EXPECT_TRUE(reasonToSkipUnlessUnderValgrind().empty());
  std::puts(PROBE_RAN);
}

/* ----------------------------- API Tests ----------------------------- */

/** @test The reason tells the reader both ways to run the case */
TEST(SkipUnlessUnderValgrindTest, ReasonSaysHowToRunTheCase) {
  const std::string reason = SKIP_UNLESS_UNDER_VALGRIND_REASON;

  EXPECT_NE(reason.find("valgrind --tool=memcheck"), std::string::npos) << reason;
  EXPECT_NE(reason.find("bench run --profile memcheck"), std::string::npos) << reason;
}

/** @test In a plain run the probe reports SKIPPED with the reason, and does not run */
TEST(SkipUnlessUnderValgrindTest, PlainRunSkipsTheProbe) {
  const ScratchDir SCRATCH;
  ASSERT_FALSE(SCRATCH.path().empty()) << "could not create a scratch directory";

  const ChildRun RUN =
      runLogged(shellQuoted(selfPath()) + probeArguments(), SCRATCH.path() + "/plain.txt");

  EXPECT_EQ(RUN.status, 0) << RUN.output;
  EXPECT_NE(RUN.output.find(std::string("[  SKIPPED ] ") + PROBE_TEST), std::string::npos)
      << RUN.output;
  EXPECT_NE(RUN.output.find(SKIP_UNLESS_UNDER_VALGRIND_REASON), std::string::npos) << RUN.output;
  EXPECT_EQ(RUN.output.find(PROBE_RAN), std::string::npos) << RUN.output;
}

/** @test Under valgrind the probe runs and passes */
TEST(SkipUnlessUnderValgrindTest, ValgrindRunRunsTheProbe) {
  if constexpr (BUILT_WITH_A_SANITIZER) {
    GTEST_SKIP() << SANITIZER_UNDER_VALGRIND_REASON;
  }
  if (!isOnPath("valgrind")) {
    GTEST_SKIP() << "valgrind is not installed; this test runs the probe under it";
  }
  const ScratchDir SCRATCH;
  ASSERT_FALSE(SCRATCH.path().empty()) << "could not create a scratch directory";

  const ChildRun RUN =
      runLogged("valgrind --tool=memcheck " + shellQuoted(selfPath()) + probeArguments(),
                SCRATCH.path() + "/valgrind.txt");

  // Only valgrind's own reason skips: its debug-information reader gave up
  // before the program ran. Any other way of not reaching the probe fails,
  // with what the run printed.
  if (RUN.output.find("[==========]") == std::string::npos) {
    const std::string GAVE_UP = debugInfoGiveUp(RUN.output);
    if (!GAVE_UP.empty()) {
      GTEST_SKIP() << "valgrind gave up reading this binary's debug information before the "
                      "program ran. It printed:\n"
                   << GAVE_UP;
    }
    FAIL() << "the probe did not start under valgrind (exit status " << RUN.status
           << "). The run printed:\n"
           << (RUN.output.empty() ? std::string("(nothing)\n") : RUN.output);
  }

  EXPECT_EQ(RUN.status, 0) << RUN.output;
  EXPECT_NE(RUN.output.find(PROBE_RAN), std::string::npos) << RUN.output;
  EXPECT_NE(RUN.output.find("[  PASSED  ] 1 test."), std::string::npos) << RUN.output;
  EXPECT_EQ(RUN.output.find("[  SKIPPED ]"), std::string::npos) << RUN.output;
}
