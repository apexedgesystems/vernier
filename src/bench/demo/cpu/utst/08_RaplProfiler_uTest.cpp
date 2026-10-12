/**
 * @file 08_RaplProfiler_uTest.cpp
 * @brief The check behind walkthrough 08: demo 08 measures both join versions
 *        and publishes a row for each.
 *
 * Not part of demo 08, whose source shows only what it teaches. Runs the demo
 * binary (BenchDemo_08_RaplProfiler, whose path the build passes in) with no
 * profiler, on fixed calls and repeats, and reads what the run left: how it
 * ended, GoogleTest's lines for its two tests and the rows of its CSV.
 * Starting the child and reading GoogleTest's lines are walkthrough 15's check
 * support (12_MemcheckProfiler_Check.hpp), used as it is. ctest runs it under
 * the demo label.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L demo
 *   ./build/bin/tests/TestDemoRapl      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The demo binary the check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_08_BINARY;

/// The demo's tests, sorted by name.
const std::vector<std::string> DEMO_TESTS = {"RaplProfiler.JoinV0", "RaplProfiler.JoinV1"};

/// Calls per repeat and repeats, fixed and few, so the run is short.
constexpr int CALLS = 20;
constexpr int REPEATS = 3;

/// The demo binary's canonical path, or an empty string when it is missing.
std::string demoPath() {
  std::error_code ec;
  const fs::path PATH = fs::canonical(DEMO_BINARY, ec);
  return ec ? std::string() : PATH.string();
}

/// A new temporary directory for the run's files; empty when it cannot be
/// made.
fs::path scratchDir() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo08-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

/// The test names of a CSV's rows, sorted: each row's first cell, read up to
/// its first comma. The demo's names hold no comma or quote, so the harness
/// writes them as they are. The header is not a row.
std::vector<std::string> rowNames(const fs::path& csv) {
  std::ifstream in(csv);
  std::vector<std::string> names;
  std::string line;
  for (bool header = true; std::getline(in, line); header = false) {
    if (!header) {
      names.push_back(line.substr(0, line.find(',')));
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

} // namespace

/* ----------------------------- Unprofiled Run ----------------------------- */

/**
 * @test The demo's two tests, run with no profiler on fixed calls and
 *       repeats, are the run's only tests and run to their end and pass, and
 *       the run's CSV holds one row for each of them and no other.
 */
TEST(RaplDemoTest, MeasuresBothVersions) {
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";
  const fs::path CSV = DIR / "run.csv";
  const fs::path OUTPUT = DIR / "program.txt";

  // Every test of the demo, whatever GTEST_FILTER this process inherited.
  const vg::ChildExit END =
      vg::runLogged({DEMO, "--gtest_filter=*", "--cycles", std::to_string(CALLS), "--repeats",
                     std::to_string(REPEATS), "--csv", CSV.string(), "--gtest_print_time=0"},
                    OUTPUT);
  const std::string PRINTED = vg::readText(OUTPUT);

  EXPECT_TRUE(vg::exitedWith(END, 0)) << "the demo " << vg::describe(END) << ". It printed:\n"
                                      << vg::lastLines(PRINTED, 40);
  EXPECT_NE(PRINTED.find("[==========] 2 tests from 1 test suite ran."), std::string::npos)
      << "the run did not run the demo's two tests and no other. It printed:\n"
      << vg::lastLines(PRINTED, 40);
  for (const std::string& name : DEMO_TESTS) {
    EXPECT_TRUE(vg::testPassed(PRINTED, name))
        << name << " did not run to its end and pass. The run printed:\n"
        << vg::lastLines(PRINTED, 40);
  }
  if (!fs::is_regular_file(CSV)) {
    ADD_FAILURE() << "the run wrote no CSV at " << CSV;
  } else {
    EXPECT_EQ(rowNames(CSV), DEMO_TESTS)
        << "the run's CSV does not hold exactly one row for each of the demo's tests:\n"
        << vg::readText(CSV);
  }

  if (HasFailure()) {
    std::printf("the run's files are kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}
