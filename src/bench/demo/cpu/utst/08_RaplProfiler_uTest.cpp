/**
 * @file 08_RaplProfiler_uTest.cpp
 * @brief The checks behind walkthrough 08: demo 08 measures both join
 *        versions and publishes a row for each, and under --profile rapl it
 *        either measures or says why it cannot.
 *
 * Not part of demo 08, whose source shows only what it teaches. The first
 * check runs the demo binary (BenchDemo_08_RaplProfiler, whose path the build
 * passes in) with no profiler, on fixed calls and repeats, and reads what the
 * run left: how it ended, GoogleTest's lines for its two tests and the rows
 * of its CSV. The second runs it under --profile rapl, as the walkthrough
 * does: where RAPL cannot measure (no access, no msr device, not an Intel
 * CPU) the run must refuse honestly, and the check skips quoting the refusal;
 * where it can, the run must measure. 08_RaplProfiler_Check.hpp holds the
 * profiled run and its reading; starting the child and reading GoogleTest's
 * lines are walkthrough 15's check support (12_MemcheckProfiler_Check.hpp),
 * used as it is. ctest runs them under the demo label, the profiled one also
 * under the rapl label, alone.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L demo
 *   ./build/bin/tests/TestDemoRapl      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/08_RaplProfiler_Check.hpp"

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace check = vernier::bench::demo::rapl_check;
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

/// The profiled run's windows: about a third of a second for each case.
const std::vector<std::string> WINDOWS = {"--target-time", "33ms", "--repeats", "10"};

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

/* ----------------------------- Profiled Run ----------------------------- */

/**
 * @test Under --profile rapl the demo measures both versions' energy, or says
 *       why it cannot and leaves nothing behind.
 *
 * Runs the demo binary on both tests under `--profile rapl` with windows of
 * about a third of a second. Where RAPL cannot measure here, the run must
 * refuse honestly: both tests pass, the run exits with status 4, the notice
 * naming the cause (denied, missing or unsupported) is printed once and
 * again in the run's closing report, and there is no capture folder and no
 * energy summary; then the check skips, quoting the notice. Where RAPL can,
 * the run must exit 0, and each case's energy report must hold one window
 * over which the counter advanced, with its energy the counter's delta in
 * units, its power the energy over the window's duration and its energy per
 * call that power times the median call time; and JoinV1's energy per call
 * must be at most a third of JoinV0's. On the x86 laptop the walkthrough is
 * captured on, JoinV0 takes about 30 times JoinV1's time per call, so a third
 * leaves JoinV1 room to draw several times JoinV0's package power. Any other
 * outcome fails: a refusal with another cause, an exit 0 without reports, a
 * report beside a refusal.
 */
TEST(Rapl, EnergyPerCall) {
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const check::ProfiledRun RUN = check::runProfiled(DEMO, WINDOWS, DIR);
  const std::string REFUSAL = check::refusalNotice(RUN.output);

  if (!REFUSAL.empty()) {
    EXPECT_TRUE(vg::exitedWith(RUN.end, 4))
        << "a refused request must fail the run once its tests pass; the demo "
        << vg::describe(RUN.end);
    for (const std::string& name : DEMO_TESTS) {
      EXPECT_TRUE(vg::testPassed(RUN.output, name)) << name << " did not run to its end and pass";
      EXPECT_FALSE(fs::exists(check::captureFolder(RUN, name)))
          << "a refused request left a capture folder for " << name;
    }
    EXPECT_EQ(check::linesStarting(RUN.output, "[FAIL] Profiler 'rapl': "), 1)
        << "the notice is printed once for the run";
    EXPECT_NE(RUN.output.find("[profile] --profile rapl failed; the run exits with status 4:"),
              std::string::npos)
        << "the run's closing report does not name the failure";
    EXPECT_EQ(RUN.output.find("=== Energy"), std::string::npos)
        << "an energy summary beside a refusal";
    if (HasFailure()) {
      std::printf("the profiled run's files are kept in %s. It printed:\n%s", DIR.c_str(),
                  vg::lastLines(RUN.output, 40).c_str());
      return;
    }
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "RAPL cannot measure here, and the run refused as it should:\n" << REFUSAL;
  }

  ASSERT_TRUE(vg::exitedWith(RUN.end, 0))
      << "the run neither measured nor refused with a cause of this machine's (denied, missing, "
         "unsupported): the demo "
      << vg::describe(RUN.end) << " (its files are in " << DIR << "). It printed:\n"
      << vg::lastLines(RUN.output, 40);

  // Each case measured once: one window in its report, the counter advanced
  // over it, and the report's numbers agree with one another to the digits
  // it prints.
  std::map<std::string, double> perCall;
  for (const std::string& name : DEMO_TESTS) {
    const std::string TEXT = vg::readText(check::captureFolder(RUN, name) / "energy.txt");
    const std::vector<check::EnergyWindow> WINDOWS = check::energyWindows(TEXT);
    if (WINDOWS.size() != 1U) {
      ADD_FAILURE() << name << "'s energy report does not hold one window:\n" << TEXT;
      continue;
    }
    const check::EnergyWindow& WINDOW = WINDOWS.front();
    EXPECT_EQ(WINDOW.number, 1) << TEXT;
    EXPECT_GT(WINDOW.deltaCounter, 0) << name << ": the energy counter did not advance over its "
                                      << WINDOW.durationSeconds << " s window:\n"
                                      << TEXT;
    EXPECT_NEAR(WINDOW.energyJoules, WINDOW.deltaCounter * WINDOW.unitMicrojoules * 1e-6,
                1e-6 + WINDOW.energyJoules * 1e-5)
        << "energy is not the delta in units:\n"
        << TEXT;
    EXPECT_NEAR(WINDOW.powerWatts, WINDOW.energyJoules / WINDOW.durationSeconds,
                1e-6 + WINDOW.powerWatts * 1e-5)
        << "power is not the energy over the duration:\n"
        << TEXT;
    EXPECT_NEAR(WINDOW.perCallMillijoules, WINDOW.powerWatts * WINDOW.medianUs * 1e-3,
                1e-6 + WINDOW.powerWatts * 5e-7 + WINDOW.perCallMillijoules * 1e-5)
        << "energy per call is not the power times the median call time:\n"
        << TEXT;
    perCall[name] = WINDOW.perCallMillijoules;
  }
  EXPECT_EQ(check::linesStarting(RUN.output, "=== Energy Measurement (RAPL, package 0) ==="), 2)
      << "a summary for each case's window";

  // The effect: the reserving version costs a fraction of the copying one's
  // energy per call.
  if (perCall.size() == DEMO_TESTS.size()) {
    const double V0 = perCall["RaplProfiler.JoinV0"];
    const double V1 = perCall["RaplProfiler.JoinV1"];
    std::printf("[Rapl.EnergyPerCall]  JoinV0 %.6f mJ/call, JoinV1 %.6f mJ/call (%.1fx)\n", V0, V1,
                V1 > 0.0 ? V0 / V1 : 0.0);
    EXPECT_LE(3.0 * V1, V0) << "JoinV1's energy per call is not at most a third of JoinV0's";
  }

  if (HasFailure()) {
    std::printf("the profiled run's files are kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/* ----------------------------- Report Reading Tests ----------------------------- */

// What decides EnergyPerCall's skip, on text taken from real runs of demo 08
// under --profile rapl: in the development container started without the msr
// device, and in the privileged one as its ordinary user.

namespace {

/// The run in a container without /dev/cpu/0/msr, from its first test's start to its
/// first result: the device is missing.
constexpr const char* MISSING_RUN =
    "[ RUN      ] RaplProfiler.JoinV0\n"
    "\n"
    "[FAIL] Profiler 'rapl': missing: cannot open /dev/cpu/0/msr: No such file or directory; "
    "the msr driver provides it\n"
    "   Load the msr driver (sudo modprobe msr); in a container, pass the device in "
    "(--device /dev/cpu/0/msr).\n"
    "   Nothing is collected for this request; the run will fail (exit status 4 if the tests "
    "pass).\n"
    "\n"
    "[target-time] 33.000 ms -> cycles=107 (calibrated 307.0000 us/call, batch of 4)\n"
    "[RaplProfiler.JoinV0]  352.808 us/call  CV=2.1%  ~2.8K calls/s  (p10=343.513 "
    "p90=354.962 sd=7.432)\n";

/// The run in the privileged development container as its ordinary user, from the
/// notice to the run's closing report: the device refuses this user.
constexpr const char* DENIED_RUN =
    "[FAIL] Profiler 'rapl': denied: cannot open /dev/cpu/0/msr: Permission denied\n"
    "   RAPL needs access set up beforehand, and Vernier does not raise its privileges: run "
    "as root, or give the process CAP_SYS_RAWIO (the msr driver checks it) and read access "
    "to /dev/cpu/0/msr (the capability does not grant it).\n"
    "   Nothing is collected for this request; the run will fail (exit status 4 if the tests "
    "pass).\n"
    "\n"
    "[target-time] 33.000 ms -> cycles=101 (calibrated 325.7500 us/call, batch of 4)\n"
    "[RaplProfiler.JoinV0]  348.376 us/call  CV=2.0%  ~2.9K calls/s  (p10=337.329 "
    "p90=352.847 sd=7.068)\n"
    "[       OK ] RaplProfiler.JoinV0\n"
    "[ RUN      ] RaplProfiler.JoinV1\n"
    "[target-time] 33.000 ms -> cycles=651 (calibrated 50.6875 us/call, batch of 32)\n"
    "\n"
    "[profile] --profile rapl failed; the run exits with status 4:\n"
    "[profile]   rapl: denied: cannot open /dev/cpu/0/msr: Permission denied\n";

/// The report the backend writes for two windows, read from a regular file
/// standing in for the msr device; over the second the counter did not advance.
constexpr const char* TWO_WINDOW_REPORT =
    "=== RAPL Energy Measurement ===\n"
    "Test: RaplProfiler.JoinV0\n"
    "CPU: Intel(R) Core(TM) Ultra 7 165H\n"
    "Family: 6, Model: 170\n"
    "Scope: package 0, read through /tmp/rapl-report-NZpLq3/msr (MSR 0x611): every core, "
    "cache and the integrated graphics of the package, whatever runs on them; nothing "
    "outside the package\n"
    "Counter: 32 bits, wrapping after 262144.000 J at this unit; a window is taken to wrap "
    "it at most once\n"
    "Energy per operation: the window's average package power x its median time per call\n"
    "\n"
    "--- Window 1 ---\n"
    "Energy consumed: 5.000000 J\n"
    "Duration: 0.330223 s\n"
    "Average power: 15.141283 W\n"
    "Energy per operation: 2.047859 mJ/call\n"
    "Median time per call: 135.250 us\n"
    "Energy unit: 61.0352 uJ\n"
    "Start counter: 123456789\n"
    "End counter: 123538709\n"
    "Delta counter: 81920\n"
    "\n"
    "--- Window 2 ---\n"
    "Energy consumed: 0.000000 J\n"
    "Duration: 0.330263 s\n"
    "Average power: 0.000000 W\n"
    "Energy per operation: 0.000000 mJ/call\n"
    "Median time per call: 140.500 us\n"
    "Energy unit: 61.0352 uJ\n"
    "Start counter: 4294960000\n"
    "End counter: 4294960000\n"
    "Delta counter: 0\n"
    "Note: the counter did not advance during this 0.330 s window\n";

} // namespace

/** @test The refusal for a missing device is read with its remedy lines, and only those. */
TEST(RaplReportTest, RefusalNoticeReadsAMissingDevice) {
  EXPECT_EQ(check::refusalNotice(MISSING_RUN),
            "[FAIL] Profiler 'rapl': missing: cannot open /dev/cpu/0/msr: No such file or "
            "directory; the msr driver provides it\n"
            "   Load the msr driver (sudo modprobe msr); in a container, pass the device in "
            "(--device /dev/cpu/0/msr).\n"
            "   Nothing is collected for this request; the run will fail (exit status 4 if the "
            "tests pass).\n");
}

/** @test The refusal for want of access is read; the closing report is not part of it. */
TEST(RaplReportTest, RefusalNoticeReadsADenial) {
  const std::string NOTICE = check::refusalNotice(DENIED_RUN);

  EXPECT_EQ(NOTICE.rfind("[FAIL] Profiler 'rapl': denied: cannot open /dev/cpu/0/msr: Permission "
                         "denied\n",
                         0),
            0U)
      << NOTICE;
  EXPECT_EQ(NOTICE.find("[profile]"), std::string::npos) << NOTICE;
  EXPECT_EQ(check::linesStarting(DENIED_RUN, "[FAIL] Profiler 'rapl': "), 1);
}

/**
 * @test A CPU that is not Intel's is a refusal of this machine's too. The
 * notice is the backend's for an ARM CPU, as its unit test for an ARM
 * cpuinfo has it, with the run's own path; no ARM board ran this check.
 */
TEST(RaplReportTest, RefusalNoticeReadsAnUnsupportedCpu) {
  const std::string RUN =
      "[FAIL] Profiler 'rapl': unsupported: rapl reads Intel's RAPL energy registers, and "
      "/proc/cpuinfo names no CPU vendor here\n"
      "   Nothing is collected for this request; the run will fail (exit status 4 if the tests "
      "pass).\n";

  EXPECT_EQ(check::refusalNotice(RUN), RUN);
}

/** @test Any other cause, and a run with no refusal, is not a refusal the check skips on. */
TEST(RaplReportTest, RefusalNoticeOnlyForThisMachinesCauses) {
  EXPECT_EQ(check::refusalNotice("[FAIL] Profiler 'rapl': unusable: read MSR 0x606 from "
                                 "/dev/cpu/0/msr returned 0 of 8 bytes\n"),
            "");
  EXPECT_EQ(check::refusalNotice("[FAIL] Profiler 'rapl': internal: the rapl check failed\n"), "");
  EXPECT_EQ(check::refusalNotice("[FAIL] Profiler 'rapl': configuration: 'dram' is not a mode of "
                                 "rapl, which takes none\n"),
            "");
  EXPECT_EQ(check::refusalNotice("[FAIL] Profiler 'perf': denied: perf_event_paranoid=4\n"), "");
  EXPECT_EQ(check::refusalNotice("[RaplProfiler.JoinV0]  4.226 us/call\n"), "");
}

/** @test An energy report's windows are read in order, each field from its own section. */
TEST(RaplReportTest, EnergyWindowsReadsEachSection) {
  const std::vector<check::EnergyWindow> WINDOWS = check::energyWindows(TWO_WINDOW_REPORT);

  ASSERT_EQ(WINDOWS.size(), 2U);
  const check::EnergyWindow& FIRST = WINDOWS[0];
  EXPECT_EQ(FIRST.number, 1);
  EXPECT_EQ(FIRST.energyJoules, 5.0);
  EXPECT_EQ(FIRST.durationSeconds, 0.330223);
  EXPECT_EQ(FIRST.powerWatts, 15.141283);
  EXPECT_EQ(FIRST.perCallMillijoules, 2.047859);
  EXPECT_EQ(FIRST.medianUs, 135.25);
  EXPECT_EQ(FIRST.unitMicrojoules, 61.0352);
  EXPECT_EQ(FIRST.startCounter, 123456789);
  EXPECT_EQ(FIRST.endCounter, 123538709);
  EXPECT_EQ(FIRST.deltaCounter, 81920);
  EXPECT_EQ(WINDOWS[1].number, 2);
  EXPECT_EQ(WINDOWS[1].deltaCounter, 0);
  EXPECT_EQ(WINDOWS[1].startCounter, 4294960000LL);
}

/** @test The header's own fields, the definition of energy per call among them, are no window. */
TEST(RaplReportTest, EnergyWindowsOfAReportWithoutWindows) {
  const std::string REPORT = TWO_WINDOW_REPORT;

  EXPECT_TRUE(check::energyWindows(REPORT.substr(0, REPORT.find("\n--- Window 1"))).empty());
  EXPECT_TRUE(check::energyWindows("").empty());
}
