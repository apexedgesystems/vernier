/**
 * @file ReadinessFixtureRun_uTest.cpp
 * @brief Runs of the readiness fixture benchmarks: what a run with a profile
 * request reports and how it ends.
 *
 * Each test starts ReadinessFixtureTarget (two cases built with the profiler
 * guard and one built without it) or ReadinessCustomMainTarget (a benchmark
 * with its own main()) as a user does, in a working directory of its own,
 * with PATH set to a FakeToolDir that holds only the fakes the test installs
 * (ReadinessFixtures.hpp), with every setting that could change a decision
 * removed from the environment, and with FAKE_LOG naming the log the fakes
 * append their invocations to. It checks what the doctor and the run print
 * for the same request, the run's exit status, what the run leaves in its
 * working directory and what the fakes were asked to do. Every run's command
 * line, exit status and output are printed, so a failure shows them.
 */

#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/types.h>
#include <sys/wait.h>

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef VERNIER_READINESS_FIXTURE_TARGET
#error "VERNIER_READINESS_FIXTURE_TARGET must name the ReadinessFixtureTarget binary"
#endif
#ifndef VERNIER_READINESS_CUSTOM_MAIN_TARGET
#error "VERNIER_READINESS_CUSTOM_MAIN_TARGET must name the ReadinessCustomMainTarget binary"
#endif
#ifndef VERNIER_READINESS_HEAP_BUILT
#error "VERNIER_READINESS_HEAP_BUILT must be 1 when heap profiling is compiled in, else 0"
#endif

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using vernier::bench::detail::shellQuote;
using vernier::bench::test::FakeToolDir;

namespace {

/// The benchmark with two guarded cases and one bare case.
constexpr const char* FIXTURE = VERNIER_READINESS_FIXTURE_TARGET;

/// The benchmark with its own main().
constexpr const char* CUSTOM_MAIN = VERNIER_READINESS_CUSTOM_MAIN_TARGET;

/// Heap profiling is compiled into libbench (VERNIER_LINK_TCMALLOC=ON).
constexpr bool HEAP_BUILT = VERNIER_READINESS_HEAP_BUILT != 0;

/// What the fake perf logs after its own directory for a run's launch, up to the pid.
constexpr const char* PERF_STAT =
    "/perf stat -e cpu-cycles,instructions,branches,branch-misses,cache-misses -p ";

/// Fast runs: the tests measure nothing that matters here.
const std::vector<std::string> QUICK = {"--cycles", "50", "--repeats", "2", "--warmup", "0"};

/// Settings removed from a run's environment, so the test alone decides.
const std::vector<std::string> UNSET = {"BENCH_SUDO",
                                        "PERF_BPF",
                                        "PERF_BPF_SUDO",
                                        "PERF_BPF_SCRIPTS",
                                        "PERF_BPF_FMT",
                                        "PERF_BPF_OUT",
                                        "VERNIER_EXTERNAL_WRAP",
                                        "VERNIER_EXTERNAL_WRAP_DIR",
                                        "FAKE_SUDO_DENY",
                                        "LD_PRELOAD",
                                        "MALLOC_CONF"};

/// @p args followed by QUICK.
std::vector<std::string> quick(std::vector<std::string> args) {
  args.insert(args.end(), QUICK.begin(), QUICK.end());
  return args;
}

/// A whole file as text; empty when it cannot be read.
std::string readText(const std::string& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// How often @p needle occurs in @p text, counting from the end of each match.
std::size_t countOf(const std::string& text, const std::string& needle) {
  std::size_t count = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

/// The names in @p dir that end with @p suffix, as the shell's *<suffix> finds them.
std::vector<std::string> namesEndingWith(const std::string& dir, const std::string& suffix) {
  std::vector<std::string> names;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    const std::string NAME = entry.path().filename().string();
    if (NAME.size() >= suffix.size() &&
        NAME.compare(NAME.size() - suffix.size(), suffix.size(), suffix) == 0) {
      names.push_back(NAME);
    }
  }
  return names;
}

/**
 * @brief The selected row's @p field in the doctor's JSON, with its escaped
 * quotes, backslashes, newlines and tabs read; empty when the document has no
 * selected row or the row has no such field.
 */
std::string selectedField(const std::string& json, const std::string& field) {
  const std::size_t ROW = json.find("\"selected\": {");
  const std::string KEY = "\"" + field + "\": \"";
  const std::size_t AT = ROW == std::string::npos ? ROW : json.find(KEY, ROW);
  if (AT == std::string::npos) {
    return "";
  }
  std::string value;
  for (std::size_t i = AT + KEY.size(); i < json.size() && json[i] != '"'; ++i) {
    if (json[i] == '\\' && i + 1 < json.size()) {
      ++i;
      value += json[i] == 'n' ? '\n' : (json[i] == 't' ? '\t' : json[i]);
    } else {
      value += json[i];
    }
  }
  return value;
}

/**
 * @brief The data rows of the CSV at @p path, each a map from the header's
 * column names to the row's cells (the line split at every comma); none when
 * the file is missing.
 */
std::vector<std::map<std::string, std::string>> csvRows(const std::string& path) {
  const auto SPLIT = [](const std::string& line) {
    std::vector<std::string> cells;
    std::size_t start = 0;
    for (std::size_t comma = line.find(','); comma != std::string::npos;
         comma = line.find(',', start)) {
      cells.push_back(line.substr(start, comma - start));
      start = comma + 1;
    }
    cells.push_back(line.substr(start));
    return cells;
  };
  std::vector<std::map<std::string, std::string>> rows;
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line)) {
    return rows;
  }
  const std::vector<std::string> COLUMNS = SPLIT(line);
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> CELLS = SPLIT(line);
    std::map<std::string, std::string> row;
    for (std::size_t i = 0; i < COLUMNS.size() && i < CELLS.size(); ++i) {
      row[COLUMNS[i]] = CELLS[i];
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

/// How one run of a fixture ended.
struct RunResult {
  int status = -1; ///< Its exit status; -1 when it did not exit.
  std::string out; ///< What it wrote to stdout.
  std::string err; ///< What it wrote to stderr.
};

/// A private PATH of fakes, a working directory, and runs of the fixtures in them.
class ReadinessFixtureRun : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok()) << "no temporary directory for the fakes";
    work_ = dir_.makeDirectory("work");
  }

  /**
   * @brief Run @p program with @p args in work_, with PATH set to the fakes'
   * directory alone, FAKE_LOG to their log, UNSET removed and env_ added, and
   * print what it did.
   */
  RunResult run(const std::vector<std::string>& args, const std::string& program = FIXTURE) const {
    const std::string OUT = dir_.path() + "/run.out";
    const std::string ERR = dir_.path() + "/run.err";
    std::string command = "cd " + shellQuote(work_) + " && exec /usr/bin/env";
    for (const std::string& name : UNSET) {
      command += " -u " + name;
    }
    command +=
        " " + shellQuote("PATH=" + dir_.path()) + " " + shellQuote("FAKE_LOG=" + dir_.logPath());
    for (const std::string& setting : env_) {
      command += " " + shellQuote(setting);
    }
    command += " " + shellQuote(program);
    for (const std::string& arg : args) {
      command += " " + shellQuote(arg);
    }
    command += " >" + shellQuote(OUT) + " 2>" + shellQuote(ERR);
    const int RAW = std::system(command.c_str());

    RunResult result;
    std::string ended = "could not be run";
    if (RAW != -1 && WIFEXITED(RAW)) {
      result.status = WEXITSTATUS(RAW);
      ended = "exit status " + std::to_string(result.status);
    } else if (RAW != -1 && WIFSIGNALED(RAW)) {
      ended = "ended by signal " + std::to_string(WTERMSIG(RAW));
    }
    result.out = readText(OUT);
    result.err = readText(ERR);
    std::cout << "run " << std::filesystem::path(program).filename().string() << ":";
    for (const std::string& setting : env_) {
      std::cout << ' ' << setting;
    }
    for (const std::string& arg : args) {
      std::cout << ' ' << arg;
    }
    std::cout << "\n"
              << ended << "\n--- stdout\n"
              << result.out << "\n--- stderr\n"
              << result.err << std::endl;
    return result;
  }

  /// The doctor's JSON document for the request @p args states.
  std::string doctorJson(std::vector<std::string> args) const {
    args.emplace_back("--profile-check-json");
    return run(args).out;
  }

  /// Why gperf cannot be tested in this build (its row is not ok: gperftools
  /// is not compiled into libbench); empty when it can.
  std::string gperfUnusable() const {
    const std::string JSON = doctorJson({});
    const std::string KEY = "\"name\": \"gperf\", \"status\": \"";
    const std::size_t AT = JSON.find(KEY);
    const std::string STATUS =
        AT == std::string::npos
            ? ""
            : JSON.substr(AT + KEY.size(), JSON.find('"', AT + KEY.size()) - AT - KEY.size());
    return STATUS == "ok" ? "" : "gperf is not usable in this build (" + STATUS + ")";
  }

  /**
   * @brief Fail for every fake tracer the log names ("pid=N") that still
   * runs, killing it, and for every pid a fake kill was asked to signal
   * ("kill -S N") that is not one of them.
   */
  void expectOwnedAndGone(const std::string& what) const {
    const std::string LOG = dir_.log();
    std::vector<std::string> pids;
    const std::regex PID("pid=([0-9]+)");
    for (std::sregex_iterator it(LOG.begin(), LOG.end(), PID), end; it != end; ++it) {
      pids.push_back((*it)[1].str());
    }
    for (const std::string& pid : pids) {
      const auto TRACER = static_cast<pid_t>(std::stol(pid));
      if (::kill(TRACER, 0) == 0) {
        (void)::kill(TRACER, SIGKILL);
        ADD_FAILURE() << what << ": tracer " << pid << " survived the run";
      }
    }
    const std::regex KILL("kill -[0-9]+ ([0-9]+)");
    for (std::sregex_iterator it(LOG.begin(), LOG.end(), KILL), end; it != end; ++it) {
      if (std::find(pids.begin(), pids.end(), (*it)[1].str()) == pids.end()) {
        ADD_FAILURE() << what << ": '" << (*it)[0].str() << "' signalled a pid it does not own";
      }
    }
  }

  FakeToolDir dir_;
  std::string work_;
  std::vector<std::string> env_; ///< Settings ("NAME=value") every later run gets.
};

} // namespace

/* ------------------------- Wrapped Tools, Unwrapped ------------------------- */

/**
 * @test valgrind is on PATH, so the doctor starts massif; the run is not under
 * valgrind, so it collects nothing and fails, printing the wrap command.
 */
TEST_F(ReadinessFixtureRun, MassifUnwrappedFails) {
  dir_.install("fake_valgrind.sh", "valgrind");
  const std::string DOCTOR = doctorJson({"--profile", "massif", "--profile-args", "pages"});
  EXPECT_EQ(selectedField(DOCTOR, "status"), "ok") << "doctor status (the tool starts)";
  EXPECT_THAT(selectedField(DOCTOR, "message"),
              HasSubstr("valgrind starts massif with --pages-as-heap=yes"))
      << "doctor message";
  const std::string BEFORE_RUN = dir_.log();
  const RunResult RUN = run(quick({"--profile", "massif", "--profile-args", "pages"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  const std::string MESSAGE = "missing: massif collects only when valgrind's massif runs the "
                              "process, and valgrind does not run this one";
  const std::string HINT =
      "Wrap it: valgrind --tool=massif --pages-as-heap=yes --massif-out-file=./massif.out "
      "<this-binary> --profile massif --profile-args pages [...]; or run it with bench run "
      "--profile massif --profile-args pages, which wraps it.";
  EXPECT_THAT(RUN.err, HasSubstr("[FAIL] Profiler 'massif': " + MESSAGE + "\n   " + HINT +
                                 "\n   Nothing is collected for this request"))
      << "run notice";
  EXPECT_EQ(countOf(RUN.err, HINT), 1U) << "the wrap command, once for two guarded cases";
  EXPECT_THAT(RUN.err, HasSubstr("[profile]   massif: " + MESSAGE)) << "run-end report";
  EXPECT_THAT(namesEndingWith(work_, ".massif"), IsEmpty())
      << "folders left by a run that collected nothing";
  EXPECT_EQ(dir_.log(), BEFORE_RUN) << "valgrind runs started by the run (none)";
}

/**
 * @test heaptrack records /bin/true for the doctor; the run is not under
 * heaptrack, so it collects nothing and fails, printing the wrap command.
 */
TEST_F(ReadinessFixtureRun, HeaptrackUnwrappedFails) {
  dir_.install("fake_heaptrack.sh", "heaptrack");
  const std::string DOCTOR = doctorJson({"--profile", "heaptrack"});
  EXPECT_EQ(selectedField(DOCTOR, "status"), "ok") << "doctor status (heaptrack records)";
  const RunResult RUN = run(quick({"--profile", "heaptrack"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  EXPECT_THAT(RUN.err,
              HasSubstr("[FAIL] Profiler 'heaptrack': missing: heaptrack collects only when "
                        "heaptrack runs the process, and heaptrack does not run this one\n"
                        "   Wrap it: heaptrack -o ./run <this-binary> --profile heaptrack [...]; "
                        "or run it with bench run --profile heaptrack, which wraps it.\n"))
      << "run notice";
  EXPECT_THAT(namesEndingWith(work_, ".heaptrack"), IsEmpty())
      << "folders left by a run that collected nothing";
}

/**
 * @test rocprof on PATH is never ok for the doctor; a run without its
 * injection fails, printing the wrap command, and creates no folder.
 */
TEST_F(ReadinessFixtureRun, RocprofUnwrappedFails) {
  dir_.install("fake_rocprof.sh", "rocprof");
  const std::string DOCTOR = doctorJson({"--profile", "rocprof", "--profile-args", "stats"});
  EXPECT_EQ(selectedField(DOCTOR, "status"), "warn") << "doctor status (never ok)";
  EXPECT_THAT(selectedField(DOCTOR, "message"),
              HasSubstr("unverified: AMD collection is not validated (legacy rocprof)"))
      << "doctor message";
  const RunResult RUN = run(quick({"--profile", "rocprof", "--profile-args", "stats"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  EXPECT_THAT(RUN.err,
              HasSubstr("[FAIL] Profiler 'rocprof': missing: rocprof collects only when rocprof "
                        "runs the process, and rocprof does not run this one\n"
                        "   Wrap it: rocprof --stats -o ./results.csv <this-binary> --profile "
                        "rocprof --profile-args stats [...]; bench run does not wrap rocprof.\n"))
      << "run notice";
  EXPECT_THAT(namesEndingWith(work_, ".rocprof"), IsEmpty())
      << "folders left by a run that collected nothing";
  EXPECT_EQ(dir_.log(), "") << "rocprof started by the doctor or the run (never)";
}

/**
 * @test The fixture is built without CUDA and knows nsight, ncu and
 * compute-sanitizer: the doctor reports each by its own tool, a run the tool
 * did not start fails with the command that captures it and starts nothing,
 * and a run under bench run's wrap is profiled by a passive profiler that
 * names the tool and the wrap's folder.
 */
TEST_F(ReadinessFixtureRun, GpuNamesOnACpuBuild) {
  for (const char* tool : {"nsys", "ncu", "compute-sanitizer"}) {
    dir_.install("fake_nvidia_tool.sh", tool);
  }
  const std::string DOCTOR = doctorJson({});
  for (const char* name : {"nsight", "ncu", "compute-sanitizer"}) {
    const std::string KEY = std::string{"\"name\": \""} + name + "\", \"status\": \"";
    const std::size_t AT = DOCTOR.find(KEY);
    const std::size_t END = AT == std::string::npos ? AT : DOCTOR.find('"', AT + KEY.size());
    const std::string MESSAGE = "\", \"message\": \"unverified: " + dir_.path() + "/";
    if (END == std::string::npos || DOCTOR.compare(END, MESSAGE.size(), MESSAGE) != 0) {
      ADD_FAILURE() << "no unverified inventory row for " << name;
    } else {
      EXPECT_EQ(DOCTOR.substr(AT + KEY.size(), END - AT - KEY.size()), "warn")
          << name << " inventory status";
    }
  }
  std::error_code ec;
  std::filesystem::remove(dir_.logPath(), ec);
  const RunResult RUN = run(quick({"--profile", "nsight"}));
  EXPECT_EQ(RUN.status, 4) << "unwrapped run exit status";
  EXPECT_THAT(RUN.err,
              HasSubstr("[FAIL] Profiler 'nsight': missing: nsight collects only when nsys starts "
                        "the process, and nsys did not start this one\n"
                        "   Wrap it: nsys profile -o ./profile -t cuda,nvtx --force-overwrite true "
                        "<this-binary> --profile nsight [...]; or run it with bench run --profile "
                        "nsight, which wraps it and writes the summary reports.\n"))
      << "unwrapped run notice";
  EXPECT_THAT(RUN.err,
              HasSubstr("[profile] --profile nsight failed; the run exits with status 4:\n"))
      << "run-end report";
  EXPECT_EQ(dir_.log(), "") << "fake log (a run starts no tool)";
  EXPECT_THAT(namesEndingWith(work_, ".nsight"), IsEmpty()) << "folders an unwrapped run made";

  const std::string WRAP_DIR = work_ + "/bench-out/ReadinessFixtureTarget.nsight";
  env_.emplace_back("VERNIER_EXTERNAL_WRAP=nsight");
  env_.push_back("VERNIER_EXTERNAL_WRAP_DIR=" + WRAP_DIR);
  const RunResult WRAPPED = run(quick({"--profile", "nsight", "--csv", "wrapped.csv"}));
  EXPECT_EQ(WRAPPED.status, 0) << "wrapped run exit status";
  EXPECT_THAT(WRAPPED.err,
              HasSubstr("[WARN] Profiler 'nsight': unverified: nsys started this process"))
      << "wrapped run notice";
  std::size_t named = 0;
  for (const auto& row : csvRows(work_ + "/wrapped.csv")) {
    const std::string TEST = row.at("test");
    if (TEST.size() >= 4 && TEST.compare(TEST.size() - 4, 4, "Bare") == 0) {
      EXPECT_EQ(row.at("profileTool"), "") << "profileTool of the bare case";
    } else {
      EXPECT_EQ(row.at("profileTool"), "nsight") << "profileTool of " << TEST;
      EXPECT_EQ(row.at("profileDir"), WRAP_DIR) << "profileDir of " << TEST;
      ++named;
    }
  }
  EXPECT_EQ(named, 2U) << "CSV rows naming the passive profiler";
}

/* ---------------------------- How a Run Ends ---------------------------- */

/**
 * @test An unknown name fails the run: one notice for the guarded cases, the
 * run-end report and exit status 4, and the CSV names no profiler. A run of
 * the bare case alone fails the same way.
 */
TEST_F(ReadinessFixtureRun, RunUnknownProfilerFails) {
  const RunResult RUN = run(quick({"--profile", "nosuch", "--csv", "run.csv"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  const std::string NOTICE = "[FAIL] Profiler 'nosuch': unknown profiler 'nosuch'\n   Available: ";
  EXPECT_THAT(RUN.err, HasSubstr(NOTICE)) << "run notice";
  EXPECT_EQ(countOf(RUN.err, NOTICE), 1U) << "notices for two guarded cases";
  EXPECT_THAT(RUN.err, HasSubstr("[profile] --profile nosuch failed; the run exits with status "
                                 "4:\n[profile]   nosuch: unknown profiler 'nosuch'\n"))
      << "run-end report";
  std::size_t rows = 0;
  for (const auto& row : csvRows(work_ + "/run.csv")) {
    EXPECT_EQ(row.at("profileTool"), "") << "profileTool of " << row.at("test");
    ++rows;
  }
  EXPECT_EQ(rows, 3U) << "CSV rows";
  const RunResult BARE =
      run(quick({"--profile", "nosuch", "--gtest_filter=ReadinessFixture.Bare"}));
  EXPECT_EQ(BARE.status, 4) << "exit status with only the bare case";
  EXPECT_THAT(BARE.err,
              HasSubstr("[profile] --profile nosuch failed; the run exits with status 4:"))
      << "run-end report (bare case)";
}

/** @test Without --profile nothing is decided or reported, and the run exits 0. */
TEST_F(ReadinessFixtureRun, UnprofiledRunExitsZero) {
  const RunResult RUN = run(quick({}));
  EXPECT_EQ(RUN.status, 0) << "run exit status";
  EXPECT_THAT(RUN.err, Not(HasSubstr("[profile]"))) << "run-end report or notice";
  EXPECT_THAT(RUN.err, Not(HasSubstr("Profiler '"))) << "run notice";
}

/** @test cupti is not a capture: its note, no failure and no notice, exit 0. */
TEST_F(ReadinessFixtureRun, CuptiNeedsNoProfile) {
  const RunResult RUN = run(quick({"--profile", "cupti"}));
  EXPECT_EQ(RUN.status, 0) << "run exit status";
  EXPECT_THAT(RUN.err, HasSubstr("[INFO] 'cupti' needs no --profile")) << "cupti note";
  EXPECT_THAT(RUN.err, Not(HasSubstr("[profile]"))) << "run-end report or notice";
  const RunResult BARE = run(quick({"--profile", "cupti", "--gtest_filter=ReadinessFixture.Bare"}));
  EXPECT_EQ(BARE.status, 0) << "exit status with only the bare case";
  EXPECT_THAT(BARE.err, Not(HasSubstr("[profile]"))) << "run-end report or notice (bare case)";
}

/**
 * @test --profile with only a case built without the guard: the run says
 * nothing was profiled, or, under the runner's wrap of that tool, that the
 * wrap recorded the whole process. Neither changes the exit status.
 */
TEST_F(ReadinessFixtureRun, NoProfilerCreatedNotice) {
  dir_.install("fake_perf.sh", "perf");
  const RunResult BARE = run(quick({"--profile", "perf", "--gtest_filter=ReadinessFixture.Bare"}));
  EXPECT_EQ(BARE.status, 0) << "run exit status";
  const std::string NOTICE = "[profile] --profile perf: no case that ran was built with the "
                             "profiler guard, so nothing was profiled.\n";
  EXPECT_THAT(BARE.err, HasSubstr(NOTICE)) << "notice";
  EXPECT_EQ(countOf(BARE.err, NOTICE), 1U) << "notices";
  EXPECT_EQ(dir_.log(), "") << "fake log (nothing decided or launched)";
  env_.emplace_back("VERNIER_EXTERNAL_WRAP=massif");
  const RunResult WRAPPED =
      run(quick({"--profile", "massif", "--gtest_filter=ReadinessFixture.Bare"}));
  EXPECT_EQ(WRAPPED.status, 0) << "wrapped run exit status";
  EXPECT_THAT(WRAPPED.err, HasSubstr("[profile] --profile massif: no case that ran was built with "
                                     "the profiler guard; the massif wrap still recorded the "
                                     "whole process.\n"))
      << "wrapped notice";
}

/**
 * @test A benchmark with its own main(), written as the advanced guide shows,
 * ends as PERF_MAIN() does: 0 unprofiled, 4 with a failed request, and the
 * tests' own status when a test fails as well.
 */
TEST_F(ReadinessFixtureRun, CustomMainReturnsTheRunStatus) {
  const RunResult PLAIN = run(quick({}), CUSTOM_MAIN);
  EXPECT_EQ(PLAIN.status, 0) << "exit status without --profile";
  const RunResult FAILED = run(quick({"--profile", "nosuch"}), CUSTOM_MAIN);
  EXPECT_EQ(FAILED.status, 4) << "exit status with a failed request";
  EXPECT_THAT(FAILED.err,
              HasSubstr("[profile] --profile nosuch failed; the run exits with status 4:\n"))
      << "run-end report";
  env_.emplace_back("READINESS_FIXTURE_FAIL=1");
  const RunResult BOTH = run(quick({"--profile", "nosuch"}), CUSTOM_MAIN);
  EXPECT_EQ(BOTH.status, 1) << "exit status with a failed test and a failed request";
  EXPECT_THAT(BOTH.err, HasSubstr("[profile] --profile nosuch failed; the run exits with the "
                                  "tests' status 1:\n"))
      << "run-end report";
}

/**
 * @test The request the doctor fails because valgrind is not on PATH fails
 * the run: the notice says nothing is collected and the run will fail, the
 * run-end report repeats the doctor's message, and the run exits 4.
 */
TEST_F(ReadinessFixtureRun, MissingToolFailsTheRun) {
  const std::string DOCTOR = doctorJson({"--profile", "massif"});
  const std::string MESSAGE = selectedField(DOCTOR, "message");
  EXPECT_EQ(MESSAGE, "missing: valgrind not found on PATH") << "selected message";
  const RunResult RUN = run(quick({"--profile", "massif"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  const std::string NOTICE = "[FAIL] Profiler 'massif': " + MESSAGE + "\n   " +
                             selectedField(DOCTOR, "hint") +
                             "\n   Nothing is collected for this request; the run will fail "
                             "(exit status 4 if the tests pass).";
  EXPECT_THAT(RUN.err, HasSubstr(NOTICE)) << "run notice";
  EXPECT_EQ(countOf(RUN.err, NOTICE), 1U) << "notices for two guarded cases";
  EXPECT_THAT(RUN.err, HasSubstr("[profile] --profile massif failed; the run exits with status "
                                 "4:\n[profile]   massif: " +
                                 MESSAGE + "\n"))
      << "run-end report";
}

/* --------------------------------- perf --------------------------------- */

/** @test A run whose perf starts and stops reports no failure and exits 0. */
TEST_F(ReadinessFixtureRun, ReadyPerfReportsNoFailure) {
  dir_.install("fake_perf.sh", "perf");
  const RunResult RUN = run(quick({"--profile", "perf"}));
  EXPECT_EQ(RUN.status, 0) << "run exit status";
  EXPECT_THAT(RUN.err, Not(HasSubstr("[profile]")))
      << "run-end report or notice (a profiler was created)";
  expectOwnedAndGone("run");
}

/** @test A perf that does not run fails the run before anything is launched: exit 4. */
TEST_F(ReadinessFixtureRun, BrokenPerfFailsTheRun) {
  dir_.install("fake_perf.sh", "perf");
  env_.emplace_back("FAKE_PERF_MODE=broken");
  const RunResult RUN = run(quick({"--profile", "perf"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  EXPECT_THAT(RUN.err, HasSubstr("[FAIL] Profiler 'perf': unusable: " + dir_.path() +
                                 "/perf --version: exit status 2: WARNING: perf not found for "
                                 "kernel"))
      << "run notice";
}

/** @test A perf that may not open the counters fails the run: exit 4. */
TEST_F(ReadinessFixtureRun, DeniedPerfFailsTheRun) {
  dir_.install("fake_perf.sh", "perf");
  env_.emplace_back("FAKE_PERF_MODE=denied");
  const RunResult RUN = run(quick({"--profile", "perf"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  EXPECT_THAT(RUN.err, HasSubstr("[FAIL] Profiler 'perf': denied: perf stat cannot open the "
                                 "counters as this user"))
      << "run notice";
}

/**
 * @test The request is ready, and the run's perf fails as it starts: each
 * case reports it with perf's own words, and the run exits 4.
 */
TEST_F(ReadinessFixtureRun, PerfStartFailureFails) {
  dir_.install("fake_perf.sh", "perf");
  env_.emplace_back("FAKE_PERF_MODE=exit-early");
  EXPECT_EQ(selectedField(doctorJson({"--profile", "perf"}), "status"), "ok")
      << "selected status (the probes pass)";
  const RunResult RUN = run(quick({"--profile", "perf"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  for (const char* testCase : {"First", "Second"}) {
    EXPECT_THAT(RUN.err,
                HasSubstr(std::string{"[FAIL] Profiler 'perf' (ReadinessFixture."} + testCase +
                          "): unusable: perf ended (exit status 1) before the measured "
                          "phase: perf: Error: failed to open counters: No such process"))
        << "run report (" << testCase << ")";
  }
}

/**
 * @test The run's metadata (git describe) is taken when the first profiler is
 * created, so git runs once and before perf is first launched, for the
 * fixture and for a benchmark with its own main().
 */
TEST_F(ReadinessFixtureRun, MetadataBeforeTheProfiledWindow) {
  dir_.install("fake_perf.sh", "perf");
  dir_.install("fake_git.sh", "git");
  const std::string LAUNCH = "perf " + dir_.path() + PERF_STAT;
  for (const std::string& program : {std::string{FIXTURE}, std::string{CUSTOM_MAIN}}) {
    const std::string NAME = std::filesystem::path(program).filename().string();
    std::error_code ec;
    std::filesystem::remove(dir_.logPath(), ec);
    const RunResult RUN = run(quick({"--profile", "perf"}), program);
    EXPECT_EQ(RUN.status, 0) << "run exit status (" << NAME << ")";
    const std::string LOG = dir_.log();
    EXPECT_EQ(countOf(LOG, "git describe "), 1U) << "git describe runs (" << NAME << ")";
    const std::size_t GIT = LOG.find("git describe ");
    const std::size_t LAUNCHED = LOG.find(LAUNCH);
    if (LAUNCHED == std::string::npos) {
      ADD_FAILURE() << "perf was not launched (" << NAME << ")";
    } else if (GIT == std::string::npos || GIT > LAUNCHED) {
      ADD_FAILURE() << "git describe ran after perf was launched (" << NAME << ")";
    }
    expectOwnedAndGone("run (" + NAME + ")");
  }
}

/* --------------------------------- gperf --------------------------------- */

/**
 * @test Without --profile-analyze a gperf run needs no analyzer, and none is on
 * PATH: no notice, and the run exits 0.
 */
TEST_F(ReadinessFixtureRun, GperfWithoutAnalyzeExitsZero) {
  if (const std::string WHY = gperfUnusable(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  const RunResult RUN = run(quick({"--profile", "gperf"}));
  EXPECT_EQ(RUN.status, 0) << "run exit status without --profile-analyze";
  EXPECT_THAT(RUN.err, Not(HasSubstr("Profiler 'gperf'"))) << "run notice";
}

/**
 * @test An analyzer that runs but fails on this profile fails the run: its
 * report says how to run it by hand, and the run-end report names the case.
 */
TEST_F(ReadinessFixtureRun, GperfAnalyzerFailureFailsTheRun) {
  if (const std::string WHY = gperfUnusable(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  dir_.install("fake_pprof.sh", "google-pprof");
  env_.emplace_back("FAKE_PPROF_MODE=fail-on-profile");
  const RunResult RUN = run(quick({"--profile", "gperf", "--profile-analyze"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  EXPECT_THAT(RUN.err, HasSubstr("\n   Run it by hand to see why: " + dir_.path() +
                                 "/google-pprof --text --cum --lines "))
      << "analysis failure remedy";
  EXPECT_THAT(RUN.err, HasSubstr("[profile] --profile gperf --profile-analyze failed; the run "
                                 "exits with status 4:\n[profile]   gperf "
                                 "(ReadinessFixture.First): analysis: unusable: "))
      << "run-end report";
}

/** @test A heap request in a build without heap profiling fails the run: exit 4. */
TEST_F(ReadinessFixtureRun, GperfHeapWithoutSupportFailsTheRun) {
  if (const std::string WHY = gperfUnusable(); !WHY.empty()) {
    GTEST_SKIP() << WHY;
  }
  if (HEAP_BUILT) {
    GTEST_SKIP() << "heap profiling is compiled in (VERNIER_LINK_TCMALLOC=ON), so a heap request "
                    "is ready";
  }
  const std::string MESSAGE =
      selectedField(doctorJson({"--profile", "gperf", "--profile-args", "heap"}), "message");
  EXPECT_THAT(MESSAGE, HasSubstr("unsupported: heap profiling is not compiled in"))
      << "selected message";
  const RunResult RUN = run(quick({"--profile", "gperf", "--profile-args", "heap"}));
  EXPECT_EQ(RUN.status, 4) << "run exit status";
  EXPECT_THAT(RUN.err, HasSubstr("[FAIL] Profiler 'gperf': " + MESSAGE)) << "run notice";
}
