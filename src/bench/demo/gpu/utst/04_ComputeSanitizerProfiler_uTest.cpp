/**
 * @file 04_ComputeSanitizerProfiler_uTest.cpp
 * @brief The check behind walkthrough 17: Compute Sanitizer names demo 04's
 *        unguarded read, and reports nothing for the shared kernel.
 *
 * Not part of demo 04, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_Gpu_04_ComputeSanitizerProfiler, whose path
 * the build passes in, with the unguarded source's) under compute-sanitizer
 * as a child and reads the report each run left;
 * 04_ComputeSanitizerProfiler_Check.hpp holds the runs and the report
 * reading, which 04_ComputeSanitizerProfiler_Report_uTest.cpp tests apart,
 * without CUDA. ctest runs it under the demo and compute-sanitizer labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L compute-sanitizer
 *   ./build/bin/tests/TestDemoComputeSanitizer      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/gpu/utst/04_ComputeSanitizerProfiler_Check.hpp"

#include "src/bench/inc/ProfilerEnv.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace check = vernier::bench::demo::sanitizer_check;
namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The demo binary the checks run, and the unguarded source; the build passes
/// both paths.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_04_BINARY;
constexpr const char* UNGUARDED_SOURCE = VERNIER_DEMO_04_UNGUARDED_SOURCE;

/// The demo's case that runs past the end, and the one the checks run as the
/// control.
constexpr const char* UNGUARDED_CASE = "ComputeSanitizer.SaxpyUnguarded";
constexpr const char* KERNEL_CASE = "ComputeSanitizer.SaxpyKernel";

/// The kernel that holds it, and its file's name as the report prints it.
constexpr const char* UNGUARDED_KERNEL = "saxpyUnguarded";
constexpr const char* UNGUARDED_FILE = "04_ComputeSanitizerProfiler_Unguarded.cu";

/// The demo's sizes: one thread of the last block past the end, so the read
/// lies right after the vectors' 1,048,575 floats.
constexpr const char* EXPECTED_THREAD = "by thread (255,0,0) in block (4095,0,0)";
constexpr long EXPECTED_BYTES_AFTER = 1;
constexpr long EXPECTED_ALLOCATION_BYTES = 1048575L * 4L;

/// What the tool exits with when it reported an error: not 1, which is also
/// what a failed test exits with.
constexpr int SANITIZER_ERROR_EXIT = 99;

/// The demo binary's canonical path, or an empty string when it is missing.
std::string demoPath() {
  std::error_code ec;
  const fs::path PATH = fs::canonical(DEMO_BINARY, ec);
  return ec ? std::string() : PATH.string();
}

/// A new temporary directory for one check's runs; empty when it cannot be
/// made.
fs::path scratchDir() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo04-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

/// Why a check that runs a kernel cannot run here, decided before anything
/// runs: a sanitizer build, the tool not on PATH, or the runtime seeing no
/// device (its own words); empty when none of these holds.
std::string reasonNotToRunAKernel() {
  if (const std::string SANITIZED = check::reasonNotToRunTheDemo(); !SANITIZED.empty()) {
    return SANITIZED;
  }
  if (!vernier::bench::profiler_env::isOnPath("compute-sanitizer")) {
    return "compute-sanitizer is not on PATH; this test runs the demo under it";
  }
  int devices = 0;
  const cudaError_t ASKED = cudaGetDeviceCount(&devices);
  if (ASKED != cudaSuccess) {
    return std::string("no CUDA device: cudaGetDeviceCount reported \"") +
           cudaGetErrorString(ASKED) + "\"";
  }
  if (devices < 1) {
    return "no CUDA device: cudaGetDeviceCount counted none";
  }
  return {};
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test memcheck reports SaxpyUnguarded's read past the end, at the
 *       unguarded line, by the one thread past the end.
 *
 * Runs the demo binary under compute-sanitizer's memcheck as `bench run
 * --profile compute-sanitizer` wraps it, plus an exit code for errors.
 * SaxpyUnguarded must run to its end and pass (it launches once and reports
 * what the device said); the report must hold exactly one invalid access,
 * the one launch's one thread past the end, and that one must be a read of
 * four bytes in the unguarded kernel at the unguarded statement's line, by
 * thread (255,0,0) in block (4095,0,0), out of bounds, one byte after an
 * allocation of the vectors' size; the summary must count it, and the tool
 * must exit with the error status it was given.
 *
 * Skipped only in a build with the address or the thread sanitizer (where the
 * demo does not run, as the check support says), where compute-sanitizer is
 * not on PATH, or where the runtime sees no device, all decided before
 * anything runs. Any other way of not reaching the case fails, with what the
 * run printed.
 */
TEST(ComputeSanitizer, FindsTheUnguardedRead) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const std::size_t LINE =
      check::lineOf(vg::readText(UNGUARDED_SOURCE), check::UNGUARDED_STATEMENT);
  ASSERT_NE(LINE, 0u) << "the unguarded source does not hold \"" << check::UNGUARDED_STATEMENT
                      << "\" on exactly one line: " << UNGUARDED_SOURCE;
  const std::string LOCATION = std::string(UNGUARDED_FILE) + ":" + std::to_string(LINE);
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const check::SanitizerRun RUN =
      check::runUnderSanitizer(DEMO, UNGUARDED_CASE, {}, DIR, SANITIZER_ERROR_EXIT);
  ASSERT_TRUE(vg::testsStarted(RUN.output))
      << "SaxpyUnguarded did not start under compute-sanitizer: the tool " << vg::describe(RUN.end)
      << " (its output is in " << DIR << "). The run printed:\n"
      << vg::lastLines(RUN.log + RUN.output, 40);
  ASSERT_TRUE(vg::testPassed(RUN.output, UNGUARDED_CASE) && vg::oneTestPassed(RUN.output))
      << "SaxpyUnguarded did not run to its end under compute-sanitizer (its output is in " << DIR
      << "):\n"
      << vg::lastLines(RUN.output);

  const std::vector<check::InvalidAccess> ACCESSES = check::invalidAccesses(RUN.log);
  const long SUMMARY = check::errorSummary(RUN.log);
  std::printf("[ComputeSanitizer.FindsTheUnguardedRead]  SaxpyUnguarded: %zu invalid access(es) "
              "read, ERROR SUMMARY %ld, looking for %s\n",
              ACCESSES.size(), SUMMARY, LOCATION.c_str());
  ASSERT_FALSE(ACCESSES.empty())
      << "compute-sanitizer reported no invalid access for SaxpyUnguarded: the unguarded kernel "
         "has stopped running past the end (log "
      << DIR / "sanitizer.log" << ")";
  EXPECT_EQ(ACCESSES.size(), 1u)
      << "one launch with one thread past the end was expected to give one access:\n"
      << ACCESSES[1].text;
  const check::InvalidAccess& READ = ACCESSES.front();
  EXPECT_EQ(READ.kind, "Invalid __global__ read of size 4 bytes") << READ.text;
  EXPECT_TRUE(check::frameAt(READ.frame, UNGUARDED_KERNEL, LOCATION))
      << "the read is not reported in " << UNGUARDED_KERNEL << " at " << LOCATION << ":\n"
      << READ.text;
  EXPECT_EQ(READ.thread, EXPECTED_THREAD) << READ.text;
  EXPECT_NE(READ.address.find("is out of bounds"), std::string::npos) << READ.text;
  EXPECT_EQ(check::bytesAfterAllocation(READ), EXPECTED_BYTES_AFTER) << READ.text;
  EXPECT_EQ(check::allocationSize(READ), EXPECTED_ALLOCATION_BYTES) << READ.text;
  EXPECT_GE(SUMMARY, 1) << "the summary does not count the read (log " << DIR / "sanitizer.log"
                        << ")";
  EXPECT_TRUE(vg::exitedWith(RUN.end, SANITIZER_ERROR_EXIT))
      << "the tool " << vg::describe(RUN.end) << ", not the --error-exitcode "
      << SANITIZER_ERROR_EXIT << " it was given for a run with errors";

  if (HasFailure()) {
    std::printf("compute-sanitizer output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test memcheck reports nothing for SaxpyKernel: the shared kernel, its
 *       guard included, stays inside its buffers.
 *
 * Runs the measured case under the same wrap with one cycle and one repeat
 * (the harness's warmup launches and one measured launch): it must run to
 * its end and pass, the report must hold no invalid access and count no
 * error, and the tool must exit 0. Skipped where FindsTheUnguardedRead
 * skips.
 */
TEST(ComputeSanitizer, KernelReportsNothing) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const check::SanitizerRun RUN = check::runUnderSanitizer(
      DEMO, KERNEL_CASE, {"--cycles", "1", "--repeats", "1"}, DIR, SANITIZER_ERROR_EXIT);
  ASSERT_TRUE(vg::testsStarted(RUN.output))
      << "SaxpyKernel did not start under compute-sanitizer: the tool " << vg::describe(RUN.end)
      << " (its output is in " << DIR << "). The run printed:\n"
      << vg::lastLines(RUN.log + RUN.output, 40);
  const std::vector<check::InvalidAccess> ACCESSES = check::invalidAccesses(RUN.log);
  const long SUMMARY = check::errorSummary(RUN.log);
  std::printf("[ComputeSanitizer.KernelReportsNothing]  SaxpyKernel: %zu invalid access(es) "
              "read, ERROR SUMMARY %ld\n",
              ACCESSES.size(), SUMMARY);
  EXPECT_TRUE(vg::testPassed(RUN.output, KERNEL_CASE) && vg::oneTestPassed(RUN.output))
      << "SaxpyKernel did not run to its end under compute-sanitizer:\n"
      << vg::lastLines(RUN.output);
  EXPECT_TRUE(ACCESSES.empty()) << "compute-sanitizer reported an invalid access for the shared "
                                   "kernel (log "
                                << DIR / "sanitizer.log" << "):\n"
                                << (ACCESSES.empty() ? std::string() : ACCESSES.front().text);
  EXPECT_EQ(SUMMARY, 0) << "the summary counts errors for the shared kernel (log "
                        << DIR / "sanitizer.log" << ")";
  EXPECT_TRUE(vg::exitedWith(RUN.end, 0))
      << "the tool " << vg::describe(RUN.end) << " for a run without errors";

  if (HasFailure()) {
    std::printf("compute-sanitizer output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test Run without the tool, SaxpyUnguarded reports SKIPPED, says how to
 *       run it, and does not run. Needs neither the tool nor a device;
 *       skipped in a build with the address or the thread sanitizer, where
 *       the demo does not run plainly.
 */
TEST(ComputeSanitizer, UnguardedSkipsOutsideTheTool) {
  if (const std::string SANITIZED = check::reasonNotToRunTheDemo(); !SANITIZED.empty()) {
    GTEST_SKIP() << SANITIZED;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const vg::ChildExit END =
      vg::runLogged({DEMO, std::string("--gtest_filter=") + UNGUARDED_CASE, "--gtest_print_time=0"},
                    DIR / "plain.txt");
  const std::string OUTPUT = vg::readText(DIR / "plain.txt");
  std::error_code ec;
  fs::remove_all(DIR, ec);

  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << OUTPUT;
  EXPECT_NE(OUTPUT.find(std::string("[  SKIPPED ] ") + UNGUARDED_CASE), std::string::npos)
      << OUTPUT;
  EXPECT_FALSE(vg::testPassed(OUTPUT, UNGUARDED_CASE)) << OUTPUT;
  EXPECT_NE(OUTPUT.find("compute-sanitizer --tool=memcheck"), std::string::npos) << OUTPUT;
  EXPECT_NE(OUTPUT.find("bench run --profile compute-sanitizer"), std::string::npos) << OUTPUT;
}

/**
 * @test Run plainly with --profile compute-sanitizer, the demo reports no
 *       wrap and prints the wrap command.
 *
 * The backend must decide "under the tool" from the tool, never from the
 * binary's name, which contains it. The run needs the tool on PATH (without
 * it the backend is not created and prints nothing of its own) and a device
 * (the measured case runs the kernel), so it skips where KernelReportsNothing
 * skips.
 */
TEST(ComputeSanitizer, PlainRunIsNotWrapped) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  // The artifact root is this run's directory, so the per-test folder the
  // backend creates lands there and not in the working directory.
  const vg::ChildExit END = vg::runLogged(
      {DEMO, "--profile", "compute-sanitizer", "--profile-output-dir", DIR.string(), "--cycles",
       "1", "--repeats", "1", std::string("--gtest_filter=") + KERNEL_CASE, "--gtest_print_time=0"},
      DIR / "plain.txt");
  const std::string OUTPUT = vg::readText(DIR / "plain.txt");
  std::error_code ec;
  fs::remove_all(DIR, ec);

  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << OUTPUT;
  EXPECT_TRUE(vg::testPassed(OUTPUT, KERNEL_CASE)) << OUTPUT;
  EXPECT_EQ(OUTPUT.find("wrapping detected"), std::string::npos)
      << "the backend reported a wrap that is not there:\n"
      << OUTPUT;
  EXPECT_NE(OUTPUT.find("compute-sanitizer --tool=memcheck"), std::string::npos)
      << "the backend did not print its wrap command:\n"
      << OUTPUT;
}

/* ----------------------------- The Hint ----------------------------- */

namespace {

/// The lines of the backend's not-wrapped hint in @p output, without their
/// "[compute-sanitizer] " prefix, in order.
std::vector<std::string> hintLines(const std::string& output) {
  constexpr const char* PREFIX = "[compute-sanitizer] ";
  std::vector<std::string> lines;
  std::istringstream in(output);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(PREFIX, 0) == 0) {
      lines.push_back(line.substr(std::string(PREFIX).size()));
    }
  }
  return lines;
}

/// The by-hand command of a hint: from the first command line that starts
/// with "mkdir -p " or "compute-sanitizer " through the lines a trailing
/// backslash continues, joined with one space; empty when there is none.
std::string byHandCommand(const std::vector<std::string>& lines) {
  std::string command;
  bool inCommand = false;
  for (const std::string& raw : lines) {
    std::string line = check::trimmedStart(raw);
    if (!inCommand) {
      if (line.rfind("mkdir -p ", 0) != 0 && line.rfind("compute-sanitizer ", 0) != 0) {
        continue;
      }
      inCommand = true;
    }
    const bool CONTINUED = !line.empty() && line.back() == '\\';
    if (CONTINUED) {
      line.pop_back();
    }
    command += (command.empty() ? "" : " ") + check::trimmedStart(line);
    if (!CONTINUED) {
      break;
    }
  }
  return command;
}

/// @p text with every @p from replaced by @p to.
std::string replaced(std::string text, const std::string& from, const std::string& to) {
  for (std::size_t at = text.find(from); at != std::string::npos;
       at = text.find(from, at + to.size())) {
    text.replace(at, from.size(), to);
  }
  return text;
}

/// True when one of @p lines is @p line.
bool hasLine(const std::vector<std::string>& lines, const std::string& line) {
  for (const std::string& candidate : lines) {
    if (candidate == line) {
      return true;
    }
  }
  return false;
}

/// True when one of @p lines contains @p text.
bool anyLineContains(const std::vector<std::string>& lines, const std::string& text) {
  for (const std::string& candidate : lines) {
    if (candidate.find(text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

/// A new temporary directory whose name holds a '%', for the hint tests:
/// the tool expands '%' in its log's name, so the hint must write it "%%".
fs::path scratchDirWithPercent() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo04-%-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

/// Runs the demo plainly with --profile compute-sanitizer, the artifact root
/// @p root and @p extraArgs, on the measured case with one cycle and one
/// repeat, and returns everything it printed.
std::string plainProfileRun(const std::string& demo, const fs::path& root,
                            const std::vector<std::string>& extraArgs, const fs::path& outputFile) {
  std::vector<std::string> args = {demo,
                                   "--profile",
                                   "compute-sanitizer",
                                   "--profile-output-dir",
                                   root.string(),
                                   "--cycles",
                                   "1",
                                   "--repeats",
                                   "1",
                                   std::string("--gtest_filter=") + KERNEL_CASE,
                                   "--gtest_print_time=0"};
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());
  vg::runLogged(args, outputFile);
  return vg::readText(outputFile);
}

/// The folder the backend names for the measured case under @p root.
std::string kernelFolder(const fs::path& root) {
  return root.string() + "/" + KERNEL_CASE + ".compute-sanitizer";
}

/// @p command, a by-hand command taken from a hint, with the binary and the
/// arguments filled in: the measured case, one cycle and one repeat.
std::string filledCommand(const std::string& command, const std::string& demo) {
  return replaced(replaced(command, "<this-binary>", demo), "[...]",
                  std::string("--cycles 1 --repeats 1 --gtest_filter=") + KERNEL_CASE +
                      " --gtest_print_time=0");
}

/// Runs @p command through the shell from @p dir, with @p path as PATH when
/// it is not empty, and with stdout and stderr in @p outputFile.
vg::ChildExit shellFrom(const fs::path& dir, const std::string& path, const std::string& command,
                        const fs::path& outputFile) {
  std::vector<std::string> args = {"env", "-C", dir.string()};
  if (!path.empty()) {
    args.push_back("PATH=" + path);
  }
  args.insert(args.end(), {"sh", "-c", command});
  return vg::runLogged(args, outputFile);
}

/// The tools compute-sanitizer runs besides memcheck, as --profile-args names
/// them.
constexpr const char* OTHER_TOOLS[] = {"racecheck", "synccheck", "initcheck"};

/// Writes @p dir/compute-sanitizer, a stand-in for the tool that writes the
/// arguments it was started with, one per line, to @p dir/arguments and
/// starts nothing. Returns false when it cannot.
bool writeRecordingStandIn(const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path SCRIPT = dir / "compute-sanitizer";
  std::ofstream out(SCRIPT);
  out << "#!/bin/sh\n"
         "# A stand-in for compute-sanitizer: records its arguments, starts nothing.\n"
         "printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/arguments\"\n";
  out.close();
  fs::permissions(SCRIPT, fs::perms::owner_all, fs::perm_options::replace, ec);
  return !out.fail() && !ec;
}

/// The lines of @p text, in order.
std::vector<std::string> linesOf(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

/// The hint's by-hand command, run as printed where its folder does not
/// exist. The demo runs plainly with --profile compute-sanitizer under the
/// artifact root @p root; the by-hand command is taken from its hint and
/// filled in, the folder the plain run made is removed, and the command runs
/// through the shell from @p dir, so whatever a misquoted command creates
/// stays inside it. The log must exist where the hint says and count no
/// error, the program must pass, and the shell must exit 0.
void expectHintRunsAsPrinted(const std::string& demo, const fs::path& dir, const fs::path& root) {
  const fs::path FOLDER = kernelFolder(root);
  const std::string OUTPUT = plainProfileRun(demo, root, {}, dir / "plain.txt");
  const std::string COMMAND = byHandCommand(hintLines(OUTPUT));
  ASSERT_FALSE(COMMAND.empty()) << "no by-hand command in the hint:\n" << OUTPUT;
  const std::string FILLED = filledCommand(COMMAND, demo);

  // The plain run made the folder; the hint must work without it.
  std::error_code ec;
  fs::remove_all(FOLDER, ec);
  ASSERT_FALSE(fs::exists(FOLDER)) << "could not remove " << FOLDER;
  const vg::ChildExit END = shellFrom(dir, "", FILLED, dir / "byhand.txt");
  const std::string BY_HAND = vg::readText(dir / "byhand.txt");
  const fs::path LOG = FOLDER / "sanitizer.log";
  const std::string LOG_TEXT = vg::readText(LOG);

  EXPECT_TRUE(fs::exists(LOG)) << "the hint's command wrote no log where it says (" << LOG << "):\n"
                               << FILLED << "\n"
                               << BY_HAND;
  EXPECT_TRUE(vg::testPassed(BY_HAND, KERNEL_CASE) && vg::oneTestPassed(BY_HAND)) << BY_HAND;
  EXPECT_EQ(check::errorSummary(LOG_TEXT), 0) << LOG_TEXT;
  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << BY_HAND;
}

} // namespace

/**
 * @test The not-wrapped hint names bench run first for memcheck only, then a
 *       by-hand command that makes the log folder before the tool opens its
 *       log, with '%' written for the tool and --profile-args only for a
 *       tool other than memcheck.
 *
 * Runs the demo plainly with --profile compute-sanitizer under an artifact
 * root whose name holds a '%', reads the hint's lines, and runs it again
 * with --profile-args racecheck, for which the hint must name no bench run:
 * the wrap bench run builds runs memcheck whatever --profile-args says.
 * Skipped where KernelReportsNothing skips: the hint is printed by the
 * backend, which the registry creates only with the tool on PATH, when the
 * measured case reaches its measurement.
 */
TEST(ComputeSanitizer, PlainRunHintShape) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDirWithPercent();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";
  const fs::path ROOT = DIR / "out";
  const std::string FOLDER = kernelFolder(ROOT);

  const std::string OUTPUT = plainProfileRun(DEMO, ROOT, {}, DIR / "plain.txt");
  const std::vector<std::string> LINES = hintLines(OUTPUT);
  EXPECT_TRUE(hasLine(LINES, "  bench run <this-binary> --profile compute-sanitizer -- [...]"))
      << OUTPUT;
  EXPECT_TRUE(hasLine(LINES, "  mkdir -p " + FOLDER +
                                 " && compute-sanitizer --tool=memcheck --log-file=" +
                                 vg::escapePercent(FOLDER) + "/sanitizer.log \\"))
      << OUTPUT;
  EXPECT_TRUE(hasLine(LINES, "      <this-binary> --profile compute-sanitizer [...]")) << OUTPUT;
  EXPECT_FALSE(anyLineContains(LINES, "--profile-args"))
      << "the default tool is passed as a mode, which the registry reports unchecked:\n"
      << OUTPUT;

  const std::string RACECHECK =
      plainProfileRun(DEMO, ROOT, {"--profile-args", "racecheck"}, DIR / "racecheck.txt");
  const std::vector<std::string> RACECHECK_LINES = hintLines(RACECHECK);
  EXPECT_FALSE(anyLineContains(RACECHECK_LINES, "bench run"))
      << "the hint offers bench run for racecheck, whose wrap runs memcheck:\n"
      << RACECHECK;
  EXPECT_TRUE(hasLine(RACECHECK_LINES, "  mkdir -p " + FOLDER +
                                           " && compute-sanitizer --tool=racecheck --log-file=" +
                                           vg::escapePercent(FOLDER) + "/sanitizer.log \\"))
      << RACECHECK;
  EXPECT_TRUE(
      hasLine(RACECHECK_LINES,
              "      <this-binary> --profile compute-sanitizer --profile-args racecheck [...]"))
      << RACECHECK;

  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test The hint's by-hand command, run as printed where its folder does not
 *       exist, writes the log where it says.
 *
 * Under an artifact root in a directory whose name holds a '%', which the
 * log's name must carry as "%%" for the tool; expectHintRunsAsPrinted() says
 * what runs and what is checked. Skipped where KernelReportsNothing skips.
 */
TEST(ComputeSanitizer, HintRunsOnTheFirstRun) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDirWithPercent();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  expectHintRunsAsPrinted(DEMO, DIR, DIR / "out");

  if (HasFailure()) {
    std::printf("the hint's run kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test The same where the artifact root's name holds spaces: the hint
 *       quotes the folder and the log for the shell, so each stays one
 *       argument.
 *
 * Unquoted, the shell splits both at the spaces: mkdir makes three folders,
 * none of them the one the hint names, and the tool is given a log name cut
 * at the first space. Skipped where KernelReportsNothing skips.
 */
TEST(ComputeSanitizer, HintRunsWithSpacesInItsPath) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDirWithPercent();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  expectHintRunsAsPrinted(DEMO, DIR, DIR / "out with spaces");

  if (HasFailure()) {
    std::printf("the hint's run kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test The same where the artifact root's name holds a quote: the hint
 *       writes it so that the shell reads the folder and the log back whole.
 *
 * Unquoted, the quotes in the folder and in the log open and close a string
 * across the "&&", so the tool's command becomes part of mkdir's arguments
 * and never runs. Skipped where KernelReportsNothing skips.
 */
TEST(ComputeSanitizer, HintRunsWithAQuoteInItsPath) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDirWithPercent();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  expectHintRunsAsPrinted(DEMO, DIR, DIR / "Bob's output");

  if (HasFailure()) {
    std::printf("the hint's run kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test For a tool named with --profile-args, the hint offers the tool by
 *       hand only, and that command, run as printed, starts
 *       compute-sanitizer with the tool named.
 *
 * The wrap bench run builds runs memcheck whatever --profile-args says, so
 * the hint for racecheck, synccheck or initcheck must not offer it. For each
 * of the three, the demo runs plainly with --profile-args naming the tool;
 * its hint must name no bench run, and its by-hand command, filled in and
 * run through the shell after the folder the plain run made is removed, with
 * a stand-in for compute-sanitizer first on PATH that records its arguments
 * and starts nothing, must make the folder and start the tool with
 * --tool=<the tool named>, the log where the hint says, and the program with
 * the same tool named. Skipped where KernelReportsNothing skips.
 */
TEST(ComputeSanitizer, NamedToolHintRunsThatTool) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDirWithPercent();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";
  const fs::path STAND_IN = DIR / "stand-in";
  ASSERT_TRUE(writeRecordingStandIn(STAND_IN)) << "cannot write a stand-in in " << STAND_IN;
  const char* const INHERITED = std::getenv("PATH");
  const std::string SEARCH_PATH =
      STAND_IN.string() + ":" + (INHERITED != nullptr ? INHERITED : "/usr/bin:/bin");

  for (const char* const TOOL : OTHER_TOOLS) {
    SCOPED_TRACE(TOOL);
    const std::string NAME = TOOL;
    const fs::path ROOT = DIR / (NAME + "-out");
    const std::string FOLDER = kernelFolder(ROOT);
    const std::string OUTPUT =
        plainProfileRun(DEMO, ROOT, {"--profile-args", NAME}, DIR / (NAME + "-plain.txt"));
    const std::vector<std::string> LINES = hintLines(OUTPUT);
    EXPECT_FALSE(anyLineContains(LINES, "bench run"))
        << "the hint offers bench run for " << NAME << ", whose wrap runs memcheck:\n"
        << OUTPUT;
    const std::string COMMAND = byHandCommand(LINES);
    EXPECT_FALSE(COMMAND.empty()) << "no by-hand command in the hint:\n" << OUTPUT;
    if (COMMAND.empty()) {
      continue;
    }

    std::error_code ec;
    fs::remove_all(FOLDER, ec);
    fs::remove(STAND_IN / "arguments", ec);
    const vg::ChildExit END =
        shellFrom(DIR, SEARCH_PATH, filledCommand(COMMAND, DEMO), DIR / (NAME + "-byhand.txt"));
    const std::string BY_HAND = vg::readText(DIR / (NAME + "-byhand.txt"));
    const std::vector<std::string> EXPECTED = {"--tool=" + NAME,
                                               "--log-file=" + vg::escapePercent(FOLDER) +
                                                   "/sanitizer.log",
                                               DEMO,
                                               "--profile",
                                               "compute-sanitizer",
                                               "--profile-args",
                                               NAME,
                                               "--cycles",
                                               "1",
                                               "--repeats",
                                               "1",
                                               std::string("--gtest_filter=") + KERNEL_CASE,
                                               "--gtest_print_time=0"};
    EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << BY_HAND;
    EXPECT_TRUE(fs::is_directory(FOLDER)) << "the command made no folder at " << FOLDER;
    EXPECT_EQ(linesOf(vg::readText(STAND_IN / "arguments")), EXPECTED)
        << "the command did not start compute-sanitizer with " << NAME << ":\n"
        << COMMAND;
  }

  if (HasFailure()) {
    std::printf("the hint's runs kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}
