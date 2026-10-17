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

/// What a run whose tests pass ends with when its profiler request failed:
/// the demo run plainly with --profile compute-sanitizer.
constexpr int REQUEST_FAILED_EXIT = 4;

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
 * @test Run plainly with --profile compute-sanitizer, the demo is not taken
 *       for wrapped: the request fails, its report naming the command that
 *       wraps the binary in memcheck, and the measured case still runs and
 *       passes.
 *
 * The backend must decide "under the tool" from the tool, never from the
 * binary's name, which contains it. compute-sanitizer checks a process only
 * when it starts it, so a plain run's request fails and the run ends with
 * status 4, its tests having passed. The run needs the tool on PATH (without
 * it the report says the tool is missing instead) and a device (the measured
 * case runs the kernel), so it skips where KernelReportsNothing skips.
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

  // The artifact root is this run's directory, so nothing the run might
  // create lands in the working directory.
  const vg::ChildExit END = vg::runLogged(
      {DEMO, "--profile", "compute-sanitizer", "--profile-output-dir", DIR.string(), "--cycles",
       "1", "--repeats", "1", std::string("--gtest_filter=") + KERNEL_CASE, "--gtest_print_time=0"},
      DIR / "plain.txt");
  const std::string OUTPUT = vg::readText(DIR / "plain.txt");
  std::error_code ec;
  fs::remove_all(DIR, ec);

  EXPECT_TRUE(vg::exitedWith(END, REQUEST_FAILED_EXIT)) << vg::describe(END) << "\n" << OUTPUT;
  EXPECT_TRUE(vg::testPassed(OUTPUT, KERNEL_CASE)) << OUTPUT;
  EXPECT_NE(OUTPUT.find("did not start this one"), std::string::npos)
      << "the request was not refused as unwrapped:\n"
      << OUTPUT;
  EXPECT_EQ(OUTPUT.find("wrapping detected"), std::string::npos)
      << "the backend reported a wrap that is not there:\n"
      << OUTPUT;
  EXPECT_NE(OUTPUT.find("compute-sanitizer --tool=memcheck"), std::string::npos)
      << "the report names no wrap command:\n"
      << OUTPUT;
}

/* ----------------------------- The Hint ----------------------------- */

namespace {

/// The wrap a plain run's report names for its refused request: the text
/// after "Wrap it: " on its line; empty when the output has none.
std::string wrapRemedy(const std::string& output) {
  const std::string MARK = "Wrap it: ";
  std::istringstream in(output);
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t AT = line.find(MARK);
    if (AT != std::string::npos) {
      return line.substr(AT + MARK.size());
    }
  }
  return {};
}

/// The by-hand command of a wrap @p remedy: its text before "; or run it
/// with "; empty when there is none.
std::string byHandCommand(const std::string& remedy) {
  const std::size_t END = remedy.find("; or run it with ");
  return END == std::string::npos ? std::string() : remedy.substr(0, END);
}

/// The bench run command a wrap @p remedy offers: its text after "or run it
/// with " and before ", which wraps it"; empty when it offers none.
std::string benchRunRoute(const std::string& remedy) {
  const std::string FROM = "or run it with ";
  const std::size_t START = remedy.find(FROM);
  if (START == std::string::npos) {
    return {};
  }
  const std::size_t BEGIN = START + FROM.size();
  const std::size_t END = remedy.find(", which wraps it", BEGIN);
  return END == std::string::npos ? std::string() : remedy.substr(BEGIN, END - BEGIN);
}

/// @p text with every @p from replaced by @p to.
std::string replaced(std::string text, const std::string& from, const std::string& to) {
  for (std::size_t at = text.find(from); at != std::string::npos;
       at = text.find(from, at + to.size())) {
    text.replace(at, from.size(), to);
  }
  return text;
}

/// A new temporary directory whose name holds a '%', for the hint tests: the
/// tool expands '%' in its log's name, and the by-hand command runs from a
/// directory under this one.
fs::path scratchDirWithPercent() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo04-%-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

/// Runs the demo plainly from the working directory @p cwd with --profile
/// compute-sanitizer, the artifact root @p root and @p extraArgs, on the
/// measured case with one cycle and one repeat, and returns everything it
/// printed.
std::string plainProfileRun(const std::string& demo, const fs::path& cwd, const fs::path& root,
                            const std::vector<std::string>& extraArgs, const fs::path& outputFile) {
  std::vector<std::string> args = {"env",
                                   "-C",
                                   cwd.string(),
                                   demo,
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

/// @p command, a by-hand command taken from a report, with the binary and the
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

/// The report's by-hand command, run as printed. The demo runs plainly with
/// --profile compute-sanitizer from @p workDir, which is made first, under
/// the artifact root @p dir / "out"; its report names the log in @p workDir.
/// The by-hand command is taken from the report, filled in, and run through
/// the shell from @p dir, so whatever a misquoted command creates stays
/// inside it. The log must exist where the command says, in @p workDir, and
/// count no error, the program must pass, and the shell must exit 0.
void expectHintRunsAsPrinted(const std::string& demo, const fs::path& dir,
                             const fs::path& workDir) {
  std::error_code ec;
  fs::create_directories(workDir, ec);
  ASSERT_TRUE(fs::is_directory(workDir)) << "cannot create " << workDir;
  const std::string OUTPUT = plainProfileRun(demo, workDir, dir / "out", {}, dir / "plain.txt");
  const std::string COMMAND = byHandCommand(wrapRemedy(OUTPUT));
  ASSERT_FALSE(COMMAND.empty()) << "no by-hand command in the report:\n" << OUTPUT;
  const std::string FILLED = filledCommand(COMMAND, demo);

  const vg::ChildExit END = shellFrom(dir, "", FILLED, dir / "byhand.txt");
  const std::string BY_HAND = vg::readText(dir / "byhand.txt");
  const fs::path LOG = workDir / "sanitizer.log";
  const std::string LOG_TEXT = vg::readText(LOG);

  EXPECT_TRUE(fs::exists(LOG)) << "the report's command wrote no log where it says (" << LOG
                               << "):\n"
                               << FILLED << "\n"
                               << BY_HAND;
  EXPECT_TRUE(vg::testPassed(BY_HAND, KERNEL_CASE) && vg::oneTestPassed(BY_HAND)) << BY_HAND;
  EXPECT_EQ(check::errorSummary(LOG_TEXT), 0) << LOG_TEXT;
  EXPECT_TRUE(vg::exitedWith(END, 0)) << vg::describe(END) << "\n" << BY_HAND;
}

} // namespace

/**
 * @test A plain run's report names the wrap by hand, then bench run, each
 *       with the tool asked for: memcheck by default, with no --profile-args,
 *       and a tool named with --profile-args with it.
 *
 * Runs the demo plainly with --profile compute-sanitizer, reads the wrap its
 * report names, and runs it again with --profile-args racecheck. The by-hand
 * command gives the tool an exit code for errors and names its log whole, in
 * the run's working directory; bench run routes the tool the request names,
 * so it is offered for racecheck too. Skipped where KernelReportsNothing
 * skips: the report names a wrap where the tool is on PATH.
 */
TEST(ComputeSanitizer, PlainRunHintShape) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  // DIR's name is mkdtemp's: nothing in it the tool or the shell would read.
  const std::string LOG = (DIR / "sanitizer.log").string();
  const std::string OUTPUT = plainProfileRun(DEMO, DIR, DIR / "out", {}, DIR / "plain.txt");
  const std::string REMEDY = wrapRemedy(OUTPUT);
  EXPECT_EQ(byHandCommand(REMEDY), "compute-sanitizer --tool=memcheck --error-exitcode 5 "
                                   "--log-file=" +
                                       LOG + " <this-binary> --profile compute-sanitizer [...]")
      << OUTPUT;
  EXPECT_EQ(benchRunRoute(REMEDY), "bench run --profile compute-sanitizer") << OUTPUT;
  EXPECT_EQ(REMEDY.find("--profile-args"), std::string::npos)
      << "the default tool is passed as a mode, which the request did not name:\n"
      << OUTPUT;

  const std::string RACECHECK = plainProfileRun(
      DEMO, DIR, DIR / "out", {"--profile-args", "racecheck"}, DIR / "racecheck.txt");
  const std::string RACECHECK_REMEDY = wrapRemedy(RACECHECK);
  EXPECT_EQ(byHandCommand(RACECHECK_REMEDY),
            "compute-sanitizer --tool=racecheck --error-exitcode 5 --log-file=" + LOG +
                " <this-binary> --profile compute-sanitizer --profile-args racecheck [...]")
      << RACECHECK;
  EXPECT_EQ(benchRunRoute(RACECHECK_REMEDY),
            "bench run --profile compute-sanitizer --profile-args racecheck")
      << RACECHECK;

  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test The report's by-hand command, run as printed with nothing prepared,
 *       writes the log where it says, named whole.
 *
 * In a directory under one whose name holds a '%', which the log's name
 * must carry as "%%" for the tool: compute-sanitizer joins a relative log
 * name to its working directory and reads a '%' there as a macro.
 * expectHintRunsAsPrinted() says what runs and what is checked. Skipped where
 * KernelReportsNothing skips.
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

  expectHintRunsAsPrinted(DEMO, DIR, DIR / "run");

  if (HasFailure()) {
    std::printf("the hint's run kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test The same in a directory whose name holds spaces: the command quotes
 *       the log's name, so the shell keeps it one argument.
 *
 * Skipped where KernelReportsNothing skips.
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

  expectHintRunsAsPrinted(DEMO, DIR, DIR / "run with spaces");

  if (HasFailure()) {
    std::printf("the hint's run kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test The same in a directory whose name holds a quote: the command writes
 *       the log's name so that the shell reads it back whole.
 *
 * Skipped where KernelReportsNothing skips.
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

  expectHintRunsAsPrinted(DEMO, DIR, DIR / "Bob's run");

  if (HasFailure()) {
    std::printf("the hint's run kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test For a tool named with --profile-args, the report's wraps run that
 *       tool: bench run with the request, and compute-sanitizer by hand with
 *       --tool=<the tool named>.
 *
 * For each of racecheck, synccheck and initcheck, the demo runs plainly with
 * --profile-args naming the tool; its report must offer bench run with that
 * request, and its by-hand command, filled in and run through the shell with
 * a stand-in for compute-sanitizer first on PATH that records its arguments
 * and starts nothing, must start the tool with --tool=<the tool named>, the
 * exit code for errors, the log in the plain run's working directory, and
 * the program with the same tool named. Skipped where KernelReportsNothing
 * skips.
 */
TEST(ComputeSanitizer, NamedToolHintRunsThatTool) {
  const std::string CANNOT = reasonNotToRunAKernel();
  if (!CANNOT.empty()) {
    GTEST_SKIP() << CANNOT;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";
  const fs::path STAND_IN = DIR / "stand-in";
  ASSERT_TRUE(writeRecordingStandIn(STAND_IN)) << "cannot write a stand-in in " << STAND_IN;
  const char* const INHERITED = std::getenv("PATH");
  const std::string SEARCH_PATH =
      STAND_IN.string() + ":" + (INHERITED != nullptr ? INHERITED : "/usr/bin:/bin");

  for (const char* const TOOL : OTHER_TOOLS) {
    SCOPED_TRACE(TOOL);
    const std::string NAME = TOOL;
    const std::string OUTPUT = plainProfileRun(
        DEMO, DIR, DIR / (NAME + "-out"), {"--profile-args", NAME}, DIR / (NAME + "-plain.txt"));
    const std::string REMEDY = wrapRemedy(OUTPUT);
    EXPECT_EQ(benchRunRoute(REMEDY), "bench run --profile compute-sanitizer --profile-args " + NAME)
        << OUTPUT;
    const std::string COMMAND = byHandCommand(REMEDY);
    EXPECT_FALSE(COMMAND.empty()) << "no by-hand command in the report:\n" << OUTPUT;
    if (COMMAND.empty()) {
      continue;
    }

    std::error_code ec;
    fs::remove(STAND_IN / "arguments", ec);
    const vg::ChildExit END =
        shellFrom(DIR, SEARCH_PATH, filledCommand(COMMAND, DEMO), DIR / (NAME + "-byhand.txt"));
    const std::string BY_HAND = vg::readText(DIR / (NAME + "-byhand.txt"));
    const std::vector<std::string> EXPECTED = {"--tool=" + NAME,
                                               "--error-exitcode",
                                               "5",
                                               "--log-file=" + (DIR / "sanitizer.log").string(),
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
