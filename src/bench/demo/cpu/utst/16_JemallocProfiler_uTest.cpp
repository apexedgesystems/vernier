/**
 * @file 16_JemallocProfiler_uTest.cpp
 * @brief The check behind walkthrough 22: jemalloc's profiler puts JoinV0's
 *        bytes under joinV0, and JoinV1 allocates a small fraction of them.
 *
 * Not part of demo 16, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_16_JemallocProfiler, whose path the build
 * passes in) as a child, as `bench run --profile jemalloc` wraps it: through
 * env, with jemalloc preloaded with the options bench run sets
 * (jemallocWrapEnvironment()) and the wrap's variables, then reads the folder
 * the run left: the one exit dump, of the demo, which jeprof reads.
 * Walkthrough 15's check support starts the child and reads its output
 * (12_MemcheckProfiler_Check.hpp), as demo 14's check uses it. ctest runs it
 * under the demo and jemalloc labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L jemalloc
 *   ./build/bin/tests/TestDemoJemalloc      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"

#include "src/bench/demo/helpers/SkipUnlessUnderValgrind.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerJemalloc.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace demo = vernier::bench::demo;
namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The demo binary the checks run; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_16_BINARY;

/// The demo's cases, and the functions jeprof names for their bytes.
constexpr const char* V0_CASE = "Jemalloc.JoinV0";
constexpr const char* V1_CASE = "Jemalloc.JoinV1";
constexpr const char* V0_FUNCTION = "vernier::bench::demo::joinV0";
constexpr const char* V1_FUNCTION = "vernier::bench::demo::joinV1";

/// Measured calls per run: with the check of the answer and the warmup,
/// 102 calls of the version.
constexpr const char* CYCLES = "100";

/// The share of the run's sampled bytes jeprof puts under joinV0 at least.
/// Measured: 99.9% to 100% in 20 runs in the dev image (Debug, x86-64),
/// 100% on the Raspberry Pi 4 rig's profiling build.
constexpr double MIN_V0_SHARE = 0.90;

/// The bytes jeprof puts under joinV0 at least, in MB, for the 102 calls:
/// about a third of what was measured (1,140 to 1,250 MB in the dev image,
/// 1,150 MB on the rig: some 11 MB per call, the one-liner's temporaries and
/// copies), so a joinV0 that stopped allocating cannot pass on what the
/// start-up left.
constexpr double MIN_V0_MB = 400.0;

/// What JoinV1's run puts under joinV1, as a fraction of what JoinV0's run
/// put under joinV0, at most. Measured: nothing, or one to four samples of
/// about 0.5 MB each, against some 1,200 MB: at most 0.17% in the dev image.
constexpr double MAX_V1_FRACTION = 0.02;

/// The demo binary's canonical path, or an empty string when it is missing.
std::string demoPath() {
  std::error_code ec;
  const fs::path PATH = fs::canonical(DEMO_BINARY, ec);
  return ec ? std::string() : PATH.string();
}

/// A new temporary directory for one check's runs; empty when it cannot be
/// made.
fs::path scratchDir() {
  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo16-XXXXXX").string();
  return ::mkdtemp(dirTemplate.data()) != nullptr ? fs::path(dirTemplate) : fs::path();
}

/// The first line of @p text that holds @p needle, or "".
std::string lineWith(const std::string& text, const std::string& needle) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(needle) != std::string::npos) {
      return line;
    }
  }
  return {};
}

/* ----------------------------- Runs ----------------------------- */

/// What one wrapped run of the demo left.
struct WrappedRun {
  vg::ChildExit end;              ///< How the demo ended
  std::string output;             ///< What it wrote to stdout and stderr
  std::vector<std::string> dumps; ///< Every *.heap in the wrap's folder, by name
};

/**
 * @brief Run @p demo on @p testCase as `bench run --profile jemalloc` wraps
 *        it, writing into @p dir: env sets the preload, jemalloc's options
 *        and the wrap's variables for the demo alone.
 */
WrappedRun runWrapped(const std::string& demo, const std::string& testCase, const fs::path& dir) {
  std::vector<std::string> args = {"env"};
  for (const auto& [NAME, VALUE] : vernier::bench::jemallocWrapEnvironment(dir.string())) {
    args.push_back(NAME + "=" + VALUE);
  }
  args.push_back("VERNIER_EXTERNAL_WRAP=jemalloc");
  args.push_back("VERNIER_EXTERNAL_WRAP_DIR=" + dir.string());
  args.push_back(demo);
  args.push_back("--gtest_filter=" + testCase);
  for (const char* ARG :
       {"--gtest_print_time=0", "--profile", "jemalloc", "--cycles", CYCLES, "--repeats", "1"}) {
    args.emplace_back(ARG);
  }

  WrappedRun run;
  const fs::path OUTPUT = dir.parent_path() / (dir.filename().string() + ".txt");
  run.end = vg::runLogged(args, OUTPUT);
  run.output = vg::readText(OUTPUT);
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    const std::string NAME = entry.path().filename().string();
    if (NAME.size() > 5 && NAME.compare(NAME.size() - 5, 5, ".heap") == 0) {
      run.dumps.push_back(NAME);
    }
  }
  return run;
}

/// True when @p dump's map (its MAPPED_LIBRARIES section) holds @p binary.
bool mapsBinary(const std::string& dump, const std::string& binary) {
  const std::size_t AT = dump.find("MAPPED_LIBRARIES:");
  if (AT == std::string::npos) {
    return false;
  }
  std::istringstream in(dump.substr(AT));
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t SLASH = line.find('/');
    if (SLASH != std::string::npos && line.substr(SLASH) == binary) {
      return true;
    }
  }
  return false;
}

/// What jeprof's cumulative text view says of one dump.
struct JeprofView {
  vg::ChildExit end;   ///< How jeprof ended
  std::string text;    ///< What it printed
  double totalMb = -1; ///< "Total: <MB> MB"; -1 when it printed none
  /// The cumulative MB of the row naming @p function exactly; 0 when no row does.
  double cumulativeMb(const std::string& function) const {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream row(line);
      double flat = 0;
      double cum = 0;
      std::string flatShare;
      std::string sumShare;
      std::string cumShare;
      std::string name;
      if ((row >> flat >> flatShare >> sumShare >> cum >> cumShare) && cumShare.back() == '%') {
        std::getline(row >> std::ws, name);
        if (name == function) {
          return cum;
        }
      }
    }
    return 0;
  }
};

/// `jeprof --text --alloc_space --cum <demo> <dump>`, its text kept beside the dump.
JeprofView readWithJeprof(const std::string& demo, const fs::path& dump) {
  JeprofView view;
  const fs::path OUTPUT = dump.string() + ".jeprof.txt";
  view.end =
      vg::runLogged({"jeprof", "--text", "--alloc_space", "--cum", demo, dump.string()}, OUTPUT);
  view.text = vg::readText(OUTPUT);
  const std::string TOTAL = lineWith(view.text, "Total: ");
  if (!TOTAL.empty()) {
    view.totalMb = std::strtod(TOTAL.c_str() + TOTAL.find("Total: ") + 7, nullptr);
  }
  return view;
}

/**
 * @brief Why this check cannot run here; empty when it can. Its own probe of
 *        the profiling jemalloc, apart from the backend's: preloaded into
 *        /bin/true with profiling and a dump at exit, jemalloc must leave
 *        that dump in @p dir.
 */
std::string cannotRun(const fs::path& dir) {
  if constexpr (demo::BUILT_WITH_ASAN_OR_TSAN) {
    return "the demo is built with the address or the thread sanitizer, whose allocator a "
           "preloaded jemalloc does not replace; run this test in a build without either";
  }
  const fs::path PROBE = dir / "probe";
  std::error_code ec;
  fs::create_directories(PROBE, ec);
  const fs::path OUTPUT = dir / "probe.txt";
  const vg::ChildExit END = vg::runLogged(
      {"env", std::string{"LD_PRELOAD="} + vernier::bench::JEMALLOC_PRELOAD,
       "MALLOC_CONF=prof:true,prof_final:true,prof_prefix:" + (PROBE / "check").string(),
       "/bin/true"},
      OUTPUT);
  const std::string SAID = vg::readText(OUTPUT);
  if (const std::string REFUSED = lineWith(SAID, "cannot be preloaded"); !REFUSED.empty()) {
    return "no jemalloc can be preloaded here. The loader said:\n" + REFUSED;
  }
  if (const std::string REJECTED = lineWith(SAID, "Invalid conf pair: prof:true");
      !REJECTED.empty()) {
    return "the jemalloc preloaded here is built without profiling. It said:\n" + REJECTED;
  }
  bool wrote = false;
  for (const auto& entry : fs::directory_iterator(PROBE, ec)) {
    wrote = wrote || entry.path().filename().string().find(".f.heap") != std::string::npos;
  }
  if (!wrote) {
    return "jemalloc preloaded into /bin/true with profiling wrote no exit dump (env " +
           vg::describe(END) + "):\n" + vg::lastLines(SAID);
  }
  if (!vernier::bench::profiler_env::isOnPath("jeprof")) {
    return "jeprof is not on PATH; this check reads the demo's dump with it";
  }
  for (const char* TOOL : {"nm", "addr2line", "objdump"}) {
    if (!vernier::bench::profiler_env::isOnPath(TOOL)) {
      return std::string{TOOL} + " is not on PATH: jeprof names functions with binutils";
    }
  }
  return {};
}

/// One case's run with its checks: it ran to its end, the folder holds one
/// dump, the exit dump of the demo, and jeprof read it. Fails the test and
/// returns a view with no total otherwise.
JeprofView profiledRun(const std::string& demo, const std::string& testCase, const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const WrappedRun RUN = runWrapped(demo, testCase, dir);
  EXPECT_TRUE(vg::exitedWith(RUN.end, 0) && vg::testPassed(RUN.output, testCase) &&
              vg::oneTestPassed(RUN.output))
      << testCase << " did not run to its end under jemalloc's wrap: the demo "
      << vg::describe(RUN.end) << ":\n"
      << vg::lastLines(RUN.output);
  // One dump, the demo's own: a process the demo started under the same
  // preload (its git describe, for one) would leave another.
  EXPECT_EQ(RUN.dumps.size(), 1U) << testCase << ": the wrap's folder holds " << RUN.dumps.size()
                                  << " dumps, not the demo's alone, in " << dir;
  if (::testing::Test::HasFailure() || RUN.dumps.empty()) {
    return {};
  }
  const fs::path DUMP = dir / RUN.dumps.front();
  const std::string TEXT = vg::readText(DUMP);
  EXPECT_NE(RUN.dumps.front().find(".f.heap"), std::string::npos)
      << RUN.dumps.front() << " is not an exit dump";
  EXPECT_EQ(TEXT.compare(0, 8, "heap_v2/"), 0) << DUMP << " is not a heap profile";
  EXPECT_TRUE(mapsBinary(TEXT, demo)) << DUMP << " does not map the demo, " << demo;
  JeprofView view = readWithJeprof(demo, DUMP);
  EXPECT_TRUE(vg::exitedWith(view.end, 0))
      << "jeprof " << vg::describe(view.end) << " on " << DUMP << ":\n"
      << vg::lastLines(view.text);
  EXPECT_GE(view.totalMb, 0.0) << "jeprof --alloc_space printed no total for " << DUMP
                               << ": the dump holds no bytes allocated over the run. It printed:\n"
                               << vg::lastLines(view.text);
  return view;
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test jemalloc's profile of JoinV0 puts its bytes under joinV0.
 *
 * Runs JoinV0 (102 calls) as `bench run --profile jemalloc` wraps it. The run
 * must end with status 0 and its case pass; the wrap's folder must then hold
 * one dump, the demo's exit dump (a heap profile mapping the demo); and
 * `jeprof --alloc_space --cum` must put at least MIN_V0_SHARE of the run's
 * sampled bytes, and at least MIN_V0_MB, under joinV0.
 *
 * Skipped in a build with the address or the thread sanitizer, where no
 * jemalloc can be preloaded or the one preloaded is built without profiling
 * (each quoting what the loader or jemalloc said), without jeprof, and
 * without binutils, which jeprof names functions with.
 */
TEST(Jemalloc, JoinV0BytesAreUnderJoinV0) {
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";
  if (const std::string WHY = cannotRun(DIR); !WHY.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << WHY;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;

  const JeprofView V0 = profiledRun(DEMO, V0_CASE, DIR / "v0");
  if (HasFailure()) {
    std::printf("jemalloc runs kept in %s\n", DIR.c_str());
    return;
  }
  const double UNDER_V0 = V0.cumulativeMb(V0_FUNCTION);
  std::printf("[Jemalloc.JoinV0BytesAreUnderJoinV0]  JoinV0: %.1f MB sampled over the run, "
              "%.1f MB (%.1f%%) under joinV0\n",
              V0.totalMb, UNDER_V0, V0.totalMb > 0 ? 100.0 * UNDER_V0 / V0.totalMb : 0.0);
  EXPECT_GE(UNDER_V0, MIN_V0_SHARE * V0.totalMb)
      << "joinV0 holds less than " << MIN_V0_SHARE * 100 << "% of JoinV0's sampled bytes:\n"
      << vg::lastLines(V0.text, 30);
  EXPECT_GE(UNDER_V0, MIN_V0_MB) << "joinV0 allocates less than " << MIN_V0_MB
                                 << " MB in 102 calls:\n"
                                 << vg::lastLines(V0.text, 30);
  if (HasFailure()) {
    std::printf("jemalloc runs kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/**
 * @test JoinV1, which allocates once per call for the final size, puts a
 *       small fraction of JoinV0's bytes under joinV1.
 *
 * Runs JoinV0 and JoinV1 as JoinV0BytesAreUnderJoinV0 runs JoinV0, each
 * checked the same way; what jeprof puts under joinV1 (nothing when JoinV1's
 * run sampled none of its allocations, an ordinary outcome) must stay below
 * MAX_V1_FRACTION of what it put under joinV0. Skipped where
 * JoinV0BytesAreUnderJoinV0 is.
 */
TEST(Jemalloc, JoinV1AllocatesAFractionOfJoinV0) {
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";
  if (const std::string WHY = cannotRun(DIR); !WHY.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << WHY;
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;

  const JeprofView V0 = profiledRun(DEMO, V0_CASE, DIR / "v0");
  const JeprofView V1 = profiledRun(DEMO, V1_CASE, DIR / "v1");
  if (HasFailure()) {
    std::printf("jemalloc runs kept in %s\n", DIR.c_str());
    return;
  }
  const double UNDER_V0 = V0.cumulativeMb(V0_FUNCTION);
  const double UNDER_V1 = V1.cumulativeMb(V1_FUNCTION);
  std::printf("[Jemalloc.JoinV1AllocatesAFractionOfJoinV0]  %.1f MB under joinV1, %.1f MB under "
              "joinV0 (JoinV1's run sampled %.1f MB in all)\n",
              UNDER_V1, UNDER_V0, V1.totalMb);
  EXPECT_GT(UNDER_V0, 0.0) << "JoinV0's run put nothing under joinV0:\n"
                           << vg::lastLines(V0.text, 30);
  EXPECT_LT(UNDER_V1, MAX_V1_FRACTION * UNDER_V0)
      << "joinV1 holds " << UNDER_V1 << " MB, not below " << MAX_V1_FRACTION * 100
      << "% of joinV0's " << UNDER_V0 << " MB:\n"
      << vg::lastLines(V1.text, 30);
  if (HasFailure()) {
    std::printf("jemalloc runs kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}
