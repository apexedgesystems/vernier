/**
 * @file 03_GperfProfiler_Demo.cpp
 * @brief Demo 03: gperftools CPU profiler -- which function has the time
 *
 * Samples the two versions of the shared join example and checks what the
 * samples say:
 *  1. JoinV0 and JoinV1 each measure one version, one CSV row each; run one
 *     of them under --profile gperf to get that version's profile
 *  2. ProfileAttribution profiles both versions itself and fails when the
 *     profile stops putting V0's time in the calls joinV0 makes, or V1's time
 *     in joinV1's own code
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_03_GperfProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Profile one version: writes GperfProfiler.JoinV0.gperf/cpu.prof
 *   bench run ./BenchDemo_03_GperfProfiler --profile gperf -- \
 *     --gtest_filter=GperfProfiler.JoinV0 --target-time 200ms --repeats 10
 *
 *   # Read it
 *   google-pprof --text ./BenchDemo_03_GperfProfiler GperfProfiler.JoinV0.gperf/cpu.prof
 *   @endcode
 *
 * @see docs/03_GPERF_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, the same input as demo 01.
static constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// CPU seconds each version runs under the profiler in ProfileAttribution.
/// gperftools samples 100 times per CPU second by default, so this gives
/// about 200 samples per version.
static constexpr double PROFILE_CPU_SECONDS = 2.0;

/// Calls between two reads of the CPU clock, so the reads take a negligible
/// share of the samples.
static constexpr int CALLS_PER_CLOCK_READ = 16;

/// Fewer samples than this and a percentage says little about the program.
static constexpr long MIN_SAMPLES = 100;

/// A version's function must be on the stack of at least this share of its
/// samples: the profile attributes the version's time to it.
static constexpr double MIN_TOTAL_PERCENT = 80.0;

/// The line between "the time is in what the function calls" and "the time
/// is in the function's own code". On the reference rig joinV0's own code
/// holds a few percent of V0's samples and joinV1's a large share of V1's, so
/// the line sits far from both readings.
static constexpr double SELF_PERCENT_SPLIT = 25.0;

static constexpr const char* JOIN_V0 = "vernier::bench::demo::joinV0";
static constexpr const char* JOIN_V1 = "vernier::bench::demo::joinV1";

/* ----------------------------- File Helpers ----------------------------- */

namespace {

/// Bytes a joined string holds: every part plus one separator each.
std::size_t joinedSize(const std::vector<std::string>& parts) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }
  return total;
}

/// One function's share of a profile, as `google-pprof --text` reports it.
struct FunctionShare {
  long samples = 0;          ///< Samples in the whole profile
  double selfPercent = 0.0;  ///< Samples taken in the function's own code
  double totalPercent = 0.0; ///< Samples taken in it or in anything it called
};

/// A temporary directory that is removed with everything in it.
class ScratchDir {
public:
  ScratchDir() {
    std::error_code ec;
    const std::filesystem::path TMP = std::filesystem::temp_directory_path(ec);
    if (ec) {
      return;
    }
    std::string pattern = (TMP / "vernier-demo03-XXXXXX").string();
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

  const std::string& path() const noexcept { return path_; }

private:
  std::string path_;
};

/// @p text as one single-quoted shell word, whatever characters it holds.
std::string shellQuoted(const std::string& text) {
  std::string quoted = "'";
  for (const char c : text) {
    quoted += (c == '\'') ? std::string("'\\''") : std::string(1, c);
  }
  return quoted + "'";
}

/// CPU time this process has used, the clock gperftools' timer counts.
double processCpuSeconds() {
  timespec now{};
  ::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
  return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) * 1e-9;
}

/// Path of the running binary, which pprof needs to name the functions.
std::string selfExePath() {
  std::array<char, 4096> buf{};
  const ssize_t LEN = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  return LEN > 0 ? std::string(buf.data(), static_cast<std::size_t>(LEN)) : std::string{};
}

/**
 * Run @p op under the gperf backend, through the same profiler facade that
 * --profile gperf uses, for PROFILE_CPU_SECONDS of CPU time. Returns the path
 * of the profile the backend wrote.
 */
std::string profileWithGperf(const std::string& root, const std::string& name,
                             const std::function<void()>& op) {
  ub::PerfConfig cfg = ub::detail::getPerfConfig();
  cfg.profileTool = "gperf";
  cfg.profileArgs.clear();
  cfg.artifactRoot = root;
  cfg.profileAnalyze = false;

  const std::unique_ptr<ub::Profiler> PROFILER = ub::Profiler::make(cfg, name);
  PROFILER->beforeMeasure();
  const double START = processCpuSeconds();
  while (processCpuSeconds() - START < PROFILE_CPU_SECONDS) {
    for (int call = 0; call < CALLS_PER_CLOCK_READ; ++call) {
      op();
    }
  }
  PROFILER->afterMeasure(ub::Stats{});
  return PROFILER->artifactDir() + "/cpu.prof";
}

/**
 * True for @p name itself and for a copy of it the compiler made: GCC may
 * clone a function whose callers it can see, for instance to specialize it
 * for a constant argument, and pprof prints the copy as
 * `<name> [clone .suffix]`.
 */
bool isFunctionOrClone(const std::string& name, const std::string& function) {
  const std::string CLONE_PREFIX = function + " [clone ";
  return name == function || name.compare(0, CLONE_PREFIX.size(), CLONE_PREFIX) == 0;
}

/**
 * Read @p profile with `google-pprof --text`, the command the walkthrough
 * uses, and return @p function's share of it, its clones included. Lines have
 * the form `flat flat% sum% cum cum% name`, after a `Total: N samples` line.
 */
FunctionShare readShare(const std::string& profile, const std::string& function) {
  FunctionShare share;
  const std::string CMD = "google-pprof --text " + shellQuoted(selfExePath()) + " " +
                          shellQuoted(profile) + " 2>/dev/null";
  std::FILE* pipe = ::popen(CMD.c_str(), "r");
  if (pipe == nullptr) {
    return share;
  }

  std::array<char, 1024> line{};
  while (std::fgets(line.data(), static_cast<int>(line.size()), pipe) != nullptr) {
    long flat = 0;
    long cum = 0;
    double flatPct = 0.0;
    double sumPct = 0.0;
    double cumPct = 0.0;
    int nameAt = 0;
    if (std::sscanf(line.data(), "Total: %ld samples", &share.samples) == 1) {
      continue;
    }
    if (std::sscanf(line.data(), " %ld %lf%% %lf%% %ld %lf%% %n", &flat, &flatPct, &sumPct, &cum,
                    &cumPct, &nameAt) < 5 ||
        nameAt == 0) {
      continue;
    }
    std::string name(line.data() + nameAt);
    while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) {
      name.pop_back();
    }
    // A function and its copies are never on one stack together, so their
    // shares add.
    if (isFunctionOrClone(name, function)) {
      share.selfPercent += flatPct;
      share.totalPercent += cumPct;
    }
  }
  ::pclose(pipe);
  return share;
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner: out = out + part + separator. */
PERF_THROUGHPUT(GperfProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version: measure once, append in place. */
PERF_THROUGHPUT(GperfProfiler, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/**
 * @test The profile attributes each version's time where the walkthrough says.
 *
 * Profiles both versions with the gperf backend and reads the profiles with
 * google-pprof. joinV0 must be on the stack of most of V0's samples while its
 * own code holds few of them: the time is in the copying and allocating it
 * calls. joinV1 must be on the stack of most of V1's samples and its own code
 * must hold a real share of them. Fails if either name drops out of the
 * profile (an inlined version) or if the versions stop differing. Writes no
 * CSV row.
 */
PERF_TEST(GperfProfiler, ProfileAttribution) {
  if (!ub::detail::getPerfConfig().profileTool.empty()) {
    GTEST_SKIP() << "runs its own profiles; run it without --profile";
  }
  const ub::EnvReport GPERF = ub::ProfilerRegistry::instance().runCheck("gperf");
  if (GPERF.status == ub::EnvReport::Status::Error) {
    GTEST_SKIP() << "gperf backend unavailable: " << GPERF.message;
  }
  if (!ub::profiler_env::isOnPath("google-pprof")) {
    GTEST_SKIP() << "google-pprof is not on PATH; it reads the profiles";
  }

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR), demo::joinV1(PARTS, SEPARATOR));

  const ScratchDir SCRATCH;
  ASSERT_FALSE(SCRATCH.path().empty()) << "could not create a scratch directory";

  volatile std::size_t sink = 0;
  const std::string V0_PROFILE =
      profileWithGperf(SCRATCH.path(), "GperfProfiler.ProfileAttribution.V0",
                       [&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  const std::string V1_PROFILE =
      profileWithGperf(SCRATCH.path(), "GperfProfiler.ProfileAttribution.V1",
                       [&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });

  const FunctionShare V0 = readShare(V0_PROFILE, JOIN_V0);
  const FunctionShare V1 = readShare(V1_PROFILE, JOIN_V1);

  std::printf("[GperfProfiler.ProfileAttribution]  V0: %ld samples, joinV0 %.1f%% self, %.1f%% "
              "total\n",
              V0.samples, V0.selfPercent, V0.totalPercent);
  std::printf("[GperfProfiler.ProfileAttribution]  V1: %ld samples, joinV1 %.1f%% self, %.1f%% "
              "total\n",
              V1.samples, V1.selfPercent, V1.totalPercent);

  ASSERT_GE(V0.samples, MIN_SAMPLES) << "google-pprof counted " << V0.samples << " samples in "
                                     << V0_PROFILE << "; the shares need more";
  ASSERT_GE(V1.samples, MIN_SAMPLES) << "google-pprof counted " << V1.samples << " samples in "
                                     << V1_PROFILE << "; the shares need more";

  // A function's own share means something only once the profile names it.
  EXPECT_GE(V0.totalPercent, MIN_TOTAL_PERCENT)
      << JOIN_V0 << " is on the stack of " << V0.totalPercent
      << "% of V0's samples: the profile no longer names it";
  if (V0.totalPercent >= MIN_TOTAL_PERCENT) {
    EXPECT_LE(V0.selfPercent, SELF_PERCENT_SPLIT)
        << JOIN_V0 << "'s own code holds " << V0.selfPercent
        << "% of V0's samples: the time is no longer in the copying and allocating it calls";
  }
  EXPECT_GE(V1.totalPercent, MIN_TOTAL_PERCENT)
      << JOIN_V1 << " is on the stack of " << V1.totalPercent
      << "% of V1's samples: the profile no longer names it";
  if (V1.totalPercent >= MIN_TOTAL_PERCENT) {
    EXPECT_GE(V1.selfPercent, SELF_PERCENT_SPLIT)
        << JOIN_V1 << "'s own code holds only " << V1.selfPercent
        << "% of V1's samples: V1 spends its time in calls, as V0 does";
  }
}

PERF_MAIN()
