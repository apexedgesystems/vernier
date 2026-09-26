#ifndef VERNIER_DEMO_GPERFPROFILER_CHECK_HPP
#define VERNIER_DEMO_GPERFPROFILER_CHECK_HPP
/**
 * @file 03_GperfProfiler_Check.hpp
 * @brief Demo 03's profile check: profile a call through the gperf backend,
 *        then read one function's share of the profile with google-pprof.
 *
 * Private to BenchDemo_03_GperfProfiler; its GperfProfiler.ProfileAttribution
 * test is the only user. The plumbing lives here (a scratch directory, the
 * profiled loop, the pprof call and its parsing) so that the demo file shows
 * the benchmarks and what the test asserts.
 */

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

#include "src/bench/inc/Perf.hpp"

namespace vernier {
namespace bench {
namespace demo {
namespace gperf_check {

/* ----------------------------- Constants ----------------------------- */

/// Calls between two reads of the CPU clock, so the reads take a negligible
/// share of the samples.
inline constexpr int CALLS_PER_CLOCK_READ = 16;

/* ----------------------------- FunctionShare ----------------------------- */

/// One function's share of a profile, as `google-pprof --text` reports it.
struct FunctionShare {
  long samples = 0;          ///< Samples in the whole profile
  double selfPercent = 0.0;  ///< Samples taken in the function's own code
  double totalPercent = 0.0; ///< Samples taken in it or in anything it called
};

/* ----------------------------- ScratchDir ----------------------------- */

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

  /** @return The directory's path; empty if it could not be created. */
  const std::string& path() const noexcept { return path_; }

private:
  std::string path_;
};

/* ----------------------------- Helpers ----------------------------- */

namespace internal {

/// @p text as one single-quoted shell word, whatever characters it holds.
inline std::string shellQuoted(const std::string& text) {
  std::string quoted = "'";
  for (const char c : text) {
    quoted += (c == '\'') ? std::string("'\\''") : std::string(1, c);
  }
  return quoted + "'";
}

/// CPU time this process has used, the clock gperftools' timer counts.
inline double processCpuSeconds() {
  timespec now{};
  ::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
  return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) * 1e-9;
}

/// Path of the running binary, which pprof needs to name the functions.
inline std::string selfExePath() {
  std::array<char, 4096> buf{};
  const ssize_t LEN = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  return LEN > 0 ? std::string(buf.data(), static_cast<std::size_t>(LEN)) : std::string{};
}

/**
 * True for @p name itself and for a copy of it the compiler made: GCC may
 * clone a function whose callers it can see, for instance to specialize it
 * for a constant argument, and pprof prints the copy as
 * `<name> [clone .suffix]`.
 */
inline bool isFunctionOrClone(const std::string& name, const std::string& function) {
  const std::string CLONE_PREFIX = function + " [clone ";
  return name == function || name.compare(0, CLONE_PREFIX.size(), CLONE_PREFIX) == 0;
}

} // namespace internal

/* ----------------------------- API ----------------------------- */

/**
 * @brief Run @p op under the gperf backend, through the same profiler facade
 *        that --profile gperf uses, for @p cpuSeconds of CPU time.
 * @param root Directory the backend writes its per-test folder into.
 * @param name Test name the backend names that folder after.
 * @return Path of the profile the backend wrote.
 */
inline std::string profileWithGperf(const std::string& root, const std::string& name,
                                    double cpuSeconds, const std::function<void()>& op) {
  ::vernier::bench::PerfConfig cfg = ::vernier::bench::detail::getPerfConfig();
  cfg.profileTool = "gperf";
  cfg.profileArgs.clear();
  cfg.artifactRoot = root;
  cfg.profileAnalyze = false;

  const std::unique_ptr<::vernier::bench::Profiler> PROFILER =
      ::vernier::bench::Profiler::make(cfg, name);
  PROFILER->beforeMeasure();
  const double START = internal::processCpuSeconds();
  while (internal::processCpuSeconds() - START < cpuSeconds) {
    for (int call = 0; call < CALLS_PER_CLOCK_READ; ++call) {
      op();
    }
  }
  PROFILER->afterMeasure(::vernier::bench::Stats{});
  return PROFILER->artifactDir() + "/cpu.prof";
}

/**
 * @brief Read @p profile with `google-pprof --text`, the command the
 *        walkthrough uses, and return @p function's share of it, its clones
 *        included.
 *
 * Lines have the form `flat flat% sum% cum cum% name`, after a
 * `Total: N samples` line.
 */
inline FunctionShare readShare(const std::string& profile, const std::string& function) {
  FunctionShare share;
  const std::string CMD = "google-pprof --text " + internal::shellQuoted(internal::selfExePath()) +
                          " " + internal::shellQuoted(profile) + " 2>/dev/null";
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
    if (internal::isFunctionOrClone(name, function)) {
      share.selfPercent += flatPct;
      share.totalPercent += cumPct;
    }
  }
  ::pclose(pipe);
  return share;
}

} // namespace gperf_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_GPERFPROFILER_CHECK_HPP
