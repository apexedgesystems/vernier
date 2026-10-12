#ifndef VERNIER_DEMO_JOIN_GPERFPROFILES_HPP
#define VERNIER_DEMO_JOIN_GPERFPROFILES_HPP
/**
 * @file GperfProfiles.hpp
 * @brief The profiling plumbing of JoinProfileAttribution: profile a call
 *        through the gperf backend, then read one function's share of the
 *        profile with google-pprof.
 *
 * Test support, private to JoinProfileAttribution_pTest.cpp. The plumbing
 * lives here (a scratch directory, the profiled loop, the pprof call and its
 * parsing); the check keeps what it asserts and when it skips.
 */

#include <cctype>
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

/// One function's share of a profile, read from its sampled stacks.
struct FunctionShare {
  long samples = 0;          ///< Samples in the whole profile
  long stackSamples = 0;     ///< Samples in the stacks read; equals samples
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
    std::string pattern = (TMP / "vernier-join-gperf-XXXXXX").string();
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

/**
 * The function a `--stacks` frame names: what follows `(address) file:line:`.
 * The file and line come first and function names hold colons, so the name
 * starts after the first colon-delimited run of digits.
 */
inline std::string frameFunction(const std::string& frame) {
  const std::size_t AFTER_ADDRESS = frame.find(") ");
  if (AFTER_ADDRESS == std::string::npos) {
    return {};
  }
  const std::string LOCATED = frame.substr(AFTER_ADDRESS + 2);
  for (std::size_t colon = LOCATED.find(':'); colon != std::string::npos;
       colon = LOCATED.find(':', colon + 1)) {
    std::size_t end = colon + 1;
    while (end < LOCATED.size() && std::isdigit(static_cast<unsigned char>(LOCATED[end])) != 0) {
      ++end;
    }
    if (end > colon + 1 && end < LOCATED.size() && LOCATED[end] == ':') {
      return LOCATED.substr(end + 1);
    }
  }
  return {};
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
 * @brief Read @p profile with `google-pprof --text --stacks` and return
 *        @p function's share of it, its clones included.
 *
 * A sample is the function's own when the instruction it caught is part of
 * the function's machine code: its stack's innermost frame, which pprof names
 * after the function holding that instruction, with the source line of
 * whatever the compiler inlined there. --text's flat column is not used: when
 * the binary carries debug information, pprof gives every callee inlined into
 * the function a row of its own, marked "(inline)", and moves its samples out
 * of the function's flat count, so the same code would read differently from
 * one build to the next.
 *
 * The report opens with a `Total: N samples` line and a `Stacks:` line, then
 * lists the stacks: one block per stack, separated by blank lines, with
 * `count (address) file:line:function` for the innermost frame, the count in
 * the first column, and an indented `(address) file:line:function` for each
 * caller after it. --text's table follows the stacks and is not read.
 */
inline FunctionShare readShare(const std::string& profile, const std::string& function) {
  FunctionShare share;
  const std::string CMD = "google-pprof --text --stacks " +
                          internal::shellQuoted(internal::selfExePath()) + " " +
                          internal::shellQuoted(profile) + " 2>/dev/null";
  std::FILE* pipe = ::popen(CMD.c_str(), "r");
  if (pipe == nullptr) {
    return share;
  }

  long own = 0;
  long onStack = 0;
  long count = 0;       // samples of the stack being read
  bool inStack = false; // its innermost frame has been read
  bool inStacks = false;
  bool pastStacks = false;
  bool matched = false; // some frame of the stack is the function
  const auto END_STACK = [&] {
    if (inStack && matched) {
      onStack += count;
    }
    inStack = false;
    matched = false;
  };

  char* raw = nullptr;
  std::size_t capacity = 0;
  while (::getline(&raw, &capacity, pipe) != -1) {
    std::string line(raw);
    while (!line.empty() && (line.back() == '\n' || line.back() == ' ')) {
      line.pop_back();
    }
    if (!inStacks) {
      if (std::sscanf(line.c_str(), "Total: %ld samples", &share.samples) == 1) {
        continue;
      }
      inStacks = (line == "Stacks:");
      continue;
    }
    if (pastStacks) {
      continue; // read to the end, so pprof never waits on a full pipe
    }
    if (line.empty()) {
      END_STACK();
      continue;
    }
    const bool INNERMOST = std::isdigit(static_cast<unsigned char>(line[0])) != 0;
    const std::size_t FIRST = line.find_first_not_of(' ');
    if (!INNERMOST && line[FIRST] != '(') {
      END_STACK();
      pastStacks = true;
      continue;
    }
    const bool IS_FUNCTION = internal::isFunctionOrClone(internal::frameFunction(line), function);
    if (INNERMOST) {
      END_STACK();
      count = std::strtol(line.c_str(), nullptr, 10);
      inStack = true;
      share.stackSamples += count;
      own += IS_FUNCTION ? count : 0;
    }
    matched = matched || IS_FUNCTION;
  }
  END_STACK();
  std::free(raw);
  ::pclose(pipe);

  if (share.samples > 0) {
    share.selfPercent = 100.0 * static_cast<double>(own) / static_cast<double>(share.samples);
    share.totalPercent = 100.0 * static_cast<double>(onStack) / static_cast<double>(share.samples);
  }
  return share;
}

} // namespace gperf_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_JOIN_GPERFPROFILES_HPP
