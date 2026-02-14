#ifndef VERNIER_PROFILERPERF_HPP
#define VERNIER_PROFILERPERF_HPP
/**
 * @file ProfilerPerf.hpp
 * @brief Linux perf backend for the benchmarking profiler facade.
 *
 * Behavior:
 *  - Default: `perf stat -e cpu-cycles,instructions,branches,branch-misses,cache-misses -p <PID>`
 *             Output is written to `<artifactDir>/stat.txt` (perf writes to stderr; we redirect).
 *  - If `cfg.profileArgs` begins with "record", we run `perf record <args> -p <PID>` instead and
 *    write `<artifactDir>/perf.data` (+ `record.err.txt`).
 *  - Only the measured window is profiled: we start in beforeMeasure() and stop in afterMeasure().
 *
 * Notes:
 *  - Linux-only. Safe no-op on other platforms (compile-time guard).
 *  - Requires `perf` to be available and sufficient permissions (CAP_PERFMON or root).
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- PerfStatProfiler ----------------------------- */

/**
 * @brief Linux perf profiler implementation.
 *
 * Supports both `perf stat` (default) and `perf record` modes.
 * Attaches to running process via `-p <PID>`.
 */
class PerfStatProfiler final : public Profiler {
public:
  /**
   * @brief Construct perf profiler.
   * @param cfg Configuration with profileArgs and artifactRoot
   * @param testName Test identifier (e.g., "Suite.Case")
   */
  PerfStatProfiler(const PerfConfig& cfg, std::string testName);
  ~PerfStatProfiler() override = default;

  std::string toolName() const noexcept override { return "perf"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  // Helper methods
  bool isPerfAvailable() const;
  static bool startsWithTrim(const std::string& s, const char* prefix);
  void launchBackground(const std::string& cmdCore, const std::string& stdoutPath,
                        const std::string& stderrPath);
  bool killChild(int sig) noexcept;

  // State
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;

#ifdef __linux__
  pid_t childPid_ = -1;
  std::string statPath_;
  std::string dataPath_;
  std::string errPath_;
#endif
};

/**
 * @brief Factory function for perf profiler.
 * @return Profiler instance, or nullptr if unsupported platform.
 */
std::unique_ptr<Profiler> makePerfProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERPERF_HPP