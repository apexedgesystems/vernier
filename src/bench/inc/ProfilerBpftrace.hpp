#ifndef VERNIER_PROFILERBPFTRACE_HPP
#define VERNIER_PROFILERBPFTRACE_HPP
/**
 * @file ProfilerBpftrace.hpp
 * @brief bpftrace backend for the benchmarking profiler facade.
 *
 * Behavior:
 *  - Builds a BpfConfig from env + PerfConfig (flags), setting outputDir to
 *    `artifactRoot`/`Suite.Case`.bpf/.
 *  - In beforeMeasure(), starts one bpftrace process per selected script
 *    (e.g., "offcpu", "syslat", "bio") with {{PID}} replaced by the current PID.
 *  - In afterMeasure(), gracefully stops all bpftrace children (SIGINT -> grace -> SIGTERM).
 *
 * Notes:
 *  - Linux-only. Safe no-op on other platforms (compile-time guard).
 *  - Requires `bpftrace` binary and sufficient privileges.
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- BpftraceProfiler ----------------------------- */

/**
 * @brief bpftrace profiler implementation.
 *
 * Runs bpftrace scripts with PID filtering. Scripts contain {{PID}} placeholder
 * which is replaced with the target process PID before execution.
 */
class BpftraceProfiler final : public Profiler {
public:
  /**
   * @brief Construct bpftrace profiler.
   * @param cfg Configuration with bpfScripts and artifactRoot
   * @param testName Test identifier (e.g., "Suite.Case")
   */
  BpftraceProfiler(const PerfConfig& cfg, std::string testName);
  ~BpftraceProfiler() override;

  std::string toolName() const noexcept override { return "bpftrace"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;

  // Forward declaration of implementation details (defined in .cpp)
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Factory function for bpftrace profiler.
 * @return Profiler instance, or nullptr if unsupported platform.
 */
std::unique_ptr<Profiler> makeBpftraceProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERBPFTRACE_HPP