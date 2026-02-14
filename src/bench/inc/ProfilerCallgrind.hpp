#ifndef VERNIER_PROFILERCALLGRIND_HPP
#define VERNIER_PROFILERCALLGRIND_HPP
/**
 * @file ProfilerCallgrind.hpp
 * @brief Valgrind Callgrind backend for the benchmarking profiler facade.
 *
 * Behavior:
 *  - Wraps the benchmark binary execution under `valgrind --tool=callgrind`
 *  - Deterministic instruction counting (no sampling noise, no frequency tuning)
 *  - Perfect for A/B comparisons: identical instruction counts across runs
 *  - Outputs callgrind.out.<pid> files for analysis with callgrind_annotate or KCachegrind
 *
 * Modes:
 *  - Default: collect instruction counts, cache simulation off (faster)
 *  - "cache" in profileArgs: enable cache simulation (--cache-sim=yes)
 *  - "branch" in profileArgs: enable branch prediction simulation
 *
 * Requirements:
 *  - valgrind installed (apt install valgrind)
 *  - Linux only (safe no-op on other platforms)
 *  - No special permissions needed (runs as normal user)
 *
 * Trade-offs vs sampling profilers:
 *  - Pro: Deterministic, zero noise, 1 repeat is sufficient, no DWARF issues
 *  - Con: 20-50x slower execution (instruction-level simulation)
 *  - Best for: A/B optimization comparison, finding exact instruction hotspots
 *  - Not for: Real-time profiling, measuring wall-clock time
 *
 * Usage:
 *   --profile callgrind                    # Basic instruction counts
 *   --profile callgrind --profile-args cache  # With cache simulation
 *   --profile callgrind --profile-analyze  # Auto-run callgrind_annotate
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- CallgrindProfiler ----------------------------- */

/**
 * @brief Valgrind Callgrind profiler implementation.
 *
 * Uses callgrind_control to toggle instrumentation around the measured window.
 * The parent process must be running under valgrind --tool=callgrind for this
 * to have any effect. If not running under valgrind, instrumentation toggles
 * are no-ops and measurement proceeds normally.
 *
 * Recommended invocation:
 *   valgrind --tool=callgrind --instr-atstart=no ./MyTest --profile callgrind
 */
class CallgrindProfiler final : public Profiler {
public:
  CallgrindProfiler(const PerfConfig& cfg, std::string testName);
  ~CallgrindProfiler() override = default;

  std::string toolName() const noexcept override { return "callgrind"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  void runAnnotateAnalysis() const;

  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  bool wantCache_{false};
  bool wantBranch_{false};
  bool runningUnderValgrind_{false};
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Factory function for callgrind profiler.
 * @return Profiler instance, or nullptr if valgrind is not available.
 */
std::unique_ptr<Profiler> makeCallgrindProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERCALLGRIND_HPP
