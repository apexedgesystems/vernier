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
 *  - Outputs a callgrind.out file (callgrind.out.<pid> when valgrind dumps at
 *    exit) for analysis with callgrind_annotate or KCachegrind
 *
 * Modes:
 *  - Default: instruction counts only (cache / branch simulation off, faster)
 *  - Cache or branch simulation: add valgrind's --cache-sim=yes /
 *    --branch-sim=yes to the wrap command
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
 *   --profile callgrind --profile-analyze  # Auto-run callgrind_annotate
 *   # Cache simulation: wrap with valgrind directly:
 *   #   valgrind --tool=callgrind --cache-sim=yes ./MyTest --profile callgrind
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
 * Under a manual wrap started with --instr-atstart=no, switches instrumentation
 * on before each measured window and off after it with callgrind_control, so
 * the profile valgrind writes at exit holds the measured calls and the
 * harness's own work between the two hooks, and not the rest of the process.
 * Under the wrap `bench run --profile callgrind` uses, the recording covers the
 * whole process and is left alone. Not under valgrind, nothing is switched and
 * measurement proceeds normally.
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
  bool runningUnderValgrind_{false};
  // True when this backend switches instrumentation around the measured
  // window: under valgrind, with callgrind_control on PATH, and not under
  // bench run's wrap, whose recording covers the whole process.
  bool canToggle_{false};
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
