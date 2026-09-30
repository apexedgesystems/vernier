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
 *  - valgrind installed (apt install valgrind), and valgrind running the
 *    process: a request run without it fails (exit status 4) and prints the
 *    wrap command
 *  - Linux only
 *  - No special permissions needed (runs as normal user)
 *
 * Trade-offs vs sampling profilers:
 *  - Pro: Deterministic, zero noise, 1 repeat is sufficient, no DWARF issues
 *  - Con: 20-50x slower execution (instruction-level simulation)
 *  - Best for: A/B optimization comparison, finding exact instruction hotspots
 *  - Not for: Real-time profiling, measuring wall-clock time
 *
 * Usage:
 *   bench run ./MyTest --profile callgrind                    # Whole-process counts
 *   bench run ./MyTest --profile callgrind --profile-analyze  # Then callgrind_annotate
 *   # Cache simulation: wrap with valgrind directly:
 *   #   valgrind --tool=callgrind --cache-sim=yes ./MyTest --profile callgrind
 *
 * The profile is complete when valgrind exits: bench run checks it and runs
 * callgrind_annotate after that exit; under a manual wrap, read it then.
 */

#include <memory>
#include <optional>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

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
  /** @brief Decide the request now; prints the decision when it cannot record. */
  CallgrindProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  CallgrindProfiler(const PerfConfig& cfg, std::string testName,
                    std::shared_ptr<const valgrind_tool::ValgrindPlan> plan);

  ~CallgrindProfiler() override = default;

  std::string toolName() const noexcept override { return "callgrind"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  void applyPlan();

  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const valgrind_tool::ValgrindPlan> plan_;
  bool runningUnderValgrind_{false};
  bool wrappedByRunner_{false};
  // True when this backend switches instrumentation around the measured
  // window: under a wrap started by hand, with callgrind_control on PATH;
  // bench run's wrap records the whole process.
  bool canToggle_{false};
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Read callgrind's mode from @p profileArgs into @p mode.
 * @return The error that refuses any word (callgrind takes none); nullopt
 *         when the mode was read.
 */
std::optional<ReadinessResult> parseCallgrindMode(const std::string& profileArgs,
                                                  valgrind_tool::ValgrindMode& mode);

/**
 * @brief The callgrind backend's readiness decision for @p request in @p ctx.
 *
 * The doctor's scopes probe the tool's start; a run reads its own memory map
 * for valgrind's callgrind (its executable: callgrind maps no preload of its
 * own). A wrap started by hand without callgrind_control is a caveat: the
 * window cannot be switched. --profile-analyze is bench run's, after the
 * process exits: ready under its wrap, an analysis-stage error under a wrap
 * started by hand, and in the doctor's scopes an analysis-stage error when
 * callgrind_annotate is missing. On success the result's plan is a
 * ValgrindPlan.
 */
ReadinessResult checkCallgrindRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/**
 * @brief checkCallgrindRequest() for a process whose memory map shows
 * @p identity at the runtime scope (the two-argument form reads the map of
 * @p ctx's process).
 */
ReadinessResult checkCallgrindRequestWithIdentity(const ReadinessRequest& request,
                                                  const ReadinessContext& ctx,
                                                  const valgrind_tool::ValgrindIdentity& identity);

/**
 * @brief Factory function for callgrind profiler: decides the request when
 * the profiler is constructed.
 */
std::unique_ptr<Profiler> makeCallgrindProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERCALLGRIND_HPP
