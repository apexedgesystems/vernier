#ifndef VERNIER_PROFILERGPERF_HPP
#define VERNIER_PROFILERGPERF_HPP
/**
 * @file ProfilerGperf.hpp
 * @brief gperftools backend for the benchmarking profiler facade.
 *
 * Modes:
 *  - CPU profiling (default): generates "<artifactDir>/cpu.prof"
 *  - Heap profiling (opt-in): parses "heap" in profileArgs -> starts HeapProfiler
 *  - Both: parse "both" or include both "cpu" and "heap" keywords in profileArgs
 *
 * Readiness (checkGperfRequest): every requested mode must be compiled in; a
 * request that is not is a collection error. With --profile-analyze the
 * analyzer is the first of google-pprof and pprof found on PATH, and it must
 * run; a missing or broken analyzer is an analysis error, and the capture
 * still runs and keeps cpu.prof. The analysis runs exactly the analyzer the
 * check found.
 *
 * Notes:
 *  - Requires gperftools headers/libraries to be available at build/link time.
 *  - Heap profiling additionally requires a build with
 *    -DVERNIER_LINK_TCMALLOC=ON: the heap profiler is part of tcmalloc, and
 *    tcmalloc replaces the allocator for the whole process, so it is never
 *    linked implicitly. Without it, a heap request is a readiness error that
 *    says how to enable it.
 *  - If unavailable, makeGperfProfiler(...) returns nullptr and the factory
 *    in Profiler.hpp will produce a named no-op.
 */

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp" // base

// Detect availability at compile time
#if __has_include(<gperftools/profiler.h>)
#define UB_HAS_GPERF_CPU 1
#else
#define UB_HAS_GPERF_CPU 0
#endif

#if __has_include(<gperftools/heap-profiler.h>) && defined(VERNIER_HAS_TCMALLOC)
#define UB_HAS_GPERF_HEAP 1
#else
#define UB_HAS_GPERF_HEAP 0
#endif

namespace vernier {
namespace bench {

/* ----------------------------- Readiness ----------------------------- */

/** @brief The modes `--profile-args` selects for gperf. */
struct GperfModes {
  bool cpu = false;
  bool heap = false;
};

/**
 * @brief One parser for the check and the profiler: empty, "cpu" or "both"
 * select CPU profiling; "heap" or "both" select heap profiling.
 */
GperfModes parseGperfModes(const std::string& profileArgs);

/** @brief What the gperf check verified, for the profiler to use. */
struct GperfPlan final : ReadinessPlan {
  GperfModes modes;
  bool analyze = false;        ///< --profile-analyze was requested.
  std::string analyzer;        ///< Absolute path of the selected analyzer; empty when none.
  bool analysisReady = false;  ///< The promised analysis can run with that analyzer.
  std::string analysisSkipped; ///< Why it cannot, for the run's note; empty when it can.
};

/** @brief The analysis half of a gperf decision. */
struct GperfAnalysis {
  std::string analyzer;                 ///< The selected analyzer's path; empty when none.
  std::optional<ReadinessResult> error; ///< The analysis-stage Error, when there is one.
};

/**
 * @brief Which analyzer a promised analysis runs, and whether it can.
 *
 * The analyzer is the first of google-pprof and pprof on the snapshot's PATH.
 * It matters only to a request that promises an analysis of a CPU capture:
 * then a missing analyzer, or one that does not answer `--help`, is an
 * ANALYSIS-stage Error naming it and the remedy, while collection still runs.
 * Otherwise the analyzer is only named, never run. Needs no gperftools, so
 * the rule holds, and is tested, on every build.
 */
GperfAnalysis decideGperfAnalysis(const GperfModes& modes, bool analyze,
                                  const ReadinessContext& ctx);

/**
 * @brief The gperf backend's readiness decision for @p request in @p ctx.
 *
 * On success, or on an analysis-stage error, the result's plan is a GperfPlan.
 */
ReadinessResult checkGperfRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/* ----------------------------- GperfProfiler ----------------------------- */

class GperfProfiler final : public Profiler {
public:
  /**
   * @brief Construct, deciding the request itself; when it cannot run,
   * prints why and does nothing in the hooks.
   */
  GperfProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Construct from a decision already made (the registry's path). */
  GperfProfiler(const PerfConfig& cfg, std::string testName, std::shared_ptr<const GperfPlan> plan);
  ~GperfProfiler() override = default;

  std::string toolName() const noexcept override { return "gperf"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const GperfPlan> plan_;

  void applyPlan();
  void runPprofAnalysis() const;

  bool wantCpu_{false};
  bool wantHeap_{false};

#if UB_HAS_GPERF_CPU
  std::string cpuPath_;
#endif
#if UB_HAS_GPERF_HEAP
  std::string heapPrefix_;
#endif
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Factory function for gperftools profiler.
 *
 * Decides the request in a snapshot of this process first.
 * @return Profiler instance, or nullptr if collection cannot run here.
 */
std::unique_ptr<Profiler> makeGperfProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERGPERF_HPP
