#ifndef VERNIER_PROFILERMASSIF_HPP
#define VERNIER_PROFILERMASSIF_HPP
/**
 * @file ProfilerMassif.hpp
 * @brief Valgrind Massif heap profiler backend.
 *
 * Massif samples heap usage over time, producing a `massif.out` file that
 * ms_print renders as a stacked timeline of allocation sites. Complements
 * callgrind (which is CPU/instruction-focused): use massif when the question
 * is "what is allocating, and how much."
 *
 * Massif collects only when valgrind runs the whole process; a request run
 * without it fails (exit status 4) and prints the wrap command:
 *
 *   bench run ./MyTest --profile massif --cycles 10 --gtest_filter='Foo.Bar'
 *
 *   # or by hand:
 *   valgrind --tool=massif --massif-out-file=run.massif.out \
 *       ./MyTest --profile massif --cycles 10 --gtest_filter='Foo.Bar'
 *
 *   ms_print run.massif.out | head -40
 *
 * Modes (--profile-args), one at most:
 *   default               heap allocations only
 *   "pages"               page-level profiling (--pages-as-heap=yes)
 *   "stacks"              include stack allocations (--stacks=yes)
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

/* ----------------------------- MassifProfiler ----------------------------- */

class MassifProfiler final : public Profiler {
public:
  /** @brief Decide the request now; prints the decision when it cannot collect. */
  MassifProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  MassifProfiler(const PerfConfig& cfg, std::string testName,
                 std::shared_ptr<const valgrind_tool::ValgrindPlan> plan);

  ~MassifProfiler() override = default;

  std::string toolName() const noexcept override { return "massif"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const valgrind_tool::ValgrindPlan> plan_;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Read massif's mode from @p profileArgs into @p mode.
 * @return The error that refuses the words: a word other than pages and
 *         stacks, or both of them; nullopt when the mode was read.
 */
std::optional<ReadinessResult> parseMassifMode(const std::string& profileArgs,
                                               valgrind_tool::ValgrindMode& mode);

/**
 * @brief The massif backend's readiness decision for @p request in @p ctx.
 *
 * The doctor's scopes probe the tool's start; a run reads its own memory map
 * for valgrind's massif. --profile-analyze is an analysis-stage error: massif
 * has no automatic analysis (ms_print reads the profile). On success the
 * result's plan is a ValgrindPlan.
 */
ReadinessResult checkMassifRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

std::unique_ptr<Profiler> makeMassifProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERMASSIF_HPP
