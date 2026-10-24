#ifndef VERNIER_PROFILERMEMCHECK_HPP
#define VERNIER_PROFILERMEMCHECK_HPP
/**
 * @file ProfilerMemcheck.hpp
 * @brief Valgrind Memcheck memory error / leak detector backend.
 *
 * Memcheck catches the classic C/C++ memory bugs:
 *  - Use of uninitialized memory
 *  - Reads/writes after free()
 *  - Reads/writes past the end of malloc'd blocks
 *  - Memory leaks (definitely lost / indirectly lost / possibly lost)
 *  - Mismatched malloc/new and free/delete
 *
 * Not a perf profiler per se -- valgrind memcheck slows execution ~20x. The
 * value is running memcheck alongside benchmarks to catch correctness
 * regressions introduced by an optimization pass.
 *
 * Memcheck checks only when valgrind runs the whole process; a request run
 * without it fails (exit status 4) and prints the wrap command:
 *
 *   bench run ./MyTest --profile memcheck --cycles 5 --gtest_filter='Foo.Bar'
 *
 *   # or by hand:
 *   valgrind --tool=memcheck --leak-check=full --error-exitcode=1 \
 *       --log-file=run.memcheck.log \
 *       ./MyTest --profile memcheck --cycles 5 --gtest_filter='Foo.Bar'
 *
 * Modes (--profile-args, words separated by spaces or commas):
 *   default               --leak-check=full, as bench run passes it
 *   "leak-full"           the same full leak check
 *   "track-origins"       --track-origins=yes (helps locate uninit-read sources)
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

/* ----------------------------- MemcheckProfiler ----------------------------- */

class MemcheckProfiler final : public Profiler {
public:
  /**
   * @brief Decide the request now and report it as the registry does: one
   * that cannot check fails the run and creates nothing.
   */
  MemcheckProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  MemcheckProfiler(const PerfConfig& cfg, std::string testName,
                   std::shared_ptr<const valgrind_tool::ValgrindPlan> plan);

  ~MemcheckProfiler() override = default;

  std::string toolName() const noexcept override { return "memcheck"; }
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
 * @brief Read memcheck's mode from @p profileArgs into @p mode.
 * @return The error that refuses a word other than leak-full and
 *         track-origins; nullopt when the mode was read.
 */
std::optional<ReadinessResult> parseMemcheckMode(const std::string& profileArgs,
                                                 valgrind_tool::ValgrindMode& mode);

/**
 * @brief The memcheck backend's readiness decision for @p request in @p ctx.
 *
 * The doctor's scopes probe the tool's start; a run reads its own memory map
 * for valgrind's memcheck. --profile-analyze is an analysis-stage error:
 * memcheck has no automatic analysis (its log is the report). On success the
 * result's plan is a ValgrindPlan.
 */
ReadinessResult checkMemcheckRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

std::unique_ptr<Profiler> makeMemcheckProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERMEMCHECK_HPP
