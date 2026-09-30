#ifndef VERNIER_PROFILERHELGRIND_HPP
#define VERNIER_PROFILERHELGRIND_HPP
/**
 * @file ProfilerHelgrind.hpp
 * @brief Valgrind Helgrind / DRD thread-error detector backend.
 *
 * The CPU analog of compute-sanitizer's race/sync checks. Helgrind catches the
 * classic multithreading bugs that on-CPU profilers can't see:
 *  - Data races (unsynchronized access to shared memory)
 *  - Lock-ordering violations (potential deadlocks)
 *  - Misuse of the POSIX pthreads API
 *
 * Complements memcheck (memory errors) and compute-sanitizer (GPU races):
 * run it alongside benchmarks of multithreaded code to catch concurrency
 * regressions introduced by an optimization pass. Slows execution heavily
 * (~20-100x), so use a low --cycles.
 *
 * Helgrind checks only when valgrind runs the whole process; a request run
 * without it fails (exit status 4) and prints the wrap command:
 *
 *   bench run ./MyTest --profile helgrind --cycles 5 --gtest_filter='Foo.Bar'
 *
 *   # or by hand:
 *   valgrind --tool=helgrind --log-file=run.helgrind.log \
 *       ./MyTest --profile helgrind --cycles 5 --gtest_filter='Foo.Bar'
 *
 * Modes (--profile-args):
 *   default    helgrind (data races + lock order + pthread misuse)
 *   "drd"      valgrind's DRD instead (lower memory, per-thread; also detects races)
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

/* ----------------------------- HelgrindProfiler ----------------------------- */

class HelgrindProfiler final : public Profiler {
public:
  /** @brief Decide the request now; prints the decision when it cannot check. */
  HelgrindProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  HelgrindProfiler(const PerfConfig& cfg, std::string testName,
                   std::shared_ptr<const valgrind_tool::ValgrindPlan> plan);

  ~HelgrindProfiler() override = default;

  std::string toolName() const noexcept override { return "helgrind"; }
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
 * @brief Read helgrind's mode from @p profileArgs into @p mode: valgrind's
 * helgrind, or its drd for the word drd.
 * @return The error that refuses any other word; nullopt when the mode was read.
 */
std::optional<ReadinessResult> parseHelgrindMode(const std::string& profileArgs,
                                                 valgrind_tool::ValgrindMode& mode);

/**
 * @brief The helgrind backend's readiness decision for @p request in @p ctx.
 *
 * The doctor's scopes probe the selected tool's start (drd as drd); a run
 * reads its own memory map for that tool. --profile-analyze is an
 * analysis-stage error: the tool's log is the report. On success the
 * result's plan is a ValgrindPlan.
 */
ReadinessResult checkHelgrindRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

std::unique_ptr<Profiler> makeHelgrindProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERHELGRIND_HPP
