#ifndef VERNIER_PROFILERROCPROF_HPP
#define VERNIER_PROFILERROCPROF_HPP
/**
 * @file ProfilerRocprof.hpp
 * @brief Backend for AMD's legacy rocprof, the command-line profiler of the
 * first ROCProfiler generation.
 *
 * Not validated: nobody on this project has run this backend against an AMD
 * GPU, and AMD deprecates rocprof in favour of newer tools. Its readiness
 * decision is therefore never Ok: with rocprof on PATH the doctor reports it
 * as unverified, and a run under rocprof's injection proceeds with the same
 * warning. rocprof writes its reports where its -o option points; which files
 * a given rocprof writes, and what they hold, is not checked here.
 *
 * Nothing is needed at build time: detection is at run time only (rocprof on
 * PATH, its injection in the process's environment). `bench run` does not
 * wrap rocprof; a request run without it fails (exit status 4) and prints the
 * command, for example:
 *
 *   rocprof --stats -o ./results.csv \
 *       ./MyTest --profile rocprof --profile-args stats [...]
 *
 * Modes (--profile-args, words separated by spaces or commas):
 *   default                no extra flag
 *   "stats"                --stats
 *   "hsa-trace"            --hsa-trace
 *   "hip-trace"            --hip-trace
 */

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- RocprofPlan ----------------------------- */

/** @brief What the rocprof check verified, for the launch to use. */
struct RocprofPlan final : ReadinessPlan {
  std::string rocprof;                              ///< The resolved rocprof (doctor scopes).
  std::vector<std::string> flags;                   ///< rocprof's flags for the requested modes.
  LaunchContext launch = LaunchContext::IN_PROCESS; ///< At the runtime scope: how the wrap began.
};

/* ----------------------------- RocprofProfiler ----------------------------- */

class RocprofProfiler final : public Profiler {
public:
  /**
   * @brief Decide the request now; prints the decision when it cannot run,
   * and creates no folder then.
   */
  RocprofProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  RocprofProfiler(const PerfConfig& cfg, std::string testName,
                  std::shared_ptr<const RocprofPlan> plan);

  ~RocprofProfiler() override = default;

  std::string toolName() const noexcept override { return "rocprof"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const RocprofPlan> plan_;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Read rocprof's flags from @p profileArgs into @p flags.
 * @return The error that refuses a word other than stats, hsa-trace and
 *         hip-trace; nullopt when the flags were read.
 */
std::optional<ReadinessResult> parseRocprofMode(const std::string& profileArgs,
                                                std::vector<std::string>& flags);

/**
 * @brief The rocprof backend's readiness decision for @p request in @p ctx.
 *
 * Never Ok. The doctor's scopes: MISSING without rocprof, otherwise
 * unverified (AMD collection is not validated). A run: unverified under
 * rocprof's injection (ROCP_TOOL_LIB, ROCPROFILER_LIBRARY, or rocprof in
 * LD_PRELOAD, from the snapshot), otherwise an error that prints the wrap
 * command. --profile-analyze is an analysis-stage error: rocprof's reports are
 * read as they are. On a result that lets the request run the plan is a
 * RocprofPlan.
 */
ReadinessResult checkRocprofRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/** @brief Factory: decides the request when the profiler is constructed. */
std::unique_ptr<Profiler> makeRocprofProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERROCPROF_HPP
