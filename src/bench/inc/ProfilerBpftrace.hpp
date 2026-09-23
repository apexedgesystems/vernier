#ifndef VERNIER_PROFILERBPFTRACE_HPP
#define VERNIER_PROFILERBPFTRACE_HPP
/**
 * @file ProfilerBpftrace.hpp
 * @brief bpftrace backend for the benchmarking profiler facade.
 *
 * Behavior:
 *  - The readiness check (checkBpftraceRequest) decides the privilege route,
 *    resolves bpftrace (and, on the sudo route, sudo and kill) on PATH,
 *    resolves every selected script, runs `bpftrace --version` as the current
 *    user, and attaches each selected script through the route in a probe
 *    that it then stops with SIGINT. The profiler launches and stops with
 *    exactly the tools and route that check verified (BpftracePlan).
 *  - In beforeMeasure(), starts one bpftrace process per selected script
 *    (e.g. "write_latency", "fsync_latency") with {{PID}} replaced by the
 *    current PID, and reports a tracer that exits during its start grace.
 *  - In afterMeasure(), stops every tracer with SIGINT, then SIGTERM, then
 *    SIGKILL through the same route, and reports each refused delivery.
 *
 * Privileges: bpftrace runs as the current user unless BENCH_SUDO opts in to
 * `sudo -n`; PERF_BPF_SUDO is a deprecated alias that BENCH_SUDO overrides.
 * Root never uses sudo.
 *
 * Notes:
 *  - Linux-only. Safe no-op on other platforms (compile-time guard).
 */

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- bpftrace route ----------------------------- */

/**
 * @brief How bpftrace is started and stopped for one request: the privilege
 * decision and the tools resolved on the snapshot's PATH.
 *
 * Shared by the backends that run bpftrace (bpftrace, offcpu).
 */
struct BpftraceRoute {
  PrivilegeDecision privilege;
  std::string bpftrace; ///< Absolute path of bpftrace.
  std::string sudo;     ///< Absolute path of sudo; set on the sudo route only.
  std::string kill;     ///< Absolute path of kill; set on the sudo route only.

  /** @brief The argv prefix that runs bpftrace through the route. */
  [[nodiscard]] std::vector<std::string> command() const;

  /** @brief Stop policy for a tracer started through the route. */
  [[nodiscard]] HelperStopPolicy stopPolicy(int interruptWaitMs, int terminateWaitMs,
                                            int killWaitMs,
                                            std::shared_ptr<const ReadinessContext> ctx) const;

  /** @brief "as the current user", "as root" or "through sudo -n (BENCH_SUDO=1)". */
  [[nodiscard]] std::string describe() const;
};

namespace bpftrace_tool {

/**
 * @brief Decide the route and resolve its tools.
 * @return The error that stops the request, or nullopt with @p route filled.
 */
std::optional<ReadinessResult> resolveRoute(const ReadinessContext& ctx,
                                            std::string_view legacyAlias, BpftraceRoute& route);

/** @brief `<bpftrace> --version` as the current user: the executable runs. */
std::optional<ReadinessResult> probeExecutable(const BpftraceRoute& route,
                                               const ReadinessContext& ctx);

/**
 * @brief Attach through the route and stop with the real first stop signal.
 *
 * Runs route.command() + @p toolArgs (a probe copy of the selected script
 * that exits by itself) as an owned helper, waits @p graceMs, and stops it
 * with SIGINT through the route (then SIGTERM and SIGKILL if needed).
 * @return nullopt when it stayed attached and stopped on SIGINT; otherwise
 *         the Error (refused, unsupported, denied, unusable) or the Warning
 *         (it ignored SIGINT).
 */
std::optional<ReadinessResult> probeAttach(const BpftraceRoute& route,
                                           const std::vector<std::string>& toolArgs,
                                           const std::string& what, int graceMs,
                                           const ReadinessContext& ctx,
                                           const std::string& scratchDir);

/** @brief The Error for a tracer that exited at start, from its stderr. */
ReadinessResult classifyAttachFailure(const BpftraceRoute& route, const std::string& what,
                                      const std::string& commandLine, const std::string& stderrText,
                                      const ReadinessContext& ctx);

/** @brief How to get access without elevation or with a scoped grant. */
std::string optInRemedy(const BpftraceRoute& route, const ReadinessContext& ctx);

/** @brief What a scoped sudoers grant must allow. */
std::string grantRemedy(const BpftraceRoute& route, const ReadinessContext& ctx);

/**
 * @brief Print how a run's tracer stop went, prefixed "[<tag>]".
 * @return True when the tracer ended on SIGINT or SIGTERM (its output flushed).
 */
bool reportStop(const char* tag, const std::string& what, const HelperStopResult& stop,
                const BpftraceRoute& route);

} // namespace bpftrace_tool

/* ----------------------------- BpftracePlan ----------------------------- */

/** @brief What the bpftrace check verified, for the launch to use. */
struct BpftracePlan final : ReadinessPlan {
  BpftraceRoute route;
  std::vector<std::string> scripts;     ///< Script names as requested.
  std::vector<std::string> scriptPaths; ///< Their resolved files, same order.
  std::string format = "text";          ///< Output format (PERF_BPF_FMT).
  std::string outputDir;                ///< PERF_BPF_OUT, or empty.
  bool envEnabled = false;              ///< PERF_BPF asked for bpftrace.
  std::shared_ptr<const ReadinessContext> context;
};

/**
 * @brief The bpftrace backend's readiness decision for @p request in @p ctx.
 *
 * Reads BENCH_SUDO, PERF_BPF_SUDO, PERF_BPF_SCRIPTS, PERF_BPF_FMT, PERF_BPF and
 * PERF_BPF_OUT from the snapshot. On success the result's plan is a
 * BpftracePlan.
 */
ReadinessResult checkBpftraceRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

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
   * @brief Construct bpftrace profiler, deciding the request itself.
   *
   * Captures the environment and runs checkBpftraceRequest(); when the
   * request cannot run, prints why and does nothing in the hooks.
   * @param cfg Configuration with bpfScripts and artifactRoot
   * @param testName Test identifier (e.g., "Suite.Case")
   */
  BpftraceProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Construct from a decision already made (the registry's path). */
  BpftraceProfiler(const PerfConfig& cfg, std::string testName,
                   std::shared_ptr<const BpftracePlan> plan);
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
 *
 * Decides the request in a snapshot of this process first.
 * @return Profiler instance, or nullptr if the request cannot run here.
 */
std::unique_ptr<Profiler> makeBpftraceProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERBPFTRACE_HPP
