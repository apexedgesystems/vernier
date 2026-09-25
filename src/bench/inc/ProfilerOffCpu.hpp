#ifndef VERNIER_PROFILEROFFCPU_HPP
#define VERNIER_PROFILEROFFCPU_HPP
/**
 * @file ProfilerOffCpu.hpp
 * @brief Off-CPU profiling backend via bpftrace.
 *
 * The on-CPU profilers (perf, gperf, callgrind, bpftrace, rapl, nsight)
 * measure work. Off-CPU profiling answers the complementary question: where
 * do threads spend time *blocked* (sleep, mutex wait, I/O wait, scheduler
 * delay)?
 *
 * Probes: an embedded bpftrace script on the sched tracepoints (a stable
 * kernel interface). At switch-out it counts the user stack of each thread of
 * this process that blocks; at switch-in it sums how long that thread was
 * off the CPU. It exits by itself when this process exits.
 *
 * Privileges: bpftrace runs as the current user unless BENCH_SUDO opts in to
 * `sudo -n` (PERF_BPF_SUDO does not apply to this backend); root never uses
 * sudo. The readiness check (checkOffCpuRequest) runs the launch's script
 * with a 5 s self-exit added through that route for the start grace and
 * stops it with SIGINT; a probe whose stop is refused ends by that self-exit,
 * and the check waits for it and reaps it. The added interval makes the
 * probe a command the run never runs, so a grant's refusal of it is
 * unverified and the run's start decides. The profiler launches and stops
 * with exactly the tools and route it verified (OffCpuPlan).
 *
 * Output: `<testName>.offcpu/offcpu.txt` (the bpftrace map dump) and
 * `offcpu.err.txt` (bpftrace's messages).
 *
 * Limitations:
 *  - The PID filter keeps this process; its threads are joined through the
 *    tid-keyed start map.
 *  - The sched tracepoints need tracefs (`/sys/kernel/tracing`). The default
 *    dev container does not mount it, so there the check reports the
 *    tracepoint as unsupported.
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerBpftrace.hpp" // BpftraceRoute

namespace vernier {
namespace bench {

/* ----------------------------- OffCpuPlan ----------------------------- */

/** @brief What the offcpu check verified, for the launch to use. */
struct OffCpuPlan final : ReadinessPlan {
  BpftraceRoute route;
  std::shared_ptr<const ReadinessContext> context;
};

/**
 * @brief The offcpu backend's readiness decision for @p request in @p ctx.
 *
 * On success the result's plan is an OffCpuPlan.
 */
ReadinessResult checkOffCpuRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/* ----------------------------- OffCpuProfiler ----------------------------- */

class OffCpuProfiler final : public Profiler {
public:
  /**
   * @brief Construct, deciding the request itself; when it cannot run, prints
   * why and does nothing in the hooks.
   */
  OffCpuProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Construct from a decision already made (the registry's path). */
  OffCpuProfiler(const PerfConfig& cfg, std::string testName,
                 std::shared_ptr<const OffCpuPlan> plan);
  ~OffCpuProfiler() override;

  std::string toolName() const noexcept override { return "offcpu"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  void spawnBpftrace();
  void stopBpftrace();

  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::string outputPath_;
  std::string errorPath_;
  std::shared_ptr<const OffCpuPlan> plan_;
  std::unique_ptr<OwnedHelper> helper_;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Factory: decides the request in a snapshot of this process first.
 * @return Profiler instance, or nullptr if the request cannot run here.
 */
std::unique_ptr<Profiler> makeOffCpuProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILEROFFCPU_HPP
