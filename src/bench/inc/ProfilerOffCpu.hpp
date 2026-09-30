#ifndef VERNIER_PROFILEROFFCPU_HPP
#define VERNIER_PROFILEROFFCPU_HPP
/**
 * @file ProfilerOffCpu.hpp
 * @brief Off-CPU profiling backend via bpftrace.
 *
 * The on-CPU profilers (perf, gperf, callgrind, bpftrace, rapl, nsight)
 * measure work. Off-CPU profiling answers the complementary question: where
 * do threads spend time *blocked* (asleep, waiting for a mutex, for I/O or
 * for another thread)?
 *
 * Probes: an embedded bpftrace script on the sched tracepoints (a stable
 * kernel interface). At switch-out it counts the user stack of each thread of
 * this process that goes to sleep; a thread that is preempted or exits is not
 * counted. At switch-in it sums how long that thread was off the CPU, which
 * includes the wait for a CPU after its wakeup. It exits by itself when this
 * process's main thread exits, not when one of its other threads does.
 *
 * Capture: beforeMeasure() starts the tracer and waits until it arms. The
 * script records nothing until it sees a thread of this process named
 * vernier-arm, which the backend starts, go to sleep, and it then prints
 * "offcpu armed <pid> <tid>"; the backend checks that the ids are its own.
 * afterMeasure() names the calling thread vernier-stop until the tracer
 * disarms and prints "offcpu disarmed <pid> <tid> <n>", n the switch-outs it
 * recorded, then stops it with SIGINT, which makes it print its maps. The
 * backend's own waits run under the name vernier-wait, which the script does
 * not record. The three names are reserved: a test's own thread given one of
 * them disturbs the capture.
 *
 * Privileges: bpftrace runs as the current user unless BENCH_SUDO opts in to
 * `sudo -n` (PERF_BPF_SUDO does not apply to this backend); root never uses
 * sudo. The run's command is `<bpftrace> -B none -e <the off-CPU script>
 * <pid>`, its output unbuffered so that each acknowledgement arrives as it is
 * printed; with BENCH_SUDO it runs as `sudo -n -- <bpftrace> -B none -e ...`,
 * and `sudo -n -- <kill> -2|-15|-9 <tracer>` stops it. The readiness check
 * (checkOffCpuRequest) runs the launch's script with a 5 s self-exit added
 * through that route for the start grace and stops it with SIGINT; a probe
 * whose stop is refused ends by that self-exit, and the check waits for it
 * and reaps it. The added interval makes the probe a command the run never
 * runs, so a grant's refusal of it is unverified and the run's start decides.
 * The profiler launches and stops with exactly the tools and route it
 * verified (OffCpuPlan).
 *
 * Output: `<testName>.offcpu/offcpu.txt` (the tracer's acknowledgements and
 * its map dump) and `offcpu.err.txt` (bpftrace's messages). Each capture ends
 * with one outcome, printed with an `[offcpu]` prefix and kept by
 * captureOutcome(): `stacks written to <path>` for a capture the tracer
 * acknowledged from its start to its stop, that ended cleanly on the stop and
 * whose dump is whole; a line of its own when no thread of this process slept
 * in that window; an error otherwise, the output kept: no arm
 * acknowledgement, or one for other ids; a tracer that ended before the stop,
 * could not be stopped, was killed or did not end cleanly; no stop
 * acknowledgement ("capture validity could not be established"); an output
 * that cannot be read, or a dump cut short (an entry cut, or no @recorded
 * line where the tracer recorded switch-outs).
 *
 * Limitations:
 *  - The PID filter keeps this process; its threads are joined through the
 *    tid-keyed start map.
 *  - bpftrace's count() creates a key with an insert that fails when another
 *    CPU creates the same key at the same moment, so threads that first sleep
 *    at one stack at the same moment can leave that stack's count short. The
 *    per-thread times, whose keys are the threads, are not affected.
 *  - Host PID view only. bpftrace in a PID namespace of its own does not
 *    number this process's threads as the process does (0.20 reports the
 *    host's ids there, 0.23.0 to 0.24.1 read pid and tid swapped), so the
 *    tracer never arms and the run reports that, naming the namespace. Run
 *    such a benchmark on the host, or in a container started with
 *    --pid=host.
 *  - The sched tracepoints need tracefs (`/sys/kernel/tracing`). The default
 *    dev container does not mount it, so there the check reports the
 *    tracepoint as unsupported.
 */

#include <memory>
#include <optional>
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
  int armWaitMs = 5000;    ///< How long a capture's start waits for the tracer to arm.
  int disarmWaitMs = 3000; ///< How long its stop waits for the tracer to disarm.
};

/**
 * @brief The sentence a report adds when this process is not in the initial
 * PID namespace, from the text of the /proc/self/ns/pid link
 * ("pid:[4026531836]" for the initial one); "" in the initial namespace.
 */
[[nodiscard]] std::string offCpuPidNamespaceNote(const std::string& namespaceLink);

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

  /**
   * @brief How the last capture ended: Ok for a verified capture (with
   * stacks, or with no thread asleep in its window), an Error saying what
   * failed otherwise, the decision's Error when the request could not run.
   * Empty before the first capture ends.
   */
  [[nodiscard]] const std::optional<ReadinessResult>& captureOutcome() const noexcept {
    return outcome_;
  }

private:
  struct Capture; ///< One running capture: its tracer and what was read of its output.

  void spawnBpftrace();
  void stopBpftrace();
  void finish(ReadinessResult outcome, const std::string& context = {});

  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::string outputPath_;
  std::string errorPath_;
  std::shared_ptr<const OffCpuPlan> plan_;
  std::unique_ptr<Capture> capture_;
  std::optional<ReadinessResult> outcome_;
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
