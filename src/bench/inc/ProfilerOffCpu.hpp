#ifndef VERNIER_PROFILEROFFCPU_HPP
#define VERNIER_PROFILEROFFCPU_HPP
/**
 * @file ProfilerOffCpu.hpp
 * @brief Off-CPU profiling backend via bpftrace.
 *
 * All six pre-existing profilers (perf, gperf, callgrind, bpftrace, rapl,
 * nsight) measure *on-CPU* work. Off-CPU profiling answers the
 * complementary question: where do threads spend time *blocked* (sleep,
 * mutex wait, I/O wait, scheduler delay)?
 *
 * Hot path: bpftrace attached to a kprobe on `finish_task_switch` collects
 * the user stack and elapsed nanoseconds every time a task is descheduled.
 * Aggregated stacks rank-ordered by total off-CPU time identify the
 * blocking call sites.
 *
 * Requires root or CAP_BPF (same constraint as the bpftrace backend).
 *
 * Output: `<testName>.offcpu/offcpu.txt` containing the bpftrace map dump.
 *
 * Limitations:
 *  - Kernel symbol shape (`finish_task_switch`, with or without `.isra.0`
 *    suffix) varies across distros; bpftrace's wildcard match handles
 *    common cases but may need tuning per kernel.
 *  - PID filter narrows the trace to this process; child threads are
 *    included via the tid-keyed start map.
 *  - Docker constraint: tracefs (`/sys/kernel/tracing`) is not mounted
 *    in the default dev container. To exercise this backend inside
 *    Docker, run the container with `--mount
 * type=bind,source=/sys/kernel/tracing,target=/sys/kernel/tracing` (and `--privileged` for kernel
 * symbol access). On bare metal this works directly under sudo.
 */

#include <memory>
#include <string>

#ifdef __linux__
#include <sys/types.h> // pid_t
#endif

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- OffCpuProfiler ----------------------------- */

class OffCpuProfiler final : public Profiler {
public:
  OffCpuProfiler(const PerfConfig& cfg, std::string testName);
  ~OffCpuProfiler() override = default;

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
#ifdef __linux__
  pid_t childPid_ = -1;
  bool viaSudo_ = false;
#endif
};

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeOffCpuProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILEROFFCPU_HPP
