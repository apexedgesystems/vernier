#ifndef VERNIER_PROFILERNSIGHT_HPP
#define VERNIER_PROFILERNSIGHT_HPP
/**
 * @file ProfilerNsight.hpp
 * @brief NVIDIA Nsight profiler backend for GPU benchmarking.
 *
 * Behavior:
 *  - Nsight Systems (nsys) and Nsight Compute (ncu) record a process only when
 *    they start it: neither can attach to one that is already running. This
 *    backend never starts either tool.
 *  - Under an nsys or ncu session (`bench run --profile nsight|ncu`, or a
 *    wrap typed by hand) the backend stays passive and says which tool owns
 *    the capture.
 *  - Without a session the request fails at its readiness check
 *    (ProfilerNsightChecks.hpp), which prints the command that captures the
 *    run, and no backend is built. A backend built directly prints that
 *    command once per test, and its measurement proceeds unprofiled.
 *  - Each measured window is marked with an NVTX range named after the test.
 *  - Artifact folder: `<artifactRoot>/<Suite.Case>.nsight/` (`.ncu/` for
 *    --profile ncu), or the runner's folder under `bench run`.
 *
 * Usage:
 *  @code{.cpp}
 *  PERF_GPU_TEST(MyKernel, Benchmark) {
 *    PERF_GPU_GUARD(perf); // attaches the profiler hooks
 *    // ... test code ...
 *  }
 *  @endcode
 *
 *  Then run with:
 *    bench run <binary> --profile nsight       # Nsight Systems (timeline)
 *    bench run <binary> --profile ncu          # Nsight Compute (kernel analysis)
 *    --profile nsight --profile-args compute   # Compute mode via the nsight name
 *    --profile ncu --profile-args replay       # Compute with the replay metric list
 *    --profile ncu --profile-args '--set full' # a section set, sections, metrics
 *                                              # (parseNsightRequest())
 *
 * Notes:
 *  - Both names dispatch to this backend; "ncu" forces Compute mode.
 *  - Nsight Compute replays every kernel launch several times: keep the
 *    launch count small (--cycles 3 --repeats 1).
 *  - Requires NVIDIA Nsight tools to be installed.
 */

#include <array>
#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Enums ----------------------------- */

/**
 * @brief Nsight profiler mode selection.
 */
enum class NsightMode {
  Systems,      ///< Nsight Systems (nsys) - timeline profiling (default)
  Compute,      ///< Nsight Compute (ncu) - kernel analysis
  ComputeReplay ///< Nsight Compute with the REPLAY_METRICS list
};

/* ----------------------------- Replay Metrics ----------------------------- */

/**
 * @brief The metrics the replay mode's ncu command collects: the
 * throughputs, achieved occupancy, warp execution and branch efficiency,
 * registers per thread and the instruction mix (the ALU, FMA and load/store
 * pipes). Every name is one ncu lists; the shared route table's replay rows
 * pin the list, which bench run's route gives too.
 */
inline constexpr std::array<const char*, 11> REPLAY_METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__thread_inst_executed_per_inst_executed.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "launch__registers_per_thread",
    "smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
};

/** @brief REPLAY_METRICS as ncu's --metrics takes them: joined with ','. */
[[nodiscard]] inline std::string replayMetricList() {
  std::string out;
  for (const char* METRIC : REPLAY_METRICS) {
    if (!out.empty()) {
      out += ',';
    }
    out += METRIC;
  }
  return out;
}

/* ----------------------------- NsightProfiler ----------------------------- */

class NsightProfiler final : public Profiler {
public:
  NsightProfiler(const PerfConfig& cfg, std::string testName);
  ~NsightProfiler() override;

  std::string toolName() const noexcept override { return "nsight"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  /// The command that captures this run in the selected mode, for a run no
  /// Nsight tool started.
  void printWrapCommand() const;
  void popRange() noexcept;

  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;
  NsightMode mode_ = NsightMode::Systems;

  // True while the NVTX range pushed by beforeMeasure() is open, so it is
  // popped exactly once, by afterMeasure() or by the destructor.
  bool nvtxRangePush_ = false;
};

/* --------------------------------- API --------------------------------- */

/** @brief Factory: creates Nsight profiler (definition in ProfilerNsight.cu). */
std::unique_ptr<Profiler> makeNsightProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERNSIGHT_HPP