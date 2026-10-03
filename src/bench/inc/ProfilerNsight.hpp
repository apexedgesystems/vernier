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
 *  - Without a session it prints the command that captures this run for the
 *    selected mode, and the measurement proceeds unprofiled.
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
 *
 * Notes:
 *  - Both names dispatch to this backend; "ncu" forces Compute mode.
 *  - Nsight Compute replays every kernel launch several times: keep the
 *    launch count small (--cycles 3 --repeats 1).
 *  - Requires NVIDIA Nsight tools to be installed.
 */

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
  ComputeReplay ///< Nsight Compute with the ReplayMetrics list
};

/* ----------------------------- ReplayMetrics ----------------------------- */

/**
 * @brief The metrics the replay mode's ncu command collects.
 */
struct ReplayMetrics {
  bool collectOccupancy = true;
  bool collectMemoryBandwidth = true;
  bool collectWarpEfficiency = true;
  bool collectBranchEfficiency = true;
  bool collectRegisterUsage = true;
  bool collectInstructionMix = true;

  [[nodiscard]] std::string toNcuMetricString() const {
    std::vector<std::string> metrics;

    if (collectOccupancy) {
      metrics.push_back("sm__throughput.avg.pct_of_peak_sustained_elapsed");
      metrics.push_back("sm__warps_active.avg.pct_of_peak_sustained_active");
    }
    if (collectMemoryBandwidth) {
      metrics.push_back("dram__throughput.avg.pct_of_peak_sustained_elapsed");
      metrics.push_back("l1tex__throughput.avg.pct_of_peak_sustained_elapsed");
      metrics.push_back("lts__throughput.avg.pct_of_peak_sustained_elapsed");
    }
    if (collectWarpEfficiency) {
      metrics.push_back("smsp__average_warps_issue_stalled_per_issue_active.pct");
    }
    if (collectBranchEfficiency) {
      metrics.push_back("smsp__sass_average_branch_targets_threads_uniform.pct");
    }
    if (collectRegisterUsage) {
      metrics.push_back("launch__registers_per_thread");
    }
    if (collectInstructionMix) {
      metrics.push_back("smsp__inst_executed_pipe_alu_pred_on.avg.pct_of_peak_sustained_elapsed");
      metrics.push_back("smsp__inst_executed_pipe_fma_type_fp32.avg.pct_of_peak_sustained_elapsed");
      metrics.push_back("smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed");
    }

    std::string result;
    for (size_t i = 0; i < metrics.size(); ++i) {
      if (i > 0)
        result += ",";
      result += metrics[i];
    }
    return result;
  }
};

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

  ReplayMetrics replayMetrics_{};

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