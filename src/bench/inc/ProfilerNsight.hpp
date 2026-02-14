#ifndef VERNIER_PROFILERNSIGHT_HPP
#define VERNIER_PROFILERNSIGHT_HPP
/**
 * @file ProfilerNsight.hpp
 * @brief NVIDIA Nsight profiler backend for GPU benchmarking.
 *
 * Behavior:
 *  - Nsight Systems (nsys): Attaches to running process via `-p <pid>` for timeline profiling
 *  - Nsight Compute (ncu): Attaches to running process for kernel analysis (limited support)
 *  - Kernel replay mode: Automatic metrics collection with detailed analysis
 *  - Artifacts written to `<artifactRoot>/<Suite.Case>.nsight/`
 *  - Integrates automatically via GPU profiler hooks
 *
 * Usage:
 *  @code{.cpp}
 *  PERF_GPU_TEST(MyKernel, Benchmark) {
 *    UB_PERF_GPU_GUARD(perf);
 *    ub::attachGpuProfilerHooks(perf, perf.cpuConfig());
 *    // ... test code ...
 *  }
 *  @endcode
 *
 *  Then run with:
 *    --profile nsight                          # Nsight Systems (timeline)
 *    --profile nsight --profile-args "replay"  # Kernel replay with metrics
 *
 * Notes:
 *  - Nsight Systems (nsys) works like perf - attaches to running process
 *  - Nsight Compute (ncu) has limited attach support (use manual wrapper for deep analysis)
 *  - Kernel replay mode automatically collects key bottleneck metrics
 *  - Requires NVIDIA Nsight tools to be installed
 *  - Safe no-op when tools unavailable
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
  Systems,      ///< Nsight Systems (nsys) - timeline profiling (default, auto-attaches)
  Compute,      ///< Nsight Compute (ncu) - kernel analysis (limited attach support)
  ComputeReplay ///< Kernel replay with detailed metrics
};

/* ----------------------------- ReplayMetrics ----------------------------- */

/**
 * @brief Kernel replay metrics configuration.
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
  bool isNsysAvailable() const;
  bool isNcuAvailable() const;
  void launchNsys();
  void launchNcu();
  void launchNcuReplay();
  void stopProfiler();
  void parseReplayMetrics();

  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;
  NsightMode mode_ = NsightMode::Systems;
  pid_t childPid_ = -1;

  ReplayMetrics replayMetrics_{};
  bool useReplayMode_ = false;
};

/* --------------------------------- API --------------------------------- */

/** @brief Factory: creates Nsight profiler (definition in ProfilerNsight.cu). */
std::unique_ptr<Profiler> makeNsightProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERNSIGHT_HPP