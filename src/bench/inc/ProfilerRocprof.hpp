#ifndef VERNIER_PROFILERROCPROF_HPP
#define VERNIER_PROFILERROCPROF_HPP
/**
 * @file ProfilerRocprof.hpp
 * @brief AMD ROCm rocprof backend -- timeline + kernel profiling on AMD GPUs.
 *
 * rocprof is the AMD analog of Nsight Systems. It wraps a HIP / OpenCL /
 * OpenMP binary and produces:
 *   - results.csv      per-kernel timing summary
 *   - results.json     timeline (Chrome trace format; open with chrome://tracing)
 *   - results.stats.csv kernel statistics
 *
 * Opens AMD MI / Radeon Instinct support for vernier without requiring ROCm
 * at build time -- detection is purely runtime (rocprof on PATH). The
 * binary's own code path stays vendor-agnostic; rocprof attaches via HSA /
 * roctracer at runtime, the way nsys attaches via CUPTI for NVIDIA.
 *
 * Modes (selectable via --profile-args):
 *   default                kernel time + API trace
 *   "stats"                kernel statistics report (--stats)
 *   "hsa-trace"            HSA trace + kernel time (--hsa-trace)
 *   "hip-trace"            HIP trace + kernel time (--hip-trace)
 *
 * Invocation (wrap externally, same pattern as nsight / compute-sanitizer):
 *
 *   rocprof --stats -o run.csv \
 *       ./MyTest --profile rocprof --cycles 10 --gtest_filter='Gpu.Kernel'
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- RocprofProfiler ----------------------------- */

class RocprofProfiler final : public Profiler {
public:
  RocprofProfiler(const PerfConfig& cfg, std::string testName);
  ~RocprofProfiler() override = default;

  std::string toolName() const noexcept override { return "rocprof"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::string mode_; // "default" | "stats" | "hsa-trace" | "hip-trace"
  bool runningUnderRocprof_{false};
};

/* --------------------------------- API --------------------------------- */

/** @brief Factory: returns a backend instance, or nullptr if rocprof is unavailable. */
std::unique_ptr<Profiler> makeRocprofProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERROCPROF_HPP
