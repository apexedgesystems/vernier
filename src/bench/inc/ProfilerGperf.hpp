#ifndef VERNIER_PROFILERGPERF_HPP
#define VERNIER_PROFILERGPERF_HPP
/**
 * @file ProfilerGperf.hpp
 * @brief gperftools backend for the benchmarking profiler facade.
 *
 * Modes:
 *  - CPU profiling (default): generates "<artifactDir>/cpu.prof"
 *  - Heap profiling (opt-in): parses "heap" in profileArgs -> starts HeapProfiler
 *  - Both: parse "both" or include both "cpu" and "heap" keywords in profileArgs
 *
 * Notes:
 *  - Requires gperftools headers/libraries to be available at build/link time.
 *  - If unavailable, makeGperfProfiler(...) returns nullptr and the factory
 *    in Profiler.hpp will produce a named no-op.
 */

#include <filesystem>
#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp" // base

// Detect availability at compile time
#if __has_include(<gperftools/profiler.h>)
#define UB_HAS_GPERF_CPU 1
#else
#define UB_HAS_GPERF_CPU 0
#endif

#if __has_include(<gperftools/heap-profiler.h>)
#define UB_HAS_GPERF_HEAP 1
#else
#define UB_HAS_GPERF_HEAP 0
#endif

namespace vernier {
namespace bench {

/* ----------------------------- GperfProfiler ----------------------------- */

class GperfProfiler final : public Profiler {
public:
  GperfProfiler(const PerfConfig& cfg, std::string testName);
  ~GperfProfiler() override = default;

  std::string toolName() const noexcept override { return "gperf"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;

  void runPprofAnalysis() const;

  bool wantCpu_{false};
  bool wantHeap_{false};

#if UB_HAS_GPERF_CPU
  std::string cpuPath_;
#endif
#if UB_HAS_GPERF_HEAP
  std::string heapPrefix_;
#endif
};

/* --------------------------------- API --------------------------------- */

/** @brief Factory function for gperftools profiler. */
std::unique_ptr<Profiler> makeGperfProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERGPERF_HPP