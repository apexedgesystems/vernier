#ifndef VERNIER_PROFILERCOMPUTESANITIZER_HPP
#define VERNIER_PROFILERCOMPUTESANITIZER_HPP
/**
 * @file ProfilerComputeSanitizer.hpp
 * @brief NVIDIA Compute Sanitizer backend -- GPU memory + concurrency checker.
 *
 * Compute Sanitizer (formerly cuda-memcheck) is the GPU analog of
 * valgrind's memcheck. It catches errors that crash kernels intermittently
 * or silently corrupt results:
 *  - Out-of-bounds device memory access
 *  - Misaligned loads / stores
 *  - Shared-memory races (--tool=racecheck)
 *  - Missed __syncthreads barriers (--tool=synccheck)
 *  - Uninitialized device memory reads (--tool=initcheck)
 *
 * Ships with the CUDA toolkit; no special permissions required.
 * Overhead is ~5-10x kernel time, so use low --cycles.
 *
 * Tools (selectable via --profile-args):
 *   memcheck   (default)  -- device access errors, leaks
 *   racecheck             -- shared-memory data races
 *   synccheck             -- __syncthreads correctness
 *   initcheck             -- reads from uninitialized device memory
 *
 * Invocation (compute-sanitizer wraps the binary externally; same pattern
 * as valgrind/callgrind):
 *
 *   compute-sanitizer --tool=memcheck \
 *       ./MyTest --profile compute-sanitizer --cycles 5 --gtest_filter='Gpu.Kernel'
 *
 * The backend itself sets up the artifact directory, prints the invocation
 * hint when the binary is run unwrapped, and stamps the chosen tool into
 * the CSV profile metadata for downstream correlation.
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------- ComputeSanitizerProfiler ----------------------- */

class ComputeSanitizerProfiler final : public Profiler {
public:
  ComputeSanitizerProfiler(const PerfConfig& cfg, std::string testName);
  ~ComputeSanitizerProfiler() override = default;

  std::string toolName() const noexcept override { return "compute-sanitizer"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::string sanitizerTool_; // memcheck | racecheck | synccheck | initcheck
  bool runningUnderSanitizer_{false};
};

/* --------------------------------- API --------------------------------- */

/** @brief Factory: returns a backend instance, or nullptr if compute-sanitizer is unavailable. */
std::unique_ptr<Profiler> makeComputeSanitizerProfiler(const PerfConfig& cfg,
                                                       const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERCOMPUTESANITIZER_HPP
