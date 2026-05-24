#ifndef VERNIER_PERFGPUMODE_HPP
#define VERNIER_PERFGPUMODE_HPP
/**
 * @file PerfGpuMode.hpp
 * @brief Tiny CUDA-free shim exposing "is this a GPU benchmark run?" to
 *        components that need to know but can't pull PerfGpuHarness.hpp.
 *
 * PERF_GPU_MAIN sets the flag during its setup phase. The CSV listener
 * reads it to decide whether to emit the GPU section of the schema; the
 * runner can read it to choose backend defaults; future code can follow
 * the same pattern without dragging CUDA into CPU-only translation units.
 */

namespace vernier {
namespace bench {
namespace detail {

inline bool& gpuModeFlag() {
  static bool active = false;
  return active;
}

/** Set by PERF_GPU_MAIN. Idempotent. */
inline void markGpuMode() { gpuModeFlag() = true; }

/** Query for downstream consumers that don't depend on CUDA. */
inline bool isGpuModeActive() { return gpuModeFlag(); }

} // namespace detail
} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUMODE_HPP
