#ifndef VERNIER_CUPTICOLLECTOR_HPP
#define VERNIER_CUPTICOLLECTOR_HPP
/**
 * @file CuptiCollector.hpp
 * @brief In-process kernel metrics via the CUPTI Activity API.
 *
 * The existing GPU harness captures wall time via CUDA events and clocks via
 * NVML. CUPTI fills the gap left by both: per-kernel register count, static
 * and dynamic shared memory, launch geometry, and a precise device-side
 * duration -- all without spawning ncu as an external process (which is
 * fragile inside container PID namespaces; see TROUBLESHOOTING.md).
 *
 * Scope: this collector exposes only the metrics CUPTI's Activity API
 * surfaces directly (`CUpti_ActivityKernel*` records). The CUPTI Profiler /
 * Perfworks API offers richer metrics (achieved occupancy, warp efficiency,
 * cache hit rates) but requires kernel replay, which is incompatible with
 * the bench harness's single-launch measurement model. Use the Nsight
 * Compute backend for replay-mode profiling.
 *
 * Build-time gate: COMPAT_CUPTI_AVAILABLE -- set by the CMake target when
 * libcupti and cupti.h are both found. When not available, the class is
 * still instantiable but every method is a no-op so callers do not need
 * conditional code paths.
 *
 * Threading: all calls are serialized on the harness thread. The CUPTI
 * activity buffers fill on whatever thread CUDA dispatches; we only read
 * them inside flush(), which runs on the harness thread after a measurement
 * window. No shared mutable state from the CUDA threads is exposed.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vernier {
namespace bench {

/* ----------------------------- CuptiKernelStats ----------------------------- */

/**
 * @brief Aggregated kernel-launch metrics over one measured window.
 *
 * Counts and medians, not per-launch records: the harness already times
 * each individual kernel via CUDA events; CUPTI adds the launch-geometry
 * + resource-usage view at the same granularity as the timing view.
 */
struct CuptiKernelStats {
  std::size_t kernelLaunches{0};     ///< Number of kernels observed in this window
  std::uint16_t registersMedian{0};  ///< Median registers/thread across launches
  std::uint16_t registersMax{0};     ///< Worst-case registers/thread
  std::uint32_t staticSmemBytes{0};  ///< Median static __shared__ allocation
  std::uint32_t dynamicSmemBytes{0}; ///< Median dynamic shared memory at launch
  std::string firstKernelName;       ///< Demangled name of the first observed kernel
};

/* ----------------------------- CuptiCollector ----------------------------- */

class CuptiCollector {
public:
  CuptiCollector();
  ~CuptiCollector();

  CuptiCollector(const CuptiCollector&) = delete;
  CuptiCollector& operator=(const CuptiCollector&) = delete;

  /** @return true when CUPTI was linked at build time AND init succeeded. */
  [[nodiscard]] bool isAvailable() const noexcept { return available_; }

  /** Start collection. Safe no-op when isAvailable() is false. Idempotent. */
  void start();

  /** Stop collection, flush activity buffers, aggregate into stats(). */
  void stop();

  /** Discard accumulated records without disabling collection. */
  void reset();

  /** @return aggregated metrics from the last start/stop window. */
  [[nodiscard]] const CuptiKernelStats& stats() const noexcept { return stats_; }

private:
  bool available_{false};
  bool running_{false};
  CuptiKernelStats stats_{};
  // The actual record buffer + CUPTI state lives in the .cu file behind a
  // pImpl so this header pulls no CUPTI symbols (and stays usable from CPU
  // TUs that never touch CUDA).
  struct Impl;
  Impl* impl_{nullptr};
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_CUPTICOLLECTOR_HPP
