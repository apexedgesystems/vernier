#ifndef VERNIER_CUPTICOLLECTOR_HPP
#define VERNIER_CUPTICOLLECTOR_HPP
/**
 * @file CuptiCollector.hpp
 * @brief In-process kernel records via the CUPTI Activity API.
 *
 * The GPU harness times kernels with CUDA events. This collector adds what
 * CUPTI's kernel activity records say about the launches in one measured
 * window: how many there were, the registers per thread each launch was
 * allocated, its static and dynamic shared memory, and the kernel's name. It
 * reads no time, launch shape or occupancy from the records.
 *
 * Scope: only what the Activity API's kernel records carry
 * (`CUpti_ActivityKernel*`). Counter metrics (achieved occupancy, warp
 * efficiency, cache hit rates) need kernel replay, which Nsight Compute does
 * (`--profile ncu`); the harness's measured launch sequence runs each launch
 * once.
 *
 * Build-time gate: COMPAT_CUPTI_AVAILABLE, which src/bench/CMakeLists.txt sets
 * to 1 when it links libcupti (VERNIER_USE_CUPTI on and the toolkit's CUPTI
 * found) and to 0 otherwise. Without CUPTI the class is still instantiable and
 * takes the same stand-down decision, but it never collects: isAvailable() is
 * false and stats() stays empty.
 *
 * Every CUPTI call is checked. A collector that cannot collect says why
 * (unavailableReason()); a window whose records are not known to be complete
 * says what went wrong (windowProblem()) and reports no launch, so no count
 * or median is published from part of a window.
 *
 * Threading: all calls are serialized on the harness thread. The CUPTI
 * activity buffers fill on whatever thread CUDA dispatches; we only read
 * them inside stop(), which runs on the harness thread after a measurement
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
 * Counts and medians, not per-launch records: the harness times the launches
 * with CUDA events; CUPTI adds their count and resource use over the same
 * window.
 */
struct CuptiKernelStats {
  std::size_t kernelLaunches{0};     ///< Number of kernels observed in this window
  std::uint16_t registersMedian{0};  ///< Median registers/thread across launches
  std::uint16_t registersMax{0};     ///< Worst-case registers/thread
  std::uint32_t staticSmemBytes{0};  ///< Median static __shared__ allocation
  std::uint32_t dynamicSmemBytes{0}; ///< Median dynamic shared memory at launch
  std::string firstKernelName;       ///< Name of the first observed kernel as reported by CUPTI
};

/* ----------------------------- CuptiCollector ----------------------------- */

class CuptiCollector {
public:
  /**
   * @param forceDisabled Skip CUPTI registration entirely. Without it the
   * collector still stands down, before registering, whenever
   * profiler_env::cuptiMustYield() says so (the explicit override, or an
   * nsys/ncu session): with the collector registered an nsys session records
   * no kernels, so the decision gates construction, not start(). The GPU
   * harness passes that same decision here.
   * @throws std::invalid_argument when @p forceDisabled is false and
   *         VERNIER_DISABLE_CUPTI is not a boolean (a configuration error),
   *         before any CUPTI call.
   */
  explicit CuptiCollector(bool forceDisabled = false);
  ~CuptiCollector();

  CuptiCollector(const CuptiCollector&) = delete;
  CuptiCollector& operator=(const CuptiCollector&) = delete;

  /**
   * @return true while the collector can collect: this build links CUPTI, the
   *         decision did not stand it down, CUPTI accepted its callbacks, and
   *         no start() or stop() has since failed to switch kernel records on
   *         or off.
   */
  [[nodiscard]] bool isAvailable() const noexcept { return available_; }

  /**
   * @return Why isAvailable() is false, as a phrase a message can quote (this
   *         build has no CUPTI; the collector stood down; CUPTI refused its
   *         callbacks, or to record kernel activity, or to stop recording it,
   *         with CUPTI's name for the result). Empty while it is available.
   */
  [[nodiscard]] const std::string& unavailableReason() const noexcept;

  /**
   * Start a window: the last window's stats() and windowProblem() are cleared,
   * then kernel records are switched on. A no-op otherwise when isAvailable()
   * is false; a refusal from CUPTI makes it false. Idempotent.
   */
  void start();

  /** Stop collection, flush activity buffers, aggregate into stats(). */
  void stop();

  /** Discard accumulated records and the last window's problem without disabling collection. */
  void reset();

  /** @return aggregated metrics from the last start/stop window. */
  [[nodiscard]] const CuptiKernelStats& stats() const noexcept { return stats_; }

  /**
   * @return What kept the last start/stop window's records from being
   *         complete, as a phrase a message can quote (the flush failed; CUPTI
   *         dropped records or could not count them; CUPTI could not read all
   *         of its records, with its error's name; CUPTI recorded no kernel
   *         launch). stats() is empty for such a window; a problem does not
   *         end collection (isAvailable() lists what does). Empty when the
   *         window's records are complete, or when the collector did not
   *         collect (see unavailableReason()).
   */
  [[nodiscard]] const std::string& windowProblem() const noexcept;

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
