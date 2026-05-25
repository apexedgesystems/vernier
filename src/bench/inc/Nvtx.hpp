#ifndef VERNIER_NVTX_HPP
#define VERNIER_NVTX_HPP
/**
 * @file Nvtx.hpp
 * @brief NVIDIA Tools Extension (NVTX) annotations for the benchmarking harness.
 *
 * NVTX is *instrumentation*, not a profiler -- it emits labeled ranges and
 * markers that any Nsight tool (Systems, Compute, also VTune) picks up
 * automatically. Annotating a benchmark with NVTX turns an unlabeled wall of
 * GPU activity in `nsys` into a timeline with named regions per stage.
 *
 * Usage:
 * @code
 *   {
 *     BENCH_NVTX_SCOPE("forward_pass");
 *     runKernel();
 *   } // pop on scope exit
 *
 *   BENCH_NVTX_MARK("epoch_start"); // instantaneous marker
 * @endcode
 *
 * Availability:
 *  - NVTX3 ships header-only with the CUDA toolkit (no separate library).
 *  - If the NVTX headers are present at compile time, the macros emit real
 *    ranges. Otherwise they compile to a no-op so the same code builds on
 *    CPU-only targets without conditional includes at the call site.
 *
 * @note RT-safe in the sense that NVTX push/pop is a userspace ringbuffer
 *       write; on profile-disabled runs it is a near-zero-cost branch.
 */

// COMPAT_NVTX_AVAILABLE is set by vernier_nvtx_enable() (see
// cmake/vernier/Cuda.cmake) when the CUDA toolkit's CUDA::nvtx3 imported
// target is available at build time. The __has_include fallback lets
// downstream consumers that bypass the vernier CMake helpers still get
// real ranges when the headers happen to be on their include path.
#if !defined(COMPAT_NVTX_AVAILABLE)
#if __has_include(<nvtx3/nvToolsExt.h>)
#define COMPAT_NVTX_AVAILABLE 1
#else
#define COMPAT_NVTX_AVAILABLE 0
#endif
#endif

#if COMPAT_NVTX_AVAILABLE
#include <nvtx3/nvToolsExt.h>
#endif

namespace vernier {
namespace bench {

/* ----------------------------- NvtxScope ----------------------------- */

/**
 * @brief RAII NVTX range. Pushes on construction, pops on destruction.
 *
 * Move-only would be tempting but unnecessary: NVTX ranges are LIFO-balanced
 * per thread, so each instance must be tied to a fixed scope.
 */
class NvtxScope {
public:
  explicit NvtxScope(const char* name) noexcept {
#if COMPAT_NVTX_AVAILABLE
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~NvtxScope() noexcept {
#if COMPAT_NVTX_AVAILABLE
    nvtxRangePop();
#endif
  }

  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;
  NvtxScope(NvtxScope&&) = delete;
  NvtxScope& operator=(NvtxScope&&) = delete;
};

/* ----------------------------- API ----------------------------- */

/** @brief Emit an instantaneous NVTX marker. No-op when NVTX is unavailable. */
inline void nvtxMark(const char* name) noexcept {
#if COMPAT_NVTX_AVAILABLE
  nvtxMarkA(name);
#else
  (void)name;
#endif
}

} // namespace bench
} // namespace vernier

/* ----------------------------- Macros ----------------------------- */

/// @brief Token-pasting helper (two levels for proper expansion of __LINE__).
#define VERNIER_NVTX_CAT_INNER(a, b) a##b
#define VERNIER_NVTX_CAT(a, b) VERNIER_NVTX_CAT_INNER(a, b)

/**
 * @brief Push an NVTX range named @p name for the enclosing scope.
 *
 * Compiles to a no-op when NVTX headers are not available; safe to leave in
 * production code.
 */
#define BENCH_NVTX_SCOPE(name)                                                                         \
  ::vernier::bench::NvtxScope VERNIER_NVTX_CAT(_bench_nvtx_scope_, __LINE__)(name)

/** @brief Emit an instantaneous NVTX marker named @p name. */
#define BENCH_NVTX_MARK(name) ::vernier::bench::nvtxMark(name)

#endif // VERNIER_NVTX_HPP
