#ifndef VERNIER_PERFVALIDATION_HPP
#define VERNIER_PERFVALIDATION_HPP
/**
 * @file PerfValidation.hpp
 * @brief Validation helpers for benchmark results (CV% thresholds, bounds checking).
 *
 * Features adaptive CV% thresholds based on payload size and quick mode support
 * with relaxed thresholds and helpful hints.
 */

#include <cstddef>
#include "src/bench/inc/PerfConfig.hpp"

namespace vernier {
namespace bench {

/* --------------------------------- API --------------------------------- */

/**
 * @brief Get recommended CV% threshold based on payload size.
 *
 * Small payloads (<64B) have higher fixed overhead, leading to naturally
 * higher CV% due to timing measurement noise and call overhead dominating.
 * This function returns empirically-derived thresholds appropriate for
 * different payload scales:
 *
 * - <64B:    20% (tiny payloads, dominated by fixed call overhead)
 * - <256B:   10% (small payloads, mixed overhead + actual work)
 * - <1KB:     5% (medium payloads, work begins to dominate)
 * - >=1KB:     3% (large payloads, very stable measurements)
 *
 * These thresholds prevent false test failures while still catching real
 * performance instability.
 *
 * @param payloadBytes Payload size in bytes (typically from cfg.msgBytes)
 * @return Recommended CV% threshold as decimal [0.03-0.20]
 * @note RT-safe (pure calculation, no allocations).
 *
 * Example usage:
 * @code
 * const double threshold = recommendedCVThreshold(64); // Returns 0.10 (10%)
 * EXPECT_LT(result.stats.cv, threshold) << "High jitter";
 * @endcode
 */
inline double recommendedCVThreshold(size_t payloadBytes) {
  if (payloadBytes < 64)
    return 0.20; // 20% for tiny payloads
  if (payloadBytes < 256)
    return 0.10; // 10% for small payloads
  if (payloadBytes < 1024)
    return 0.05; // 5% for medium payloads
  return 0.03;   // 3% for large payloads
}

/**
 * @brief Get recommended CV% threshold from PerfConfig (convenience).
 *
 * Extracts msgBytes from config and determines appropriate threshold.
 * Automatically adjusts for quick mode:
 * - Normal mode: Standard thresholds based on payload size
 * - Quick mode: Relaxed by 50% (fewer samples = higher variance expected)
 * - Capped at 30% to catch real pathological issues
 *
 * @param cfg Performance configuration with msgBytes and quickMode settings
 * @return Recommended CV% threshold, adjusted for quick mode if applicable
 * @note RT-safe (pure calculation, no allocations).
 *
 * Example usage:
 * @code
 * EXPECT_LT(result.stats.cv, recommendedCVThreshold(perf.config()));
 * @endcode
 */
inline double recommendedCVThreshold(const PerfConfig& cfg) {
  double threshold = recommendedCVThreshold(static_cast<size_t>(cfg.msgBytes));

  // Quick mode relaxation
  // Quick mode uses fewer cycles/repeats -> less statistical confidence
  // Relax threshold by 50% to avoid spurious failures during dev iteration
  if (cfg.quickMode) {
    threshold *= 1.5;
    // Cap at 30% (beyond this, something is likely wrong even in quick mode)
    if (threshold > 0.30) {
      threshold = 0.30;
    }
  }

  return threshold;
}

/**
 * @brief Check if CV% is acceptable for the given configuration.
 *
 * Returns true if CV% is within acceptable bounds. Used by validation
 * macros to provide helpful context when tests fail.
 *
 * @param cv Coefficient of variation (e.g., 0.15 = 15%)
 * @param cfg Configuration to determine thresholds
 * @return true if CV% is acceptable, false otherwise
 * @note RT-safe (pure calculation, no allocations).
 */
inline bool isStableCV(double cv, const PerfConfig& cfg) {
  return cv < recommendedCVThreshold(cfg);
}

} // namespace bench
} // namespace vernier

/* ----------------------------- Test Macros ----------------------------- */

/**
 * @brief GoogleTest assertion macros for CV% validation with adaptive thresholds.
 *
 * Automatically selects appropriate CV% threshold based on payload size
 * and quick mode settings. Provides helpful error message showing actual
 * vs expected threshold.
 *
 * Features adaptive thresholds based on payload size and quick mode awareness
 * with relaxed thresholds.
 *
 * Use EXPECT_STABLE_CV_CPU for CPU tests, EXPECT_STABLE_CV_GPU for GPU tests.
 *
 * @param result PerfResult (CPU) or PerfGpuResult (GPU)
 * @param cfg PerfConfig with msgBytes and quickMode
 *
 * CPU example:
 * @code
 * auto result = perf.throughputLoop([&]{ work(); });
 * EXPECT_STABLE_CV_CPU(result, perf.config());
 * @endcode
 *
 * GPU example:
 * @code
 * auto gpuResult = perf.cudaKernel(...).measure();
 * EXPECT_STABLE_CV_GPU(gpuResult, perf.cpuConfig());
 * @endcode
 */

// CPU version: result.stats.cv
#define EXPECT_STABLE_CV_CPU(result, cfg)                                                          \
  EXPECT_LT((result).stats.cv, vernier::bench::recommendedCVThreshold(cfg))                        \
      << "High jitter: CV% " << (result).stats.cv * 100.0 << "% exceeds "                          \
      << vernier::bench::recommendedCVThreshold(cfg) * 100.0 << "% threshold for "                 \
      << (cfg).msgBytes << "B payload"                                                             \
      << ((cfg).quickMode ? " (quick mode: relaxed threshold)" : "")

// GPU version: result.stats.cpuStats.cv
#define EXPECT_STABLE_CV_GPU(result, cfg)                                                          \
  EXPECT_LT((result).stats.cpuStats.cv, vernier::bench::recommendedCVThreshold(cfg))               \
      << "High jitter: CV% " << (result).stats.cpuStats.cv * 100.0 << "% exceeds "                 \
      << vernier::bench::recommendedCVThreshold(cfg) * 100.0 << "% threshold for "                 \
      << (cfg).msgBytes << "B payload"                                                             \
      << ((cfg).quickMode ? " (quick mode: relaxed threshold)" : "")

// Generic fallback (tries CPU first)
#define EXPECT_STABLE_CV(result, cfg) EXPECT_STABLE_CV_CPU(result, cfg)

#endif // VERNIER_PERFVALIDATION_HPP