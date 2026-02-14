#ifndef VERNIER_PERFGPUSTATS_HPP
#define VERNIER_PERFGPUSTATS_HPP
/**
 * @file PerfGpuStats.hpp
 * @brief GPU-specific performance metrics including multi-GPU scaling and Unified Memory profiling.
 */

#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include "src/bench/inc/PerfStats.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Device Info ----------------------------- */

/**
 * @brief GPU device information captured at test start.
 */
struct GpuDeviceInfo {
  std::string name;           ///< Device name (e.g., "NVIDIA RTX 4090")
  int computeCapability[2]{}; ///< Compute capability (major, minor)
  size_t totalMemoryMB{};     ///< Total global memory (MB)
  int smCount{};              ///< Number of streaming multiprocessors
  int maxThreadsPerSM{};      ///< Max threads per SM
  int clockRateMHz{};         ///< GPU clock rate (MHz)
  int memoryClockRateMHz{};   ///< Memory clock rate (MHz)
  int memoryBusWidthBits{};   ///< Memory bus width (bits)
};

/* ----------------------------- Transfer Metrics ----------------------------- */

/**
 * @brief Memory transfer profile for a kernel execution.
 */
struct MemoryTransferProfile {
  size_t h2dBytes{};  ///< Host-to-device bytes transferred
  size_t d2hBytes{};  ///< Device-to-host bytes transferred
  double h2dTimeUs{}; ///< H2D transfer time (us)
  double d2hTimeUs{}; ///< D2H transfer time (us)

  [[nodiscard]] double transferOverheadPct(double kernelTimeUs) const {
    const double TOTAL_TRANSFER_TIME = h2dTimeUs + d2hTimeUs;
    const double TOTAL_TIME = TOTAL_TRANSFER_TIME + kernelTimeUs;
    return (TOTAL_TIME > 0.0) ? (100.0 * TOTAL_TRANSFER_TIME / TOTAL_TIME) : 0.0;
  }

  [[nodiscard]] double bandwidthGBs() const {
    const double TOTAL_BYTES = static_cast<double>(h2dBytes + d2hBytes);
    const double TOTAL_TIME_S = (h2dTimeUs + d2hTimeUs) / 1e6;
    if (TOTAL_TIME_S <= 0.0)
      return 0.0;
    return TOTAL_BYTES / TOTAL_TIME_S / 1e9;
  }

  [[nodiscard]] double h2dBandwidthGBs() const {
    if (h2dTimeUs <= 0.0)
      return 0.0;
    return static_cast<double>(h2dBytes) / (h2dTimeUs / 1e6) / 1e9;
  }

  [[nodiscard]] double d2hBandwidthGBs() const {
    if (d2hTimeUs <= 0.0)
      return 0.0;
    return static_cast<double>(d2hBytes) / (d2hTimeUs / 1e6) / 1e9;
  }
};

/* ----------------------------- Occupancy & Clocks ----------------------------- */

/**
 * @brief Kernel occupancy metrics.
 */
struct OccupancyMetrics {
  int blockSize{};            ///< Threads per block used
  int gridSize{};             ///< Blocks per grid used
  int activeWarpsPerSM{};     ///< Active warps per SM (from occupancy calc)
  int maxWarpsPerSM{};        ///< Max possible warps per SM
  double achievedOccupancy{}; ///< Fraction of max occupancy achieved [0.0-1.0]

  enum class LimitingFactor { Unknown, Registers, SharedMemory, Warps, BlockSize };
  LimitingFactor limitingFactor = LimitingFactor::Unknown;
};

/**
 * @brief Clock speed monitoring (throttling detection).
 */
struct ClockSpeedProfile {
  int smClockMHzStart{};
  int smClockMHzEnd{};
  int memClockMHzStart{};
  int memClockMHzEnd{};
  int boostClockMHz{};

  [[nodiscard]] bool isThrottling(double threshold = 0.10) const {
    if (boostClockMHz == 0 || smClockMHzStart == 0)
      return false;
    const double DROP_PCT = 1.0 - (static_cast<double>(smClockMHzEnd) / smClockMHzStart);
    return DROP_PCT > threshold;
  }
};

/* ----------------------------- Multi-GPU ----------------------------- */

/**
 * @brief Multi-GPU scaling metrics.
 */
struct MultiGpuMetrics {
  int deviceCount = 1;
  double scalingEfficiency = 0.0; ///< Speedup / N_gpus (ideal=1.0)
  double loadImbalance = 0.0;     ///< Max/Min device time ratio (ideal=1.0)
  bool p2pEnabled = false;
  double p2pBandwidthGBs = 0.0;

  [[nodiscard]] double scalingQuality() const {
    const double EFFICIENCY_SCORE = std::min(1.0, scalingEfficiency);
    const double BALANCE_SCORE = 1.0 / std::max(1.0, loadImbalance);
    return (EFFICIENCY_SCORE + BALANCE_SCORE) / 2.0;
  }
};

/**
 * @brief Peer-to-peer transfer measurement.
 */
struct P2PTransferProfile {
  int srcDevice = -1;
  int dstDevice = -1;
  size_t bytes = 0;
  double timeUs = 0.0;
  bool accessEnabled = false;

  [[nodiscard]] double bandwidthGBs() const {
    if (timeUs <= 0.0)
      return 0.0;
    return static_cast<double>(bytes) / (timeUs / 1e6) / 1e9;
  }
};

/* ----------------------------- Unified Memory ----------------------------- */

/**
 * @brief Unified Memory profiling metrics.
 *
 * Tracks page faults and migration patterns for managed memory.
 */
struct UnifiedMemoryProfile {
  size_t pageFaults = 0;        ///< Total page faults
  size_t h2dMigrations = 0;     ///< Host->Device page migrations
  size_t d2hMigrations = 0;     ///< Device->Host page migrations
  double migrationTimeUs = 0.0; ///< Estimated time spent migrating pages
  size_t thrashingEvents = 0;   ///< Rapid back-and-forth migrations (H<->D<->H)

  /**
   * @brief Detect thrashing (excessive back-and-forth migration).
   * @param threshold Ratio of total migrations to unique pages (default: 2.0)
   * @return True if thrashing detected
   */
  [[nodiscard]] bool isThrashing(double threshold = 2.0) const {
    const size_t TOTAL_MIGRATIONS = h2dMigrations + d2hMigrations;
    return (pageFaults > 0) && (static_cast<double>(TOTAL_MIGRATIONS) / pageFaults > threshold);
  }

  /**
   * @brief Calculate migration overhead as percentage of kernel time.
   *
   * Divides total migration time by the number of cycles to get per-call overhead.
   *
   * @param kernelTimeUs Median kernel time per call (us)
   * @param cycles Number of operations measured
   * @return Migration overhead percentage [0-100+]
   */
  [[nodiscard]] double migrationOverheadPct(double kernelTimeUs, int cycles) const {
    if (kernelTimeUs <= 0.0 || cycles <= 0)
      return 0.0;
    const double MIGRATION_PER_CALL = migrationTimeUs / cycles;
    return (MIGRATION_PER_CALL / kernelTimeUs) * 100.0;
  }
};

/* ----------------------------- GpuStats ----------------------------- */

/**
 * @brief Comprehensive GPU performance statistics.
 */
struct GpuStats {
  Stats cpuStats{};
  GpuDeviceInfo deviceInfo{};
  MemoryTransferProfile transfers{};
  OccupancyMetrics occupancy{};
  ClockSpeedProfile clocks{};

  double kernelTimeMedianUs{};
  double transferTimeMedianUs{};
  double totalTimeMedianUs{};

  // Multi-GPU metrics
  std::optional<MultiGpuMetrics> multiGpu{};
  std::optional<P2PTransferProfile> p2pProfile{};

  // Unified Memory profiling
  std::optional<UnifiedMemoryProfile> unifiedMemory{};

  [[nodiscard]] double memoryBandwidthUtilization(double peakBandwidthGBs) const {
    const double ACHIEVED = transfers.bandwidthGBs();
    return (peakBandwidthGBs > 0.0) ? (100.0 * ACHIEVED / peakBandwidthGBs) : 0.0;
  }
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUSTATS_HPP