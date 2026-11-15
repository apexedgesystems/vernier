#ifndef VERNIER_PERFGPUSTATS_HPP
#define VERNIER_PERFGPUSTATS_HPP
/**
 * @file PerfGpuStats.hpp
 * @brief GPU-specific performance metrics including multi-GPU scaling.
 */

#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/CuptiCollector.hpp"

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
 * @brief The harness's occupancy estimate for a launch configuration.
 *
 * Filled only for a kernel measured with withLaunchConfig(); all zero
 * otherwise. The estimate counts the SM's thread and block limits and the
 * shared memory the launch configuration declares, not registers or static
 * shared memory, and is not a measurement: Nsight Compute measures the
 * occupancy a kernel achieves.
 */
struct OccupancyMetrics {
  int blockSize{};        ///< Threads per block used
  int gridSize{};         ///< Blocks per grid used
  int activeWarpsPerSM{}; ///< Warps per SM the launch shape can keep resident (estimated)
  int maxWarpsPerSM{};    ///< Max possible warps per SM
  /// The estimate, activeWarpsPerSM over maxWarpsPerSM [0.0-1.0], published as
  /// the CSV's `occupancy` cell. Despite the name, not a measured achieved
  /// occupancy.
  double achievedOccupancy{};

  enum class LimitingFactor { Unknown, Registers, SharedMemory, Warps, BlockSize };
  LimitingFactor limitingFactor = LimitingFactor::Unknown;
};

/**
 * @brief Clock speed monitoring (throttling detection).
 *
 * NVML samples at the measured window's start and end. A field is 0 where NVML
 * reported nothing; the CSV cells built from these fields are then empty and
 * the run names the readings NVML did not report.
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

/* ----------------------------- Power & Thermal ----------------------------- */

/**
 * @brief GPU power draw and temperature samples around the measured window.
 *
 * Thermal / power drift is a common confounder in GPU benchmark comparison:
 * a kernel that looks faster might just be running on a colder card. Capturing
 * these alongside kernel time lets `bench compare` flag the difference without
 * a separate run of `bench gpu-monitor`.
 *
 * A field is zero where NVML is unavailable or did not report the reading;
 * the CSV cells built from these fields are then empty and the run says why.
 */
struct PowerThermalProfile {
  /// NVML's power reading at the measured window's start (W): not an
  /// instant, an average over the second before it (Ampere and newer).
  double powerDrawWStart{};
  /// NVML's power reading at the window's end (W), an average over the last
  /// second: the window's power only for a window of at least 1.5 s, which is
  /// when the CSV's powerDrawW carries it.
  double powerDrawWEnd{};
  double powerLimitW{};    ///< Configured power limit (W)
  int temperatureCStart{}; ///< GPU core temperature at start (Celsius)
  int temperatureCEnd{};   ///< GPU core temperature at end (Celsius)

  /** The mean of the two readings: not the window's power, which no mean of them is. */
  [[nodiscard]] double avgPowerDrawW() const { return (powerDrawWStart + powerDrawWEnd) * 0.5; }

  /** Temperature delta over the measure window (positive = warmed up). */
  [[nodiscard]] int temperatureDeltaC() const { return temperatureCEnd - temperatureCStart; }
};

/* ----------------------------- Multi-GPU ----------------------------- */

/**
 * @brief Multi-GPU scaling metrics.
 */
struct MultiGpuMetrics {
  int deviceCount = 1;
  /// Not measured: always 0. A scaling efficiency needs one device's time on
  /// the whole job, which a multi-GPU measurement does not take; the field
  /// keeps its place in the layout, so a check against it fails rather than
  /// pass on another number.
  double scalingEfficiency = 0.0;
  double loadImbalance = 0.0; ///< Max/Min device time ratio (ideal=1.0)
  bool p2pEnabled = false;
  double p2pBandwidthGBs = 0.0;

  /// The mean of min(1, scalingEfficiency) and the balance score: with the
  /// efficiency not measured (0), at most 0.5 and only the balance moves it.
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
 * @brief Unified Memory counters over one measured window: what CUPTI's
 *        unified-memory counter records add up to
 *        (CuptiCollector::unifiedMemory()).
 *
 * Each record of the window adds to one field:
 *  - a GPU page-fault record (CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT)
 *    adds its value, the fault groups for one page, to pageFaults, and a CPU
 *    page-fault record (..._CPU_PAGE_FAULT_COUNT) adds one (its value is the
 *    faulting instruction's address);
 *  - a host-to-device transfer (..._BYTES_TRANSFER_HTOD) adds one to
 *    h2dMigrations and its bytes to h2dBytes, a device-to-host transfer
 *    (..._BYTES_TRANSFER_DTOH) the same to d2hMigrations and d2hBytes, and
 *    either adds its end minus start to migrationTimeUs; transfers that
 *    overlap each count their own time, so the sum can exceed the time spent
 *    migrating;
 *  - a thrashing record (..._THRASHING) adds one to thrashingEvents.
 * A window that migrated nothing has zero counts: observed, not missing.
 * isThrashing() and migrationOverheadPct() are arithmetic over these counts,
 * not CUPTI's report.
 */
struct UnifiedMemoryProfile {
  size_t pageFaults = 0;        ///< GPU page-fault groups plus CPU page faults
  size_t h2dMigrations = 0;     ///< Host->Device transfers
  size_t d2hMigrations = 0;     ///< Device->Host transfers
  double migrationTimeUs = 0.0; ///< The transfers' times, summed (us)
  size_t thrashingEvents = 0;   ///< Thrashing records
  size_t h2dBytes = 0;          ///< Bytes moved Host->Device
  size_t d2hBytes = 0;          ///< Bytes moved Device->Host

  /**
   * @brief Whether the migrations exceed @p threshold times the page faults:
   *        a ratio of these counts, not CUPTI's thrashing report
   *        (thrashingEvents).
   * @param threshold Ratio of total migrations to page faults (default: 2.0)
   * @return True above the ratio; false without page faults
   */
  [[nodiscard]] bool isThrashing(double threshold = 2.0) const {
    const size_t TOTAL_MIGRATIONS = h2dMigrations + d2hMigrations;
    return (pageFaults > 0) && (static_cast<double>(TOTAL_MIGRATIONS) / pageFaults > threshold);
  }

  /**
   * @brief Calculate migration overhead as percentage of kernel time.
   *
   * Divides the total migration time by the number of launches it spans to get
   * per-call overhead.
   *
   * @param kernelTimeUs Median kernel time per call (us)
   * @param cycles Number of launches migrationTimeUs spans (for one harness
   *        measurement, its cycles times its repeats)
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
  PowerThermalProfile powerThermal{};
  /// In-process kernel metrics of the untimed batch that follows a kernel
  /// measurement's timed repeats (PerfGpuCase::cudaKernel()); zeroed when
  /// CUPTI is unavailable.
  CuptiKernelStats cupti{};

  double kernelTimeMedianUs{};
  double transferTimeMedianUs{};
  double totalTimeMedianUs{};

  // Multi-GPU metrics
  std::optional<MultiGpuMetrics> multiGpu{};
  std::optional<P2PTransferProfile> p2pProfile{};

  /// Unified Memory counters over the measurement's window (see
  /// UnifiedMemoryProfile): set when PerfGpuConfig::captureUnifiedMemory asks
  /// for them and CUPTI records them completely; empty otherwise. The window
  /// spans a kernel measurement's declared copies and timed launches, or a
  /// multi-GPU measurement's device launches, not the warmup or the host code
  /// around them.
  std::optional<UnifiedMemoryProfile> unifiedMemory{};

  [[nodiscard]] double memoryBandwidthUtilization(double peakBandwidthGBs) const {
    const double ACHIEVED = transfers.bandwidthGBs();
    return (peakBandwidthGBs > 0.0) ? (100.0 * ACHIEVED / peakBandwidthGBs) : 0.0;
  }
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUSTATS_HPP