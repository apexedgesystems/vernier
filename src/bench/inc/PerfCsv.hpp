#ifndef VERNIER_PERFCSV_HPP
#define VERNIER_PERFCSV_HPP
/**
 * @file PerfCsv.hpp
 * @brief CSV helpers for writing benchmark results with support for CPU, GPU, multi-GPU, and
 * Unified Memory metrics.
 */

#include <fstream>
#include <optional>
#include <string>

#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfStats.hpp"

namespace vernier {
namespace bench {

/* --------------------------------- API --------------------------------- */

/**
 * @brief Write a standard header for per-call wall-time results.
 * @param csv            Output stream (opened in text mode).
 * @param includeProfile When true, append the columns: `profileTool,profileDir`.
 * @param includeMetadata When true, append metadata columns: `timestamp,gitHash,hostname,platform`.
 * @param includeGpu     When true, append GPU-specific columns (20 columns: device info, multi-GPU
 * metrics, and Unified Memory stats).
 * @note NOT RT-safe (file I/O).
 */
inline void writeCsvHeader(std::ofstream& csv, bool includeProfile = false,
                           bool includeMetadata = false, bool includeGpu = false) {
  csv << "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,"
         "wallMedian,wallP10,wallP90,wallMin,wallMax,wallMean,wallStddev,wallCV,callsPerSecond,"
         "stable,cvThreshold";

  if (includeProfile) {
    csv << ",profileTool,profileDir";
  }
  if (includeMetadata) {
    csv << ",timestamp,gitHash,hostname,platform";
  }
  if (includeGpu) {
    csv << ",gpuModel,computeCapability,kernelTimeUs,transferTimeUs,"
           "h2dBytes,d2hBytes,speedupVsCpu,memBandwidthGBs,occupancy,"
           "smClockMHz,throttling";
    csv << ",deviceId,deviceCount,multiGpuEfficiency,p2pBandwidthGBs";
    csv << ",umPageFaults,umH2DMigrations,umD2HMigrations,umMigrationTimeUs,umThrashing";
  }
  csv << "\n";
}

/**
 * @brief Write a single result row from a PerfRow struct.
 * @note NOT RT-safe (file I/O, heap allocation).
 */
inline void writeCsvRow(std::ofstream& csv, const PerfRow& row) {
  csv << row.testName << "," << row.cycles << "," << row.repeats << "," << row.warmup << ","
      << row.threads << "," << row.msgBytes << "," << (row.console ? "1" : "0") << ","
      << (row.nonBlocking ? "1" : "0") << "," << row.minLevel << "," << row.stats.median << ","
      << row.stats.p10 << "," << row.stats.p90 << "," << row.stats.min << "," << row.stats.max
      << "," << row.stats.mean << "," << row.stats.stddev << "," << row.stats.cv << ","
      << row.callsPerSecond << "," << (row.stable ? "1" : "0") << "," << row.cvThreshold;

  // Profile columns
  if (row.profileTool.has_value() || row.profileDir.has_value()) {
    csv << "," << (row.profileTool ? *row.profileTool : "");
    csv << "," << (row.profileDir ? *row.profileDir : "");
  }

  // Metadata columns
  if (!row.timestamp.empty() || !row.gitHash.empty() || !row.hostname.empty() ||
      !row.platform.empty()) {
    csv << "," << row.timestamp << "," << row.gitHash << "," << row.hostname << "," << row.platform;
  }

  // GPU columns (base + multi-GPU + Unified Memory)
  if (row.gpuModel.has_value() || row.computeCapability.has_value() ||
      row.kernelTimeUs.has_value() || row.transferTimeUs.has_value() || row.h2dBytes.has_value() ||
      row.d2hBytes.has_value() || row.speedupVsCpu.has_value() || row.memBandwidthGBs.has_value() ||
      row.occupancy.has_value() || row.smClockMHz.has_value() || row.throttling.has_value() ||
      row.deviceId.has_value() || row.deviceCount.has_value() ||
      row.multiGpuEfficiency.has_value() || row.p2pBandwidthGBs.has_value() ||
      row.umPageFaults.has_value() || row.umH2DMigrations.has_value() ||
      row.umD2HMigrations.has_value() || row.umMigrationTimeUs.has_value() ||
      row.umThrashing.has_value()) {

    csv << "," << (row.gpuModel ? *row.gpuModel : "");
    csv << "," << (row.computeCapability ? *row.computeCapability : "");
    csv << "," << (row.kernelTimeUs ? std::to_string(*row.kernelTimeUs) : "");
    csv << "," << (row.transferTimeUs ? std::to_string(*row.transferTimeUs) : "");
    csv << "," << (row.h2dBytes ? std::to_string(*row.h2dBytes) : "");
    csv << "," << (row.d2hBytes ? std::to_string(*row.d2hBytes) : "");
    csv << "," << (row.speedupVsCpu ? std::to_string(*row.speedupVsCpu) : "");
    csv << "," << (row.memBandwidthGBs ? std::to_string(*row.memBandwidthGBs) : "");
    csv << "," << (row.occupancy ? std::to_string(*row.occupancy) : "");
    csv << "," << (row.smClockMHz ? std::to_string(*row.smClockMHz) : "");
    csv << "," << (row.throttling ? (*row.throttling ? "1" : "0") : "");

    csv << "," << (row.deviceId ? std::to_string(*row.deviceId) : "");
    csv << "," << (row.deviceCount ? std::to_string(*row.deviceCount) : "");
    csv << "," << (row.multiGpuEfficiency ? std::to_string(*row.multiGpuEfficiency) : "");
    csv << "," << (row.p2pBandwidthGBs ? std::to_string(*row.p2pBandwidthGBs) : "");

    csv << "," << (row.umPageFaults ? std::to_string(*row.umPageFaults) : "");
    csv << "," << (row.umH2DMigrations ? std::to_string(*row.umH2DMigrations) : "");
    csv << "," << (row.umD2HMigrations ? std::to_string(*row.umD2HMigrations) : "");
    csv << "," << (row.umMigrationTimeUs ? std::to_string(*row.umMigrationTimeUs) : "");
    csv << "," << (row.umThrashing ? (*row.umThrashing ? "1" : "0") : "");
  }

  csv << "\n";
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFCSV_HPP
