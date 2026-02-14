#ifndef VERNIER_PROFILERRAPL_HPP
#define VERNIER_PROFILERRAPL_HPP
/**
 * @file ProfilerRAPL.hpp
 * @brief Intel RAPL (Running Average Power Limit) profiler backend.
 *
 * Behavior:
 *  - Reads energy counters from MSR registers on Intel CPUs
 *  - Tracks package energy (CPU + iGPU + DRAM on some models)
 *  - Calculates energy consumed (Joules) and average power (Watts)
 *  - Writes detailed energy report to artifact directory
 *
 * Requirements:
 *  - Intel CPU (Haswell or newer, ~2013+)
 *  - Linux with MSR module: `sudo modprobe msr`
 *  - Root access or CAP_SYS_RAWIO: `sudo setcap cap_sys_rawio=ep ./test`
 *
 * Usage:
 *  Run benchmark with: --profile rapl
 *
 * Notes:
 *  - Safe no-op when RAPL unavailable
 *  - Package-level measurement only (entire CPU socket)
 *  - Does not include PCIe devices or discrete GPUs
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Enums ----------------------------- */

/**
 * @brief RAPL energy domain types.
 */
enum class RAPLDomain {
  Package, ///< Entire CPU package (most common)
  Core,    ///< CPU cores only
  DRAM,    ///< Memory controller
  GPU      ///< Integrated GPU
};

/* ----------------------------- RAPLMeasurement ----------------------------- */

/**
 * @brief RAPL measurement result.
 */
struct RAPLMeasurement {
  double energyJoules = 0.0;
  double durationSeconds = 0.0;
  double avgPowerWatts = 0.0;
  double energyPerOpMillijoules = 0.0;
};

/* ----------------------------- RAPLProfiler ----------------------------- */

class RAPLProfiler final : public Profiler {
public:
  RAPLProfiler(const PerfConfig& cfg, std::string testName);
  ~RAPLProfiler() override = default;

  std::string toolName() const noexcept override { return "rapl"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

  [[nodiscard]] static bool isAvailable() noexcept;

private:
  [[nodiscard]] std::optional<uint64_t> readEnergyCounter(RAPLDomain domain) const;
  void detectCPUModel();
  void printSummary(const RAPLMeasurement& measurement) const;

  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;

  RAPLDomain domain_ = RAPLDomain::Package;
  double energyUnit_ = 0.0;
  uint64_t energyStart_ = 0;
  uint64_t energyEnd_ = 0;
  double startTimeUs_ = 0.0;
  double endTimeUs_ = 0.0;

  std::string cpuModel_;
  int cpuFamily_ = 0;
  int cpuModelNum_ = 0;
};

/* --------------------------------- API --------------------------------- */

/** @brief Factory function for RAPL profiler. */
std::unique_ptr<Profiler> makeRAPLProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERRAPL_HPP