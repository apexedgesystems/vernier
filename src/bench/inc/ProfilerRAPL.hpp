#ifndef VERNIER_PROFILERRAPL_HPP
#define VERNIER_PROFILERRAPL_HPP
/**
 * @file ProfilerRAPL.hpp
 * @brief Intel RAPL (Running Average Power Limit) energy backend.
 *
 * Reads package 0's energy-status counter (MSR 0x611) through the msr
 * driver's /dev/cpu/0/msr before and after each measured window, and writes
 * the window's energy, average package power and energy per call to
 * <Test>.rapl/energy.txt. The counter covers the whole package (its cores,
 * caches, integrated graphics and the rest of it), not the case's thread
 * alone, and nothing off the package: no DRAM counter is read.
 *
 * One decision for the doctor rows and the run: rapl takes no mode words and
 * has no analysis; the CPU must be Intel's (vendor_id in /proc/cpuinfo); and
 * this process must open /dev/cpu/0/msr and read the power-unit register
 * (MSR 0x606) and the package counter. A refusal names its cause:
 * unsupported (not Linux, not Intel, a register the CPU does not implement),
 * missing (no msr device), denied (the device or the driver refused this
 * process) or unusable (any other failure), and fails the run (exit status 4
 * once the tests pass).
 *
 * Access is set up beforehand and never raised by Vernier: run as root, or
 * give the process CAP_SYS_RAWIO, which the msr driver checks, and read
 * access to /dev/cpu/0/msr, which the capability does not grant.
 *
 * Usage: run a benchmark with --profile rapl.
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"

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

/* ----------------------------- RaplPlan ----------------------------- */

/** @brief What the RAPL check verified, for the capture to use. */
struct RaplPlan final : ReadinessPlan {
  std::string device;            ///< The msr device the check read.
  double energyUnitJoules = 0.0; ///< One count of the energy counter, from MSR 0x606.
  std::string cpuModel;          ///< /proc/cpuinfo's model name.
  int cpuFamily = 0;             ///< /proc/cpuinfo's cpu family.
  int cpuModelNumber = 0;        ///< /proc/cpuinfo's model.
};

/** @brief The files the RAPL check reads; a test points them at files it makes. */
struct RaplSources {
  std::string cpuinfo = "/proc/cpuinfo"; ///< Where the CPU's vendor and model are read.
  std::string device = "/dev/cpu/0/msr"; ///< The msr device of CPU 0, in package 0.
};

/* ----------------------------- RAPLProfiler ----------------------------- */

class RAPLProfiler final : public Profiler {
public:
  /**
   * @brief Decide the request now and report it as the registry does: one
   * that cannot run fails the run and creates no folder.
   */
  RAPLProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  RAPLProfiler(const PerfConfig& cfg, std::string testName, std::shared_ptr<const RaplPlan> plan);

  ~RAPLProfiler() override = default;

  std::string toolName() const noexcept override { return "rapl"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

  /** @brief True when a default rapl request is ready here: the check's decision. */
  [[nodiscard]] static bool isAvailable() noexcept;

private:
  [[nodiscard]] std::optional<uint64_t> readEnergyCounter(RAPLDomain domain) const;
  void printSummary(const RAPLMeasurement& measurement) const;

  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const RaplPlan> plan_;

  RAPLDomain domain_ = RAPLDomain::Package;
  uint64_t energyStart_ = 0;
  uint64_t energyEnd_ = 0;
  double startTimeUs_ = 0.0;
  double endTimeUs_ = 0.0;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief The refusal of a failed open or read of the msr device: @p step is
 * what failed ("open /dev/cpu/0/msr", "read MSR 0x611 from /dev/cpu/0/msr")
 * and @p err its errno, which the message quotes.
 *
 * DENIED for EACCES and EPERM, with the access RAPL needs as the remedy;
 * MISSING for ENOENT and ENODEV (the msr driver is not loaded, or a container
 * was not given the device); UNSUPPORTED for EIO (the CPU does not implement
 * the register, or has no model-specific registers); UNUSABLE for any other
 * value.
 */
ReadinessResult raplAccessRefusal(const std::string& step, int err);

/**
 * @brief The RAPL backend's readiness decision for @p request in @p ctx; the
 * doctor rows, the doctor's selected row and the run all ask this one.
 *
 * In order: a mode word is a configuration error (rapl takes none); a system
 * other than Linux, or a CPU whose vendor_id is not GenuineIntel, is
 * unsupported; then /dev/cpu/0/msr is opened and MSR 0x606 and MSR 0x611 are
 * read, each failure refused by raplAccessRefusal() (a read of fewer than 8
 * bytes is unusable). With both registers read the request is ready and the
 * plan is a RaplPlan. --profile-analyze is an analysis-stage error on a
 * request that can collect: the capture still runs, and the run fails at its
 * end.
 */
ReadinessResult checkRaplRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/** @brief checkRaplRequest() reading @p sources in place of /proc/cpuinfo and /dev/cpu/0/msr. */
ReadinessResult checkRaplRequestFrom(const ReadinessRequest& request, const RaplSources& sources);

/**
 * @brief Factory: decides the request and reports it as the registry does;
 * nullptr, with no folder created, when it cannot run.
 */
std::unique_ptr<Profiler> makeRAPLProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERRAPL_HPP
