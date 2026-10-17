/**
 * @file ProfilerRAPL.cpp
 * @brief The RAPL energy backend: its readiness decision and its capture.
 */

#include "src/bench/inc/ProfilerRAPL.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/PerfUtils.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vernier {
namespace bench {

namespace {

constexpr uint32_t MSR_RAPL_POWER_UNIT = 0x606;
constexpr uint32_t MSR_PKG_ENERGY_STATUS = 0x611;
constexpr uint32_t MSR_PP0_ENERGY_STATUS = 0x639;
constexpr uint32_t MSR_DRAM_ENERGY_STATUS = 0x619;
constexpr uint32_t MSR_PP1_ENERGY_STATUS = 0x641;

/// What a refusal for want of access tells the user to set up.
constexpr const char* ACCESS_REMEDY =
    "RAPL needs access set up beforehand, and Vernier does not raise its privileges: run as "
    "root, or give the process CAP_SYS_RAWIO (the msr driver checks it) and read access to "
    "/dev/cpu/0/msr (the capability does not grant it).";

/// What a refusal for want of the device tells the user to do.
constexpr const char* DEVICE_REMEDY =
    "Load the msr driver (sudo modprobe msr); in a container, pass the device in "
    "(--device /dev/cpu/0/msr).";

/** @brief "MSR 0x611" for @p msr. */
std::string msrName(uint32_t msr) {
  char text[24];
  std::snprintf(text, sizeof(text), "MSR 0x%x", msr);
  return text;
}

/** @brief One read of a model-specific register through an msr device. */
struct MsrRead {
  std::optional<uint64_t> value; ///< The register, when all 8 bytes were read.
  std::string step;              ///< What failed: the open or the read.
  int error = 0;                 ///< Its errno; 0 for a short read.
  long bytes = 0;                ///< What a short read returned.
};

/**
 * @brief Read @p msr through @p device: open, read the 8 bytes at the
 * register's offset, close. Nothing is held between reads.
 */
MsrRead readMsr(const std::string& device, uint32_t msr) {
  MsrRead out;
#ifdef __linux__
  const int FD = ::open(device.c_str(), O_RDONLY | O_CLOEXEC);
  if (FD < 0) {
    out.error = errno;
    out.step = "open " + device;
    return out;
  }
  uint64_t value = 0;
  ssize_t got = 0;
  do {
    got = ::pread(FD, &value, sizeof(value), static_cast<off_t>(msr));
  } while (got < 0 && errno == EINTR);
  const int READ_ERROR = got < 0 ? errno : 0;
  ::close(FD);
  if (got == static_cast<ssize_t>(sizeof(value))) {
    out.value = value;
    return out;
  }
  out.step = "read " + msrName(msr) + " from " + device;
  out.error = READ_ERROR;
  out.bytes = got < 0 ? 0 : static_cast<long>(got);
#else
  (void)msr;
  out.step = "open " + device;
  out.error = ENOSYS;
#endif
  return out;
}

/** @brief The refusal of a failed read: raplAccessRefusal(), or unusable for a short read. */
ReadinessResult refusalOf(const MsrRead& read) {
  if (read.error != 0) {
    return raplAccessRefusal(read.step, read.error);
  }
  return readinessResult(ReadinessCause::UNUSABLE,
                         read.step + " returned " + std::to_string(read.bytes) + " of 8 bytes", "");
}

/** @brief What /proc/cpuinfo says of the first CPU it lists. */
struct CpuDescription {
  bool readable = false; ///< The file could be opened.
  int error = 0;         ///< Why it could not.
  std::string vendor;    ///< vendor_id; empty when there is none (ARM, for one).
  std::string model;     ///< model name.
  int family = 0;        ///< cpu family.
  int modelNumber = 0;   ///< model.
};

/** @brief @p text without the spaces and tabs at its ends. */
std::string trimmed(const std::string& text) {
  const std::size_t FIRST = text.find_first_not_of(" \t");
  if (FIRST == std::string::npos) {
    return {};
  }
  return text.substr(FIRST, text.find_last_not_of(" \t") - FIRST + 1);
}

/** @brief The first CPU's fields of @p cpuinfo, up to the blank line that ends them. */
CpuDescription describeCpu(const std::string& cpuinfo) {
  CpuDescription cpu;
  errno = 0;
  std::ifstream in(cpuinfo);
  if (!in) {
    cpu.error = errno != 0 ? errno : ENOENT;
    return cpu;
  }
  cpu.readable = true;
  std::string line;
  bool seen = false;
  while (std::getline(in, line)) {
    if (trimmed(line).empty()) {
      if (seen) {
        break;
      }
      continue;
    }
    const std::size_t COLON = line.find(':');
    if (COLON == std::string::npos) {
      continue;
    }
    seen = true;
    const std::string KEY = trimmed(line.substr(0, COLON));
    const std::string VALUE = trimmed(line.substr(COLON + 1));
    if (KEY == "vendor_id") {
      cpu.vendor = VALUE;
    } else if (KEY == "model name") {
      cpu.model = VALUE;
    } else if (KEY == "cpu family") {
      cpu.family = std::atoi(VALUE.c_str());
    } else if (KEY == "model") {
      cpu.modelNumber = std::atoi(VALUE.c_str());
    }
  }
  return cpu;
}

/** @brief Whether the CPU and the msr device described by @p sources let rapl collect. */
ReadinessResult decideCollection(const RaplSources& sources) {
#ifndef __linux__
  (void)sources;
  return readinessResult(ReadinessCause::UNSUPPORTED,
                         "rapl reads the RAPL registers through Linux's msr driver, and this "
                         "system is not Linux",
                         "");
#else
  const CpuDescription CPU = describeCpu(sources.cpuinfo);
  if (!CPU.readable) {
    return readinessResult(ReadinessCause::UNUSABLE,
                           "cannot read " + sources.cpuinfo + ": " + std::strerror(CPU.error), "");
  }
  if (CPU.vendor.empty()) {
    return readinessResult(ReadinessCause::UNSUPPORTED,
                           "rapl reads Intel's RAPL energy registers, and " + sources.cpuinfo +
                               " names no CPU vendor here",
                           "");
  }
  if (CPU.vendor != "GenuineIntel") {
    return readinessResult(
        ReadinessCause::UNSUPPORTED,
        "rapl reads Intel's RAPL energy registers, and this CPU's vendor is " + CPU.vendor, "");
  }
  const MsrRead UNIT = readMsr(sources.device, MSR_RAPL_POWER_UNIT);
  if (!UNIT.value) {
    return refusalOf(UNIT);
  }
  const MsrRead ENERGY = readMsr(sources.device, MSR_PKG_ENERGY_STATUS);
  if (!ENERGY.value) {
    return refusalOf(ENERGY);
  }

  // The energy-status unit is bits 12:8 of MSR 0x606: one count is 2^-unit J.
  const int ESU = static_cast<int>((*UNIT.value >> 8) & 0x1F);
  auto plan = std::make_shared<RaplPlan>();
  plan->device = sources.device;
  plan->energyUnitJoules = std::ldexp(1.0, -ESU);
  plan->cpuModel = CPU.model;
  plan->cpuFamily = CPU.family;
  plan->cpuModelNumber = CPU.modelNumber;
  char unit[48];
  std::snprintf(unit, sizeof(unit), "%.3f uJ (2^-%d J, from MSR 0x606)",
                plan->energyUnitJoules * 1e6, ESU);
  ReadinessResult ready = readinessResult(ReadinessCause::READY,
                                          "read package 0's energy counter (MSR 0x611) through " +
                                              sources.device + "; one count is " + unit,
                                          "");
  ready.plan = std::move(plan);
  return ready;
#endif
}

std::shared_ptr<const RaplPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const RaplPlan>(result.plan);
}

/** @brief The decision for the request @p cfg states, in a snapshot of this process. */
ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "rapl";
  return checkRaplRequest(request, CTX);
}

} // namespace

/* ----------------------------- Decision ----------------------------- */

ReadinessResult raplAccessRefusal(const std::string& step, int err) {
  const std::string DETAIL = "cannot " + step + ": " + std::strerror(err);
  switch (err) {
  case EACCES:
  case EPERM:
    return readinessResult(ReadinessCause::DENIED, DETAIL, ACCESS_REMEDY);
  case ENOENT:
  case ENODEV:
    return readinessResult(ReadinessCause::MISSING, DETAIL + "; the msr driver provides it",
                           DEVICE_REMEDY);
  case EIO:
    return readinessResult(ReadinessCause::UNSUPPORTED,
                           DETAIL + (step.rfind("open ", 0) == 0
                                         ? "; this CPU has no model-specific registers"
                                         : "; this CPU does not implement that register"),
                           "");
  default:
    return readinessResult(ReadinessCause::UNUSABLE, DETAIL, "");
  }
}

ReadinessResult checkRaplRequestFrom(const ReadinessRequest& request, const RaplSources& sources) {
  const std::vector<std::string> WORDS = valgrind_tool::modeWords(request.profileArgs);
  if (!WORDS.empty()) {
    return valgrind_tool::refusedWord("rapl", WORDS.front(), {});
  }
  ReadinessResult result = decideCollection(sources);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: rapl has no automatic analysis; its energy report is "
                      "the result, and the capture still runs",
                      "Read each case's <Test>.rapl/energy.txt, and drop --profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

ReadinessResult checkRaplRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  (void)ctx; // The decision reads files, not the environment.
  return checkRaplRequestFrom(request, RaplSources{});
}

/* ----------------------------- RAPLProfiler ----------------------------- */

RAPLProfiler::RAPLProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Reported as the registry reports its own decisions. An empty folder is
  // not capture evidence: a request that cannot collect creates none.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (!ProfilerRegistry::instance().reportDecision("rapl", DECISION)) {
    return;
  }
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    return;
  }
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "rapl");
}

RAPLProfiler::RAPLProfiler(const PerfConfig& cfg, std::string testName,
                           std::shared_ptr<const RaplPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  if (plan_) {
    artifactDir_ =
        profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "rapl");
  }
}

bool RAPLProfiler::isAvailable() noexcept {
  try {
    ReadinessRequest request;
    request.backend = "rapl";
    request.scope = ReadinessScope::RUNTIME;
    return checkRaplRequest(request, ReadinessContext::capture()).collectionReady();
  } catch (...) {
    return false;
  }
}

std::optional<uint64_t> RAPLProfiler::readEnergyCounter(RAPLDomain domain) const {
  uint32_t msr = 0;

  switch (domain) {
  case RAPLDomain::Package:
    msr = MSR_PKG_ENERGY_STATUS;
    break;
  case RAPLDomain::Core:
    msr = MSR_PP0_ENERGY_STATUS;
    break;
  case RAPLDomain::DRAM:
    msr = MSR_DRAM_ENERGY_STATUS;
    break;
  case RAPLDomain::GPU:
    msr = MSR_PP1_ENERGY_STATUS;
    break;
  default:
    return std::nullopt;
  }

  return readMsr(plan_->device, msr).value;
}

void RAPLProfiler::beforeMeasure() {
  if (!plan_) {
    return;
  }

  startTimeUs_ = nowUs();

  auto counter = readEnergyCounter(domain_);
  if (counter) {
    energyStart_ = *counter;
  } else {
    energyStart_ = 0;
  }
}

void RAPLProfiler::afterMeasure(const Stats& s) {
  if (!plan_ || energyStart_ == 0) {
    return;
  }

  endTimeUs_ = nowUs();

  auto counter = readEnergyCounter(domain_);
  if (!counter) {
    return;
  }
  energyEnd_ = *counter;

  uint32_t start32 = static_cast<uint32_t>(energyStart_ & 0xFFFFFFFF);
  uint32_t end32 = static_cast<uint32_t>(energyEnd_ & 0xFFFFFFFF);

  double deltaCounter = 0.0;
  if (end32 >= start32) {
    deltaCounter = static_cast<double>(end32 - start32);
  } else {
    deltaCounter = static_cast<double>(0xFFFFFFFFUL - start32 + end32 + 1);
  }

  const double ENERGY_UNIT = plan_->energyUnitJoules;
  double energyJoules = deltaCounter * ENERGY_UNIT;
  double durationSeconds = (endTimeUs_ - startTimeUs_) / 1e6;
  double avgPowerWatts = energyJoules / durationSeconds;

  double totalOperations = (durationSeconds * 1e6) / s.median;
  double energyPerOpMillijoules = (energyJoules * 1000.0) / totalOperations;

  RAPLMeasurement measurement{.energyJoules = energyJoules,
                              .durationSeconds = durationSeconds,
                              .avgPowerWatts = avgPowerWatts,
                              .energyPerOpMillijoules = energyPerOpMillijoules};

  printSummary(measurement);

  std::string resultPath = artifactDir_ + "/energy.txt";
  std::ofstream out(resultPath);
  if (out) {
    out << "=== RAPL Energy Measurement ===\n";
    out << "Test: " << testName_ << "\n";
    out << "CPU: " << plan_->cpuModel << "\n";
    out << "Family: " << plan_->cpuFamily << ", Model: " << plan_->cpuModelNumber << "\n";
    out << "\n";
    out << "Energy consumed: " << energyJoules << " J\n";
    out << "Duration: " << durationSeconds << " s\n";
    out << "Average power: " << avgPowerWatts << " W\n";
    out << "Energy per operation: " << energyPerOpMillijoules << " mJ/call\n";
    out << "\n";
    out << "Energy unit: " << (ENERGY_UNIT * 1e6) << " uJ\n";
    out << "Start counter: " << energyStart_ << "\n";
    out << "End counter: " << energyEnd_ << "\n";
    out << "Delta counter: " << deltaCounter << "\n";
  }
}

void RAPLProfiler::printSummary(const RAPLMeasurement& measurement) const {
  std::printf("\n=== Energy Measurement (RAPL) ===\n");
  std::printf("Energy consumed:     %.3f Joules\n", measurement.energyJoules);
  std::printf("Average power:       %.2f Watts\n", measurement.avgPowerWatts);
  std::printf("Energy per op:       %.3f mJ/call\n", measurement.energyPerOpMillijoules);
  std::printf("Duration:            %.3f seconds\n", measurement.durationSeconds);
  std::printf("\n");

  if (measurement.avgPowerWatts > 50.0) {
    std::printf("Hint: High power draw (>50W) - CPU-intensive workload\n");
  } else if (measurement.avgPowerWatts < 10.0) {
    std::printf("Hint: Low power draw (<10W) - Idle or I/O-bound workload\n");
  }

  if (measurement.energyPerOpMillijoules > 1.0) {
    std::printf("Hint: High energy per operation (>1mJ) - Consider optimization\n");
  }

  std::printf("\n");
  std::printf("Note: RAPL measures package energy (CPU + iGPU + DRAM on some models)\n");
  std::printf("      Does not include PCIe devices, fans, or other system components\n");
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeRAPLProfiler(const PerfConfig& cfg, const std::string& testName) {
  const ReadinessResult DECISION = decideNow(cfg);
  auto plan =
      ProfilerRegistry::instance().reportDecision("rapl", DECISION) ? readyPlan(DECISION) : nullptr;
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<RAPLProfiler>(cfg, testName, std::move(plan));
}

namespace {

std::unique_ptr<Profiler> makePlannedRaplProfiler(const PerfConfig& cfg,
                                                  const std::string& testName,
                                                  const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<RAPLProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("rapl", ::vernier::bench::checkRaplRequest,
                                   ::vernier::bench::makePlannedRaplProfiler,
                                   "RAPL needs an Intel CPU, the msr driver's /dev/cpu/0/msr and "
                                   "read access to it.")
