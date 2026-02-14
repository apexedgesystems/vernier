/**
 * @file ProfilerRAPL.cpp
 * @brief Implementation of RAPL energy profiler backend.
 */

#include "src/bench/inc/ProfilerRAPL.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <system_error>

#include "src/bench/inc/PerfUtils.hpp"

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

std::optional<uint64_t> readMSR(int cpu, uint32_t msr) {
#ifdef __linux__
  char path[64];
  std::snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);

  int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    return std::nullopt;
  }

  uint64_t value = 0;
  ssize_t ret = ::pread(fd, &value, sizeof(value), msr);
  ::close(fd);

  if (ret != sizeof(value)) {
    return std::nullopt;
  }

  return value;
#else
  (void)cpu;
  (void)msr;
  return std::nullopt;
#endif
}

struct CPUInfo {
  std::string modelName;
  int family = 0;
  int model = 0;
};

CPUInfo readCPUInfo() {
  CPUInfo info;

#ifdef __linux__
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;

  while (std::getline(cpuinfo, line)) {
    if (line.find("model name") == 0) {
      size_t colon = line.find(':');
      if (colon != std::string::npos) {
        info.modelName = line.substr(colon + 2);
      }
    } else if (line.find("cpu family") == 0) {
      size_t colon = line.find(':');
      if (colon != std::string::npos) {
        info.family = std::atoi(line.substr(colon + 2).c_str());
      }
    } else if (line.find("model") == 0 && line.find("model name") == std::string::npos) {
      size_t colon = line.find(':');
      if (colon != std::string::npos) {
        info.model = std::atoi(line.substr(colon + 2).c_str());
      }
    }

    if (!info.modelName.empty() && info.family > 0 && info.model > 0) {
      break;
    }
  }
#endif

  return info;
}

bool isIntelCPUWithRAPL() {
#ifdef __linux__
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;

  bool isIntel = false;
  while (std::getline(cpuinfo, line)) {
    if (line.find("vendor_id") == 0 && line.find("GenuineIntel") != std::string::npos) {
      isIntel = true;
      break;
    }
  }

  if (!isIntel) {
    return false;
  }

  return std::filesystem::exists("/dev/cpu/0/msr");
#else
  return false;
#endif
}

} // namespace

RAPLProfiler::RAPLProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {

  if (!cfg_.artifactRoot.empty()) {
    artifactDir_ = cfg_.artifactRoot + "/" + testName_ + ".rapl";
  } else {
    artifactDir_ = "./" + testName_ + ".rapl";
  }

  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);

  detectCPUModel();
  domain_ = RAPLDomain::Package;
}

bool RAPLProfiler::isAvailable() noexcept { return isIntelCPUWithRAPL(); }

void RAPLProfiler::detectCPUModel() {
  auto cpuInfo = readCPUInfo();
  cpuModel_ = cpuInfo.modelName;
  cpuFamily_ = cpuInfo.family;
  cpuModelNum_ = cpuInfo.model;

  auto powerUnit = readMSR(0, MSR_RAPL_POWER_UNIT);
  if (powerUnit) {
    uint32_t esu = (*powerUnit >> 8) & 0x1F;
    energyUnit_ = 1.0 / std::pow(2.0, static_cast<double>(esu));
  } else {
    energyUnit_ = 1.0 / std::pow(2.0, 16.0);
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

  return readMSR(0, msr);
}

void RAPLProfiler::beforeMeasure() {
  if (!isAvailable()) {
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
  if (!isAvailable() || energyStart_ == 0) {
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

  double energyJoules = deltaCounter * energyUnit_;
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
    out << "CPU: " << cpuModel_ << "\n";
    out << "Family: " << cpuFamily_ << ", Model: " << cpuModelNum_ << "\n";
    out << "\n";
    out << "Energy consumed: " << energyJoules << " J\n";
    out << "Duration: " << durationSeconds << " s\n";
    out << "Average power: " << avgPowerWatts << " W\n";
    out << "Energy per operation: " << energyPerOpMillijoules << " mJ/call\n";
    out << "\n";
    out << "Energy unit: " << (energyUnit_ * 1e6) << " uuJ\n";
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

std::unique_ptr<Profiler> makeRAPLProfiler(const PerfConfig& cfg, const std::string& testName) {
  if (!RAPLProfiler::isAvailable()) {
    return nullptr;
  }

  return std::make_unique<RAPLProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier