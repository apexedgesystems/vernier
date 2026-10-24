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

/// A refusal's parts, before a stage and its cause word make it a result.
struct Refusal {
  ReadinessCause cause = ReadinessCause::UNUSABLE;
  std::string detail;
  std::string remedy;
};

/** @brief The cause of errno @p err from @p step, an open of the device or a read of it. */
Refusal refusalFor(const std::string& step, int err) {
  const std::string DETAIL = "cannot " + step + ": " + std::strerror(err);
  switch (err) {
  case EACCES:
  case EPERM:
    return {ReadinessCause::DENIED, DETAIL, ACCESS_REMEDY};
  case ENOENT:
  case ENODEV:
    return {ReadinessCause::MISSING, DETAIL + "; the msr driver provides it", DEVICE_REMEDY};
  case EIO:
    return {ReadinessCause::UNSUPPORTED,
            DETAIL + (step.rfind("open ", 0) == 0 ? "; this CPU has no model-specific registers"
                                                  : "; this CPU does not implement that register"),
            ""};
  default:
    return {ReadinessCause::UNUSABLE, DETAIL, ""};
  }
}

/** @brief The cause of a failed read: refusalFor() its errno, or unusable for a short read. */
Refusal refusalFor(const MsrRead& read) {
  if (read.error != 0) {
    return refusalFor(read.step, read.error);
  }
  return {ReadinessCause::UNUSABLE,
          read.step + " returned " + std::to_string(read.bytes) + " of 8 bytes", ""};
}

/** @brief A failed read as the decision refuses it. */
ReadinessResult refusalOf(const MsrRead& read) {
  const Refusal REFUSAL = refusalFor(read);
  return readinessResult(REFUSAL.cause, REFUSAL.detail, REFUSAL.remedy);
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
  const Refusal REFUSAL = refusalFor(step, err);
  return readinessResult(REFUSAL.cause, REFUSAL.detail, REFUSAL.remedy);
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

void RAPLProfiler::beforeMeasure() {
  // Each window's start is its own: a window ended by an exception gets no
  // after hook, and its reading goes here. Nothing else outlives a window:
  // the device is opened for each read.
  start_.reset();
  if (!plan_) {
    return;
  }
  const double TIME_US = nowUs();
  const MsrRead READ = readMsr(plan_->device, MSR_PKG_ENERGY_STATUS);
  if (!READ.value) {
    const Refusal REFUSAL = refusalFor(READ);
    fail(REFUSAL.cause, "window " + std::to_string(windows_ + 1) + "'s start: " + REFUSAL.detail,
         REFUSAL.remedy, ReadinessStage::COLLECTION);
    return;
  }
  // The energy-status counter is bits 31:0; bits 63:32 are reserved.
  start_ = Reading{static_cast<uint32_t>(*READ.value & 0xFFFFFFFFULL), TIME_US};
}

void RAPLProfiler::afterMeasure(const Stats& s) {
  if (!plan_) {
    return;
  }
  // The end reading comes first: nothing else is counted in the window.
  const MsrRead READ = readMsr(plan_->device, MSR_PKG_ENERGY_STATUS);
  const double END_US = nowUs();

  // Every window that ends published a row: the sections follow the rows.
  const int WINDOW = ++windows_;
  if (!start_) {
    return; // The start's failure was reported when it happened.
  }
  const Reading START = *start_;
  start_.reset();
  if (!READ.value) {
    const Refusal REFUSAL = refusalFor(READ);
    fail(REFUSAL.cause, "window " + std::to_string(WINDOW) + "'s end: " + REFUSAL.detail,
         REFUSAL.remedy, ReadinessStage::COLLECTION);
    return;
  }
  const Reading END{static_cast<uint32_t>(*READ.value & 0xFFFFFFFFULL), END_US};

  // Unsigned 32-bit subtraction is right across one wrap of the counter;
  // two wraps in one window cannot be told from none.
  const uint32_t DELTA = END.counter - START.counter;
  RAPLMeasurement measurement;
  measurement.energyJoules = static_cast<double>(DELTA) * plan_->energyUnitJoules;
  measurement.durationSeconds = (END.timeUs - START.timeUs) / 1e6;
  measurement.avgPowerWatts = measurement.durationSeconds > 0.0
                                  ? measurement.energyJoules / measurement.durationSeconds
                                  : 0.0;
  // Energy per call: the window's average package power times the median
  // call time (W x us = uJ, so 1e-3 for mJ). The hook is given the window's
  // statistics, not its count of calls.
  measurement.energyPerOpMillijoules = measurement.avgPowerWatts * s.median * 1e-3;

  if (!writeWindow(WINDOW, START, END, DELTA, measurement, s.median)) {
    return;
  }
  printSummary(WINDOW, measurement, DELTA);
}

void RAPLProfiler::fail(ReadinessCause cause, const std::string& detail, const std::string& remedy,
                        ReadinessStage stage) const {
  ProfilerRegistry::instance().reportFailure("rapl", testName_,
                                             readinessResult(cause, detail, remedy, stage));
}

bool RAPLProfiler::writeWindow(int window, const Reading& start, const Reading& end, uint32_t delta,
                               const RAPLMeasurement& measurement, double medianUs) {
  char line[256];
  std::string text;
  if (!reportStarted_) {
    text += "=== RAPL Energy Measurement ===\n";
    text += "Test: " + testName_ + "\n";
    text += "CPU: " + plan_->cpuModel + "\n";
    text += "Family: " + std::to_string(plan_->cpuFamily) +
            ", Model: " + std::to_string(plan_->cpuModelNumber) + "\n";
    text += "Scope: package 0, read through " + plan_->device +
            " (MSR 0x611): every core, cache and the integrated graphics of the package, "
            "whatever runs on them; nothing outside the package\n";
    std::snprintf(line, sizeof(line),
                  "Counter: 32 bits, wrapping after %.3f J at this unit; a window is taken to "
                  "wrap it at most once\n",
                  4294967296.0 * plan_->energyUnitJoules);
    text += line;
    text += "Energy per operation: the window's average package power x its median time per "
            "call\n";
  }
  text += "\n--- Window " + std::to_string(window) + " ---\n";
  std::snprintf(line, sizeof(line), "Energy consumed: %.6f J\n", measurement.energyJoules);
  text += line;
  std::snprintf(line, sizeof(line), "Duration: %.6f s\n", measurement.durationSeconds);
  text += line;
  std::snprintf(line, sizeof(line), "Average power: %.6f W\n", measurement.avgPowerWatts);
  text += line;
  std::snprintf(line, sizeof(line), "Energy per operation: %.6f mJ/call\n",
                measurement.energyPerOpMillijoules);
  text += line;
  std::snprintf(line, sizeof(line), "Median time per call: %.3f us\n", medianUs);
  text += line;
  std::snprintf(line, sizeof(line), "Energy unit: %.4f uJ\n", plan_->energyUnitJoules * 1e6);
  text += line;
  text += "Start counter: " + std::to_string(start.counter) + "\n";
  text += "End counter: " + std::to_string(end.counter) + "\n";
  text += "Delta counter: " + std::to_string(delta) + "\n";
  if (delta == 0) {
    std::snprintf(line, sizeof(line),
                  "Note: the counter did not advance during this %.3f s window\n",
                  measurement.durationSeconds);
    text += line;
  }

  // Each step checked: a report that is not on disk must not be announced.
  const std::string PATH = artifactDir_ + "/energy.txt";
  const std::string WHERE = "window " + std::to_string(window) + "'s energy report: ";
  const std::string REMEDY =
      "Make sure the output folder, which --profile-output-dir sets, can be created and written.";
  errno = 0;
  std::FILE* file = std::fopen(PATH.c_str(), reportStarted_ ? "a" : "w");
  if (file == nullptr) {
    fail(ReadinessCause::UNUSABLE,
         WHERE + "cannot open " + PATH + " for writing: " + std::strerror(errno), REMEDY,
         ReadinessStage::COMPLETION);
    return false;
  }
  errno = 0;
  const bool WRITTEN = std::fputs(text.c_str(), file) >= 0;
  const int WRITE_ERROR = errno;
  errno = 0;
  const bool CLOSED = std::fclose(file) == 0;
  const int CLOSE_ERROR = errno;
  if (!WRITTEN || !CLOSED) {
    const int ERR = !WRITTEN ? WRITE_ERROR : CLOSE_ERROR;
    fail(ReadinessCause::UNUSABLE,
         WHERE + "cannot write " + PATH + ": " +
             (ERR != 0 ? std::strerror(ERR) : "the write did not complete"),
         REMEDY, ReadinessStage::COMPLETION);
    return false;
  }
  reportStarted_ = true;
  return true;
}

void RAPLProfiler::printSummary(int window, const RAPLMeasurement& measurement,
                                uint32_t delta) const {
  std::printf("\n=== Energy Measurement (RAPL, package 0) ===\n");
  std::printf("Window:              %d\n", window);
  std::printf("Energy consumed:     %.3f Joules\n", measurement.energyJoules);
  std::printf("Average power:       %.2f Watts\n", measurement.avgPowerWatts);
  std::printf("Energy per op:       %.3f mJ/call (average power x median call time)\n",
              measurement.energyPerOpMillijoules);
  std::printf("Duration:            %.3f seconds\n", measurement.durationSeconds);
  if (delta == 0) {
    std::printf("Counter:             did not advance during the window\n");
  }
  std::printf("Scope:               package 0, every core included; nothing outside it\n");
  std::printf("Report:              %s/energy.txt\n\n", artifactDir_.c_str());
  std::fflush(stdout);
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
