/**
 * @file PerfGpuHarness.cu
 * @brief Implementation of GPU performance test harness with multi-GPU support
 *        and Unified Memory profiling.
 */

#include "src/bench/inc/PerfGpuHarness.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfGpuStats.hpp"
#include "src/bench/inc/PerfGpuTestMacros.hpp"
#include "src/bench/inc/PerfUtils.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/src/NvmlTelemetry.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <thread>

namespace vernier {
namespace bench {

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = call;                                                                        \
    if (err != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                        \
                   cudaGetErrorString(err));                                                       \
      throw std::runtime_error(cudaGetErrorString(err));                                           \
    }                                                                                              \
  } while (0)

// ============================================================================
// Occupancy calculation helper
// ============================================================================

void calculateOccupancy(OccupancyMetrics& occ, dim3 grid, dim3 block, size_t sharedMemBytes,
                        const cudaDeviceProp& prop) {
  occ.blockSize = block.x * block.y * block.z;
  occ.gridSize = grid.x * grid.y * grid.z;
  occ.maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;

  const int WARPS_PER_BLOCK = (occ.blockSize + 31) / 32;
  int blocksPerSM = prop.maxThreadsPerMultiProcessor / occ.blockSize;

  const int MAX_BLOCKS_PER_SM = prop.maxBlocksPerMultiProcessor;
  if (blocksPerSM > MAX_BLOCKS_PER_SM) {
    blocksPerSM = MAX_BLOCKS_PER_SM;
  }

  if (sharedMemBytes > 0) {
    const int SHMEM_BLOCKS_PER_SM = prop.sharedMemPerMultiprocessor / sharedMemBytes;
    if (blocksPerSM > SHMEM_BLOCKS_PER_SM) {
      blocksPerSM = SHMEM_BLOCKS_PER_SM;
    }
  }

  occ.activeWarpsPerSM = WARPS_PER_BLOCK * blocksPerSM;
  occ.achievedOccupancy =
      static_cast<double>(occ.activeWarpsPerSM) / static_cast<double>(occ.maxWarpsPerSM);

  if (occ.achievedOccupancy > 1.0) {
    occ.achievedOccupancy = 1.0;
  }

  if (occ.blockSize < 64) {
    occ.limitingFactor = OccupancyMetrics::LimitingFactor::BlockSize;
  } else if (sharedMemBytes > 0 &&
             (prop.sharedMemPerMultiprocessor / sharedMemBytes) < MAX_BLOCKS_PER_SM) {
    occ.limitingFactor = OccupancyMetrics::LimitingFactor::SharedMemory;
  } else if (occ.achievedOccupancy < 0.5) {
    occ.limitingFactor = OccupancyMetrics::LimitingFactor::Warps;
  } else {
    occ.limitingFactor = OccupancyMetrics::LimitingFactor::Unknown;
  }
}

// ============================================================================
// Unified Memory Profiling Helpers
// ============================================================================

#if CUDART_VERSION >= 8000

struct UMSnapshot {
  size_t totalMem = 0;
  size_t freeMem = 0;
  double timestamp = 0.0;
};

UMSnapshot captureUMSnapshot(int deviceId) {
  UMSnapshot snap;
  cudaSetDevice(deviceId);
  cudaMemGetInfo(&snap.freeMem, &snap.totalMem);
  snap.timestamp = nowUs();
  return snap;
}

void trackUnifiedMemory(UnifiedMemoryProfile& profile, const UMSnapshot& before,
                        const UMSnapshot& after, size_t managedBytes) {
  const size_t MEM_CHANGE = (before.freeMem > after.freeMem) ? (before.freeMem - after.freeMem)
                                                             : (after.freeMem - before.freeMem);

  const size_t PAGE_SIZE = 4096;
  profile.pageFaults = managedBytes / PAGE_SIZE;

  if (MEM_CHANGE > managedBytes / 2) {
    profile.h2dMigrations = profile.pageFaults / 2;
  }

  profile.migrationTimeUs = static_cast<double>(profile.pageFaults) * 1.0;
  profile.thrashingEvents = (profile.h2dMigrations > profile.pageFaults / 4) ? 1 : 0;
}

#else

struct UMSnapshot {
  double timestamp = 0.0;
};

UMSnapshot captureUMSnapshot(int) { return UMSnapshot{}; }

void trackUnifiedMemory(UnifiedMemoryProfile&, const UMSnapshot&, const UMSnapshot&, size_t) {}

#endif

// ============================================================================
// Suite-wide CPU baseline
// ============================================================================
// A GoogleTest suite runs each of its cases in a PerfGpuCase of its own, so a
// baseline measured in one case is out of reach of the next unless it is kept
// beside the case objects. The rule: a GPU case is compared against the
// baseline its own case measured; without one it is compared against its
// suite's baseline only while exactly one case of that suite has recorded one.
// Once a second case records a different baseline there is no answer to "which
// one", so a GPU case without its own baseline reports no speedup and the
// suite is named once on stderr. Keying by suite keeps two suites in one
// binary from reading each other's baseline.
// ============================================================================

namespace {

/** @brief "Suite.Case" -> "Suite". */
std::string suiteOf(const std::string& testName) {
  const auto DOT = testName.find('.');
  return (DOT == std::string::npos) ? testName : testName.substr(0, DOT);
}

/** @brief What one suite recorded as its CPU baseline. */
struct SuiteBaseline {
  std::string recordedBy; ///< Full name of the case that recorded the median
  double medianUs = 0.0;  ///< Baseline median (us per call)
  bool ambiguous = false; ///< A second case recorded one: no shared answer
  bool reported = false;  ///< The ambiguity has been named on stderr
};

std::mutex& baselineMutex() {
  static std::mutex mu;
  return mu;
}

std::map<std::string, SuiteBaseline>& baselineTable() {
  static std::map<std::string, SuiteBaseline> table;
  return table;
}

/**
 * @brief Record @p medianUs as the CPU baseline of @p testName's suite.
 *
 * A case that measures a baseline twice (a repeated run) keeps one entry; a
 * second, different case of the same suite makes the shared value ambiguous.
 */
void recordSuiteBaseline(const std::string& testName, double medianUs) {
  if (medianUs <= 0.0) {
    return;
  }
  const std::lock_guard<std::mutex> LOCK(baselineMutex());
  SuiteBaseline& entry = baselineTable()[suiteOf(testName)];
  if (entry.medianUs > 0.0 && entry.recordedBy != testName) {
    entry.ambiguous = true;
    return;
  }
  entry.recordedBy = testName;
  entry.medianUs = medianUs;
}

/**
 * @brief The baseline @p testName's suite shares, for a case with none of its own.
 * @return Median microseconds per call, 0.0 when the suite has no baseline or
 *         more than one case recorded one.
 */
double sharedSuiteBaselineUs(const std::string& testName) {
  const std::string SUITE = suiteOf(testName);
  const std::lock_guard<std::mutex> LOCK(baselineMutex());
  const auto IT = baselineTable().find(SUITE);
  if (IT == baselineTable().end()) {
    return 0.0;
  }
  if (IT->second.ambiguous) {
    if (!IT->second.reported) {
      IT->second.reported = true;
      std::fprintf(stderr,
                   "[gpu] suite %s measures a CPU baseline in more than one test, so a GPU test "
                   "of that suite has no single baseline to be compared against and reports no "
                   "speedup. Call cpuBaseline() in the GPU test itself, or keep one baseline "
                   "test per suite.\n",
                   SUITE.c_str());
    }
    return 0.0;
  }
  return IT->second.medianUs;
}

} // namespace

// ============================================================================
// A GPU cell a run cannot measure is left empty, and the run says why. What
// holds for the whole run (this build has no CUPTI, a provider refused) is
// said once per process, at the first measurement it empties cells of; what
// went wrong in one measured window is said for that window, naming the test.
// ============================================================================

namespace {

/// The CSV columns filled from the CUPTI collector's stats.
constexpr const char* CUPTI_CELLS = "cuptiKernelLaunches, cuptiRegistersMedian, "
                                    "cuptiRegistersMax, cuptiStaticSmemBytes and "
                                    "cuptiDynamicSmemBytes";

/** @brief Writes @p line and a newline to stderr, the first time this process asks for it. */
void stateOnce(const std::string& line) {
  static std::mutex mu;
  static std::set<std::string> stated;
  const std::lock_guard<std::mutex> LOCK(mu);
  if (stated.insert(line).second) {
    std::fprintf(stderr, "%s\n", line.c_str());
  }
}

} // namespace

// ============================================================================
// PerfGpuCaseImpl - PIMPL
// ============================================================================

class PerfGpuCaseImpl {
public:
  PerfGpuCaseImpl(std::string testName, PerfConfig cpuCfg)
      : testName_(std::move(testName)), cpuCfg_(std::move(cpuCfg)),
        gpuCfg_(detail::getGlobalGpuConfig()) {

    CUDA_CHECK(cudaSetDevice(gpuCfg_.deviceId));

    if (gpuCfg_.useHighPriorityStream) {
      int leastPriority, greatestPriority;
      CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatestPriority));
    } else {
      CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    CUDA_CHECK(cudaEventCreate(&eventStart_));
    CUDA_CHECK(cudaEventCreate(&eventStop_));

    queryDeviceInfo();

    nvml_.emplace(gpuCfg_.captureClockSpeeds, nvml_telemetry::uuidText(deviceProp_.uuid.bytes));
  }

  ~PerfGpuCaseImpl() {
    cudaEventDestroy(eventStart_);
    cudaEventDestroy(eventStop_);
    cudaStreamDestroy(stream_);
  }

  PerfGpuCaseImpl(const PerfGpuCaseImpl&) = delete;
  PerfGpuCaseImpl& operator=(const PerfGpuCaseImpl&) = delete;

  void setBeforeMeasureHook(PerfGpuCase::BeforeHook h) { beforeHook_ = std::move(h); }

  void setAfterMeasureHook(PerfGpuCase::AfterHook h) { afterHook_ = std::move(h); }

  PerfResult cpuBaseline(std::function<void()> fn, std::string label) {
    PerfCase cpuPerf(testName_, cpuCfg_);
    // The baseline is a measured window of this case like any other: the
    // inner PerfCase runs the case's hooks around it, and its after hook
    // follows the row it publishes, so the profiler stamps the baseline row.
    if (beforeHook_) {
      cpuPerf.setBeforeMeasureHook([this](const PerfCase&) { beforeHook_(*owner_); });
    }
    if (afterHook_) {
      cpuPerf.setAfterMeasureHook([this](const PerfCase&, const Stats& s) {
        GpuStats stats{};
        stats.cpuStats = s;
        afterHook_(*owner_, stats);
      });
    }
    cpuPerf.warmup([&]() {
      for (int i = 0; i < cpuCfg_.cycles; ++i) {
        fn();
      }
    });

    auto result = cpuPerf.throughputLoop(fn, label);
    cpuBaselineMedianUs_ = result.stats.median;
    recordSuiteBaseline(testName_, cpuBaselineMedianUs_);

    return result;
  }

  void cudaWarmup(std::function<void(cudaStream_t)> kernel) {
    kernel(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    for (int i = 0; i < gpuCfg_.gpuWarmup; ++i) {
      kernel(stream_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  PerfGpuResult measureKernel(std::function<void(cudaStream_t)> kernel,
                              const std::vector<CudaKernelBuilder::Transfer>& h2d,
                              const std::vector<CudaKernelBuilder::Transfer>& d2h, dim3 grid,
                              dim3 block, size_t sharedMemBytes, bool hasLaunchConfig, int deviceId,
                              std::string label) {
    // The profiler window opens here, before anything is timed, and closes
    // after the row is published (below), so every hook pair brackets one
    // measurement.
    if (beforeHook_) {
      beforeHook_(*owner_);
    }

    std::vector<double> kernelTimes, h2dTimes, d2hTimes, totalTimes;
    kernelTimes.reserve(cpuCfg_.repeats);
    h2dTimes.reserve(cpuCfg_.repeats);
    d2hTimes.reserve(cpuCfg_.repeats);
    totalTimes.reserve(cpuCfg_.repeats);

    // NVML readings at the window's start and end; each one checked.
    nvml_telemetry::WindowReadings nvmlReadings;
    nvml_->readStart(nvmlReadings);

    // Start in-process kernel metrics. Off when this case yielded to an
    // nsys/ncu session or to the explicit override (cuptiYields_, decided
    // before the collector registered). A collector that cannot collect for
    // any other reason (no CUPTI in this build, a refusal from CUPTI) is
    // stated once per process.
    const bool cuptiEnabled = !cuptiYields_;
    if (cuptiEnabled) {
      cupti_.start();
      if (!cupti_.isAvailable()) {
        stateOnce("[gpu] " + cupti_.unavailableReason() + ": " + CUPTI_CELLS + " stay empty.");
      }
    } else {
      std::fprintf(stderr, "[gpu] in-process CUPTI collection disabled for this run "
                           "(external Nsight session or VERNIER_DISABLE_CUPTI); "
                           "CUPTI CSV columns will be empty.\n");
    }

    UnifiedMemoryProfile umProfile{};
    UMSnapshot umBefore, umAfter;
    size_t totalManagedBytes = 0;

    if (gpuCfg_.captureUnifiedMemory) {
      for (const auto& xfer : h2d) {
        totalManagedBytes += xfer.bytes;
      }
      umBefore = captureUMSnapshot(gpuCfg_.deviceId);
    }

    for (int r = 0; r < cpuCfg_.repeats; ++r) {
      // A test that declares no transfer has no leg to time: an event pair
      // around nothing measures the event round trip itself and books it as
      // transfer time.
      float h2dMs = 0.0f;
      if (!h2d.empty()) {
        CUDA_CHECK(cudaEventRecord(eventStart_, stream_));
        for (const auto& xfer : h2d) {
          CUDA_CHECK(
              cudaMemcpyAsync(xfer.dst, xfer.src, xfer.bytes, cudaMemcpyHostToDevice, stream_));
        }
        CUDA_CHECK(cudaEventRecord(eventStop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(eventStop_));
        CUDA_CHECK(cudaEventElapsedTime(&h2dMs, eventStart_, eventStop_));
      }
      h2dTimes.push_back(h2dMs * 1000.0);

      CUDA_CHECK(cudaEventRecord(eventStart_, stream_));
      for (int c = 0; c < cpuCfg_.cycles; ++c) {
        kernel(stream_);
      }
      CUDA_CHECK(cudaEventRecord(eventStop_, stream_));
      CUDA_CHECK(cudaEventSynchronize(eventStop_));
      float kernelMs = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&kernelMs, eventStart_, eventStop_));
      kernelTimes.push_back(kernelMs * 1000.0 / cpuCfg_.cycles);

      float d2hMs = 0.0f;
      if (!d2h.empty()) {
        CUDA_CHECK(cudaEventRecord(eventStart_, stream_));
        for (const auto& xfer : d2h) {
          CUDA_CHECK(
              cudaMemcpyAsync(xfer.dst, xfer.src, xfer.bytes, cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaEventRecord(eventStop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(eventStop_));
        CUDA_CHECK(cudaEventElapsedTime(&d2hMs, eventStart_, eventStop_));
      }
      d2hTimes.push_back(d2hMs * 1000.0);

      // One round trip is one H2D leg, one kernel launch and one D2H leg.
      // kernelTimes holds the per-launch time (already divided by the cycle
      // count above) and the transfer legs run once per repeat, so the sum is
      // the per-call wall time.
      totalTimes.push_back(h2dTimes.back() + kernelTimes.back() + d2hTimes.back());
    }

    if (gpuCfg_.captureUnifiedMemory && totalManagedBytes > 0) {
      umAfter = captureUMSnapshot(gpuCfg_.deviceId);
      trackUnifiedMemory(umProfile, umBefore, umAfter, totalManagedBytes);
    }

    nvml_->readEnd(nvmlReadings);
    ClockSpeedProfile clocks{};
    PowerThermalProfile powerThermal{};
    nvml_telemetry::fillProfiles(nvmlReadings, clocks, powerThermal);
    // NVML cells nothing was read for are stated once per process: every
    // cell, for a session that samples nothing, or the readings NVML did not
    // report and the cells they leave empty.
    const std::string NVML_STATEMENT =
        nvml_->ready() ? nvml_telemetry::missingReadingsStatement(nvmlReadings)
                       : nvml_telemetry::absenceStatement(nvml_->unavailableReason());
    if (!NVML_STATEMENT.empty()) {
      stateOnce(NVML_STATEMENT);
    }

    // Drain CUPTI activity buffers and aggregate before publishing the result.
    // A window whose records are not known to be complete has no stats.
    if (cuptiEnabled) {
      cupti_.stop();
      if (!cupti_.windowProblem().empty()) {
        std::fprintf(stderr, "[gpu] %s in %s's measured window, so its %s stay empty.\n",
                     cupti_.windowProblem().c_str(), testName_.c_str(), CUPTI_CELLS);
      }
    }

    auto kernelVals = kernelTimes;
    auto h2dVals = h2dTimes;
    auto d2hVals = d2hTimes;
    auto totalVals = totalTimes;

    Stats kernelStats = summarize(kernelVals);
    Stats h2dStats = summarize(h2dVals);
    Stats d2hStats = summarize(d2hVals);
    Stats totalStats = summarize(totalVals);

    PerfGpuResult result;
    result.label = std::move(label);
    result.kernelTimeUs = kernelStats.median;
    result.transferTimeUs = h2dStats.median + d2hStats.median;
    result.totalTimeUs = totalStats.median;
    result.callsPerSecond = (totalStats.median > 0.0) ? 1e6 / totalStats.median : 0.0;
    result.deviceId = deviceId;

    const double BASELINE_US = baselineUs();
    if (BASELINE_US > 0.0 && totalStats.median > 0.0) {
      result.speedupVsCpu = BASELINE_US / totalStats.median;
    }

    result.stats.cpuStats = totalStats;
    result.stats.deviceInfo = deviceInfo_;
    result.stats.clocks = clocks;
    result.stats.powerThermal = powerThermal;
    if (cuptiEnabled) {
      result.stats.cupti = cupti_.stats();
    }

    size_t totalH2D = 0, totalD2H = 0;
    for (const auto& xfer : h2d)
      totalH2D += xfer.bytes;
    for (const auto& xfer : d2h)
      totalD2H += xfer.bytes;

    result.stats.transfers.h2dBytes = totalH2D;
    result.stats.transfers.d2hBytes = totalD2H;
    result.stats.transfers.h2dTimeUs = h2dStats.median;
    result.stats.transfers.d2hTimeUs = d2hStats.median;

    result.stats.kernelTimeMedianUs = kernelStats.median;
    result.stats.transferTimeMedianUs = result.transferTimeUs;
    result.stats.totalTimeMedianUs = totalStats.median;

    if (gpuCfg_.captureUnifiedMemory && totalManagedBytes > 0) {
      result.stats.unifiedMemory = umProfile;
    }

    if (hasLaunchConfig) {
      calculateOccupancy(result.stats.occupancy, grid, block, sharedMemBytes, deviceProp_);
    }

    const double CV_THRESHOLD = recommendedCVThreshold(cpuCfg_);
    const bool IS_STABLE = totalStats.cv < CV_THRESHOLD;

    const std::string LABEL_STR = "[" + testName_ + "]";
    printStatsWithHints(LABEL_STR.c_str(), totalStats, result.callsPerSecond, cpuCfg_, IS_STABLE);

    const std::string THROTTLING = nvml_telemetry::throttlingWarning(nvmlReadings);
    if (!THROTTLING.empty()) {
      std::fprintf(stderr, "%s\n", THROTTLING.c_str());
    }

    if (!hasLaunchConfig) {
      std::fprintf(
          stderr,
          "Hint: Occupancy is 0%% - add .withLaunchConfig(grid, block) for accurate metrics\n");
    }

    if (result.stats.unifiedMemory.has_value()) {
      const auto& um = *result.stats.unifiedMemory;
      std::printf("\n=== Unified Memory Profile ===\n");
      std::printf("Page faults: %zu\n", um.pageFaults);
      std::printf("H->D migrations: %zu\n", um.h2dMigrations);
      std::printf("D->H migrations: %zu\n", um.d2hMigrations);
      std::printf("Migration time: %.2f us (%.1f%% overhead)\n", um.migrationTimeUs,
                  um.migrationOverheadPct(result.kernelTimeUs, cpuCfg_.cycles));

      if (um.isThrashing()) {
        std::fprintf(stderr, "Warning: UM thrashing detected - consider prefetching\n");
      }

      if (um.migrationOverheadPct(result.kernelTimeUs, cpuCfg_.cycles) > 20.0) {
        std::printf("Hint: High migration overhead - consider:\n");
        std::printf("   - cudaMemPrefetchAsync() for predictable access\n");
        std::printf("   - cudaMemAdvise() for access hints\n");
        std::printf("   - Explicit H2D/D2H transfers if pattern is regular\n");
      }
    }

    publishResult(result, nvml_telemetry::cellsOf(nvmlReadings));

    // After publishResult(): the hook stamps profileTool/profileDir onto the
    // row just published.
    if (afterHook_) {
      afterHook_(*owner_, result.stats);
    }

    return result;
  }

  MultiGpuResult measureMultiGpu(int deviceCount, std::function<void(int, cudaStream_t)> kernel,
                                 dim3 grid, dim3 block, size_t sharedMemBytes, bool hasLaunchConfig,
                                 bool enableP2P, int p2pSrcDevice, int p2pDstDevice,
                                 size_t p2pTestBytes, std::string label) {
    if (beforeHook_) {
      beforeHook_(*owner_);
    }

    MultiGpuResult result;
    result.label = std::move(label);
    result.perDevice.reserve(deviceCount);

    if (enableP2P) {
      enablePeerToPeer(deviceCount);
    }

    if (p2pTestBytes > 0 && p2pSrcDevice >= 0 && p2pDstDevice >= 0) {
      result.aggregatedStats.p2pProfile =
          measureP2PBandwidth(p2pSrcDevice, p2pDstDevice, p2pTestBytes);
    }

    std::vector<std::thread> threads;
    std::vector<PerfGpuResult> deviceResults(deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
      threads.emplace_back([&, dev]() {
        cudaSetDevice(dev);

        cudaStream_t devStream;
        CUDA_CHECK(cudaStreamCreate(&devStream));

        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));

        std::vector<double> kernelTimes;
        kernelTimes.reserve(cpuCfg_.repeats);

        for (int r = 0; r < cpuCfg_.repeats; ++r) {
          CUDA_CHECK(cudaEventRecord(startEvent, devStream));
          for (int c = 0; c < cpuCfg_.cycles; ++c) {
            kernel(dev, devStream);
          }
          CUDA_CHECK(cudaEventRecord(stopEvent, devStream));
          CUDA_CHECK(cudaEventSynchronize(stopEvent));

          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
          kernelTimes.push_back(ms * 1000.0 / cpuCfg_.cycles);
        }

        auto vals = kernelTimes;
        Stats stats = summarize(vals);

        PerfGpuResult devResult;
        devResult.deviceId = dev;
        devResult.kernelTimeUs = stats.median;
        devResult.totalTimeUs = stats.median;
        devResult.callsPerSecond = (stats.median > 0.0) ? 1e6 / stats.median : 0.0;
        devResult.stats.cpuStats = stats;

        const double BASELINE_US = baselineUs();
        if (BASELINE_US > 0.0 && stats.median > 0.0) {
          devResult.speedupVsCpu = BASELINE_US / stats.median;
        }

        if (hasLaunchConfig) {
          cudaDeviceProp prop;
          cudaGetDeviceProperties(&prop, dev);
          calculateOccupancy(devResult.stats.occupancy, grid, block, sharedMemBytes, prop);
        }

        deviceResults[dev] = devResult;

        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        cudaStreamDestroy(devStream);
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    result.perDevice = std::move(deviceResults);

    double minTime = 1e9, maxTime = 0.0;
    double totalSpeedup = 0.0;

    for (const auto& devRes : result.perDevice) {
      minTime = std::min(minTime, devRes.kernelTimeUs);
      maxTime = std::max(maxTime, devRes.kernelTimeUs);
      totalSpeedup += devRes.speedupVsCpu;
    }

    result.totalSpeedupVsCpu = totalSpeedup;

    MultiGpuMetrics mgpu;
    mgpu.deviceCount = deviceCount;
    mgpu.loadImbalance = (minTime > 0.0) ? (maxTime / minTime) : 1.0;
    mgpu.scalingEfficiency = (totalSpeedup > 0.0 && deviceCount > 0)
                                 ? totalSpeedup / static_cast<double>(deviceCount)
                                 : 0.0;
    mgpu.p2pEnabled = enableP2P;
    if (result.aggregatedStats.p2pProfile.has_value()) {
      mgpu.p2pBandwidthGBs = result.aggregatedStats.p2pProfile->bandwidthGBs();
    }

    result.aggregatedStats.multiGpu = mgpu;

    std::printf("\n=== Multi-GPU Results ===\n");
    std::printf("Devices: %d\n", deviceCount);
    std::printf("Total speedup: %.2fx\n", totalSpeedup);
    std::printf("Scaling efficiency: %.2f (ideal=1.0)\n", mgpu.scalingEfficiency);
    std::printf("Load imbalance: %.2f (ideal=1.0)\n", mgpu.loadImbalance);
    if (enableP2P) {
      std::printf("P2P enabled: yes\n");
      if (mgpu.p2pBandwidthGBs > 0.0) {
        std::printf("P2P bandwidth: %.2f GB/s\n", mgpu.p2pBandwidthGBs);
      }
    }

    publishMultiGpuResult(result);

    // The published row carries the first device's times, so the hook gets
    // the same: the aggregated struct holds no timing of its own.
    if (afterHook_) {
      GpuStats stats = result.aggregatedStats;
      if (!result.perDevice.empty()) {
        stats.cpuStats = result.perDevice.front().stats.cpuStats;
      }
      afterHook_(*owner_, stats);
    }

    return result;
  }

  int cycles() const noexcept { return cpuCfg_.cycles; }
  int repeats() const noexcept { return cpuCfg_.repeats; }
  const PerfConfig& cpuConfig() const noexcept { return cpuCfg_; }
  const PerfGpuConfig& gpuConfig() const noexcept { return gpuCfg_; }
  const std::string& testName() const noexcept { return testName_; }
  cudaStream_t stream() const noexcept { return stream_; }

private:
  /**
   * @brief The CPU baseline a speedup is measured against: this case's own
   *        when it ran one, otherwise the one its suite recorded.
   * @return Median microseconds per call, 0.0 when the suite has no baseline.
   */
  [[nodiscard]] double baselineUs() const {
    return (cpuBaselineMedianUs_ > 0.0) ? cpuBaselineMedianUs_ : sharedSuiteBaselineUs(testName_);
  }

  void queryDeviceInfo() {
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp_, gpuCfg_.deviceId));

    deviceInfo_.name = deviceProp_.name;
    deviceInfo_.computeCapability[0] = deviceProp_.major;
    deviceInfo_.computeCapability[1] = deviceProp_.minor;
    deviceInfo_.totalMemoryMB = deviceProp_.totalGlobalMem / (1024 * 1024);
    deviceInfo_.smCount = deviceProp_.multiProcessorCount;
    deviceInfo_.maxThreadsPerSM = deviceProp_.maxThreadsPerMultiProcessor;

#if CUDART_VERSION >= 13000
    int clockKHz = 0, memClockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, gpuCfg_.deviceId);
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, gpuCfg_.deviceId);
    deviceInfo_.clockRateMHz = clockKHz / 1000;
    deviceInfo_.memoryClockRateMHz = memClockKHz / 1000;
    int busWidth = 0;
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, gpuCfg_.deviceId);
    deviceInfo_.memoryBusWidthBits = busWidth;
#else
    deviceInfo_.clockRateMHz = deviceProp_.clockRate / 1000;
    deviceInfo_.memoryClockRateMHz = deviceProp_.memoryClockRate / 1000;
    deviceInfo_.memoryBusWidthBits = deviceProp_.memoryBusWidth;
#endif
  }

  void enablePeerToPeer(int deviceCount) {
    for (int i = 0; i < deviceCount; ++i) {
      cudaSetDevice(i);
      for (int j = 0; j < deviceCount; ++j) {
        if (i != j) {
          int canAccess = 0;
          cudaDeviceCanAccessPeer(&canAccess, i, j);
          if (canAccess) {
            cudaDeviceEnablePeerAccess(j, 0);
          }
        }
      }
    }
  }

  P2PTransferProfile measureP2PBandwidth(int srcDev, int dstDev, size_t bytes) {
    P2PTransferProfile profile;
    profile.srcDevice = srcDev;
    profile.dstDevice = dstDev;
    profile.bytes = bytes;

    int canAccess = 0;
    cudaDeviceCanAccessPeer(&canAccess, srcDev, dstDev);
    profile.accessEnabled = (canAccess != 0);

    if (!profile.accessEnabled) {
      return profile;
    }

    void *srcPtr, *dstPtr;
    cudaSetDevice(srcDev);
    CUDA_CHECK(cudaMalloc(&srcPtr, bytes));
    cudaSetDevice(dstDev);
    CUDA_CHECK(cudaMalloc(&dstPtr, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaStream_t stream;
    cudaSetDevice(srcDev);
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyPeerAsync(dstPtr, dstDev, srcPtr, srcDev, bytes, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    profile.timeUs = ms * 1000.0;

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaSetDevice(srcDev);
    cudaFree(srcPtr);
    cudaSetDevice(dstDev);
    cudaFree(dstPtr);

    return profile;
  }

  void publishResult(const PerfGpuResult& result, const nvml_telemetry::Cells& nvml) {
    // The row builder the CPU path uses: the config, metadata and stability
    // columns of a GPU row are then the same columns, filled the same way,
    // and the CSV `stable` verdict is the one the console printed.
    PerfRow row = buildPerfRow(testName_, cpuCfg_, cpuCfg_.warmup, /*threadCount=*/1,
                               result.stats.cpuStats, result.callsPerSecond);

    row.gpuModel = result.stats.deviceInfo.name;
    row.computeCapability = std::to_string(result.stats.deviceInfo.computeCapability[0]) + "." +
                            std::to_string(result.stats.deviceInfo.computeCapability[1]);
    row.kernelTimeUs = result.kernelTimeUs;
    row.transferTimeUs = result.transferTimeUs;
    row.h2dBytes = result.stats.transfers.h2dBytes;
    row.d2hBytes = result.stats.transfers.d2hBytes;
    // An unknown speedup is an empty cell, not a zero a reader would take for
    // a measurement.
    if (result.speedupVsCpu > 0.0) {
      row.speedupVsCpu = result.speedupVsCpu;
    }
    row.memBandwidthGBs = result.stats.transfers.bandwidthGBs();
    row.occupancy = result.stats.occupancy.achievedOccupancy;
    // The NVML cells, each from the readings NVML reported (empty otherwise).
    row.smClockMHz = nvml.smClockMHz;
    row.throttling = nvml.throttling;
    row.powerDrawW = nvml.powerDrawW;
    row.powerLimitW = nvml.powerLimitW;
    row.temperatureC = nvml.temperatureC;
    row.temperatureDeltaC = nvml.temperatureDeltaC;

    // CUPTI columns are only populated when at least one launch was captured;
    // otherwise the CSV cells stay empty rather than reporting zeros.
    const auto& cu = result.stats.cupti;
    if (cu.kernelLaunches > 0) {
      row.cuptiKernelLaunches = cu.kernelLaunches;
      row.cuptiRegistersMedian = cu.registersMedian;
      row.cuptiRegistersMax = cu.registersMax;
      row.cuptiStaticSmemBytes = cu.staticSmemBytes;
      row.cuptiDynamicSmemBytes = cu.dynamicSmemBytes;
    }

    if (result.deviceId >= 0) {
      row.deviceId = result.deviceId;
    }
    row.deviceCount = 1;

    if (result.stats.unifiedMemory.has_value()) {
      const auto& um = *result.stats.unifiedMemory;
      row.umPageFaults = um.pageFaults;
      row.umH2DMigrations = um.h2dMigrations;
      row.umD2HMigrations = um.d2hMigrations;
      row.umMigrationTimeUs = um.migrationTimeUs;
      row.umThrashing = um.isThrashing();
    }

    PerfRegistry::instance().set(row);
  }

  void publishMultiGpuResult(const MultiGpuResult& result) {
    if (result.perDevice.empty())
      return;

    const auto& firstDev = result.perDevice[0];

    PerfRow row = buildPerfRow(testName_, cpuCfg_, cpuCfg_.warmup, /*threadCount=*/1,
                               firstDev.stats.cpuStats, firstDev.callsPerSecond);

    row.gpuModel = firstDev.stats.deviceInfo.name;
    row.computeCapability = std::to_string(firstDev.stats.deviceInfo.computeCapability[0]) + "." +
                            std::to_string(firstDev.stats.deviceInfo.computeCapability[1]);
    row.kernelTimeUs = firstDev.kernelTimeUs;
    if (result.totalSpeedupVsCpu > 0.0) {
      row.speedupVsCpu = result.totalSpeedupVsCpu;
    }
    row.occupancy = firstDev.stats.occupancy.achievedOccupancy;

    row.deviceId = -1;
    row.deviceCount = result.aggregatedStats.multiGpu->deviceCount;
    row.multiGpuEfficiency = result.aggregatedStats.multiGpu->scalingEfficiency;
    if (result.aggregatedStats.p2pProfile.has_value()) {
      row.p2pBandwidthGBs = result.aggregatedStats.p2pProfile->bandwidthGBs();
    }

    PerfRegistry::instance().set(row);
  }

  std::string testName_;
  PerfConfig cpuCfg_;
  PerfGpuConfig gpuCfg_;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t eventStart_ = nullptr;
  cudaEvent_t eventStop_ = nullptr;

  cudaDeviceProp deviceProp_{};
  GpuDeviceInfo deviceInfo_{};
  double cpuBaselineMedianUs_ = 0.0;

  PerfGpuCase::BeforeHook beforeHook_{};
  PerfGpuCase::AfterHook afterHook_{};
  /// The case this implementation belongs to, for the hook signatures
  /// (non-owning; set by the PerfGpuCase constructor).
  const PerfGpuCase* owner_ = nullptr;

  // NVML opened for this case's device, found by its UUID (or the reason it
  // samples nothing); set at the end of the constructor, once the device's
  // properties are read.
  std::optional<nvml_telemetry::Session> nvml_;

  // Whether this case's CUPTI collection stands down (profiler_env::
  // cuptiMustYield(): an nsys/ncu session, or the explicit override). Decided
  // once, before the collector below is built, because registering the
  // collector already keeps an nsys session from recording kernels; the
  // collector, the measurement and its diagnostic all follow this value. An
  // invalid VERNIER_DISABLE_CUPTI throws here, a configuration error, before
  // the collector registers and before the constructor touches the device.
  const bool cuptiYields_ = profiler_env::cuptiMustYield();

  // In-process kernel records; a no-op in a build without CUPTI.
  CuptiCollector cupti_{cuptiYields_};

  friend class PerfGpuCase;
};

// ============================================================================
// CudaKernelBuilder implementation
// ============================================================================

PerfGpuResult CudaKernelBuilder::measure() {
  return impl_->measureKernel(kernel_, h2d_, d2h_, grid_, block_, sharedMemBytes_, hasLaunchConfig_,
                              deviceId_, label_);
}

// ============================================================================
// MultiGpuKernelBuilder implementation
// ============================================================================

MultiGpuResult MultiGpuKernelBuilder::measure() {
  return impl_->measureMultiGpu(deviceCount_, kernel_, grid_, block_, sharedMemBytes_,
                                hasLaunchConfig_, enableP2P_, p2pSrcDevice_, p2pDstDevice_,
                                p2pTestBytes_, label_);
}

// ============================================================================
// PerfGpuCase implementation
// ============================================================================

PerfGpuCase::PerfGpuCase(std::string testName, PerfConfig cpuCfg)
    : impl_(std::make_unique<PerfGpuCaseImpl>(std::move(testName), std::move(cpuCfg))) {
  // PerfGpuCase is neither copyable nor movable, so this address stays valid.
  impl_->owner_ = this;
}

PerfGpuCase::~PerfGpuCase() = default;

PerfResult PerfGpuCase::cpuBaseline(CpuFn fn, std::string label) {
  return impl_->cpuBaseline(std::move(fn), std::move(label));
}

CudaKernelBuilder PerfGpuCase::cudaKernel(KernelFn kernel, std::string label) {
  return CudaKernelBuilder(impl_.get(), std::move(kernel), std::move(label));
}

MultiGpuKernelBuilder PerfGpuCase::cudaKernelMultiGpu(int deviceCount, MultiGpuKernelFn kernel,
                                                      std::string label) {
  return MultiGpuKernelBuilder(impl_.get(), deviceCount, std::move(kernel), std::move(label));
}

void PerfGpuCase::cudaWarmup(KernelFn kernel) { impl_->cudaWarmup(std::move(kernel)); }

void PerfGpuCase::setBeforeMeasureHook(BeforeHook h) { impl_->setBeforeMeasureHook(std::move(h)); }

void PerfGpuCase::setAfterMeasureHook(AfterHook h) { impl_->setAfterMeasureHook(std::move(h)); }

int PerfGpuCase::cycles() const noexcept { return impl_->cycles(); }
int PerfGpuCase::repeats() const noexcept { return impl_->repeats(); }
const PerfConfig& PerfGpuCase::cpuConfig() const noexcept { return impl_->cpuConfig(); }
const PerfGpuConfig& PerfGpuCase::gpuConfig() const noexcept { return impl_->gpuConfig(); }
const std::string& PerfGpuCase::testName() const noexcept { return impl_->testName(); }
cudaStream_t PerfGpuCase::stream() const noexcept { return impl_->stream(); }

} // namespace bench
} // namespace vernier