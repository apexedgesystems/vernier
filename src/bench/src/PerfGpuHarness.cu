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

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <thread>

#ifdef COMPAT_NVML_AVAILABLE
#include <nvml.h>
#endif

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

#ifdef COMPAT_NVML_AVAILABLE
    if (gpuCfg_.captureClockSpeeds) {
      if (nvmlInit() == NVML_SUCCESS) {
        nvmlDeviceGetHandleByIndex(gpuCfg_.deviceId, &nvmlDevice_);
        nvmlInitialized_ = true;
      }
    }
#endif
  }

  ~PerfGpuCaseImpl() {
    cudaEventDestroy(eventStart_);
    cudaEventDestroy(eventStop_);
    cudaStreamDestroy(stream_);

#ifdef COMPAT_NVML_AVAILABLE
    if (nvmlInitialized_) {
      nvmlShutdown();
    }
#endif
  }

  PerfGpuCaseImpl(const PerfGpuCaseImpl&) = delete;
  PerfGpuCaseImpl& operator=(const PerfGpuCaseImpl&) = delete;

  void setBeforeMeasureHook(PerfGpuCase::BeforeHook h) { beforeHook_ = std::move(h); }

  void setAfterMeasureHook(PerfGpuCase::AfterHook h) { afterHook_ = std::move(h); }

  PerfResult cpuBaseline(std::function<void()> fn, std::string label) {
    PerfCase cpuPerf(testName_, cpuCfg_);
    cpuPerf.warmup([&]() {
      for (int i = 0; i < cpuCfg_.cycles; ++i) {
        fn();
      }
    });

    auto result = cpuPerf.throughputLoop(fn, label);
    cpuBaselineMedianUs_ = result.stats.median;

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

    std::vector<double> kernelTimes, h2dTimes, d2hTimes, totalTimes;
    kernelTimes.reserve(cpuCfg_.repeats);
    h2dTimes.reserve(cpuCfg_.repeats);
    d2hTimes.reserve(cpuCfg_.repeats);
    totalTimes.reserve(cpuCfg_.repeats);

    ClockSpeedProfile clocks{};
    if (gpuCfg_.captureClockSpeeds) {
      captureClockSpeed(clocks, true);
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
      CUDA_CHECK(cudaEventRecord(eventStart_, stream_));
      for (const auto& xfer : h2d) {
        CUDA_CHECK(
            cudaMemcpyAsync(xfer.dst, xfer.src, xfer.bytes, cudaMemcpyHostToDevice, stream_));
      }
      CUDA_CHECK(cudaEventRecord(eventStop_, stream_));
      CUDA_CHECK(cudaEventSynchronize(eventStop_));
      float h2dMs = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&h2dMs, eventStart_, eventStop_));
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

      CUDA_CHECK(cudaEventRecord(eventStart_, stream_));
      for (const auto& xfer : d2h) {
        CUDA_CHECK(
            cudaMemcpyAsync(xfer.dst, xfer.src, xfer.bytes, cudaMemcpyDeviceToHost, stream_));
      }
      CUDA_CHECK(cudaEventRecord(eventStop_, stream_));
      CUDA_CHECK(cudaEventSynchronize(eventStop_));
      float d2hMs = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&d2hMs, eventStart_, eventStop_));
      d2hTimes.push_back(d2hMs * 1000.0);

      totalTimes.push_back((h2dTimes.back() + kernelTimes.back() + d2hTimes.back()) /
                           cpuCfg_.cycles);
    }

    if (gpuCfg_.captureUnifiedMemory && totalManagedBytes > 0) {
      umAfter = captureUMSnapshot(gpuCfg_.deviceId);
      trackUnifiedMemory(umProfile, umBefore, umAfter, totalManagedBytes);
    }

    if (gpuCfg_.captureClockSpeeds) {
      captureClockSpeed(clocks, false);
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

    if (cpuBaselineMedianUs_ > 0.0) {
      result.speedupVsCpu = cpuBaselineMedianUs_ / totalStats.median;
    }

    result.stats.cpuStats = totalStats;
    result.stats.deviceInfo = deviceInfo_;
    result.stats.clocks = clocks;

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

    if (clocks.isThrottling()) {
      std::fprintf(stderr, "Warning: GPU throttling detected (%d -> %d MHz)\n",
                   clocks.smClockMHzStart, clocks.smClockMHzEnd);
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

    publishResult(result);

    return result;
  }

  MultiGpuResult measureMultiGpu(int deviceCount, std::function<void(int, cudaStream_t)> kernel,
                                 dim3 grid, dim3 block, size_t sharedMemBytes, bool hasLaunchConfig,
                                 bool enableP2P, int p2pSrcDevice, int p2pDstDevice,
                                 size_t p2pTestBytes, std::string label) {

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

        if (cpuBaselineMedianUs_ > 0.0) {
          devResult.speedupVsCpu = cpuBaselineMedianUs_ / stats.median;
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
    mgpu.scalingEfficiency =
        (cpuBaselineMedianUs_ > 0.0 && deviceCount > 0) ? totalSpeedup / deviceCount : 0.0;
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

    return result;
  }

  int cycles() const noexcept { return cpuCfg_.cycles; }
  int repeats() const noexcept { return cpuCfg_.repeats; }
  const PerfConfig& cpuConfig() const noexcept { return cpuCfg_; }
  const PerfGpuConfig& gpuConfig() const noexcept { return gpuCfg_; }
  const std::string& testName() const noexcept { return testName_; }
  cudaStream_t stream() const noexcept { return stream_; }

private:
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

  void captureClockSpeed(ClockSpeedProfile& clocks, bool isStart) {
#ifdef COMPAT_NVML_AVAILABLE
    if (!nvmlInitialized_)
      return;

    unsigned int smClock = 0, memClock = 0;
    if (nvmlDeviceGetClockInfo(nvmlDevice_, NVML_CLOCK_SM, &smClock) == NVML_SUCCESS) {
      if (isStart) {
        clocks.smClockMHzStart = static_cast<int>(smClock);
      } else {
        clocks.smClockMHzEnd = static_cast<int>(smClock);
      }
    }

    if (nvmlDeviceGetClockInfo(nvmlDevice_, NVML_CLOCK_MEM, &memClock) == NVML_SUCCESS) {
      if (isStart) {
        clocks.memClockMHzStart = static_cast<int>(memClock);
      } else {
        clocks.memClockMHzEnd = static_cast<int>(memClock);
      }
    }

    if (isStart) {
      unsigned int boostClock = 0;
      if (nvmlDeviceGetMaxClockInfo(nvmlDevice_, NVML_CLOCK_SM, &boostClock) == NVML_SUCCESS) {
        clocks.boostClockMHz = static_cast<int>(boostClock);
      }
    }
#else
    (void)clocks;
    (void)isStart;
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

  void publishResult(const PerfGpuResult& result) {
    auto [timestamp, gitHash, hostname, platform] = captureMetadata(true);

    PerfRow row{};
    row.testName = testName_;
    row.cycles = cpuCfg_.cycles;
    row.repeats = cpuCfg_.repeats;
    row.warmup = cpuCfg_.warmup;
    row.threads = 1;
    row.msgBytes = cpuCfg_.msgBytes;
    row.console = cpuCfg_.console;
    row.nonBlocking = cpuCfg_.nonBlocking;
    row.minLevel = cpuCfg_.minLevel;
    row.stats = result.stats.cpuStats;
    row.callsPerSecond = result.callsPerSecond;
    row.timestamp = timestamp;
    row.gitHash = gitHash;
    row.hostname = hostname;
    row.platform = platform;

    row.gpuModel = result.stats.deviceInfo.name;
    row.computeCapability = std::to_string(result.stats.deviceInfo.computeCapability[0]) + "." +
                            std::to_string(result.stats.deviceInfo.computeCapability[1]);
    row.kernelTimeUs = result.kernelTimeUs;
    row.transferTimeUs = result.transferTimeUs;
    row.h2dBytes = result.stats.transfers.h2dBytes;
    row.d2hBytes = result.stats.transfers.d2hBytes;
    row.speedupVsCpu = result.speedupVsCpu;
    row.memBandwidthGBs = result.stats.transfers.bandwidthGBs();
    row.occupancy = result.stats.occupancy.achievedOccupancy;
    row.smClockMHz = result.stats.clocks.smClockMHzEnd;
    row.throttling = result.stats.clocks.isThrottling();

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
    auto [timestamp, gitHash, hostname, platform] = captureMetadata(true);

    if (result.perDevice.empty())
      return;

    const auto& firstDev = result.perDevice[0];

    PerfRow row{};
    row.testName = testName_;
    row.cycles = cpuCfg_.cycles;
    row.repeats = cpuCfg_.repeats;
    row.warmup = cpuCfg_.warmup;
    row.threads = 1;
    row.msgBytes = cpuCfg_.msgBytes;
    row.console = cpuCfg_.console;
    row.nonBlocking = cpuCfg_.nonBlocking;
    row.minLevel = cpuCfg_.minLevel;
    row.stats = firstDev.stats.cpuStats;
    row.callsPerSecond = firstDev.callsPerSecond;
    row.timestamp = timestamp;
    row.gitHash = gitHash;
    row.hostname = hostname;
    row.platform = platform;

    row.gpuModel = firstDev.stats.deviceInfo.name;
    row.computeCapability = std::to_string(firstDev.stats.deviceInfo.computeCapability[0]) + "." +
                            std::to_string(firstDev.stats.deviceInfo.computeCapability[1]);
    row.kernelTimeUs = firstDev.kernelTimeUs;
    row.speedupVsCpu = result.totalSpeedupVsCpu;
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

#ifdef COMPAT_NVML_AVAILABLE
  nvmlDevice_t nvmlDevice_{};
  bool nvmlInitialized_ = false;
#endif

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
    : impl_(std::make_unique<PerfGpuCaseImpl>(std::move(testName), std::move(cpuCfg))) {}

PerfGpuCase::~PerfGpuCase() = default;

PerfResult PerfGpuCase::cpuBaseline(CpuFn fn, std::string label) {
  return impl_->cpuBaseline(std::move(fn), std::move(label));
}

CudaKernelBuilder PerfGpuCase::cudaKernel(KernelFn kernel, std::string label) {
  if (impl_->beforeHook_) {
    impl_->beforeHook_(*this);
  }
  return CudaKernelBuilder(impl_.get(), std::move(kernel), std::move(label));
}

MultiGpuKernelBuilder PerfGpuCase::cudaKernelMultiGpu(int deviceCount, MultiGpuKernelFn kernel,
                                                      std::string label) {
  if (impl_->beforeHook_) {
    impl_->beforeHook_(*this);
  }
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