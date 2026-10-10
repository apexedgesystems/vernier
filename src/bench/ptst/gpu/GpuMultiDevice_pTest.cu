/**
 * @file GpuMultiDevice_pTest.cu
 * @brief Multi-GPU execution, P2P transfers and device selection
 *
 * This test suite validates multi-GPU functionality including load distribution
 * and peer-to-peer transfers, and shows device selection with --gpu-device.
 *
 * Features tested:
 *  - Multi-GPU kernel execution with cudaKernelMultiGpu()
 *  - Load balancing across multiple GPUs
 *  - P2P transfer bandwidth measurement
 *  - Device selection with --gpu-device
 *
 * Expected behavior:
 *  - Kernels execute on all specified devices
 *  - Load is distributed evenly
 *  - P2P transfers work when supported
 *  - The suite measures no CPU baseline, so it reports no speedup or scaling
 *    efficiency
 *
 * Every case but DeviceSelection needs two or more GPUs and skips with fewer.
 * A skip validates nothing, and none of this project's rigs
 * (src/bench/docs/rigs/) has two GPUs, so those cases are not validated there.
 *
 * Usage:
 *   @code{.sh}
 *   # Run all multi-GPU tests
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuMultiDevice.*"
 *
 *   # Device selection on another device
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuMultiDevice.DeviceSelection" --gpu-device 1
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~15 seconds total (if 2+ GPUs available)
 *
 * @see PerfGpuCase
 * @see MultiGpuKernelBuilder
 * @see MultiGpuResult
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Simple vector addition kernel for multi-GPU testing */
__global__ void multiGpuVectorAdd(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

/** @brief Get number of available CUDA devices */
int getDeviceCount() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  return deviceCount;
}

/** @brief Check if P2P is supported between two devices */
bool isP2PSupported(int dev1, int dev2) {
  int canAccess = 0;
  cudaDeviceCanAccessPeer(&canAccess, dev1, dev2);
  return canAccess != 0;
}

} // anonymous namespace

/**
 * @brief Basic multi-GPU kernel execution
 *
 * Validates that kernels can execute across multiple GPUs simultaneously
 * with proper load distribution and synchronization. The suite measures no
 * CPU baseline, so the harness reports no speedup or scaling efficiency for
 * it (both stay 0); the case prints each device's kernel time and the load
 * imbalance instead.
 *
 * @test BasicMultiGpu
 *
 * Validates:
 *  - cudaKernelMultiGpu() API executes on all devices
 *  - Per-device results are collected
 *  - Load balancing metrics are computed
 *
 * Expected performance:
 *  - All devices execute successfully
 *  - Load imbalance < 1.2 (20% imbalance acceptable)
 */
PERF_GPU_TEST(GpuMultiDevice, BasicMultiGpu) {
  UB_PERF_GPU_GUARD(perf);

  const int deviceCount = getDeviceCount();
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires 2+ GPUs, found " << deviceCount;
  }

  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate host data
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  // Allocate per-device data
  std::vector<float*> d_a(deviceCount);
  std::vector<float*> d_b(deviceCount);
  std::vector<float*> d_c(deviceCount);

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaMalloc(&d_a[dev], SIZE);
    cudaMalloc(&d_b[dev], SIZE);
    cudaMalloc(&d_c[dev], SIZE);
    cudaMemcpy(d_a[dev], h_a.data(), SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b[dev], h_b.data(), SIZE, cudaMemcpyHostToDevice);
  }

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    cudaSetDevice(0);
    multiGpuVectorAdd<<<grid, block, 0, s>>>(d_a[0], d_b[0], d_c[0], N);
  });

  // Multi-GPU execution
  auto result = perf.cudaKernelMultiGpu(
                        deviceCount,
                        [&](int deviceId, cudaStream_t stream) {
                          cudaSetDevice(deviceId);
                          multiGpuVectorAdd<<<grid, block, 0, stream>>>(
                              d_a[deviceId], d_b[deviceId], d_c[deviceId], N);
                        },
                        "multi_gpu_vector_add")
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate results
  EXPECT_EQ(result.perDevice.size(), static_cast<size_t>(deviceCount))
      << "Should have results for each device";

  ASSERT_TRUE(result.aggregatedStats.multiGpu.has_value())
      << "Multi-GPU metrics should be populated";

  const auto& mgpu = result.aggregatedStats.multiGpu.value();

  EXPECT_EQ(mgpu.deviceCount, deviceCount) << "Device count should match";

  EXPECT_LT(mgpu.loadImbalance, 1.2) << "Load should be reasonably balanced (< 20% imbalance)";

  for (const auto& devResult : result.perDevice) {
    std::printf("[GpuMultiDevice.BasicMultiGpu] device %d: %.3f us per launch\n",
                devResult.deviceId, devResult.kernelTimeUs);
  }
  std::printf("[GpuMultiDevice.BasicMultiGpu] load imbalance %.3f (slowest device over fastest); "
              "no CPU baseline, so no speedup or scaling efficiency\n",
              mgpu.loadImbalance);

  // Verify per-device execution
  for (size_t i = 0; i < result.perDevice.size(); ++i) {
    const auto& devResult = result.perDevice[i];
    EXPECT_EQ(devResult.deviceId, static_cast<int>(i)) << "Device ID should match index";
    EXPECT_GT(devResult.callsPerSecond, 0.0) << "Device " << i << " should have valid throughput";
  }

  // Cleanup
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaFree(d_a[dev]);
    cudaFree(d_b[dev]);
    cudaFree(d_c[dev]);
  }
}

/**
 * @brief Multi-GPU load balancing validation
 *
 * Validates that work is distributed evenly across GPUs and that
 * load imbalance metrics accurately reflect the distribution.
 *
 * @test LoadBalancing
 *
 * Validates:
 *  - Work distributed to all devices
 *  - Load imbalance metric calculation
 *  - Per-device timing consistency
 *
 * Expected performance:
 *  - Load imbalance < 1.5 for identical GPUs
 */
PERF_GPU_TEST(GpuMultiDevice, LoadBalancing) {
  UB_PERF_GPU_GUARD(perf);

  const int deviceCount = getDeviceCount();
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires 2+ GPUs, found " << deviceCount;
  }

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float*> d_a(deviceCount);

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaMalloc(&d_a[dev], SIZE);
    cudaMemcpy(d_a[dev], h_a.data(), SIZE, cudaMemcpyHostToDevice);
  }

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Multi-GPU execution with identical workload
  auto result = perf.cudaKernelMultiGpu(
                        deviceCount,
                        [&](int deviceId, cudaStream_t stream) {
                          cudaSetDevice(deviceId);
                          // Simple kernel that should take similar time on all devices
                          for (int i = 0; i < 10; ++i) {
                            multiGpuVectorAdd<<<grid, block, 0, stream>>>(
                                d_a[deviceId], d_a[deviceId], d_a[deviceId], N);
                          }
                        },
                        "load_balance_test")
                    .withLaunchConfig(grid, block)
                    .measure();

  ASSERT_TRUE(result.aggregatedStats.multiGpu.has_value());
  const auto& mgpu = result.aggregatedStats.multiGpu.value();

  // For identical workloads on similar GPUs, imbalance should be low
  EXPECT_LT(mgpu.loadImbalance, 1.5) << "Load imbalance should be minimal for identical workloads";

  // Check that all devices completed work
  for (const auto& devResult : result.perDevice) {
    EXPECT_GT(devResult.kernelTimeUs, 0.0)
        << "Device " << devResult.deviceId << " should have completed work";
  }

  // Cleanup
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaFree(d_a[dev]);
  }
}

/**
 * @brief P2P transfer bandwidth measurement
 *
 * Validates peer-to-peer transfer functionality and measures bandwidth
 * between GPU devices.
 *
 * @test P2PTransfers
 *
 * Validates:
 *  - P2P access enablement
 *  - P2P bandwidth measurement
 *  - P2P metrics collection
 *
 * Expected performance:
 *  - P2P bandwidth > 20 GB/s (NVLink) or > 10 GB/s (PCIe)
 */
PERF_GPU_TEST(GpuMultiDevice, P2PTransfers) {
  UB_PERF_GPU_GUARD(perf);

  const int deviceCount = getDeviceCount();
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires 2+ GPUs, found " << deviceCount;
  }

  if (!isP2PSupported(0, 1)) {
    GTEST_SKIP() << "P2P not supported between GPU 0 and GPU 1";
  }

  const size_t TRANSFER_SIZE = 64 * 1024 * 1024; // 64 MB

  float *d_src, *d_dst;
  cudaSetDevice(0);
  cudaMalloc(&d_src, TRANSFER_SIZE);
  cudaSetDevice(1);
  cudaMalloc(&d_dst, TRANSFER_SIZE);

  // Dummy kernel for framework compatibility
  auto dummyKernel = [](int, cudaStream_t) {};

  // Measure P2P bandwidth
  auto result = perf.cudaKernelMultiGpu(2, dummyKernel, "p2p_test")
                    .withP2PAccess()
                    .measureP2PBandwidth(0, 1, TRANSFER_SIZE)
                    .measure();

  ASSERT_TRUE(result.aggregatedStats.p2pProfile.has_value()) << "P2P profile should be available";

  const auto& p2p = result.aggregatedStats.p2pProfile.value();

  EXPECT_TRUE(p2p.accessEnabled) << "P2P access should be enabled";

  EXPECT_EQ(p2p.srcDevice, 0);
  EXPECT_EQ(p2p.dstDevice, 1);
  EXPECT_EQ(p2p.bytes, TRANSFER_SIZE);

  const double bandwidth = p2p.bandwidthGBs();
  EXPECT_GT(bandwidth, 5.0) << "P2P bandwidth should be > 5 GB/s (minimum for PCIe)";

  // Cleanup
  cudaSetDevice(0);
  cudaFree(d_src);
  cudaSetDevice(1);
  cudaFree(d_dst);
}

/**
 * @brief P2P vs host-mediated transfer comparison
 *
 * Moves the same number of bytes from a buffer on device 0 to a buffer on
 * device 1 twice: through pageable host memory (a copy to the host, then a
 * copy on to device 1, timed on the host clock from before the first copy
 * until device 1 has finished, CUDA API time included) and peer to peer (the
 * harness's copy, timed with events on a stream of device 0).
 *
 * @test P2PVsHost
 *
 * Validates:
 *  - Every CUDA call succeeds
 *  - P2P is faster than the host-mediated path
 *
 * Expected performance:
 *  - P2P bandwidth > 2x host-mediated (for NVLink)
 */
PERF_GPU_TEST(GpuMultiDevice, P2PVsHost) {
  UB_PERF_GPU_GUARD(perf);

  const int deviceCount = getDeviceCount();
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires 2+ GPUs, found " << deviceCount;
  }

  if (!isP2PSupported(0, 1)) {
    GTEST_SKIP() << "P2P not supported between GPU 0 and GPU 1";
  }

  const size_t SIZE = 32 * 1024 * 1024; // 32 MB

  std::vector<float> h_data(SIZE / sizeof(float));
  float* d_src = nullptr;
  float* d_dst = nullptr;

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_src, SIZE), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_data.data(), SIZE, cudaMemcpyHostToDevice), cudaSuccess);

  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, SIZE), cudaSuccess);

  // Host-mediated: device 0 to the host, then the host to device 1. A pageable
  // copy to the device can return before it lands, so device 1 is synchronized
  // before the clock stops.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  const auto HOST_START = std::chrono::steady_clock::now();
  ASSERT_EQ(cudaMemcpy(h_data.data(), d_src, SIZE, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_dst, h_data.data(), SIZE, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const double hostMediatedS =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - HOST_START).count();
  const double hostBandwidth = (SIZE / hostMediatedS) / 1e9;

  // Peer to peer: the harness's event-timed copy of SIZE bytes
  auto dummyKernel = [](int, cudaStream_t) {};
  auto result = perf.cudaKernelMultiGpu(2, dummyKernel, "p2p_comparison")
                    .withP2PAccess()
                    .measureP2PBandwidth(0, 1, SIZE)
                    .measure();

  ASSERT_TRUE(result.aggregatedStats.p2pProfile.has_value());
  const double p2pBandwidth = result.aggregatedStats.p2pProfile.value().bandwidthGBs();

  std::printf("[GpuMultiDevice.P2PVsHost] %zu bytes from device 0 to device 1: host-mediated "
              "%.2f GB/s (host clock), peer to peer %.2f GB/s (events)\n",
              SIZE, hostBandwidth, p2pBandwidth);

  EXPECT_GT(p2pBandwidth, hostBandwidth) << "P2P should be faster than host-mediated transfer";

  // Cleanup
  EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
  EXPECT_EQ(cudaFree(d_src), cudaSuccess);
  EXPECT_EQ(cudaSetDevice(1), cudaSuccess);
  EXPECT_EQ(cudaFree(d_dst), cudaSuccess);
}

/**
 * @brief Device selection with --gpu-device
 *
 * A GPU case measures on the device --gpu-device chooses (device 0 by
 * default), which it reads back as perf.gpuConfig().deviceId. The case checks
 * that this device is current, launches c = a + b into the case's stream and
 * compares c element by element. This project's rigs have one GPU each and
 * run it on device 0; another --gpu-device is not verified on them.
 *
 * @test DeviceSelection
 *
 * Validates:
 *  - The configured device is the current device
 *  - Every CUDA call and the launch succeed
 *  - The output is correct element by element
 *
 * Expected performance:
 *  - Execution on the configured device
 */
PERF_GPU_TEST(GpuMultiDevice, DeviceSelection) {
  UB_PERF_GPU_GUARD(perf);

  const int DEVICE = perf.gpuConfig().deviceId;
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  ASSERT_EQ(current, DEVICE) << "The case's device is not the current device";

  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  ASSERT_EQ(cudaMalloc(&d_a, SIZE), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_b, SIZE), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_c, SIZE), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), SIZE, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), SIZE, cudaMemcpyHostToDevice), cudaSuccess);
  // An element the kernel never writes reads 0, whatever the allocation held
  ASSERT_EQ(cudaMemset(d_c, 0, SIZE), cudaSuccess);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  auto result = perf.cudaKernel(
                        [&](cudaStream_t stream) {
                          multiGpuVectorAdd<<<grid, block, 0, stream>>>(d_a, d_b, d_c, N);
                        },
                        "device_selection")
                    .withLaunchConfig(grid, block)
                    .measure();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "The kernel launch failed";

  ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, SIZE, cudaMemcpyDeviceToHost), cudaSuccess);
  int firstWrong = -1;
  for (int i = 0; i < N && firstWrong < 0; ++i) {
    if (h_c[i] != 3.0f) {
      firstWrong = i;
    }
  }
  EXPECT_EQ(firstWrong, -1) << "c[" << firstWrong << "] = " << h_c[firstWrong] << ", expected 3";

  std::printf("[GpuMultiDevice.DeviceSelection] device %d (%s): %.3f us per launch\n", DEVICE,
              result.stats.deviceInfo.name.c_str(), result.kernelTimeUs);

  EXPECT_EQ(cudaFree(d_a), cudaSuccess);
  EXPECT_EQ(cudaFree(d_b), cudaSuccess);
  EXPECT_EQ(cudaFree(d_c), cudaSuccess);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
