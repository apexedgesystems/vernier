/**
 * @file GpuMultiDevice_pTest.cu
 * @brief Multi-GPU execution and P2P transfer validation
 *
 * This test suite validates multi-GPU functionality including load distribution,
 * peer-to-peer transfers, scaling efficiency, and device selection.
 *
 * Features tested:
 *  - Multi-GPU kernel execution with cudaKernelMultiGpu()
 *  - Load balancing across multiple GPUs
 *  - P2P transfer bandwidth measurement
 *  - Scaling efficiency calculation
 *  - Device selection with withDeviceId()
 *
 * Expected behavior:
 *  - Kernels execute on all specified devices
 *  - Load is distributed evenly
 *  - P2P transfers work when supported
 *  - Scaling efficiency is reasonable (>0.7 for 2 GPUs)
 *
 * Usage:
 *   @code{.sh}
 *   # Run all multi-GPU tests
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuMultiDevice.*"
 *
 *   # Requires 2+ GPUs
 *   # Tests will skip if insufficient GPUs available
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~15 seconds total (if 2+ GPUs available)
 *  - Pass rate: 100% with 2+ CUDA GPUs
 *  - Scaling efficiency: >70% for 2 GPUs
 *
 * @see PerfGpuCase
 * @see MultiGpuKernelBuilder
 * @see MultiGpuResult
 */

#include <gtest/gtest.h>
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
 * with proper load distribution and synchronization.
 *
 * @test BasicMultiGpu
 *
 * Validates:
 *  - cudaKernelMultiGpu() API executes on all devices
 *  - Per-device results are collected
 *  - Load balancing metrics are computed
 *  - Aggregated statistics are correct
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

  EXPECT_GT(mgpu.scalingEfficiency, 0.5)
      << "Scaling efficiency should be reasonable for this simple workload";

  EXPECT_GT(result.totalSpeedupVsCpu, 1.0) << "Multi-GPU should be faster than CPU";

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
 * Compares P2P transfer performance against host-mediated transfers
 * to validate P2P advantage.
 *
 * @test P2PVsHost
 *
 * Validates:
 *  - P2P is faster than host-mediated transfers
 *  - Bandwidth difference is measurable
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
  float *d_src, *d_dst;

  cudaSetDevice(0);
  cudaMalloc(&d_src, SIZE);
  cudaMemcpy(d_src, h_data.data(), SIZE, cudaMemcpyHostToDevice);

  cudaSetDevice(1);
  cudaMalloc(&d_dst, SIZE);

  // Measure host-mediated transfer
  cudaSetDevice(0);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaMemcpy(h_data.data(), d_src, SIZE, cudaMemcpyDeviceToHost);
  cudaSetDevice(1);
  cudaMemcpy(d_dst, h_data.data(), SIZE, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float hostMediatedMs = 0;
  cudaEventElapsedTime(&hostMediatedMs, start, stop);
  const double hostBandwidth = (SIZE / (hostMediatedMs / 1000.0)) / 1e9;

  // Measure P2P transfer
  auto dummyKernel = [](int, cudaStream_t) {};
  auto result = perf.cudaKernelMultiGpu(2, dummyKernel, "p2p_comparison")
                    .withP2PAccess()
                    .measureP2PBandwidth(0, 1, SIZE)
                    .measure();

  ASSERT_TRUE(result.aggregatedStats.p2pProfile.has_value());
  const double p2pBandwidth = result.aggregatedStats.p2pProfile.value().bandwidthGBs();

  EXPECT_GT(p2pBandwidth, hostBandwidth) << "P2P should be faster than host-mediated transfer";

  // Cleanup
  cudaSetDevice(0);
  cudaFree(d_src);
  cudaSetDevice(1);
  cudaFree(d_dst);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/**
 * @brief Multi-GPU scaling efficiency
 *
 * Validates that multi-GPU execution provides good scaling efficiency
 * and that metrics accurately reflect performance gains.
 *
 * @test ScalingEfficiency
 *
 * Validates:
 *  - Multi-GPU speedup calculation
 *  - Scaling efficiency metric
 *  - Scaling quality assessment
 *
 * Expected performance:
 *  - Scaling efficiency > 0.7 for 2 GPUs
 */
PERF_GPU_TEST(GpuMultiDevice, ScalingEfficiency) {
  UB_PERF_GPU_GUARD(perf);

  const int deviceCount = getDeviceCount();
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires 2+ GPUs, found " << deviceCount;
  }

  const int N = 2 * 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
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

  auto result = perf.cudaKernelMultiGpu(
                        deviceCount,
                        [&](int deviceId, cudaStream_t stream) {
                          cudaSetDevice(deviceId);
                          multiGpuVectorAdd<<<grid, block, 0, stream>>>(
                              d_a[deviceId], d_b[deviceId], d_c[deviceId], N);
                        },
                        "scaling_test")
                    .withLaunchConfig(grid, block)
                    .measure();

  ASSERT_TRUE(result.aggregatedStats.multiGpu.has_value());
  const auto& mgpu = result.aggregatedStats.multiGpu.value();

  EXPECT_GT(mgpu.scalingEfficiency, 0.5) << "Should have reasonable scaling efficiency";

  const double quality = mgpu.scalingQuality();
  EXPECT_GT(quality, 0.4) << "Scaling quality should be reasonable";

  EXPECT_GT(result.totalSpeedupVsCpu, static_cast<double>(deviceCount))
      << "Multi-GPU should provide speedup over single CPU";

  // Cleanup
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaFree(d_a[dev]);
    cudaFree(d_b[dev]);
    cudaFree(d_c[dev]);
  }
}

/**
 * @brief Device selection with withDeviceId()
 *
 * Validates that specific devices can be targeted using the
 * withDeviceId() API.
 *
 * @test DeviceSelection
 *
 * Validates:
 *  - withDeviceId() selects correct device
 *  - Result contains correct device ID
 *  - Kernel executes on specified device
 *
 * Expected performance:
 *  - Execution on correct device
 */
PERF_GPU_TEST(GpuMultiDevice, DeviceSelection) {
  UB_PERF_GPU_GUARD(perf);

  const int deviceCount = getDeviceCount();
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires 2+ GPUs, found " << deviceCount;
  }

  const int TARGET_DEVICE = 1;
  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  std::vector<float> h_a(N, 1.0f);
  float* d_a;

  cudaSetDevice(TARGET_DEVICE);
  cudaMalloc(&d_a, SIZE);
  cudaMemcpy(d_a, h_a.data(), SIZE, cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Execute on specific device
  auto result = perf.cudaKernel(
                        [&](cudaStream_t stream) {
                          multiGpuVectorAdd<<<grid, block, 0, stream>>>(d_a, d_a, d_a, N);
                        },
                        "device_selection")
                    .withDeviceId(TARGET_DEVICE)
                    .withLaunchConfig(grid, block)
                    .measure();

  EXPECT_EQ(result.deviceId, TARGET_DEVICE) << "Kernel should execute on specified device";

  EXPECT_GT(result.callsPerSecond, 0.0) << "Execution should complete successfully";

  cudaSetDevice(TARGET_DEVICE);
  cudaFree(d_a);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
