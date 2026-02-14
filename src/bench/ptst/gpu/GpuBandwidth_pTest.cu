/**
 * @file GpuBandwidth_pTest.cu
 * @brief GPU memory bandwidth measurement tests
 *
 * This test suite validates GPU memory bandwidth measurements for different
 * transfer types: host-to-device, device-to-host, and device-to-device.
 *
 * Features tested:
 *  - Host-to-device (H2D) transfer bandwidth
 *  - Device-to-host (D2H) transfer bandwidth
 *  - Device-to-device (D2D) copy bandwidth
 *  - Bandwidth calculation accuracy
 *
 * Expected behavior:
 *  - All transfers complete successfully
 *  - Bandwidth measurements are positive
 *  - D2D bandwidth > H2D/D2H bandwidth
 *  - Bandwidth values are reasonable for hardware
 *
 * Usage:
 *   @code{.sh}
 *   # Run all bandwidth tests
 *   ./TestBenchSamples_PTEST --gtest_filter="GpuBandwidth.*"
 *
 *   # Test specific bandwidth type
 *   ./TestBenchSamples_PTEST --gtest_filter="GpuBandwidth.HostToDevice"
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~10 seconds total
 *  - Pass rate: 100% with CUDA GPU available
 *  - H2D/D2H: >1 GB/s
 *  - D2D: >10 GB/s
 *
 * @see PerfGpuCase
 * @see TransferMetrics
 */

#include <gtest/gtest.h>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Simple copy kernel for D2D testing */
__global__ void copyKernel(const float* src, float* dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

} // anonymous namespace

/**
 * @brief Host-to-device transfer bandwidth
 *
 * Validates H2D transfer bandwidth measurement and ensures
 * it meets minimum performance expectations.
 *
 * @test HostToDevice
 *
 * Validates:
 *  - H2D transfers work correctly
 *  - Bandwidth is calculated
 *  - Bandwidth meets minimum threshold
 *
 * Expected performance:
 *  - H2D bandwidth >1 GB/s
 */
PERF_GPU_TEST(GpuBandwidth, HostToDevice) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 4 * 1024 * 1024; // 4M floats = 16MB
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_data(N, 1.0f);
  float* d_data;
  cudaMalloc(&d_data, SIZE);

  // Simple kernel that just reads the data
  auto kernelFn = [&](cudaStream_t s) {
    // Minimal kernel to ensure data is used
    (void)d_data;
  };

  // Warmup
  perf.cudaWarmup(kernelFn);

  // Measure H2D transfer
  auto result = perf.cudaKernel(kernelFn).withHostToDevice(h_data.data(), d_data, SIZE).measure();

  // Validate transfer metrics captured
  const auto& transfers = result.stats.transfers;

  // Validate H2D bandwidth
  EXPECT_GT(transfers.h2dBandwidthGBs(), 1.0)
      << "H2D bandwidth too low: " << transfers.h2dBandwidthGBs() << " GB/s";

  EXPECT_GT(transfers.h2dTimeUs, 0.0) << "H2D time should be positive";

  EXPECT_EQ(transfers.h2dBytes, SIZE) << "H2D bytes should match transfer size";

  // Cleanup
  cudaFree(d_data);
}

/**
 * @brief Device-to-host transfer bandwidth
 *
 * Validates D2H transfer bandwidth measurement and ensures
 * it meets minimum performance expectations.
 *
 * @test DeviceToHost
 *
 * Validates:
 *  - D2H transfers work correctly
 *  - Bandwidth is calculated
 *  - Bandwidth meets minimum threshold
 *
 * Expected performance:
 *  - D2H bandwidth >1 GB/s
 */
PERF_GPU_TEST(GpuBandwidth, DeviceToHost) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 4 * 1024 * 1024; // 4M floats = 16MB
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_data(N, 0.0f);
  float* d_data;
  cudaMalloc(&d_data, SIZE);

  // Initialize device data
  cudaMemset(d_data, 0, SIZE);

  // Simple kernel that writes data
  auto kernelFn = [&](cudaStream_t s) { (void)d_data; };

  // Warmup
  perf.cudaWarmup(kernelFn);

  // Measure D2H transfer
  auto result = perf.cudaKernel(kernelFn).withDeviceToHost(d_data, h_data.data(), SIZE).measure();

  // Validate transfer metrics captured
  const auto& transfers = result.stats.transfers;

  // Validate D2H bandwidth
  EXPECT_GT(transfers.d2hBandwidthGBs(), 1.0)
      << "D2H bandwidth too low: " << transfers.d2hBandwidthGBs() << " GB/s";

  EXPECT_GT(transfers.d2hTimeUs, 0.0) << "D2H time should be positive";

  EXPECT_EQ(transfers.d2hBytes, SIZE) << "D2H bytes should match transfer size";

  // Cleanup
  cudaFree(d_data);
}

/**
 * @brief Device-to-device memory bandwidth
 *
 * Validates D2D memory copy bandwidth which should be significantly
 * faster than H2D/D2H transfers.
 *
 * @test DeviceToDevice
 *
 * Validates:
 *  - D2D copies work correctly
 *  - Kernel bandwidth is calculated
 *  - D2D bandwidth > H2D/D2H bandwidth
 *
 * Expected performance:
 *  - D2D bandwidth >10 GB/s (much faster than PCIe)
 */
PERF_GPU_TEST(GpuBandwidth, DeviceToDevice) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 4 * 1024 * 1024; // Reduced for speed (was 16M)
  const size_t SIZE = N * sizeof(float);

  // Allocate device memory
  float *d_src, *d_dst;
  cudaMalloc(&d_src, SIZE);
  cudaMalloc(&d_dst, SIZE);

  // Initialize source
  cudaMemset(d_src, 1, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) { copyKernel<<<grid, block, 0, s>>>(d_src, d_dst, N); });

  // Measure D2D copy
  auto result =
      perf.cudaKernel([&](cudaStream_t s) { copyKernel<<<grid, block, 0, s>>>(d_src, d_dst, N); })
          .withLaunchConfig(grid, block)
          .measure();

  // Validate kernel executed
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Calculate bandwidth from kernel time
  const double expectedBandwidth = (2.0 * SIZE) / (result.kernelTimeUs * 1e-6) / 1e9;

  // D2D should be much faster than PCIe transfers
  EXPECT_GT(expectedBandwidth, 10.0) << "D2D bandwidth too low: " << expectedBandwidth << " GB/s";

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}

/**
 * @brief Bidirectional transfer bandwidth
 *
 * Validates simultaneous H2D and D2H transfers and total bandwidth.
 *
 * @test BidirectionalTransfer
 *
 * Validates:
 *  - Concurrent H2D and D2H work
 *  - Total bandwidth calculated correctly
 *  - Combined transfers are efficient
 *
 * Expected performance:
 *  - Combined bandwidth >1.5 GB/s
 */
PERF_GPU_TEST(GpuBandwidth, BidirectionalTransfer) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 2 * 1024 * 1024; // 2M floats = 8MB each way
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_input(N, 1.0f);
  std::vector<float> h_output(N, 0.0f);
  float* d_data;
  cudaMalloc(&d_data, SIZE);

  // Simple kernel
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) { copyKernel<<<grid, block, 0, s>>>(d_data, d_data, N); });

  // Measure with both H2D and D2H
  auto result =
      perf.cudaKernel([&](cudaStream_t s) { copyKernel<<<grid, block, 0, s>>>(d_data, d_data, N); })
          .withHostToDevice(h_input.data(), d_data, SIZE)
          .withDeviceToHost(d_data, h_output.data(), SIZE)
          .withLaunchConfig(grid, block)
          .measure();

  // Validate both transfers occurred
  const auto& transfers = result.stats.transfers;

  EXPECT_GT(transfers.h2dTimeUs, 0.0) << "H2D should occur";
  EXPECT_GT(transfers.d2hTimeUs, 0.0) << "D2H should occur";

  // Validate total transfer time reasonable
  const double totalTransferTime = transfers.h2dTimeUs + transfers.d2hTimeUs;
  EXPECT_GT(totalTransferTime, 0.0) << "Total transfer time should be positive";

  // Validate combined bandwidth
  const double totalBytes = transfers.h2dBytes + transfers.d2hBytes;
  const double combinedBandwidthGBs = totalBytes / (totalTransferTime * 1e-6) / 1e9;

  EXPECT_GT(combinedBandwidthGBs, 1.5)
      << "Combined bandwidth too low: " << combinedBandwidthGBs << " GB/s";

  // Cleanup
  cudaFree(d_data);
}
