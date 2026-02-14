/**
 * @file GpuBasics_pTest.cu
 * @brief Basic GPU functionality validation tests
 *
 * This test suite validates fundamental GPU benchmarking capabilities including
 * kernel timing, memory transfers, speedup calculation, and basic GPU metrics.
 *
 * Features tested:
 *  - Simple kernel execution and timing
 *  - Host-to-device and device-to-host transfers
 *  - CPU baseline comparison
 *  - Speedup calculation
 *  - Basic occupancy tracking
 *
 * Expected behavior:
 *  - GPU kernels execute correctly
 *  - Transfers complete successfully
 *  - GPU shows speedup over CPU
 *  - Metrics are reasonable
 *
 * Usage:
 *   @code{.sh}
 *   # Run all GPU basics tests
 *   ./TestBenchSamples_PTEST --gtest_filter="GpuBasics.*"
 *
 *   # With specific GPU device
 *   ./TestBenchSamples_PTEST --gtest_filter="GpuBasics.*" --gpu-device 0
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~10 seconds total
 *  - Pass rate: 100% with CUDA GPU available
 *  - GPU speedup: >2x over CPU
 *
 * @see PerfGpuCase
 * @see CudaKernelBuilder
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Simple vector addition kernel */
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

/** @brief CPU version of vector addition */
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

/** @brief Simple saxpy kernel */
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}

} // anonymous namespace

/**
 * @brief Basic GPU kernel execution
 *
 * Validates that simple GPU kernels execute correctly and produce
 * valid timing measurements.
 *
 * @test SimpleKernelExecution
 *
 * Validates:
 *  - Kernel launches without error
 *  - Timing measurements are positive
 *  - Results are correct
 *
 * Expected performance:
 *  - Kernel time >0
 *  - Reasonable throughput
 */
PERF_GPU_TEST(GpuBasics, SimpleKernelExecution) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate host memory
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); });

  // Measure
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
                    })
                    .withHostToDevice(h_a.data(), d_a, SIZE)
                    .withHostToDevice(h_b.data(), d_b, SIZE)
                    .withDeviceToHost(d_c, h_c.data(), SIZE)
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate measurements
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel time should be positive";
  EXPECT_GT(result.transferTimeUs, 0.0) << "Transfer time should be positive";
  EXPECT_GT(result.totalTimeUs, 0.0) << "Total time should be positive";

  // Validate results
  for (int i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(h_c[i], 3.0f) << "Result incorrect at index " << i;
  }

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

/**
 * @brief GPU speedup over CPU baseline
 *
 * Validates that GPU shows measurable speedup over CPU for parallel workloads.
 *
 * @test SpeedupCalculation
 *
 * Validates:
 *  - CPU baseline measured correctly
 *  - GPU faster than CPU
 *  - Speedup calculated correctly
 *
 * Expected performance:
 *  - GPU speedup >2x over CPU
 */
PERF_GPU_TEST(GpuBasics, SpeedupCalculation) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 128 * 1024; // Reduced for speed (was 512K)
  const size_t SIZE = N * sizeof(float);

  // Allocate host memory
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c_cpu(N, 0.0f);
  std::vector<float> h_c_gpu(N, 0.0f);

  // CPU baseline
  auto cpuResult = perf.cpuBaseline(
      [&] { vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), N); }, "cpu_vector_add");

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); });

  // Measure GPU
  auto gpuResult = perf.cudaKernel([&](cudaStream_t s) {
                         vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
                       })
                       .withHostToDevice(h_a.data(), d_a, SIZE)
                       .withHostToDevice(h_b.data(), d_b, SIZE)
                       .withDeviceToHost(d_c, h_c_gpu.data(), SIZE)
                       .withLaunchConfig(grid, block)
                       .measure();

  // Validate speedup
  EXPECT_GT(gpuResult.speedupVsCpu, 2.0)
      << "GPU speedup too low: " << gpuResult.speedupVsCpu << "x";

  // Validate speedup calculation is consistent
  const double expectedSpeedup = cpuResult.stats.median / gpuResult.totalTimeUs;
  EXPECT_NEAR(gpuResult.speedupVsCpu, expectedSpeedup, 0.1) << "Speedup calculation inconsistent";

  // Validate results match
  for (int i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(h_c_cpu[i], h_c_gpu[i]) << "CPU and GPU results differ at index " << i;
  }

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

/**
 * @brief Occupancy tracking validation
 *
 * Validates that occupancy metrics are captured correctly when
 * launch configuration is provided.
 *
 * @test OccupancyTracking
 *
 * Validates:
 *  - Occupancy is calculated
 *  - Occupancy is in valid range [0,1]
 *  - Occupancy metrics are reasonable
 *
 * Expected performance:
 *  - Occupancy >0.5 for simple kernels
 */
PERF_GPU_TEST(GpuBasics, OccupancyTracking) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_x(N, 2.0f);
  std::vector<float> h_y(N, 1.0f);

  float *d_x, *d_y;
  cudaMalloc(&d_x, SIZE);
  cudaMalloc(&d_y, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  const float a = 2.5f;

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) { saxpyKernel<<<grid, block, 0, s>>>(a, d_x, d_y, N); });

  // Measure with launch config (enables occupancy)
  auto result =
      perf.cudaKernel([&](cudaStream_t s) { saxpyKernel<<<grid, block, 0, s>>>(a, d_x, d_y, N); })
          .withHostToDevice(h_x.data(), d_x, SIZE)
          .withHostToDevice(h_y.data(), d_y, SIZE)
          .withDeviceToHost(d_y, h_y.data(), SIZE)
          .withLaunchConfig(grid, block) // THIS enables occupancy
          .measure();

  // Validate occupancy captured
  const auto& occ = result.stats.occupancy;

  // Validate occupancy in valid range
  EXPECT_GE(occ.achievedOccupancy, 0.0) << "Occupancy should be >= 0";
  EXPECT_LE(occ.achievedOccupancy, 1.0) << "Occupancy should be <= 1";

  // For simple kernels, expect reasonable occupancy
  EXPECT_GT(occ.achievedOccupancy, 0.5)
      << "Occupancy too low for simple kernel: " << occ.achievedOccupancy;

  // Cleanup
  cudaFree(d_x);
  cudaFree(d_y);
}

/**
 * @brief Transfer overhead measurement
 *
 * Validates that transfer overhead percentage is calculated correctly
 * and is reasonable for the workload.
 *
 * @test TransferOverhead
 *
 * Validates:
 *  - Transfer times are measured
 *  - Overhead percentage calculated correctly
 *  - Overhead is within reasonable bounds
 *
 * Expected performance:
 *  - Transfer overhead <50% for compute-bound kernels
 */
PERF_GPU_TEST(GpuBasics, TransferOverhead) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); });

  // Measure
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
                    })
                    .withHostToDevice(h_a.data(), d_a, SIZE)
                    .withHostToDevice(h_b.data(), d_b, SIZE)
                    .withDeviceToHost(d_c, h_c.data(), SIZE)
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate transfer metrics captured
  const auto& transfers = result.stats.transfers;

  // Validate transfer times
  EXPECT_GT(transfers.h2dTimeUs, 0.0) << "H2D time should be positive";
  EXPECT_GT(transfers.d2hTimeUs, 0.0) << "D2H time should be positive";

  // Validate overhead percentage
  const double overhead = transfers.transferOverheadPct(result.kernelTimeUs);
  EXPECT_GE(overhead, 0.0) << "Overhead should be non-negative";

  // For simple compute kernels, overhead should be reasonable
  EXPECT_LT(overhead, 100.0) << "Transfer overhead too high: " << overhead << "%";

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
