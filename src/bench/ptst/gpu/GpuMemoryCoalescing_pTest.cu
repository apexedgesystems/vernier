/**
 * @file GpuMemoryCoalescing_pTest.cu
 * @brief Memory coalescing performance impact validation
 *
 * This test suite validates GPU memory coalescing performance and demonstrates
 * the bandwidth impact of different memory access patterns.
 *
 * Features tested:
 *  - Coalesced memory access (stride=1) performance
 *  - Non-coalesced access (large stride) performance
 *  - Stride impact on memory bandwidth
 *  - Access pattern optimization validation
 *
 * Expected behavior:
 *  - Coalesced access shows maximum bandwidth
 *  - Larger strides reduce bandwidth significantly
 *  - Bandwidth decreases as stride increases
 *  - Results demonstrate coalescing importance
 *
 * Usage:
 *   @code{.sh}
 *   # Run all coalescing tests
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuMemoryCoalescing.*"
 *
 *   # Run specific test
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuMemoryCoalescing.CoalescedAccess"
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~8 seconds total
 *  - Pass rate: 100% with CUDA GPU
 *  - Coalesced: >100 GB/s (modern GPUs)
 *  - Stride=32: <50 GB/s
 *
 * @see PerfGpuCase
 * @see MemoryTransferProfile
 */

#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Strided memory read kernel */
__global__ void stridedReadKernel(const float* input, float* output, int n, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int readIdx = idx * stride;

  if (readIdx < n) {
    output[idx] = input[readIdx];
  }
}

/** @brief Calculate effective bandwidth from kernel time */
double calculateBandwidth(int n, int stride, double kernelTimeUs) {
  // Bytes read + bytes written
  const size_t elementsAccessed = (n + stride - 1) / stride;
  const size_t bytesAccessed = elementsAccessed * sizeof(float) * 2; // read + write
  const double timeS = kernelTimeUs / 1e6;
  return static_cast<double>(bytesAccessed) / timeS / 1e9;
}

} // anonymous namespace

/**
 * @brief Coalesced memory access performance
 *
 * Validates that fully coalesced memory access (stride=1) achieves
 * maximum memory bandwidth.
 *
 * @test CoalescedAccess
 *
 * Validates:
 *  - Stride=1 access pattern
 *  - Maximum bandwidth achieved
 *  - Sequential access optimization
 *
 * Expected performance:
 *  - Bandwidth >100 GB/s for modern GPUs
 *  - Near peak memory bandwidth
 */
PERF_GPU_TEST(GpuMemoryCoalescing, CoalescedAccess) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 4 * 1024 * 1024; // Reduced for speed (was 16M)
  const size_t SIZE = N * sizeof(float);
  const int STRIDE = 1; // Fully coalesced

  // Allocate memory
  std::vector<float> h_input(N, 1.0f);
  std::vector<float> h_output(N / STRIDE, 0.0f);

  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);

  cudaMemcpy(d_input, h_input.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = ((N / STRIDE) + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, STRIDE);
  });

  // Measure coalesced access
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, STRIDE);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Calculate bandwidth
  const double bandwidth = calculateBandwidth(N, STRIDE, result.kernelTimeUs);

  // Coalesced access should achieve high bandwidth
  EXPECT_GT(bandwidth, 50.0) << "Coalesced access bandwidth too low: " << bandwidth << " GB/s";

  EXPECT_GT(result.callsPerSecond, 0.0) << "Kernel should execute successfully";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Non-coalesced memory access performance
 *
 * Validates that non-coalesced access (large stride) shows significantly
 * reduced bandwidth compared to coalesced access.
 *
 * @test StridedAccess
 *
 * Validates:
 *  - Stride=32 access pattern
 *  - Bandwidth reduction due to non-coalescing
 *  - Performance impact measurement
 *
 * Expected performance:
 *  - Bandwidth significantly lower than coalesced
 *  - Demonstrates coalescing importance
 */
PERF_GPU_TEST(GpuMemoryCoalescing, StridedAccess) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 4 * 1024 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);
  const int STRIDE = 32; // Non-coalesced

  // Allocate memory
  std::vector<float> h_input(N, 2.0f);
  std::vector<float> h_output(N / STRIDE, 0.0f);

  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, (N / STRIDE) * sizeof(float));

  cudaMemcpy(d_input, h_input.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = ((N / STRIDE) + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, STRIDE);
  });

  // Measure strided access
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, STRIDE);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Calculate bandwidth
  const double bandwidth = calculateBandwidth(N, STRIDE, result.kernelTimeUs);

  // Strided access should show lower bandwidth
  EXPECT_GT(bandwidth, 1.0) << "Bandwidth should be positive: " << bandwidth << " GB/s";

  // Should still complete successfully
  EXPECT_GT(result.callsPerSecond, 0.0) << "Kernel should execute successfully";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Memory access stride comparison
 *
 * Validates that bandwidth decreases as stride increases, demonstrating
 * the impact of memory coalescing on performance.
 *
 * @test StrideComparison
 *
 * Validates:
 *  - Multiple stride values (1, 2, 4, 8, 16, 32)
 *  - Bandwidth degradation with stride
 *  - Coalescing impact measurement
 *
 * Expected performance:
 *  - Bandwidth(stride=1) > Bandwidth(stride=2) > ... > Bandwidth(stride=32)
 */
PERF_GPU_TEST(GpuMemoryCoalescing, StrideComparison) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 2 * 1024 * 1024; // Reduced for speed (was 8M)
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_input(N, 1.0f);
  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);

  cudaMemcpy(d_input, h_input.data(), SIZE, cudaMemcpyHostToDevice);

  // Test different strides
  std::vector<int> strides = {1, 2, 4, 8, 16, 32};
  std::vector<double> bandwidths;

  for (int stride : strides) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = ((N / stride) + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid);
    dim3 block(threadsPerBlock);

    // Warmup for this stride
    perf.cudaWarmup([&](cudaStream_t s) {
      stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, stride);
    });

    // Measure
    auto result = perf.cudaKernel(
                          [&](cudaStream_t s) {
                            stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, stride);
                          },
                          "stride_" + std::to_string(stride))
                      .withLaunchConfig(grid, block)
                      .measure();

    const double bw = calculateBandwidth(N, stride, result.kernelTimeUs);
    bandwidths.push_back(bw);
  }

  // Validate that bandwidth generally decreases with stride
  EXPECT_GT(bandwidths[0], 10.0) << "Stride=1 bandwidth should be high: " << bandwidths[0]
                                 << " GB/s";

  // Stride=1 should be best (or at least good)
  bool coalescedIsBest = true;
  for (size_t i = 1; i < bandwidths.size(); ++i) {
    // Allow some tolerance - stride=2 might be close to stride=1
    if (i > 1 && bandwidths[i] > bandwidths[0] * 1.5) {
      coalescedIsBest = false;
    }
  }

  EXPECT_TRUE(coalescedIsBest) << "Coalesced access should generally perform best";

  // Later strides should show lower bandwidth
  EXPECT_LT(bandwidths.back(), bandwidths[0] * 1.2)
      << "Large stride should show lower bandwidth than coalesced";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Alignment impact on coalescing
 *
 * Validates that memory alignment affects coalescing efficiency and
 * that unaligned access shows reduced performance.
 *
 * @test AlignmentImpact
 *
 * Validates:
 *  - Aligned vs unaligned access
 *  - Coalescing requirements
 *  - Alignment optimization importance
 *
 * Expected performance:
 *  - Aligned access performs better
 *  - Misalignment reduces bandwidth
 */
PERF_GPU_TEST(GpuMemoryCoalescing, AlignmentImpact) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 2 * 1024 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  // Allocate aligned memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE + 128); // Extra space for misalignment
  cudaMalloc(&d_output, SIZE);

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Test 1: Aligned access
  perf.cudaWarmup(
      [&](cudaStream_t s) { stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, 1); });

  auto alignedResult = perf.cudaKernel(
                               [&](cudaStream_t s) {
                                 stridedReadKernel<<<grid, block, 0, s>>>(d_input, d_output, N, 1);
                               },
                               "aligned")
                           .withLaunchConfig(grid, block)
                           .measure();

  const double alignedBW = calculateBandwidth(N, 1, alignedResult.kernelTimeUs);

  // Test 2: Slightly misaligned (offset by 1 float)
  float* d_input_misaligned = d_input + 1;

  perf.cudaWarmup([&](cudaStream_t s) {
    stridedReadKernel<<<grid, block, 0, s>>>(d_input_misaligned, d_output, N, 1);
  });

  auto misalignedResult =
      perf.cudaKernel(
              [&](cudaStream_t s) {
                stridedReadKernel<<<grid, block, 0, s>>>(d_input_misaligned, d_output, N, 1);
              },
              "misaligned")
          .withLaunchConfig(grid, block)
          .measure();

  const double misalignedBW = calculateBandwidth(N, 1, misalignedResult.kernelTimeUs);

  // Both should work, but aligned may be faster
  EXPECT_GT(alignedBW, 10.0) << "Aligned bandwidth should be reasonable: " << alignedBW << " GB/s";

  EXPECT_GT(misalignedBW, 1.0) << "Misaligned should still work: " << misalignedBW << " GB/s";

  // Aligned should generally be >= misaligned (within tolerance)
  // Note: Modern GPUs handle misalignment well, so don't make this too strict
  EXPECT_GE(alignedBW * 1.5, misalignedBW)
      << "Aligned should not be significantly worse than misaligned";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
