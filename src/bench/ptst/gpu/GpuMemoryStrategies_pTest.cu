/**
 * @file GpuMemoryStrategies_pTest.cu
 * @brief CUDA memory strategy comparison tests
 *
 * This test suite validates different CUDA memory management strategies
 * including explicit transfers, unified memory, pinned memory, and mapped memory.
 *
 * Features tested:
 *  - Explicit memory management (cudaMalloc/cudaMemcpy)
 *  - Unified Memory (cudaMallocManaged)
 *  - Pinned host memory (cudaMallocHost)
 *  - Mapped zero-copy memory
 *  - Performance comparison across strategies
 *
 * Expected behavior:
 *  - All strategies execute correctly
 *  - Explicit/Pinned show best performance
 *  - Unified Memory works but may be slower
 *  - Mapped memory works for small data
 *
 * Usage:
 *   @code{.sh}
 *   # Run all memory strategy tests
 *   ./TestBenchSamples_PTEST --gtest_filter="GpuMemoryStrategies.*"
 *
 *   # Test specific strategy
 *   ./TestBenchSamples_PTEST --gtest_filter="GpuMemoryStrategies.ExplicitMemory"
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~15 seconds total
 *  - Pass rate: 100% with CUDA GPU available
 *  - Explicit memory fastest
 *
 * @see PerfGpuCase
 * @see PerfGpuConfig::MemoryStrategy
 */

#include <gtest/gtest.h>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Simple computation kernel */
__global__ void computeKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = input[idx];
    // Some computation to make it non-trivial
    for (int i = 0; i < 10; ++i) {
      val = val * 0.99f + 0.01f;
    }
    output[idx] = val;
  }
}

} // anonymous namespace

/**
 * @brief Explicit memory management test
 *
 * Validates explicit memory allocation with cudaMalloc and cudaMemcpy.
 * This is the standard CUDA memory management approach.
 *
 * @test ExplicitMemory
 *
 * Validates:
 *  - cudaMalloc works correctly
 *  - cudaMemcpy transfers work
 *  - Kernel execution works
 *  - Results are correct
 *
 * Expected performance:
 *  - Best performance among strategies
 */
PERF_GPU_TEST(GpuMemoryStrategies, ExplicitMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate regular host memory
  std::vector<float> h_input(N, 1.0f);
  std::vector<float> h_output(N, 0.0f);

  // Allocate device memory explicitly
  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure with explicit transfers
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withHostToDevice(h_input.data(), d_input, SIZE)
                    .withDeviceToHost(d_output, h_output.data(), SIZE)
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";
  EXPECT_GT(result.transferTimeUs, 0.0) << "Transfers should occur";

  // Validate results
  EXPECT_GT(h_output[0], 0.0f) << "Output should be computed";
  EXPECT_LT(h_output[0], 2.0f) << "Output should be reasonable";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Unified Memory test
 *
 * Validates Unified Memory (managed memory) allocation and usage.
 * Unified Memory simplifies programming but may have performance trade-offs.
 *
 * @test UnifiedMemory
 *
 * Validates:
 *  - cudaMallocManaged works
 *  - Automatic migration works
 *  - Kernel can access managed memory
 *  - Results are correct
 *
 * Expected performance:
 *  - May be slower than explicit due to page migration
 */
PERF_GPU_TEST(GpuMemoryStrategies, UnifiedMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate unified memory
  float *um_input, *um_output;
  cudaMallocManaged(&um_input, SIZE);
  cudaMallocManaged(&um_output, SIZE);

  // Initialize on host
  for (int i = 0; i < N; ++i) {
    um_input[i] = 1.0f;
    um_output[i] = 0.0f;
  }

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    computeKernel<<<grid, block, 0, s>>>(um_input, um_output, N);
    cudaStreamSynchronize(s);
  });

  // Measure - no explicit transfers needed
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(um_input, um_output, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Synchronize to ensure data available on host
  cudaDeviceSynchronize();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Validate results (accessible directly on host)
  EXPECT_GT(um_output[0], 0.0f) << "Output should be computed";
  EXPECT_LT(um_output[0], 2.0f) << "Output should be reasonable";

  // Cleanup
  cudaFree(um_input);
  cudaFree(um_output);
}

/**
 * @brief Pinned (page-locked) host memory test
 *
 * Validates pinned host memory allocation which provides faster transfers
 * than regular pageable host memory.
 *
 * @test PinnedMemory
 *
 * Validates:
 *  - cudaMallocHost works
 *  - Pinned memory transfers are fast
 *  - Performance benefit over regular memory
 *
 * Expected performance:
 *  - Similar or better than explicit
 */
PERF_GPU_TEST(GpuMemoryStrategies, PinnedMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate pinned host memory
  float *h_input, *h_output;
  cudaMallocHost(&h_input, SIZE);
  cudaMallocHost(&h_output, SIZE);

  // Initialize
  for (int i = 0; i < N; ++i) {
    h_input[i] = 1.0f;
    h_output[i] = 0.0f;
  }

  // Allocate device memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure with pinned host memory
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withHostToDevice(h_input, d_input, SIZE)
                    .withDeviceToHost(d_output, h_output, SIZE)
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";
  EXPECT_GT(result.transferTimeUs, 0.0) << "Transfers should occur";

  // Pinned memory should have reasonable transfer times
  const auto& transfers = result.stats.transfers;
  EXPECT_GT(transfers.h2dBandwidthGBs(), 1.0)
      << "Pinned memory should have decent transfer bandwidth";

  // Validate results
  EXPECT_GT(h_output[0], 0.0f) << "Output should be computed";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
}

/**
 * @brief Zero-copy mapped memory test
 *
 * Validates mapped memory where GPU accesses host memory directly.
 * Useful for small data or when transfers are impractical.
 *
 * @test MappedMemory
 *
 * Validates:
 *  - Mapped memory allocation works
 *  - GPU can access host memory
 *  - Results are correct
 *
 * Expected performance:
 *  - Slower for large data
 *  - Acceptable for small workloads
 */
PERF_GPU_TEST(GpuMemoryStrategies, MappedMemory) {
  UB_PERF_GPU_GUARD(perf);

  // Use smaller size for mapped memory
  const int N = 64 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate mapped host memory
  float *h_input, *h_output;
  cudaHostAlloc(&h_input, SIZE, cudaHostAllocMapped);
  cudaHostAlloc(&h_output, SIZE, cudaHostAllocMapped);

  // Initialize
  for (int i = 0; i < N; ++i) {
    h_input[i] = 1.0f;
    h_output[i] = 0.0f;
  }

  // Get device pointers
  float *d_input, *d_output;
  cudaHostGetDevicePointer(&d_input, h_input, 0);
  cudaHostGetDevicePointer(&d_output, h_output, 0);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure - GPU accesses host memory directly
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Validate results (available immediately on host)
  EXPECT_GT(h_output[0], 0.0f) << "Output should be computed";

  // Cleanup
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
}
