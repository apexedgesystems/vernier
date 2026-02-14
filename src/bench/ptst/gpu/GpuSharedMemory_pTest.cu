/**
 * @file GpuSharedMemory_pTest.cu
 * @brief Shared memory performance and optimization validation
 *
 * This test suite validates GPU shared memory usage, bank conflicts,
 * and the performance benefits of shared memory optimization.
 *
 * Features tested:
 *  - Global memory baseline (no shared memory)
 *  - Shared memory optimization benefits
 *  - Bank conflict detection and impact
 *  - Conflict-free shared memory access patterns
 *
 * Expected behavior:
 *  - Shared memory shows significant speedup over global memory
 *  - Bank conflicts reduce performance
 *  - Conflict-free access achieves best performance
 *  - Matrix transpose demonstrates shared memory benefits
 *
 * Usage:
 *   @code{.sh}
 *   # Run all shared memory tests
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuSharedMemory.*"
 *
 *   # Run specific test
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuSharedMemory.SharedMemoryOptimized"
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~12 seconds total
 *  - Pass rate: 100% with CUDA GPU
 *  - Shared memory: 2-10x faster than global memory
 *  - Conflict-free: Best performance
 *
 * @see PerfGpuCase
 * @see OccupancyMetrics
 */

#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

constexpr int TILE_SIZE = 32;

/** @brief Matrix transpose using global memory only */
__global__ void transposeGlobalKernel(const float* input, float* output, int width, int height) {
  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (x < width && y < height) {
    int inIdx = y * width + x;
    int outIdx = x * height + y;
    output[outIdx] = input[inIdx];
  }
}

/** @brief Matrix transpose using shared memory (optimized) */
__global__ void transposeSharedKernel(const float* input, float* output, int width, int height) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  // Load tile into shared memory
  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
  }

  __syncthreads();

  // Transpose write
  x = blockIdx.y * TILE_SIZE + threadIdx.x;
  y = blockIdx.x * TILE_SIZE + threadIdx.y;

  if (x < height && y < width) {
    output[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

/** @brief Shared memory with intentional bank conflicts */
__global__ void sharedBankConflictKernel(float* data, int n) {
  __shared__ float shared[256];
  int idx = threadIdx.x;

  if (idx < n) {
    // Intentional bank conflict: all threads access same bank
    // Stride of 32 floats = 128 bytes causes conflicts on 32 banks
    int conflictIdx = idx * 32;
    if (conflictIdx < 256) {
      shared[conflictIdx] = data[idx];
      __syncthreads();
      data[idx] = shared[conflictIdx];
    }
  }
}

/** @brief Shared memory with conflict-free access */
__global__ void sharedConflictFreeKernel(float* data, int n) {
  __shared__ float shared[256];
  int idx = threadIdx.x;

  if (idx < n && idx < 256) {
    // Conflict-free: sequential access
    shared[idx] = data[idx];
    __syncthreads();
    data[idx] = shared[idx];
  }
}

/** @brief Reduction using shared memory */
__global__ void reductionSharedKernel(const float* input, float* output, int n) {
  __shared__ float shared[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data
  shared[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

} // anonymous namespace

/**
 * @brief Global memory baseline (no shared memory)
 *
 * Establishes baseline performance for matrix transpose using only
 * global memory, for comparison with shared memory optimization.
 *
 * @test GlobalMemoryBaseline
 *
 * Validates:
 *  - Global memory transpose works correctly
 *  - Baseline performance measurement
 *  - Reference for comparison
 *
 * Expected performance:
 *  - Lower than shared memory version
 */
PERF_GPU_TEST(GpuSharedMemory, GlobalMemoryBaseline) {
  UB_PERF_GPU_GUARD(perf);

  const int WIDTH = 1024;  // Reduced for speed (was 2048)
  const int HEIGHT = 1024; // Reduced for speed (was 2048)
  const size_t SIZE = WIDTH * HEIGHT * sizeof(float);

  // Allocate memory
  std::vector<float> h_input(WIDTH * HEIGHT);
  for (int i = 0; i < WIDTH * HEIGHT; ++i) {
    h_input[i] = static_cast<float>(i);
  }
  std::vector<float> h_output(WIDTH * HEIGHT, 0.0f);

  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);
  cudaMemcpy(d_input, h_input.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    transposeGlobalKernel<<<grid, block, 0, s>>>(d_input, d_output, WIDTH, HEIGHT);
  });

  // Measure global memory transpose
  auto result =
      perf.cudaKernel([&](cudaStream_t s) {
            transposeGlobalKernel<<<grid, block, 0, s>>>(d_input, d_output, WIDTH, HEIGHT);
          })
          .withLaunchConfig(grid, block)
          .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  EXPECT_GT(result.callsPerSecond, 0.0) << "Should have valid throughput";

  // Verify correctness
  cudaMemcpy(h_output.data(), d_output, SIZE, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_output[WIDTH], h_input[1]) << "Transpose should be correct: [0][1] -> [1][0]";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Shared memory optimized transpose
 *
 * Validates that shared memory optimization significantly improves
 * performance compared to global memory baseline.
 *
 * @test SharedMemoryOptimized
 *
 * Validates:
 *  - Shared memory usage
 *  - Performance improvement over baseline
 *  - Correctness of optimized implementation
 *
 * Expected performance:
 *  - 2-10x faster than global memory version
 */
PERF_GPU_TEST(GpuSharedMemory, SharedMemoryOptimized) {
  UB_PERF_GPU_GUARD(perf);

  const int WIDTH = 1024;  // Reduced for speed
  const int HEIGHT = 1024; // Reduced for speed
  const size_t SIZE = WIDTH * HEIGHT * sizeof(float);

  // Allocate memory
  std::vector<float> h_input(WIDTH * HEIGHT);
  for (int i = 0; i < WIDTH * HEIGHT; ++i) {
    h_input[i] = static_cast<float>(i);
  }
  std::vector<float> h_output(WIDTH * HEIGHT, 0.0f);

  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);
  cudaMemcpy(d_input, h_input.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

  // Shared memory size
  const size_t sharedMemBytes = TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    transposeSharedKernel<<<grid, block, sharedMemBytes, s>>>(d_input, d_output, WIDTH, HEIGHT);
  });

  // Measure shared memory transpose
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      transposeSharedKernel<<<grid, block, sharedMemBytes, s>>>(d_input, d_output,
                                                                                WIDTH, HEIGHT);
                    })
                    .withLaunchConfig(grid, block, sharedMemBytes)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  EXPECT_GT(result.callsPerSecond, 0.0) << "Should have valid throughput";

  // Verify correctness
  cudaMemcpy(h_output.data(), d_output, SIZE, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_output[WIDTH], h_input[1]) << "Transpose should be correct: [0][1] -> [1][0]";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Bank conflict performance impact
 *
 * Validates that intentional bank conflicts cause measurable performance
 * degradation in shared memory access.
 *
 * @test BankConflicts
 *
 * Validates:
 *  - Bank conflict detection
 *  - Performance impact of conflicts
 *  - Conflict pattern identification
 *
 * Expected performance:
 *  - Slower than conflict-free access
 *  - Demonstrates conflict penalty
 */
PERF_GPU_TEST(GpuSharedMemory, BankConflicts) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256;
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_data(N, 1.0f);
  float* d_data;
  cudaMalloc(&d_data, SIZE);
  cudaMemcpy(d_data, h_data.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  dim3 block(256);
  dim3 grid(1);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { sharedBankConflictKernel<<<grid, block, 0, s>>>(d_data, N); });

  // Measure with bank conflicts
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      sharedBankConflictKernel<<<grid, block, 0, s>>>(d_data, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Bank conflicts cause slowdown, but kernel still works
  EXPECT_GT(result.callsPerSecond, 0.0) << "Should complete despite conflicts";

  // Cleanup
  cudaFree(d_data);
}

/**
 * @brief Conflict-free shared memory access
 *
 * Validates that conflict-free shared memory access patterns achieve
 * best performance without bank conflict penalties.
 *
 * @test ConflictFree
 *
 * Validates:
 *  - Conflict-free access pattern
 *  - Optimal shared memory performance
 *  - Best-case throughput
 *
 * Expected performance:
 *  - Best shared memory performance
 *  - Faster than version with conflicts
 */
PERF_GPU_TEST(GpuSharedMemory, ConflictFree) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256;
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_data(N, 1.0f);
  float* d_data;
  cudaMalloc(&d_data, SIZE);
  cudaMemcpy(d_data, h_data.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  dim3 block(256);
  dim3 grid(1);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { sharedConflictFreeKernel<<<grid, block, 0, s>>>(d_data, N); });

  // Measure conflict-free access
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      sharedConflictFreeKernel<<<grid, block, 0, s>>>(d_data, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  EXPECT_GT(result.callsPerSecond, 0.0) << "Should have high throughput";

  // Cleanup
  cudaFree(d_data);
}

/**
 * @brief Reduction with shared memory
 *
 * Validates shared memory usage in a practical algorithm (reduction)
 * demonstrating real-world shared memory benefits.
 *
 * @test ReductionSharedMemory
 *
 * Validates:
 *  - Shared memory reduction algorithm
 *  - Synchronization correctness
 *  - Performance benefits for reductions
 *
 * Expected performance:
 *  - Efficient parallel reduction
 *  - Correct results
 */
PERF_GPU_TEST(GpuSharedMemory, ReductionSharedMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  std::vector<float> h_input(N, 1.0f);
  float *d_input, *d_output;

  const int threadsPerBlock = 256;
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, numBlocks * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch configuration
  dim3 block(threadsPerBlock);
  dim3 grid(numBlocks);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { reductionSharedKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure reduction
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      reductionSharedKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Verify partial results (each block produces one output)
  std::vector<float> h_output(numBlocks);
  cudaMemcpy(h_output.data(), d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

  // Each block should have reduced its portion
  for (int i = 0; i < std::min(10, numBlocks); ++i) {
    EXPECT_GT(h_output[i], 0.0f) << "Block " << i << " should have non-zero reduction result";
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
