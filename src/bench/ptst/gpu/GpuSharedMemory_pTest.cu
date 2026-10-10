/**
 * @file GpuSharedMemory_pTest.cu
 * @brief Shared memory usage: a tiled transpose and a block reduction
 *
 * This test suite measures a matrix transpose with and without a shared-memory
 * tile, and a reduction in shared memory, and checks every result.
 *
 * Features tested:
 *  - Global memory baseline (no shared memory)
 *  - Shared memory transpose through a padded static tile
 *  - Reduction in shared memory
 *
 * Expected behavior:
 *  - Both transposes match the CPU's transpose element by element
 *  - Every block of the reduction sums its inputs exactly
 *  - The two transposes' rows show what the tile changes on the GPU at hand;
 *    no case asserts by how much
 *
 * Bank conflicts, and the padding that avoids them in the tile, are the
 * subject of walkthrough 12 (src/bench/demo/docs/12_SHARED_MEMORY_OPT.md).
 *
 * Usage:
 *   @code{.sh}
 *   # Run all shared memory tests
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuSharedMemory.*"
 *
 *   # Run specific test
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuSharedMemory.SharedMemoryOptimized"
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~12 seconds total
 *  - Pass rate: 100% with CUDA GPU
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

/** @brief The transpose of a row-major width x height matrix, computed on the CPU. */
std::vector<float> transposeOnCpu(const std::vector<float>& input, int width, int height) {
  std::vector<float> output(input.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      output[x * height + y] = input[y * width + x];
    }
  }
  return output;
}

/** @brief Index of the first element where @p got differs from @p want, or -1 when all agree. */
int firstMismatch(const std::vector<float>& got, const std::vector<float>& want) {
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      return static_cast<int>(i);
    }
  }
  return -1;
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
 *  - Global memory transpose is correct element by element
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
  // An element the kernel never writes reads 0, whatever the allocation held
  ASSERT_EQ(cudaMemset(d_output, 0, SIZE), cudaSuccess);

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

  // Verify correctness: every element against the CPU's transpose
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, SIZE, cudaMemcpyDeviceToHost), cudaSuccess);
  const std::vector<float> expected = transposeOnCpu(h_input, WIDTH, HEIGHT);
  const int WRONG = firstMismatch(h_output, expected);
  EXPECT_EQ(WRONG, -1) << "output[" << WRONG << "] (row " << WRONG / HEIGHT << ", column "
                       << WRONG % HEIGHT << ") = " << h_output[WRONG] << ", expected "
                       << expected[WRONG];

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Shared memory optimized transpose
 *
 * Transposes through a tile in shared memory: each block reads a 32x32 tile
 * along rows and writes it out transposed, also along rows. The tile is
 * declared static, padded by one column, and the launch asks for no dynamic
 * shared memory.
 *
 * @test SharedMemoryOptimized
 *
 * Validates:
 *  - Shared memory usage
 *  - Correctness of optimized implementation, element by element
 *
 * Expected performance:
 *  - Faster than GlobalMemoryBaseline; how much depends on the GPU and the
 *    matrix size
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
  // An element the kernel never writes reads 0, whatever the allocation held
  ASSERT_EQ(cudaMemset(d_output, 0, SIZE), cudaSuccess);

  // Launch configuration. The kernel's tile is static shared memory, so the
  // launch requests no dynamic shared memory.
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    transposeSharedKernel<<<grid, block, 0, s>>>(d_input, d_output, WIDTH, HEIGHT);
  });

  // Measure shared memory transpose
  auto result =
      perf.cudaKernel([&](cudaStream_t s) {
            transposeSharedKernel<<<grid, block, 0, s>>>(d_input, d_output, WIDTH, HEIGHT);
          })
          .withLaunchConfig(grid, block)
          .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  EXPECT_GT(result.callsPerSecond, 0.0) << "Should have valid throughput";

  // Verify correctness: every element against the CPU's transpose
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, SIZE, cudaMemcpyDeviceToHost), cudaSuccess);
  const std::vector<float> expected = transposeOnCpu(h_input, WIDTH, HEIGHT);
  const int WRONG = firstMismatch(h_output, expected);
  EXPECT_EQ(WRONG, -1) << "output[" << WRONG << "] (row " << WRONG / HEIGHT << ", column "
                       << WRONG % HEIGHT << ") = " << h_output[WRONG] << ", expected "
                       << expected[WRONG];

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Reduction with shared memory
 *
 * Validates shared memory usage in a practical algorithm (reduction): each
 * block sums its 256 inputs in shared memory.
 *
 * @test ReductionSharedMemory
 *
 * Validates:
 *  - Shared memory reduction algorithm
 *  - Synchronization correctness: every block's sum is exact
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
  // A block that never writes its sum reads 0, whatever the allocation held
  ASSERT_EQ(cudaMemset(d_output, 0, numBlocks * sizeof(float)), cudaSuccess);

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

  // Every block sums its threadsPerBlock inputs of 1.0, exactly
  std::vector<float> h_output(numBlocks);
  ASSERT_EQ(
      cudaMemcpy(h_output.data(), d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess);
  const std::vector<float> expected(numBlocks, static_cast<float>(threadsPerBlock));
  const int WRONG = firstMismatch(h_output, expected);
  EXPECT_EQ(WRONG, -1) << "Block " << WRONG << " of " << numBlocks << " summed to "
                       << h_output[WRONG] << ", expected " << expected[WRONG];

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
