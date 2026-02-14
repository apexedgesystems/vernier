/**
 * @file 03_SharedMemoryOpt_Demo.cu
 * @brief Demo 12: Shared memory optimization and bank conflicts
 *
 * Demonstrates the progression from global memory to shared memory,
 * showing how bank conflicts degrade performance and how padding
 * eliminates them. Three-way comparison:
 *
 *  1. Global memory only (baseline)
 *  2. Shared memory with bank conflicts (partial win)
 *  3. Shared memory without bank conflicts (full win)
 *
 * Workload: Matrix transpose (demonstrates the classic shared memory use case)
 *
 * Usage:
 *   @code{.sh}
 *   # Run all three
 *   ./BenchDemo_Gpu_03_SharedMemoryOpt --csv results.csv
 *
 *   # Compare
 *   bench summary results.csv --sort median
 *   @endcode
 *
 * @see docs/12_SHARED_MEMORY_OPT.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/* ----------------------------- Constants ----------------------------- */

static constexpr int TILE_DIM = 32;
static constexpr int MATRIX_DIM = 1024; // 1024x1024 matrix
static constexpr int N = MATRIX_DIM * MATRIX_DIM;
static constexpr std::size_t SIZE = N * sizeof(float);

/* ----------------------------- Kernels ----------------------------- */

/**
 * @brief Naive transpose: global memory only.
 *
 * Reads are coalesced (row-major) but writes are strided (column-major).
 * Each warp writes to addresses MATRIX_DIM apart, causing uncoalesced writes.
 */
__global__ void transposeNaive(const float* input, float* output, int dim) {
  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (x < dim && y < dim) {
    output[x * dim + y] = input[y * dim + x];
  }
}

/**
 * @brief Shared memory transpose WITH bank conflicts.
 *
 * Loads a tile into shared memory (coalesced reads), then writes
 * from shared memory (coalesced writes). However, the shared memory
 * access pattern during the write phase causes 32-way bank conflicts
 * because all threads in a warp access the same bank.
 */
__global__ void transposeSharedConflict(const float* input, float* output, int dim) {
  __shared__ float tile[TILE_DIM][TILE_DIM]; // Bank conflicts on column access

  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;

  // Coalesced read from global -> shared
  if (x < dim && y < dim) {
    tile[threadIdx.y][threadIdx.x] = input[y * dim + x];
  }
  __syncthreads();

  // Write from shared -> global (transposed)
  const int outX = blockIdx.y * TILE_DIM + threadIdx.x;
  const int outY = blockIdx.x * TILE_DIM + threadIdx.y;
  if (outX < dim && outY < dim) {
    // tile[threadIdx.x][threadIdx.y] causes bank conflicts:
    // all threads in warp read same column (same bank)
    output[outY * dim + outX] = tile[threadIdx.x][threadIdx.y];
  }
}

/**
 * @brief Shared memory transpose WITHOUT bank conflicts (padded).
 *
 * Same algorithm as above, but shared memory is padded by 1 element
 * per row. This shifts column accesses across different banks,
 * eliminating all bank conflicts.
 */
__global__ void transposeSharedNoPadding(const float* input, float* output, int dim) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 padding eliminates bank conflicts

  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x < dim && y < dim) {
    tile[threadIdx.y][threadIdx.x] = input[y * dim + x];
  }
  __syncthreads();

  const int outX = blockIdx.y * TILE_DIM + threadIdx.x;
  const int outY = blockIdx.x * TILE_DIM + threadIdx.y;
  if (outX < dim && outY < dim) {
    output[outY * dim + outX] = tile[threadIdx.x][threadIdx.y];
  }
}

} // anonymous namespace

/**
 * @test Baseline: Naive global memory transpose.
 *
 * Writes are uncoalesced (column-major pattern). This is the worst
 * case for memory throughput. Nsight will show low store efficiency.
 */
PERF_GPU_BANDWIDTH(SharedMemoryOpt, NaiveGlobalMemory) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  std::vector<float> h_in(N);
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(TILE_DIM, TILE_DIM);
  const dim3 grid((MATRIX_DIM + TILE_DIM - 1) / TILE_DIM, (MATRIX_DIM + TILE_DIM - 1) / TILE_DIM);

  perf.cudaWarmup(
      [&](cudaStream_t s) { transposeNaive<<<grid, block, 0, s>>>(d_in, d_out, MATRIX_DIM); });

  auto result = perf.cudaKernel(
                        [&](cudaStream_t s) {
                          transposeNaive<<<grid, block, 0, s>>>(d_in, d_out, MATRIX_DIM);
                        },
                        "naive_global")
                    .withLaunchConfig(grid, block)
                    .measure();

  EXPECT_GT(result.callsPerSecond, 1.0);

  cudaFree(d_in);
  cudaFree(d_out);
}

/**
 * @test Intermediate: Shared memory with bank conflicts.
 *
 * Both reads and writes to global memory are coalesced (via shared
 * memory staging). However, the shared memory column access pattern
 * causes bank conflicts, limiting speedup to ~2-3x over naive.
 */
PERF_GPU_BANDWIDTH(SharedMemoryOpt, SharedWithBankConflicts) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  std::vector<float> h_in(N);
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(TILE_DIM, TILE_DIM);
  const dim3 grid((MATRIX_DIM + TILE_DIM - 1) / TILE_DIM, (MATRIX_DIM + TILE_DIM - 1) / TILE_DIM);

  perf.cudaWarmup([&](cudaStream_t s) {
    transposeSharedConflict<<<grid, block, 0, s>>>(d_in, d_out, MATRIX_DIM);
  });

  auto result = perf.cudaKernel(
                        [&](cudaStream_t s) {
                          transposeSharedConflict<<<grid, block, 0, s>>>(d_in, d_out, MATRIX_DIM);
                        },
                        "shared_with_conflicts")
                    .withLaunchConfig(grid, block)
                    .measure();

  EXPECT_GT(result.callsPerSecond, 1.0);

  cudaFree(d_in);
  cudaFree(d_out);
}

/**
 * @test Optimized: Shared memory without bank conflicts (padded).
 *
 * Adding +1 padding to shared memory declarations shifts column
 * accesses across different banks. All bank conflicts are eliminated.
 *
 * Expected improvement over naive: 5-10x.
 * Expected improvement over bank-conflict version: 1.5-3x.
 */
PERF_GPU_BANDWIDTH(SharedMemoryOpt, SharedConflictFree) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  std::vector<float> h_in(N);
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(TILE_DIM, TILE_DIM);
  const dim3 grid((MATRIX_DIM + TILE_DIM - 1) / TILE_DIM, (MATRIX_DIM + TILE_DIM - 1) / TILE_DIM);

  perf.cudaWarmup([&](cudaStream_t s) {
    transposeSharedNoPadding<<<grid, block, 0, s>>>(d_in, d_out, MATRIX_DIM);
  });

  auto result = perf.cudaKernel(
                        [&](cudaStream_t s) {
                          transposeSharedNoPadding<<<grid, block, 0, s>>>(d_in, d_out, MATRIX_DIM);
                        },
                        "shared_conflict_free")
                    .withLaunchConfig(grid, block)
                    .measure();

  EXPECT_GT(result.callsPerSecond, 1.0);

  cudaFree(d_in);
  cudaFree(d_out);
}

/* ----------------------------- Main ----------------------------- */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfListener.hpp"
#include "src/bench/inc/PerfTestMacros.hpp"
#include "src/bench/inc/PerfGpuTestMacros.hpp"

int main(int argc, char** argv) {
  auto& cfg = vernier::bench::detail::perfConfigSingleton();
  vernier::bench::parsePerfFlags(cfg, &argc, argv);

  vernier::bench::PerfGpuConfig gpuCfg;
  vernier::bench::parseGpuFlags(gpuCfg, &argc, argv);

  vernier::bench::detail::setGlobalGpuConfig(gpuCfg);
  vernier::bench::setGlobalPerfConfig(&cfg);
  vernier::bench::installPerfEventListener(cfg);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
