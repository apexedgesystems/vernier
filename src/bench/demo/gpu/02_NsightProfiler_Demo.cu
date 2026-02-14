/**
 * @file 02_NsightProfiler_Demo.cu
 * @brief Demo 11: Nsight profiler for GPU memory coalescing
 *
 * Demonstrates using NVIDIA Nsight to identify and fix uncoalesced
 * global memory access patterns. Coalesced access combines warp
 * memory requests into minimal transactions.
 *
 * Slow: Strided access (thread i reads element i*stride) -- uncoalesced
 * Fast: Sequential access (thread i reads element i) -- perfectly coalesced
 *
 * Usage:
 *   @code{.sh}
 *   # Baseline
 *   ./BenchDemo_Gpu_02_NsightProfiler --csv baseline.csv
 *
 *   # Profile with Nsight Systems
 *   ./BenchDemo_Gpu_02_NsightProfiler --profile nsight --gtest_filter="*Uncoalesced*"
 *
 *   # Compare
 *   bench summary baseline.csv
 *   @endcode
 *
 * @see docs/11_NSIGHT_PROFILER.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/* ----------------------------- Kernels ----------------------------- */

/**
 * @brief Coalesced read: thread i reads element i (sequential).
 *
 * All threads in a warp access consecutive addresses. The memory
 * controller combines 32 requests into 1-2 cache line transactions.
 */
__global__ void coalescedReadKernel(const float* input, float* output, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}

/**
 * @brief Uncoalesced read: thread i reads element i*32 (strided).
 *
 * Threads in a warp access addresses 32*sizeof(float)=128 bytes apart.
 * Each thread's request hits a different cache line, requiring 32
 * separate memory transactions per warp instead of 1.
 */
__global__ void uncoalescedReadKernel(const float* input, float* output, int n, int stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int srcIdx = (idx * stride) % n;
  if (idx < n) {
    output[idx] = input[srcIdx] * 2.0f;
  }
}

/* ----------------------------- Constants ----------------------------- */

static constexpr int N = 1024 * 1024;
static constexpr std::size_t SIZE = N * sizeof(float);
static constexpr int BLOCK_SIZE = 256;
static constexpr int STRIDE = 32; // One warp width

} // anonymous namespace

/**
 * @test Slow: Uncoalesced memory access (strided reads).
 *
 * Each warp reads 32 non-consecutive locations, generating 32 memory
 * transactions instead of 1. Nsight Compute will show low global memory
 * load efficiency and high L2 cache traffic.
 */
PERF_GPU_BANDWIDTH(NsightProfiler, UncoalescedAccess) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  // Initialize
  std::vector<float> h_in(N, 1.0f);
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  perf.cudaWarmup([&](cudaStream_t s) {
    uncoalescedReadKernel<<<grid, block, 0, s>>>(d_in, d_out, N, STRIDE);
  });

  auto result = perf.cudaKernel(
                        [&](cudaStream_t s) {
                          uncoalescedReadKernel<<<grid, block, 0, s>>>(d_in, d_out, N, STRIDE);
                        },
                        "uncoalesced_stride32")
                    .withLaunchConfig(grid, block)
                    .measure();

  EXPECT_GT(result.callsPerSecond, 1.0);

  cudaFree(d_in);
  cudaFree(d_out);
}

/**
 * @test Fast: Coalesced memory access (sequential reads).
 *
 * Each warp reads 32 consecutive elements. The memory controller
 * serves the entire warp in 1-2 transactions. Nsight Compute will
 * show high global memory load efficiency (close to 100%).
 *
 * Expected improvement: 5-20x depending on GPU architecture.
 */
PERF_GPU_BANDWIDTH(NsightProfiler, CoalescedAccess) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  std::vector<float> h_in(N, 1.0f);
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  perf.cudaWarmup(
      [&](cudaStream_t s) { coalescedReadKernel<<<grid, block, 0, s>>>(d_in, d_out, N); });

  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { coalescedReadKernel<<<grid, block, 0, s>>>(d_in, d_out, N); },
              "coalesced_sequential")
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
