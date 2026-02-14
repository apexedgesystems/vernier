/**
 * @file 01_GpuBasicWorkflow_Demo.cu
 * @brief Demo 10: Basic GPU benchmarking workflow
 *
 * Teaches the fundamental GPU measurement workflow:
 *  1. Measure CPU baseline with cpuBaseline()
 *  2. Measure GPU kernel with cudaKernel()
 *  3. Compare speedup automatically
 *  4. Understand transfer overhead (H2D + D2H)
 *
 * Workload: Vector addition (a[i] + b[i] = c[i])
 *
 * Usage:
 *   @code{.sh}
 *   # Run all GPU workflow demos
 *   ./BenchDemo_Gpu_01_GpuBasicWorkflow --csv gpu_results.csv
 *
 *   # Summary with GPU columns
 *   bench summary gpu_results.csv
 *   @endcode
 *
 * @see docs/10_GPU_BASIC_WORKFLOW.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/* ----------------------------- Kernels ----------------------------- */

/** @brief Simple vector addition kernel. */
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

/** @brief CPU version of vector addition. */
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

/* ----------------------------- Constants ----------------------------- */

static constexpr int N = 1024 * 1024; // 1M elements
static constexpr std::size_t SIZE = N * sizeof(float);
static constexpr int BLOCK_SIZE = 256;

} // anonymous namespace

/**
 * @test CPU baseline: vector addition on CPU.
 *
 * Measures the CPU implementation to establish a reference point.
 * The framework automatically calculates GPU speedup relative to
 * this baseline when both tests are in the same CSV.
 */
PERF_GPU_COMPARISON(GpuBasicWorkflow, CpuBaseline) {
  UB_PERF_GPU_GUARD(perf);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  auto result = perf.cpuBaseline([&] { vectorAddCPU(h_a.data(), h_b.data(), h_c.data(), N); },
                                 "cpu_vector_add");

  EXPECT_GT(result.callsPerSecond, 10.0);
  EXPECT_NEAR(h_c[0], 3.0f, 0.01f);
}

/**
 * @test GPU kernel: vector addition with transfer overhead.
 *
 * Measures the full GPU pipeline: H2D transfer, kernel execution,
 * D2H transfer. The result includes transfer times separately so
 * you can see where time is spent.
 *
 * For small workloads, transfer overhead dominates and GPU may be
 * slower than CPU. This is expected and educational.
 */
PERF_GPU_COMPARISON(GpuBasicWorkflow, GpuWithTransfers) {
  UB_PERF_GPU_GUARD(perf);

  // Host memory
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  // Device memory
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    cudaMemcpyAsync(d_a, h_a.data(), SIZE, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_b, h_b.data(), SIZE, cudaMemcpyHostToDevice, s);
    vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
    cudaMemcpyAsync(h_c.data(), d_c, SIZE, cudaMemcpyDeviceToHost, s);
  });

  // Measure with transfers
  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); },
              "gpu_vector_add")
          .withHostToDevice(h_a.data(), d_a, SIZE)
          .withHostToDevice(h_b.data(), d_b, SIZE)
          .withDeviceToHost(d_c, h_c.data(), SIZE)
          .withLaunchConfig(grid, block)
          .measure();

  EXPECT_GT(result.callsPerSecond, 1.0);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

/**
 * @test GPU kernel only: no transfer overhead.
 *
 * Measures kernel execution time alone (data stays on device).
 * This shows the "best case" GPU performance when data is already
 * resident. Compare with GpuWithTransfers to see transfer overhead.
 */
PERF_GPU_COMPARISON(GpuBasicWorkflow, GpuKernelOnly) {
  UB_PERF_GPU_GUARD(perf);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  // Pre-transfer data (not measured)
  cudaMemcpy(d_a, h_a.data(), SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  perf.cudaWarmup(
      [&](cudaStream_t s) { vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); });

  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); },
              "gpu_kernel_only")
          .withLaunchConfig(grid, block)
          .measure();

  EXPECT_GT(result.callsPerSecond, 10.0);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
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
