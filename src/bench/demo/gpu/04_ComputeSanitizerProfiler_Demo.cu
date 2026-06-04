/**
 * @file 04_ComputeSanitizerProfiler_Demo.cu
 * @brief Demo 04: NVIDIA Compute Sanitizer for GPU memory + race checking.
 *
 * Compute Sanitizer is the GPU analog of valgrind memcheck. It catches
 * device-side bugs that often don't crash but silently corrupt results:
 *  - Out-of-bounds reads / writes (--tool=memcheck, default)
 *  - Shared-memory races (--tool=racecheck)
 *  - Missed __syncthreads (--tool=synccheck)
 *  - Reads from uninitialized device memory (--tool=initcheck)
 *
 * Compute Sanitizer wraps the binary externally, the same way valgrind
 * wraps a CPU binary for callgrind.
 *
 * Usage:
 *   @code{.sh}
 *   # 1) Run unwrapped (printable hint shown by the backend):
 *   ./BenchDemo_Gpu_04_ComputeSanitizerProfiler --profile compute-sanitizer
 *
 *   # 2) Wrap externally so the sanitizer actually runs:
 *   compute-sanitizer --tool=memcheck \
 *       --log-file=ComputeSanitizer.SafeKernel.compute-sanitizer/sanitizer.log \
 *       ./BenchDemo_Gpu_04_ComputeSanitizerProfiler \
 *       --profile compute-sanitizer --cycles 5 \
 *       --gtest_filter='ComputeSanitizer.SafeKernel'
 *
 *   # 3) Inspect the log -- on the safe kernel it should report
 *   #    "0 errors". Switch the filter to '*WithDeliberateOob' to see
 *   #    compute-sanitizer flag the OOB read.
 *   cat ComputeSanitizer.SafeKernel.compute-sanitizer/sanitizer.log | tail -20
 *   @endcode
 */

#include <gtest/gtest.h>

#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/* ----------------------------- Kernels ----------------------------- */

/** @brief Bounds-checked element-wise scale -- nothing for the sanitizer to find. */
__global__ void scaleKernel(const float* in, float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] * 2.0f;
  }
}

/**
 * @brief Deliberately reads ONE element past the end on the last thread of the
 *        grid. Demonstrates a class of off-by-one bug that often "works" on
 *        real hardware (the read is within the allocation's page) but compute-
 *        sanitizer flags as out-of-bounds.
 *
 * Do not copy this pattern; it is here so the sanitizer has something to find.
 */
__global__ void scaleKernelWithOob(const float* in, float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] * 2.0f;
  }
  // One OOB write per launch: thread 0 of the last block writes at out[n],
  // one past the logical end of the buffer. cudaMalloc tracks logical size,
  // so compute-sanitizer memcheck flags this even though the address may
  // still fall inside the allocated page.
  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
    out[n] = 99.0f;
  }
}

/* ----------------------------- Constants ----------------------------- */

static constexpr int N = 1 << 20;
static constexpr std::size_t SIZE = static_cast<std::size_t>(N) * sizeof(float);
static constexpr int BLOCK_SIZE = 256;

} // anonymous namespace

/* ----------------------------- Tests ----------------------------- */

/** @test Safe kernel: clean run; sanitizer report shows 0 errors. */
PERF_GPU_BANDWIDTH(ComputeSanitizer, SafeKernel) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  std::vector<float> h_in(N, 1.0f);
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  perf.cudaWarmup([&](cudaStream_t s) { scaleKernel<<<grid, block, 0, s>>>(d_in, d_out, N); });

  auto result =
      perf.cudaKernel([&](cudaStream_t s) { scaleKernel<<<grid, block, 0, s>>>(d_in, d_out, N); },
                      "scale_safe")
          .withLaunchConfig(grid, block)
          .measure();

  EXPECT_GT(result.callsPerSecond, 1.0);

  cudaFree(d_in);
  cudaFree(d_out);
}

/**
 * @test Kernel with a deliberate one-element OOB write on the last thread.
 *
 * Standalone behavior is driver- and architecture-dependent: many recent
 * CUDA runtimes detect the page-boundary overrun and surface it as "an
 * illegal memory access", which fails the gtest case; older runtimes
 * silently return a benign value (the next page) and the case passes.
 *
 * Either outcome leaves you with a buggy kernel. compute-sanitizer
 * --tool=memcheck is what actually pinpoints it: "Invalid __global__
 * write of size 4 bytes" with the exact source line, thread, and block.
 */
PERF_GPU_BANDWIDTH(ComputeSanitizer, WithDeliberateOob) {
  UB_PERF_GPU_GUARD(perf);

  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, SIZE);
  cudaMalloc(&d_out, SIZE);

  std::vector<float> h_in(N, 1.0f);
  cudaMemcpy(d_in, h_in.data(), SIZE, cudaMemcpyHostToDevice);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  perf.cudaWarmup(
      [&](cudaStream_t s) { scaleKernelWithOob<<<grid, block, 0, s>>>(d_in, d_out, N); });

  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { scaleKernelWithOob<<<grid, block, 0, s>>>(d_in, d_out, N); },
              "scale_with_oob")
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
