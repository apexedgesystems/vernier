/**
 * @file GpuProfilers_pTest.cu
 * @brief GPU profiler samples: kernels measured with the profiler the guard
 *        attaches
 *
 * UB_PERF_GPU_GUARD attaches the profiler a --profile request names to every
 * measurement of the case; the cases attach nothing else. Under --profile
 * nsight the Nsight backend only marks each measurement with an NVTX range:
 * nsys records the process it started and writes its report when the process
 * exits, so whether a capture succeeded is decided then, by nsys and by the
 * run's outcome, never by a case. A request nsys cannot serve (--profile
 * nsight in a process nsys did not start) fails the run, which reports why and
 * exits with status 4 when its tests pass; bench run --profile nsight starts
 * the process under nsys and checks the report after it exits.
 *
 * Usage:
 *   @code{.sh}
 *   # The samples, no profiler
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuProfilers.*"
 *
 *   # Under Nsight Systems
 *   bench run ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --profile nsight -- --gtest_filter="GpuProfilers.*"
 *   @endcode
 *
 * @see ProfilerNsight
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "../../inc/PerfGpu.hpp"
#include "../helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;

namespace {

/**
 * @brief Simple vector addition kernel for profiling tests
 */
__global__ void simpleKernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

} // anonymous namespace

/**
 * @brief A kernel measured with the guard's profiler
 *
 * Under --profile nsight in a process nsys started, the guard's Nsight
 * backend marks the measurement with an NVTX range in nsys's timeline.
 *
 * @test NsightIntegration
 */
PERF_GPU_TEST(GpuProfilers, NsightIntegration) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  // Initialize
  cudaMemset(d_a, 1, SIZE);
  cudaMemset(d_b, 2, SIZE);
  cudaMemset(d_c, 0, SIZE);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  perf.cudaWarmup([&](cudaStream_t s) { simpleKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); });

  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { simpleKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); },
              "nsight_test")
          .withLaunchConfig(grid, block)
          .measure();

  // Validation
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

/**
 * @brief One window measured with the guard's hooks
 *
 * The guard attaches its hooks to every measurement of the case, so the
 * process has no unprofiled window to compare with. A profiler's cost on the
 * measured window shows between two runs that differ only in --profile, with
 * everything else given explicitly and the in-process CUPTI collector off in
 * both:
 *   export VERNIER_DISABLE_CUPTI=1
 *   bench run BenchmarkGPU_PTEST --taskset 2 --cycles 1000 --repeats 10 --csv base.csv \
 *       -- --gtest_filter=GpuProfilers.ProfilerOverhead --gpu-device 0 --warmup 1 --gpu-warmup 10
 *   bench run BenchmarkGPU_PTEST --taskset 2 --cycles 1000 --repeats 10 --csv nsight.csv \
 *       --profile nsight \
 *       -- --gtest_filter=GpuProfilers.ProfilerOverhead --gpu-device 0 --warmup 1 --gpu-warmup 10
 *   bench compare base.csv nsight.csv
 * The difference leaves out nsys's start-up, its report export and the
 * command's own time, and no threshold applies to it.
 *
 * @test ProfilerOverhead
 */
PERF_GPU_TEST(GpuProfilers, ProfilerOverhead) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256 * 1024;
  const size_t SIZE = N * sizeof(float);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  cudaMemset(d_a, 1, SIZE);
  cudaMemset(d_b, 2, SIZE);
  cudaMemset(d_c, 0, SIZE);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  perf.cudaWarmup([&](cudaStream_t s) { simpleKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); });

  // Measure with profiling hooks (even if no profiler active)
  auto resultWithHooks =
      perf.cudaKernel(
              [&](cudaStream_t s) { simpleKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); },
              "with_hooks")
          .withLaunchConfig(grid, block)
          .measure();

  // Validation - Use generous thresholds since this is a micro-benchmark
  // measuring extremely fast operations where variance is naturally higher
  const double GENEROUS_CV_THRESHOLD = 0.30; // 30% for tiny micro-benchmarks

  EXPECT_LT(resultWithHooks.stats.cpuStats.cv, GENEROUS_CV_THRESHOLD)
      << "High jitter with profiler hooks: CV% " << (resultWithHooks.stats.cpuStats.cv * 100.0)
      << "%";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
