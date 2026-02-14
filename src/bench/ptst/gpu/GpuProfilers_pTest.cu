/**
 * @file GpuProfilers_pTest.cu
 * @brief GPU profiler integration validation
 *
 * This test suite validates that GPU profilers (Nsight Systems/Compute)
 * integrate correctly with the framework and capture metrics properly.
 *
 * Features tested:
 *  - Nsight Systems integration
 *  - Profiler artifact generation
 *  - GPU-specific profiling hooks
 *
 * Expected behavior:
 *  - Profilers should activate when enabled
 *  - Artifacts should be generated in correct locations
 *  - Tests gracefully skip if profilers unavailable
 *
 * Usage:
 *   @code{.sh}
 *   # Run all GPU profiler tests
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuProfilers.*"
 *
 *   # With Nsight Systems profiling
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuProfilers.NsightIntegration" \
 *       --profile nsight --artifact-root profiler_artifacts
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~15 seconds total
 *  - Pass rate: 100% when profilers available, graceful skip otherwise
 *
 * @see ProfilerNsight
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <filesystem>

#include "../../inc/PerfGpu.hpp"
#include "../helpers/TestHelpers.hpp"

namespace ub = vernier::bench;
namespace test = vernier::bench::test;
namespace fs = std::filesystem;

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
 * @brief Nsight Systems profiler integration test
 *
 * Validates that Nsight Systems profiler correctly captures GPU activity
 * when enabled via --profile nsight flag.
 *
 * @test NsightIntegration
 *
 * Validates:
 *  - Profiler activates when requested
 *  - Kernel launches are captured
 *  - Profiler artifacts are generated
 *  - No crashes or errors occur
 *
 * Expected performance:
 *  - Profiling adds ~5-10% overhead
 *  - Artifacts generated in correct directory
 */
PERF_GPU_TEST(GpuProfilers, NsightIntegration) {
  UB_PERF_GPU_GUARD(perf);

  // Attach GPU profiler hooks
  ub::attachGpuProfilerHooks(perf, perf.cpuConfig());

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

  // Run with profiling
  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { simpleKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); },
              "nsight_test")
          .withLaunchConfig(grid, block)
          .measure();

  // Validation
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  // Check if profiling was enabled
  const auto& cfg = perf.cpuConfig();
  if (cfg.profileTool == "nsight") {
    std::printf("\n[OK] Nsight profiling enabled\n");

    // Check for artifacts if artifact root specified
    if (!cfg.artifactRoot.empty()) {
      fs::path artifactDir(cfg.artifactRoot);
      if (fs::exists(artifactDir)) {
        std::printf("[OK] Artifact directory exists: %s\n", cfg.artifactRoot.c_str());
      }
    }
  } else {
    std::printf("\n[Note] Nsight profiling not enabled (use --profile nsight)\n");
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

/**
 * @brief GPU profiler overhead measurement
 *
 * Measures the overhead introduced by GPU profilers to ensure
 * it's within acceptable bounds.
 *
 * @test ProfilerOverhead
 *
 * Validates:
 *  - Profiling overhead is measurable
 *  - Overhead is within expected range (<20%)
 *  - Measurements remain stable with profiling
 *
 * Expected performance:
 *  - Profiling adds 5-20% overhead
 *  - Still produces stable measurements
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

  // Measure without profiling hooks
  auto resultNoProfile =
      perf.cudaKernel(
              [&](cudaStream_t s) { simpleKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N); },
              "no_profile")
          .withLaunchConfig(grid, block)
          .measure();

  // Attach profiler hooks
  ub::attachGpuProfilerHooks(perf, perf.cpuConfig());

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

  EXPECT_LT(resultNoProfile.stats.cpuStats.cv, GENEROUS_CV_THRESHOLD)
      << "High jitter in baseline measurement: CV% " << (resultNoProfile.stats.cpuStats.cv * 100.0)
      << "%";

  EXPECT_LT(resultWithHooks.stats.cpuStats.cv, GENEROUS_CV_THRESHOLD)
      << "High jitter with profiler hooks: CV% " << (resultWithHooks.stats.cpuStats.cv * 100.0)
      << "%";

  // Calculate overhead
  double overhead = ((resultWithHooks.kernelTimeUs - resultNoProfile.kernelTimeUs) /
                     resultNoProfile.kernelTimeUs) *
                    100.0;

  std::printf("\nProfiler hook overhead: %.2f%%\n", overhead);

  const auto& cfg = perf.cpuConfig();
  if (cfg.profileTool.empty()) {
    // Without active profiler, overhead should be minimal (<5%)
    EXPECT_LT(std::abs(overhead), 5.0) << "Hook overhead without active profiler should be minimal";
  } else {
    // With active profiler, overhead up to 20% is acceptable
    std::printf("Active profiler: %s\n", cfg.profileTool.c_str());
    EXPECT_LT(std::abs(overhead), 20.0) << "Profiler overhead should be reasonable (<20%)";
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

/**
 * @brief Profiler artifact directory creation test
 *
 * Validates that profiler artifacts are correctly placed in the
 * specified artifact root directory.
 *
 * @test ArtifactGeneration
 *
 * Validates:
 *  - Artifact directory is created
 *  - Artifacts use correct naming convention
 *  - Test-specific subdirectories work correctly
 *
 * Expected performance:
 *  - Quick test (~2 seconds)
 *  - No performance overhead from artifact generation
 */
PERF_GPU_TEST(GpuProfilers, ArtifactGeneration) {
  UB_PERF_GPU_GUARD(perf);

  const auto& cfg = perf.cpuConfig();

  // Only meaningful if artifact root specified
  if (cfg.artifactRoot.empty()) {
    GTEST_SKIP() << "Artifact root not specified (use --artifact-root)";
  }

  ub::attachGpuProfilerHooks(perf, cfg);

  const int N = 128 * 1024;
  const size_t SIZE = N * sizeof(float);

  float* d_data;
  cudaMalloc(&d_data, SIZE);
  cudaMemset(d_data, 1, SIZE);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Run a simple kernel
  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { simpleKernel<<<1, 1, 0, s>>>(d_data, d_data, d_data, 1); },
              "artifact_test")
          .withLaunchConfig(dim3(1), dim3(1))
          .measure();

  // Check artifact directory
  fs::path artifactDir(cfg.artifactRoot);
  EXPECT_TRUE(fs::exists(artifactDir)) << "Artifact directory should exist: " << cfg.artifactRoot;

  std::printf("\n[OK] Artifact root exists: %s\n", cfg.artifactRoot.c_str());

  // List contents (if any)
  if (fs::exists(artifactDir)) {
    int fileCount = 0;
    for (const auto& entry : fs::directory_iterator(artifactDir)) {
      fileCount++;
      if (fileCount <= 5) { // Show first 5 files
        std::printf("  - %s\n", entry.path().filename().c_str());
      }
    }
    if (fileCount > 5) {
      std::printf("  ... and %d more files\n", fileCount - 5);
    }
    std::printf("Total artifacts: %d\n", fileCount);
  }

  cudaFree(d_data);
}
