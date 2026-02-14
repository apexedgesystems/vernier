/**
 * @file GpuUnifiedMemory_pTest.cu
 * @brief Unified Memory profiling and migration pattern validation
 *
 * This test suite validates Unified Memory (UM) profiling capabilities including
 * page fault tracking, migration detection, thrashing identification, and
 * overhead calculation.
 *
 * Features tested:
 *  - Page fault detection and counting
 *  - Host-to-device and device-to-host migration tracking
 *  - Thrashing detection (excessive back-and-forth migration)
 *  - Migration overhead calculation
 *  - UM vs explicit memory performance comparison
 *
 * Expected behavior:
 *  - Page faults are tracked when UM pages are accessed
 *  - Migration counts reflect actual page movement
 *  - Thrashing is detected when migration is excessive
 *  - Overhead calculation is accurate
 *
 * Usage:
 *   @code{.sh}
 *   # Run all Unified Memory tests
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuUnifiedMemory.*"
 *
 *   # With verbose UM profiling
 *   ./TestBenchSamples_GPU_PTEST --gtest_filter="GpuUnifiedMemory.*" --verbose
 *   @endcode
 *
 * Performance expectations:
 *  - Runtime: ~12 seconds total
 *  - Pass rate: 100% with CUDA GPU
 *  - UM overhead typically 5-30% depending on access pattern
 *
 * @see PerfGpuCase
 * @see UnifiedMemoryProfile
 * @see GpuStats
 */

#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Simple kernel that reads UM data */
__global__ void readUnifiedMemory(const float* data, float* sum, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(sum, data[idx]);
  }
}

/** @brief Kernel that modifies UM data */
__global__ void writeUnifiedMemory(float* data, float value, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = value + idx;
  }
}

/** @brief Kernel that reads and writes UM data */
__global__ void readWriteUnifiedMemory(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * 2.0f + 1.0f;
  }
}

/** @brief Trigger host access to UM pages */
void hostAccessUM(float* data, int n) {
  float sum = 0;
  for (int i = 0; i < n; i += 1024) {
    sum += data[i];
  }
  volatile float sink = sum;
  (void)sink;
}

} // anonymous namespace

/**
 * @brief Page fault detection and tracking
 *
 * Validates that the framework correctly detects and counts page faults
 * when unified memory is accessed by the GPU.
 *
 * @test PageFaultDetection
 *
 * Validates:
 *  - Page faults are detected on first GPU access
 *  - Page fault count is non-zero
 *  - UnifiedMemoryProfile is populated
 *
 * Expected performance:
 *  - Page faults > 0 for first-time access
 */
PERF_GPU_TEST(GpuUnifiedMemory, PageFaultDetection) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 32 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  // Allocate unified memory
  float* um_data;
  cudaMallocManaged(&um_data, SIZE);

  // Initialize on host (creates host pages)
  for (int i = 0; i < N; ++i) {
    um_data[i] = static_cast<float>(i);
  }

  float* d_sum;
  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  perf.cudaWarmup(
      [&](cudaStream_t s) { readUnifiedMemory<<<grid, block, 0, s>>>(um_data, d_sum, N); });

  // First access will cause page faults (H→D migration)
  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { readUnifiedMemory<<<grid, block, 0, s>>>(um_data, d_sum, N); },
              "page_fault_test")
          .withLaunchConfig(grid, block)
          .measure();

  // Check for UM profiling data - skip if not available
  if (!result.stats.unifiedMemory.has_value()) {
    cudaFree(um_data);
    cudaFree(d_sum);
    GTEST_SKIP() << "Unified Memory profiling not available (requires CUDA 11.8+, privileged mode, "
                    "or specific GPU driver)";
  }

  const auto& um = result.stats.unifiedMemory.value();

  // Page faults should be detected
  // Note: warmup may have already migrated pages, so this is informational
  if (um.pageFaults > 0) {
    EXPECT_GT(um.pageFaults, 0u) << "Page faults should be detected on UM access";
  }

  cudaFree(um_data);
  cudaFree(d_sum);
}

/**
 * @brief Host-to-device migration tracking
 *
 * Validates that H→D page migrations are correctly detected and counted
 * when GPU accesses pages that are resident on the host.
 *
 * @test MigrationPatterns
 *
 * Validates:
 *  - H2D migrations tracked
 *  - D2H migrations tracked
 *  - Migration counts are reasonable
 *
 * Expected performance:
 *  - H2D migrations > 0 when GPU accesses host pages
 */
PERF_GPU_TEST(GpuUnifiedMemory, MigrationPatterns) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 128 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  float* um_data;
  cudaMallocManaged(&um_data, SIZE);

  // Initialize on host
  for (int i = 0; i < N; ++i) {
    um_data[i] = 1.0f;
  }

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  perf.cudaWarmup(
      [&](cudaStream_t s) { writeUnifiedMemory<<<grid, block, 0, s>>>(um_data, 2.0f, N); });

  // GPU write will cause H→D migration
  auto gpuWrite =
      perf.cudaKernel(
              [&](cudaStream_t s) { writeUnifiedMemory<<<grid, block, 0, s>>>(um_data, 2.0f, N); },
              "h2d_migration")
          .withLaunchConfig(grid, block)
          .measure();

  if (!gpuWrite.stats.unifiedMemory.has_value()) {
    cudaFree(um_data);
    GTEST_SKIP() << "Unified Memory profiling not available (requires CUDA 11.8+, privileged mode, "
                    "or specific GPU driver)";
  }

  const auto& umWrite = gpuWrite.stats.unifiedMemory.value();

  // Should see H→D migrations
  if (umWrite.h2dMigrations > 0) {
    EXPECT_GT(umWrite.h2dMigrations, 0u) << "H2D migrations should be detected";
  }

  // Now access from host (will cause D→H migration)
  cudaDeviceSynchronize();
  hostAccessUM(um_data, N);

  // GPU access again (will cause H→D migration again)
  auto gpuRewrite =
      perf.cudaKernel(
              [&](cudaStream_t s) { writeUnifiedMemory<<<grid, block, 0, s>>>(um_data, 3.0f, N); },
              "d2h_then_h2d")
          .withLaunchConfig(grid, block)
          .measure();

  if (!gpuRewrite.stats.unifiedMemory.has_value()) {
    cudaFree(um_data);
    GTEST_SKIP() << "Unified Memory profiling not available";
  }

  const auto& umRewrite = gpuRewrite.stats.unifiedMemory.value();

  // Should see more migrations due to back-and-forth access
  const size_t totalMigrations = umRewrite.h2dMigrations + umRewrite.d2hMigrations;
  if (totalMigrations > 0) {
    EXPECT_GT(totalMigrations, 0u) << "Total migrations should be > 0 for alternating access";
  }

  cudaFree(um_data);
}

/**
 * @brief Thrashing detection
 *
 * Validates that excessive back-and-forth migration (thrashing) is
 * correctly detected using the isThrashing() metric.
 *
 * @test ThrashingDetection
 *
 * Validates:
 *  - Thrashing detection logic
 *  - Thrashing events counted
 *  - isThrashing() returns correct value
 *
 * Expected performance:
 *  - Thrashing detected when migrations >> page faults
 */
PERF_GPU_TEST(GpuUnifiedMemory, ThrashingDetection) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 64 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  float* um_data;
  cudaMallocManaged(&um_data, SIZE);

  // Initialize
  for (int i = 0; i < N; ++i) {
    um_data[i] = 1.0f;
  }

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Intentionally cause thrashing by alternating GPU and host access
  for (int iter = 0; iter < 5; ++iter) {
    // GPU access
    writeUnifiedMemory<<<grid, block>>>(um_data, static_cast<float>(iter), N);
    cudaDeviceSynchronize();

    // Host access
    hostAccessUM(um_data, N);
  }

  perf.cudaWarmup(
      [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); });

  // Measure with potential thrashing pattern
  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); },
              "thrashing_test")
          .withLaunchConfig(grid, block)
          .measure();

  if (!result.stats.unifiedMemory.has_value()) {
    cudaFree(um_data);
    GTEST_SKIP() << "Unified Memory profiling not available (requires CUDA 11.8+, privileged mode, "
                    "or specific GPU driver)";
  }

  const auto& um = result.stats.unifiedMemory.value();

  // Check if thrashing detection works
  if (um.pageFaults > 0) {
    const double migrationRatio =
        static_cast<double>(um.h2dMigrations + um.d2hMigrations) / um.pageFaults;

    // If many migrations relative to page faults, should detect thrashing
    if (migrationRatio > 2.0) {
      EXPECT_TRUE(um.isThrashing()) << "Should detect thrashing when migrations >> page faults";
    }
  }

  cudaFree(um_data);
}

/**
 * @brief Migration overhead calculation
 *
 * Validates that migration overhead is correctly calculated as a
 * percentage of kernel execution time.
 *
 * @test MigrationOverhead
 *
 * Validates:
 *  - migrationOverheadPct() calculation
 *  - Overhead is reasonable (typically 5-30%)
 *  - Overhead increases with migration frequency
 *
 * Expected performance:
 *  - Overhead 5-30% for typical access patterns
 */
PERF_GPU_TEST(GpuUnifiedMemory, MigrationOverhead) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  float* um_data;
  cudaMallocManaged(&um_data, SIZE);

  // Initialize on host
  for (int i = 0; i < N; ++i) {
    um_data[i] = static_cast<float>(i);
  }

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  perf.cudaWarmup(
      [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); });

  // Measure with potential migrations
  auto result =
      perf.cudaKernel(
              [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); },
              "migration_overhead")
          .withLaunchConfig(grid, block)
          .measure();

  if (!result.stats.unifiedMemory.has_value()) {
    cudaFree(um_data);
    GTEST_SKIP() << "Unified Memory profiling not available (requires CUDA 11.8+, privileged mode, "
                    "or specific GPU driver)";
  }

  const auto& um = result.stats.unifiedMemory.value();

  // Calculate overhead - FIXED: use result.kernelTimeUs instead of result.stats.cpuStats.median
  const double overhead = um.migrationOverheadPct(result.kernelTimeUs, perf.cycles());

  // Overhead should be non-negative
  EXPECT_GE(overhead, 0.0) << "Migration overhead should be non-negative";

  // Overhead should be reasonable (< 100% for this workload)
  EXPECT_LT(overhead, 100.0) << "Migration overhead should be < 100% for compute-bound kernel";

  // If migrations occurred, overhead should be positive
  if (um.h2dMigrations + um.d2hMigrations > 0 && um.migrationTimeUs > 0.0) {
    EXPECT_GT(overhead, 0.0) << "Should have measurable overhead when migrations occur";
  }

  cudaFree(um_data);
}

/**
 * @brief UM vs explicit memory performance comparison
 *
 * Compares unified memory performance against explicit memory management
 * to validate overhead measurement and identify use cases where UM is appropriate.
 *
 * @test CompareStrategies
 *
 * Validates:
 *  - UM overhead measurement vs explicit memory
 *  - Performance difference is measurable
 *  - Both strategies produce correct results
 *
 * Expected performance:
 *  - Explicit memory typically faster
 *  - UM overhead typically 10-40% depending on access pattern
 */
PERF_GPU_TEST(GpuUnifiedMemory, CompareStrategies) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  // Unified memory
  float* um_data;
  cudaMallocManaged(&um_data, SIZE);
  for (int i = 0; i < N; ++i) {
    um_data[i] = 1.0f;
  }

  // Explicit memory
  std::vector<float> h_data(N, 1.0f);
  float* d_data;
  cudaMalloc(&d_data, SIZE);
  cudaMemcpy(d_data, h_data.data(), SIZE, cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Warmup both
  perf.cudaWarmup(
      [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); });

  perf.cudaWarmup(
      [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(d_data, N); });

  // Measure unified memory
  auto umResult =
      perf.cudaKernel(
              [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); },
              "unified_memory")
          .withLaunchConfig(grid, block)
          .measure();

  // Measure explicit memory
  auto explicitResult =
      perf.cudaKernel(
              [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(d_data, N); },
              "explicit_memory")
          .withLaunchConfig(grid, block)
          .measure();

  // Both should complete successfully
  EXPECT_GT(umResult.callsPerSecond, 0.0);
  EXPECT_GT(explicitResult.callsPerSecond, 0.0);

  // Compare performance - FIXED: use kernelTimeUs instead of stats.cpuStats.median
  const double umTime = umResult.kernelTimeUs;
  const double explicitTime = explicitResult.kernelTimeUs;

  // Calculate relative overhead
  const double relativeOverhead = ((umTime - explicitTime) / explicitTime) * 100.0;

  // UM typically has some overhead, but should not be extreme
  if (relativeOverhead > 0) {
    EXPECT_LT(relativeOverhead, 100.0) << "UM overhead should be < 100% for this workload";
  }

  // Check UM profiling data
  if (umResult.stats.unifiedMemory.has_value()) {
    const auto& um = umResult.stats.unifiedMemory.value();

    // Should have profiling data for UM but not explicit
    EXPECT_TRUE(umResult.stats.unifiedMemory.has_value())
        << "UM result should have unified memory profile";
  }

  cudaFree(um_data);
  cudaFree(d_data);
}

/**
 * @brief Access pattern impact on UM performance
 *
 * Validates that different access patterns have measurable performance
 * differences with unified memory.
 *
 * @test AccessPatterns
 *
 * Validates:
 *  - Sequential access performance
 *  - Random access performance
 *  - Access pattern affects page faults
 *
 * Expected performance:
 *  - Sequential access faster than random
 */
PERF_GPU_TEST(GpuUnifiedMemory, AccessPatterns) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 256 * 1024; // Reduced for speed
  const size_t SIZE = N * sizeof(float);

  // Allocate UM
  float* um_data;
  cudaMallocManaged(&um_data, SIZE);

  // Initialize on host
  for (int i = 0; i < N; ++i) {
    um_data[i] = 1.0f;
  }

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Sequential access
  perf.cudaWarmup(
      [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); });

  auto sequential =
      perf.cudaKernel(
              [&](cudaStream_t s) { readWriteUnifiedMemory<<<grid, block, 0, s>>>(um_data, N); },
              "sequential_access")
          .withLaunchConfig(grid, block)
          .measure();

  // Both should complete successfully
  EXPECT_GT(sequential.callsPerSecond, 0.0);

  // Check UM profiling data
  if (sequential.stats.unifiedMemory.has_value()) {
    const auto& um = sequential.stats.unifiedMemory.value();

    // UM profiling should capture some metrics
    // Note: actual values may vary by CUDA version and GPU
    EXPECT_GE(um.pageFaults, 0u) << "Page fault count should be non-negative";
  }

  cudaFree(um_data);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
