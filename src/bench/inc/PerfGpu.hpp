#ifndef VERNIER_PERFGPU_HPP
#define VERNIER_PERFGPU_HPP
/**
 * @file PerfGpu.hpp
 * @brief All-in-one convenience header for GPU performance benchmarking.
 *
 * Scope: Includes all GPU-specific headers for writing CUDA performance tests.
 * Works alongside the CPU benchmarking framework (Perf.hpp).
 *
 * Typical usage:
 * @code{.cpp}
 *   #include "src/bench/inc/Perf.hpp"     // CPU baseline
 *   #include "src/bench/inc/PerfGpu.hpp"  // GPU extensions
 *
 *   PERF_GPU_TEST(MatrixMul, CudaKernel) {
 *     UB_PERF_GPU_GUARD(perf);
 *
 *     // CPU baseline
 *     auto cpuResult = perf.cpuBaseline([&]{ matmul_cpu(A, B, C, N); });
 *
 *     // GPU kernel
 *     auto gpuResult = perf.cudaKernel([&](cudaStream_t s){
 *       matmul_kernel<<<grid, block, 0, s>>>(d_A, d_B, d_C, N);
 *     }).measure();
 *
 *     EXPECT_GT(gpuResult.speedupVsCpu, 10.0);
 *   }
 *
 *   PERF_MAIN()
 * @endcode
 */

// Core CPU framework (needed for base types and config)
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/PerfTestMacros.hpp"
#include "src/bench/inc/PerfValidation.hpp"

// GPU-specific components (PerfGpuHarness includes profiler hooks and Nsight)
#include "src/bench/inc/PerfGpuConfig.hpp"     // GPU config extensions
#include "src/bench/inc/PerfGpuHarness.hpp"    // GPU timing harness + profiler hooks
#include "src/bench/inc/PerfGpuStats.hpp"      // GPU metrics
#include "src/bench/inc/PerfGpuTestMacros.hpp" // PERF_GPU_TEST macros

#endif // VERNIER_PERFGPU_HPP