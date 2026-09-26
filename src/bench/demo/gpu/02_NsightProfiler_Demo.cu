/**
 * @file 02_NsightProfiler_Demo.cu
 * @brief Demo 11: Nsight Systems and Nsight Compute on the shared SAXPY example.
 *
 * Two ways of driving the same kernel, and what the profilers say about them:
 *  - G0 allocates both device buffers, copies, launches one thread per block
 *    and frees, on every call;
 *  - G1 keeps its buffers and launches 256 threads per block.
 * Nsight Systems shows where G0's time goes (an allocation and a free per
 * buffer per call, and a slow kernel); Nsight Compute shows what limits that
 * kernel (its launch shape).
 *
 * Tests:
 *  - G0, G1: one call of each version, end to end, per measured cycle.
 *  - KernelOneThreadPerBlock, Kernel256ThreadsPerBlock: the bare kernel in
 *    G0's and in G1's launch shape, timed by the GPU harness; each fails when
 *    its occupancy stops matching its shape.
 *  - LaunchShapeSpeedup: times both shapes itself, writes no CSV row, and
 *    fails when one thread per block stops being the slow shape. It compares
 *    timings, so leave it out of a run under Nsight Compute, which replays
 *    every launch (the filter NsightProfiler.Kernel* does).
 *
 * G0 against G1 end to end is not asserted: where the copies dominate (a
 * discrete GPU over PCIe) the two take about as long; the counts that tell
 * them apart are pinned by the example's host test (SaxpyDriving_uTest.cpp).
 *
 * Workload: y = a*x + y over 1M floats
 * ([examples/saxpy](../examples/saxpy/inc/Saxpy.hpp)).
 *
 * Give the run a cycle count: G0 and the one-thread-per-block kernel take
 * milliseconds per call, so the default 10,000 cycles per repeat would keep
 * each of them busy for minutes.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure all four, write the CSV the walkthrough reads
 *   ./BenchDemo_Gpu_02_NsightProfiler --cycles 20 --repeats 10 --csv nsight_profiler.csv
 *
 *   # Nsight Systems on G0: what happened, and when
 *   bench run ./BenchDemo_Gpu_02_NsightProfiler --profile nsight -- \
 *     --gtest_filter=NsightProfiler.G0 --cycles 20 --repeats 3
 *
 *   # Nsight Compute on the two kernel shapes: what limits the kernel (root on Jetson)
 *   bench run ./BenchDemo_Gpu_02_NsightProfiler --profile ncu --cycles 3 --repeats 1 -- \
 *     --gtest_filter='NsightProfiler.Kernel*'
 *   @endcode
 *
 * @see docs/11_NSIGHT_PROFILER.md for the walkthrough this demo backs
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"
#include "src/bench/demo/gpu/02_NsightProfiler_Timing.hpp"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;
namespace ubd = vernier::bench::demo;

namespace {

/* ----------------------------- Constants ----------------------------- */

constexpr std::size_t N = 1024 * 1024;           ///< Elements per call
constexpr std::size_t BYTES = N * sizeof(float); ///< One vector, in bytes
constexpr float A = 2.5F;                        ///< The scalar of a*x + y
constexpr float X_VALUE = 1.5F;                  ///< Every element of x
constexpr float Y_VALUE = 0.5F;                  ///< Every element of y at the start

/// G0's launch shape: one thread per block, so a million blocks.
constexpr int ONE_THREAD = 1;
/// G1's launch shape: 256 threads per block.
constexpr int WIDE_BLOCK = 256;

/// Launches per sample, and samples per shape, in the guard test.
constexpr int GUARD_CALLS = 5;
constexpr int GUARD_SAMPLES = 5;

/**
 * @brief The kernel in G0's launch shape takes at least this many times as
 * long as in G1's.
 *
 * The bound states the direction the walkthrough teaches, far inside what the
 * GPUs measure. Reference rig (Jetson AGX Thor, Release): 141.6x to 143.0x over
 * five runs with the clocks locked per the rig document, 137.8x to 139.0x over
 * four with them as found; an RTX 5000 Ada over PCIe, 99.1x to 99.7x over four
 * runs. The lowest reading is 3.3 times the bound.
 */
constexpr double MIN_KERNEL_SHAPE_SPEEDUP = 30.0;

/**
 * @brief Largest occupancy the harness estimates for one thread per block.
 *
 * A one-thread block is one warp, and an SM holds only as many blocks as its
 * block limit, so the estimate is that limit over the warps an SM can hold:
 * 0.5 on the reference rig (compute capability 11.0) and on an RTX 5000 Ada
 * (8.9).
 */
constexpr double MAX_ONE_THREAD_OCCUPANCY = 0.5;

/**
 * @brief Smallest occupancy the harness estimates for 256 threads per block.
 *
 * Eight warps per block reach the SM's warp limit before its block limit: 1.0
 * on the reference rig and on an RTX 5000 Ada.
 */
constexpr double MIN_WIDE_OCCUPANCY = 0.75;

/* ----------------------------- Helpers ----------------------------- */

/** @brief The answer a*x + y after @p calls calls, for the constants above. */
double expectedAfter(std::size_t calls) {
  return static_cast<double>(Y_VALUE) + static_cast<double>(calls) * A * X_VALUE;
}

/** @brief Device buffers for one kernel test, freed however the test leaves. */
class DeviceVectors {
public:
  DeviceVectors() {
    if (cudaMalloc(&dX_, BYTES) != cudaSuccess) {
      dX_ = nullptr;
    }
    if (cudaMalloc(&dY_, BYTES) != cudaSuccess) {
      dY_ = nullptr;
    }
  }

  ~DeviceVectors() {
    cudaFree(dY_);
    cudaFree(dX_);
  }

  DeviceVectors(const DeviceVectors&) = delete;
  DeviceVectors& operator=(const DeviceVectors&) = delete;

  [[nodiscard]] bool ok() const { return dX_ != nullptr && dY_ != nullptr; }
  [[nodiscard]] float* x() const { return dX_; }
  [[nodiscard]] float* y() const { return dY_; }

  /** @brief Fill both vectors with the demo's values. */
  [[nodiscard]] bool fill() const {
    const std::vector<float> X(N, X_VALUE);
    const std::vector<float> Y(N, Y_VALUE);
    return cudaMemcpy(dX_, X.data(), BYTES, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(dY_, Y.data(), BYTES, cudaMemcpyHostToDevice) == cudaSuccess;
  }

private:
  float* dX_ = nullptr;
  float* dY_ = nullptr;
};

/** @brief Grid for @p threadsPerBlock threads per block over N elements. */
dim3 gridFor(int threadsPerBlock) {
  return dim3(static_cast<unsigned>((N + static_cast<std::size_t>(threadsPerBlock) - 1) /
                                    static_cast<std::size_t>(threadsPerBlock)));
}

/** @brief The bare kernel in one launch shape, measured by the GPU harness. */
ub::PerfGpuResult measureKernelShape(ub::PerfGpuCase& perf, const DeviceVectors& device,
                                     int threadsPerBlock, const char* label) {
  const auto LAUNCH = [&device, threadsPerBlock](cudaStream_t s) {
    ubd::launchSaxpy(A, device.x(), device.y(), N, threadsPerBlock, s);
  };
  perf.cudaWarmup(LAUNCH);
  return perf.cudaKernel(LAUNCH, label)
      .withLaunchConfig(gridFor(threadsPerBlock), dim3(threadsPerBlock))
      .measure();
}

} // namespace

/* ----------------------------- End to End ----------------------------- */

/**
 * @test G0, one call end to end: allocate, copy in, launch one thread per
 *       block, copy back, free. Under Nsight Systems: two cudaMalloc and two
 *       cudaFree per call, and a slow kernel.
 */
PERF_TEST(NsightProfiler, G0) {
  PERF_GUARD(perf);

  const std::vector<float> X(N, X_VALUE);
  std::vector<float> y(N, Y_VALUE);
  std::size_t calls = 0;
  const auto CALL = [&] {
    ubd::saxpyG0(A, X, y);
    ++calls;
  };

  perf.warmup(CALL);
  (void)perf.throughputLoop(CALL, "saxpy_g0");

  // Every call computed a*x + y and brought y back.
  ASSERT_GT(calls, 0U);
  EXPECT_NEAR(static_cast<double>(y[0]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
  EXPECT_NEAR(static_cast<double>(y[N - 1]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
}

/**
 * @test G1, one call end to end through buffers set up once, outside the
 *       measurement: copy in, launch 256 threads per block, copy back, wait.
 */
PERF_TEST(NsightProfiler, G1) {
  PERF_GUARD(perf);

  const std::vector<float> X(N, X_VALUE);
  std::vector<float> y(N, Y_VALUE);
  ubd::SaxpyG1 runner(N);
  std::size_t calls = 0;
  const auto CALL = [&] {
    runner.apply(A, X, y);
    ++calls;
  };

  perf.warmup(CALL);
  (void)perf.throughputLoop(CALL, "saxpy_g1");

  ASSERT_GT(calls, 0U);
  EXPECT_NEAR(static_cast<double>(y[0]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
  EXPECT_NEAR(static_cast<double>(y[N - 1]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
}

/* ----------------------------- Kernel Shapes ----------------------------- */

/**
 * @test The kernel in G0's launch shape, (1048576, 1, 1) x (1, 1, 1): one warp
 *       per block with one busy lane, so the SM's block limit caps occupancy.
 */
PERF_GPU_TEST(NsightProfiler, KernelOneThreadPerBlock) {
  PERF_GPU_GUARD(perf);

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  ASSERT_TRUE(device.fill()) << "device fill failed";

  const ub::PerfGpuResult RESULT =
      measureKernelShape(perf, device, ONE_THREAD, "saxpy_kernel_one_thread");

  EXPECT_LE(RESULT.stats.occupancy.achievedOccupancy, MAX_ONE_THREAD_OCCUPANCY)
      << "one thread per block no longer limits occupancy on this GPU";
}

/**
 * @test The kernel in G1's launch shape, (4096, 1, 1) x (256, 1, 1): eight warps
 *       per block fill the SM's warp slots.
 */
PERF_GPU_TEST(NsightProfiler, Kernel256ThreadsPerBlock) {
  PERF_GPU_GUARD(perf);

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  ASSERT_TRUE(device.fill()) << "device fill failed";

  const ub::PerfGpuResult RESULT =
      measureKernelShape(perf, device, WIDE_BLOCK, "saxpy_kernel_256_threads");

  EXPECT_GE(RESULT.stats.occupancy.achievedOccupancy, MIN_WIDE_OCCUPANCY)
      << "256 threads per block no longer fill the SM on this GPU";
}

/* ----------------------------- What Must Stay True ----------------------------- */

/**
 * @test The kernel in G0's launch shape stays slower than in G1's by at least
 *       MIN_KERNEL_SHAPE_SPEEDUP, timed on the device.
 *
 * The two measured kernel tests cannot make this comparison without carrying
 * state from one test to the next, which --gtest_shuffle would break, so this
 * test times both shapes itself and writes no CSV row.
 */
PERF_TEST(NsightProfiler, LaunchShapeSpeedup) {
  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  ASSERT_TRUE(device.fill()) << "device fill failed";

  const double ONE_US =
      ubd::medianLaunchUs(A, device.x(), device.y(), N, ONE_THREAD, GUARD_CALLS, GUARD_SAMPLES);
  const double WIDE_US =
      ubd::medianLaunchUs(A, device.x(), device.y(), N, WIDE_BLOCK, GUARD_CALLS, GUARD_SAMPLES);
  ASSERT_GT(ONE_US, 0.0) << "timing the one-thread-per-block kernel failed";
  ASSERT_GT(WIDE_US, 0.0) << "timing the 256-thread kernel failed";

  std::printf("[NsightProfiler.LaunchShapeSpeedup]  1 thread/block %.3f us/launch  256 "
              "threads/block %.3f us/launch  %.1fx\n",
              ONE_US, WIDE_US, ONE_US / WIDE_US);
  EXPECT_GT(ONE_US, MIN_KERNEL_SHAPE_SPEEDUP * WIDE_US)
      << "the one-thread-per-block kernel is not " << MIN_KERNEL_SHAPE_SPEEDUP
      << "x slower than the 256-thread one: the demo has stopped demonstrating";
}

/* ----------------------------- Main ----------------------------- */

PERF_GPU_MAIN()
