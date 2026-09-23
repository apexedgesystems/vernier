/**
 * @file 01_GpuBasicWorkflow_Demo.cu
 * @brief Demo 10: the first GPU measurement, on the shared SAXPY example.
 *
 * Three measurements of one workload:
 *  1. the CPU loop, which every GPU result in the suite is compared against;
 *  2. the kernel with its transfers, which is what a GPU call costs;
 *  3. the kernel alone, which is what the GPU is capable of once the data is
 *     already there.
 *
 * Workload: y = a*x + y over 1M floats
 * ([examples/saxpy](../examples/saxpy/inc/Saxpy.hpp)).
 *
 * The suite must run whole: the speedup column of the GPU tests comes from the
 * CpuBaseline test, which has to have run in the same process. A filtered run
 * measures the same times and skips the comparison.
 *
 * Usage:
 *   @code{.sh}
 *   # Run all three, write the CSV the walkthrough reads
 *   ./BenchDemo_Gpu_01_GpuBasicWorkflow --repeats 10 --csv gpu_results.csv
 *
 *   # The three rows, by name
 *   bench summary gpu_results.csv
 *   @endcode
 *
 * @see docs/10_GPU_BASIC_WORKFLOW.md for the walkthrough this demo backs
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;
namespace ubd = vernier::bench::demo;

namespace {

/* ----------------------------- Constants ----------------------------- */

constexpr std::size_t N = 1024 * 1024;           ///< Elements per call
constexpr std::size_t BYTES = N * sizeof(float); ///< One vector, in bytes
constexpr int BLOCK_SIZE = 256;                  ///< Threads per block
constexpr float A = 2.5F;                        ///< The scalar of a*x + y
constexpr float X_VALUE = 1.5F;                  ///< Every element of x
constexpr float Y_VALUE = 0.5F;                  ///< Every element of y at the start

/**
 * @brief The copies of a round trip cost at least this many times the kernel.
 *
 * The walkthrough's claim is that transfers dominate a round trip of this
 * shape, so the test fails where they stop dominating, not where a machine is
 * noisy. Reference rig (Jetson AGX Thor, Release): 6.60 to 7.00 over sixteen
 * runs with the clocks locked per the rig document, 7.90 to 8.16 over six with
 * them left to the governor; a discrete GPU over PCIe, 234 to 255. The lowest
 * reading is 2.64 times the bound.
 */
constexpr double MIN_TRANSFER_TO_KERNEL = 2.5;

/**
 * @brief The kernel alone beats the CPU loop by at least this much.
 *
 * The bound states the direction the demo teaches, not the reference rig's
 * figure, which depends on the memory system and on whether the clocks are
 * pinned. Reference rig: 8.91x to 9.60x over sixteen runs with the clocks
 * locked, 8.61x to 8.75x over six with them left to the governor; a discrete
 * GPU over PCIe, roughly 19x to 22x. The lowest reading is 2.87 times the bound.
 */
constexpr double MIN_KERNEL_ONLY_SPEEDUP = 3.0;

/* ----------------------------- Helpers ----------------------------- */

/** @brief The answer a*x + y after @p calls calls, for the constants above. */
double expectedAfter(std::size_t calls) {
  return static_cast<double>(Y_VALUE) + static_cast<double>(calls) * A * X_VALUE;
}

/** @brief Device buffers for one measurement, freed however the test leaves. */
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

private:
  float* dX_ = nullptr;
  float* dY_ = nullptr;
};

} // namespace

/**
 * @test The CPU loop: the reference every GPU number here is read against.
 *
 * GPU tests of this suite that run after it in the same process take their
 * `speedupVsCpu` column from this baseline, so keep it the first test of the
 * suite and do not filter it out.
 */
PERF_GPU_COMPARISON(GpuBasicWorkflow, CpuBaseline) {
  UB_PERF_GPU_GUARD(perf);

  const std::vector<float> X(N, X_VALUE);
  std::vector<float> y(N, Y_VALUE);
  std::size_t calls = 0;

  (void)perf.cpuBaseline(
      [&] {
        ubd::saxpyCpu(A, X, y);
        ++calls;
      },
      "saxpy_cpu");

  // The loop really ran, and every call computed a*x + y: the first and last
  // elements hold the scalar applied once per call, to the tolerance single
  // precision leaves after that many additions.
  ASSERT_GT(calls, 0U);
  EXPECT_NEAR(static_cast<double>(y[0]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
  EXPECT_NEAR(static_cast<double>(y[N - 1]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
}

/**
 * @test The kernel with its transfers: what one GPU call costs end to end.
 *
 * The harness times the host-to-device leg, the launches and the
 * device-to-host leg separately, and the wall columns report one round trip.
 * On a rig whose CPU and GPU share DRAM the transfers are memory-to-memory
 * copies and still dominate; on a discrete GPU they cross PCIe and dominate
 * by more.
 */
PERF_GPU_COMPARISON(GpuBasicWorkflow, GpuWithTransfers) {
  UB_PERF_GPU_GUARD(perf);

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";

  const std::vector<float> X(N, X_VALUE);
  std::vector<float> y(N, Y_VALUE);

  const dim3 BLOCK(BLOCK_SIZE);
  const dim3 GRID(static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE));

  const auto LAUNCH = [&](cudaStream_t s) {
    ubd::launchSaxpy(A, device.x(), device.y(), N, BLOCK_SIZE, s);
  };

  // The warmup launches the kernel, which reads both vectors, so the device
  // gets real inputs first; the measured copies declared below repeat them.
  ASSERT_EQ(cudaMemcpy(device.x(), X.data(), BYTES, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device.y(), y.data(), BYTES, cudaMemcpyHostToDevice), cudaSuccess);
  perf.cudaWarmup(LAUNCH);

  const ub::PerfGpuResult RESULT = perf.cudaKernel(LAUNCH, "saxpy_gpu_with_transfers")
                                       .withHostToDevice(X.data(), device.x(), BYTES)
                                       .withHostToDevice(y.data(), device.y(), BYTES)
                                       .withDeviceToHost(device.y(), y.data(), BYTES)
                                       .withLaunchConfig(GRID, BLOCK)
                                       .measure();

  // The effect this test demonstrates: the copies cost more than the kernel.
  ASSERT_GT(RESULT.kernelTimeUs, 0.0);
  EXPECT_GT(RESULT.transferTimeUs, MIN_TRANSFER_TO_KERNEL * RESULT.kernelTimeUs)
      << "transfers no longer dominate a round trip on this machine";
  EXPECT_EQ(RESULT.stats.transfers.h2dBytes, 2 * BYTES);
  EXPECT_EQ(RESULT.stats.transfers.d2hBytes, BYTES);
}

/**
 * @test The kernel alone: what the GPU does once the data is already there.
 *
 * No transfer is declared, so nothing but the launches is timed and the wall
 * time is the kernel time.
 */
PERF_GPU_COMPARISON(GpuBasicWorkflow, GpuKernelOnly) {
  UB_PERF_GPU_GUARD(perf);

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";

  const std::vector<float> X(N, X_VALUE);
  std::vector<float> y(N, Y_VALUE);
  ASSERT_EQ(cudaMemcpy(device.x(), X.data(), BYTES, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device.y(), y.data(), BYTES, cudaMemcpyHostToDevice), cudaSuccess);

  const dim3 BLOCK(BLOCK_SIZE);
  const dim3 GRID(static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE));

  const auto LAUNCH = [&](cudaStream_t s) {
    ubd::launchSaxpy(A, device.x(), device.y(), N, BLOCK_SIZE, s);
  };
  perf.cudaWarmup(LAUNCH);

  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(LAUNCH, "saxpy_gpu_kernel_only").withLaunchConfig(GRID, BLOCK).measure();

  // Nothing was copied, so nothing was booked as transfer time.
  EXPECT_DOUBLE_EQ(RESULT.transferTimeUs, 0.0);
  EXPECT_DOUBLE_EQ(RESULT.totalTimeUs, RESULT.kernelTimeUs);

  // The effect this test demonstrates. The speedup needs the CpuBaseline test
  // of this suite to have run in this process; a filtered run has no baseline
  // to compare against and says so instead of asserting nothing.
  if (RESULT.speedupVsCpu <= 0.0) {
    GTEST_SKIP() << "no CPU baseline in this run: run the whole suite for the speedup";
  }
  EXPECT_GT(RESULT.speedupVsCpu, MIN_KERNEL_ONLY_SPEEDUP)
      << "the kernel no longer beats the CPU loop on this machine";
}

/* ----------------------------- Main ----------------------------- */

PERF_GPU_MAIN()
