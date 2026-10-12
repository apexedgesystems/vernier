/**
 * @file 04_ComputeSanitizerProfiler_Demo.cu
 * @brief Demo 04: Compute Sanitizer -- is the device code correct
 *
 * Measures the shared SAXPY kernel and carries a copy of it without its
 * bounds guard for compute-sanitizer to find:
 *  1. SaxpyKernel times the example's kernel on its own and checks its
 *     answer, one CSV row
 *  2. SaxpyUnguarded launches the unguarded copy once on a grid that rounds
 *     up, so one thread runs past the end of both vectors; it runs only under
 *     compute-sanitizer and skips itself anywhere else
 *
 * Both cases use 1,048,575 elements, one less than 4,096 blocks of 256
 * cover: the guard is the only difference between the two kernels.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_Gpu_04_ComputeSanitizerProfiler --repeats 10 --csv compute_sanitizer.csv
 *
 *   # Find the bug: the tool's log lands in
 *   # bench-out/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log
 *   bench run ./BenchDemo_Gpu_04_ComputeSanitizerProfiler --profile compute-sanitizer -- \
 *     --gtest_filter=ComputeSanitizer.SaxpyUnguarded
 *
 *   # Read it
 *   cat bench-out/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log
 *   @endcode
 *
 * What the tool reports for this binary is checked apart from it, by
 * utst/04_ComputeSanitizerProfiler_uTest.cpp.
 *
 * @see docs/17_COMPUTE_SANITIZER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <vector>

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"
#include "src/bench/demo/gpu/04_ComputeSanitizerProfiler_Unguarded.hpp"
#include "src/bench/demo/helpers/SkipUnlessUnderComputeSanitizer.hpp"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;
namespace ubd = vernier::bench::demo;
namespace wrong = vernier::bench::demo::sanitizer_demo;

namespace {

/* ----------------------------- Constants ----------------------------- */

constexpr int BLOCK_SIZE = 256;                  ///< Threads per block
constexpr std::size_t N = 4096 * BLOCK_SIZE - 1; ///< One less than the grid covers
constexpr std::size_t BYTES = N * sizeof(float); ///< One vector, in bytes
constexpr float A = 2.5F;                        ///< The scalar of a*x + y
constexpr float X_VALUE = 1.5F;                  ///< Every element of x
constexpr float Y_VALUE = 0.5F;                  ///< Every element of y at the start

/* ----------------------------- Helpers ----------------------------- */

/** @brief The answer a*x + y after @p launches launches, for the constants above. */
double expectedAfter(std::size_t launches) {
  return static_cast<double>(Y_VALUE) + static_cast<double>(launches) * A * X_VALUE;
}

/** @brief Device buffers for one case, freed however the test leaves. */
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

  /** @brief Fill x and y on the device from the constants above. */
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

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test The shared kernel on its own: what it costs, and that every launch
 *       reaches both ends of the vectors, its guard included.
 *
 * No transfer is declared, so the wall time is the kernel time. After the
 * measurement the first and the last element hold the scalar applied once
 * per launch, so a kernel that missed either would not pass. A stray access
 * past the end can leave both right: staying in bounds is memcheck's to check.
 */
PERF_GPU_TEST(ComputeSanitizer, SaxpyKernel) {
  UB_PERF_GPU_GUARD(perf);

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  ASSERT_TRUE(device.fill()) << "copy to the device failed";

  const dim3 BLOCK(BLOCK_SIZE);
  const dim3 GRID(static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE));
  std::size_t launches = 0;
  const auto LAUNCH = [&](cudaStream_t s) {
    ubd::launchSaxpy(A, device.x(), device.y(), N, BLOCK_SIZE, s);
    ++launches;
  };
  perf.cudaWarmup(LAUNCH);

  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(LAUNCH, "saxpy_kernel").withLaunchConfig(GRID, BLOCK).measure();

  // Nothing was copied inside the measurement, so the wall time is the kernel's.
  EXPECT_DOUBLE_EQ(RESULT.transferTimeUs, 0.0);
  EXPECT_DOUBLE_EQ(RESULT.totalTimeUs, RESULT.kernelTimeUs);

  // The effect this case checks: every launch computed a*x + y at both ends,
  // to the tolerance single precision leaves after that many adds.
  std::vector<float> y(N, 0.0F);
  ASSERT_EQ(cudaMemcpy(y.data(), device.y(), BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_GT(launches, 0U);
  EXPECT_NEAR(static_cast<double>(y[0]), expectedAfter(launches), 1e-3 * expectedAfter(launches));
  EXPECT_NEAR(static_cast<double>(y[N - 1]), expectedAfter(launches),
              1e-3 * expectedAfter(launches));
}

/**
 * @test The unguarded copy, launched once past the end of its buffers.
 *       Runs only under compute-sanitizer.
 *
 * Measures nothing: what happens to the kernel is the tool's doing (memcheck
 * stops it at its first invalid access and, by default, ends the CUDA
 * context, so the sync and the frees that follow report a launch failure),
 * and what the tool reports is read by the check beside this demo. The case
 * asserts that the launch was accepted and prints what the device reported.
 * Anywhere else it skips itself and says how to run it.
 */
PERF_GPU_TEST(ComputeSanitizer, SaxpyUnguarded) {
  DEMO_SKIP_UNLESS_UNDER_COMPUTE_SANITIZER();

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  ASSERT_TRUE(device.fill()) << "copy to the device failed";

  wrong::launchSaxpyUnguarded(A, device.x(), device.y(), N, BLOCK_SIZE, nullptr);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "the launch was refused";
  const cudaError_t END = cudaDeviceSynchronize();
  std::printf("[ComputeSanitizer.SaxpyUnguarded]  one launch over %zu elements, %u blocks of %d; "
              "the device reported: %s\n",
              N, static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE,
              cudaGetErrorString(END));
}

/* ----------------------------- Main ----------------------------- */

PERF_GPU_MAIN()
