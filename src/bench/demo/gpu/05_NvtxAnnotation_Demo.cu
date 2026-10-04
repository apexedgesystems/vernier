/**
 * @file 05_NvtxAnnotation_Demo.cu
 * @brief Demo 13: NVTX ranges that show SAXPY G1's three phases on an Nsight
 *        Systems timeline.
 *
 * An NVTX range is a name the program records around a stretch of its own
 * code; Nsight Systems draws it above the CUDA work of the same moment. One G1
 * call, two ways:
 *  - G1: the example's SaxpyG1::apply as it is, with no range of its own.
 *    Under --profile nsight the timeline shows only the range Vernier opens
 *    around each test's measurement.
 *  - G1Phases: the same work, driven by this demo one phase at a time, each
 *    phase in a named range: copy_in, kernel, copy_out.
 *
 * A range is opened and closed by the host thread, and a CUDA copy or launch
 * returns once it is queued. A range around those calls alone would only be
 * sure to hold the queueing: the GPU might run some or all of the work while
 * it is open, but nothing would make the work finish inside it. So each phase
 * waits for its stream before its range closes. The waits are what let the
 * timeline show the phases; they cost G1Phases two more waits a call than G1.
 *
 * Workload: y = a*x + y over 1M floats
 * ([examples/saxpy](../examples/saxpy/inc/Saxpy.hpp)).
 *
 * Give the run a cycle count: a call takes a fraction of a millisecond, so the
 * default 10,000 cycles per repeat would keep each test busy for over a minute.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure both, write the CSV the walkthrough reads
 *   ./BenchDemo_Gpu_05_NvtxAnnotation --cycles 20 --repeats 10 --csv nvtx_annotation.csv
 *
 *   # Nsight Systems on the annotated call, then its ranges
 *   bench run ./BenchDemo_Gpu_05_NvtxAnnotation --profile nsight -- \
 *     --gtest_filter=NvtxAnnotation.G1Phases --cycles 20 --repeats 3
 *   nsys stats --force-export=true --report nvtx_pushpop_sum \
 *     bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile.nsys-rep
 *   @endcode
 *
 * @see docs/13_NVTX_ANNOTATION.md for the walkthrough this demo backs
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"
#include "src/bench/demo/gpu/05_NvtxAnnotation_Phases.hpp"
#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ubd = vernier::bench::demo;
namespace phase = vernier::bench::demo::nvtx_annotation;

namespace {

/* ----------------------------- Constants ----------------------------- */

constexpr std::size_t N = 1024 * 1024;           ///< Elements per call
constexpr std::size_t BYTES = N * sizeof(float); ///< One vector, in bytes
constexpr int THREADS_PER_BLOCK = 256;           ///< G1's launch shape
constexpr float A = 2.5F;                        ///< The scalar of a*x + y
constexpr float X_VALUE = 1.5F;                  ///< Every element of x
constexpr float Y_VALUE = 0.5F;                  ///< Every element of y at the start

/* ----------------------------- Helpers ----------------------------- */

/** @brief The answer a*x + y after @p calls calls, for the constants above. */
double expectedAfter(std::size_t calls) {
  return static_cast<double>(Y_VALUE) + static_cast<double>(calls) * A * X_VALUE;
}

/** @brief Turn a failed CUDA call into an exception naming the call. */
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

/**
 * @brief G1 driven one phase at a time. Its buffers are G1's, set up once as
 *        the example's SaxpyG1 sets up its own (x and y on the device, pinned
 *        host staging for both, one stream), and freed however the test leaves.
 */
class SaxpyG1InPhases {
public:
  SaxpyG1InPhases() {
    if (cudaMalloc(&dX_, BYTES) != cudaSuccess) {
      dX_ = nullptr;
    }
    if (cudaMalloc(&dY_, BYTES) != cudaSuccess) {
      dY_ = nullptr;
    }
    if (cudaHostAlloc(&hX_, BYTES, cudaHostAllocDefault) != cudaSuccess) {
      hX_ = nullptr;
    }
    if (cudaHostAlloc(&hY_, BYTES, cudaHostAllocDefault) != cudaSuccess) {
      hY_ = nullptr;
    }
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      stream_ = nullptr;
    }
  }

  ~SaxpyG1InPhases() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    cudaFreeHost(hY_);
    cudaFreeHost(hX_);
    cudaFree(dY_);
    cudaFree(dX_);
  }

  SaxpyG1InPhases(const SaxpyG1InPhases&) = delete;
  SaxpyG1InPhases& operator=(const SaxpyG1InPhases&) = delete;

  [[nodiscard]] bool ok() const {
    return dX_ != nullptr && dY_ != nullptr && hX_ != nullptr && hY_ != nullptr &&
           stream_ != nullptr;
  }

  /**
   * @brief One G1 call, y = a*x + y, in three named phases. Each phase waits
   *        for the stream before its range closes, so the range holds the
   *        phase's GPU work.
   */
  void apply(float a, const std::vector<float>& x, std::vector<float>& y) {
    {
      BENCH_NVTX_SCOPE(phase::COPY_IN);
      std::memcpy(hX_, x.data(), BYTES);
      std::memcpy(hY_, y.data(), BYTES);
      check(cudaMemcpyAsync(dX_, hX_, BYTES, cudaMemcpyHostToDevice, stream_), "HtoD x");
      check(cudaMemcpyAsync(dY_, hY_, BYTES, cudaMemcpyHostToDevice, stream_), "HtoD y");
      check(cudaStreamSynchronize(stream_), "copy in");
    }
    {
      BENCH_NVTX_SCOPE(phase::KERNEL);
      ubd::launchSaxpy(a, dX_, dY_, N, THREADS_PER_BLOCK, stream_);
      check(cudaGetLastError(), "launch");
      check(cudaStreamSynchronize(stream_), "kernel");
    }
    {
      BENCH_NVTX_SCOPE(phase::COPY_OUT);
      check(cudaMemcpyAsync(hY_, dY_, BYTES, cudaMemcpyDeviceToHost, stream_), "DtoH y");
      check(cudaStreamSynchronize(stream_), "copy out");
      std::memcpy(y.data(), hY_, BYTES);
    }
  }

private:
  float* dX_ = nullptr;
  float* dY_ = nullptr;
  float* hX_ = nullptr;
  float* hY_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test G1 as the example drives it: both copies in, the launch and the copy
 *       back queued at once, then one wait. It opens no range of its own.
 */
PERF_TEST(NvtxAnnotation, G1) {
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

  // Every call computed a*x + y and brought y back.
  ASSERT_GT(calls, 0U);
  EXPECT_NEAR(static_cast<double>(y[0]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
  EXPECT_NEAR(static_cast<double>(y[N - 1]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
}

/**
 * @test The same work in three named phases, copy_in, kernel and copy_out,
 *       each waiting for its stream before its range closes.
 */
PERF_TEST(NvtxAnnotation, G1Phases) {
  PERF_GUARD(perf);

  const std::vector<float> X(N, X_VALUE);
  std::vector<float> y(N, Y_VALUE);
  SaxpyG1InPhases runner;
  ASSERT_TRUE(runner.ok()) << "allocating G1's buffers failed";
  std::size_t calls = 0;
  const auto CALL = [&] {
    runner.apply(A, X, y);
    ++calls;
  };

  perf.warmup(CALL);
  (void)perf.throughputLoop(CALL, "saxpy_g1_phases");

  // The phased call computes what G1 does.
  ASSERT_GT(calls, 0U);
  EXPECT_NEAR(static_cast<double>(y[0]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
  EXPECT_NEAR(static_cast<double>(y[N - 1]), expectedAfter(calls), 1e-3 * expectedAfter(calls));
}

/* ----------------------------- Main ----------------------------- */

PERF_GPU_MAIN()
