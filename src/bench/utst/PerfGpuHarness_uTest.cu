/**
 * @file PerfGpuHarness_uTest.cu
 * @brief Unit tests for what the GPU harness measures and publishes.
 *
 * These need a CUDA device: they run a small kernel through the public
 * PerfGpuCase API and check the result and the row the harness leaves in
 * PerfRegistry. Without a device every test skips.
 *
 * Each test uses a test-name prefix of its own where process-wide state is
 * involved (the per-suite CPU baseline), so the order tests run in, and
 * repeated runs in one process, cannot change an outcome.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

constexpr int ELEMENTS = 1 << 16;
constexpr int BLOCK = 256;

/** @brief y = a * x + y over ELEMENTS floats. */
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    y[IDX] = a * x[IDX] + y[IDX];
  }
}

/** @brief A distinct suite name per call, so process-wide state never leaks between tests. */
std::string uniqueSuite(const char* stem) {
  static std::atomic<int> counter{0};
  return std::string(stem) + std::to_string(counter.fetch_add(1));
}

/** @brief Device and host buffers for one measurement. */
class SaxpyFixtureData {
public:
  SaxpyFixtureData() : hostX_(ELEMENTS, 1.0F), hostY_(ELEMENTS, 2.0F) {
    cudaMalloc(&deviceX_, bytes());
    cudaMalloc(&deviceY_, bytes());
    cudaMemcpy(deviceX_, hostX_.data(), bytes(), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY_, hostY_.data(), bytes(), cudaMemcpyHostToDevice);
  }

  ~SaxpyFixtureData() {
    cudaFree(deviceX_);
    cudaFree(deviceY_);
  }

  SaxpyFixtureData(const SaxpyFixtureData&) = delete;
  SaxpyFixtureData& operator=(const SaxpyFixtureData&) = delete;

  [[nodiscard]] static std::size_t bytes() { return ELEMENTS * sizeof(float); }
  [[nodiscard]] float* deviceX() const { return deviceX_; }
  [[nodiscard]] float* deviceY() const { return deviceY_; }
  [[nodiscard]] const float* hostX() const { return hostX_.data(); }
  [[nodiscard]] float* hostY() { return hostY_.data(); }

  /** @brief The kernel launch the tests measure. */
  [[nodiscard]] ub::PerfGpuCase::KernelFn launch() const {
    float* x = deviceX_;
    float* y = deviceY_;
    return [x, y](cudaStream_t s) {
      saxpyKernel<<<(ELEMENTS + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(2.0F, x, y, ELEMENTS);
    };
  }

private:
  std::vector<float> hostX_;
  std::vector<float> hostY_;
  float* deviceX_ = nullptr;
  float* deviceY_ = nullptr;
};

} // namespace

/* ----------------------------- Fixture ----------------------------- */

/** @brief Skips when no CUDA device is present; keeps the registry slot clean. */
class PerfGpuHarnessTest : public ::testing::Test {
protected:
  ub::PerfConfig cfg_{};

  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ub::ensureBenchGpuAbi();

    cfg_.cycles = 50;
    cfg_.repeats = 10;
    cfg_.warmup = 1;
    cfg_.msgBytes = 64; // recommendedCVThreshold(cfg) is 0.10 here, not the 0.05 default
    (void)ub::PerfRegistry::instance().take();
  }

  void TearDown() override { (void)ub::PerfRegistry::instance().take(); }

  /** @brief The row the harness published for the measurement just taken. */
  static ub::PerfRow lastRow() {
    auto row = ub::PerfRegistry::instance().take();
    EXPECT_TRUE(row.has_value()) << "the harness published no row";
    return row.value_or(ub::PerfRow{});
  }
};

/* ----------------------------- Wall time ----------------------------- */

/**
 * @test The wall time of a transferring test is one round trip: the H2D leg,
 *       one kernel launch and the D2H leg, not that sum divided by the cycle
 *       count.
 */
TEST_F(PerfGpuHarnessTest, RoundTripIsTransfersPlusOneLaunch) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuRoundTrip") + ".Transfers", cfg_};
  perf.cudaWarmup(data.launch());

  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(data.launch(), "saxpy")
          .withHostToDevice(data.hostX(), data.deviceX(), SaxpyFixtureData::bytes())
          .withDeviceToHost(data.deviceY(), data.hostY(), SaxpyFixtureData::bytes())
          .measure();

  const double LEGS = RESULT.kernelTimeUs + RESULT.transferTimeUs;
  ASSERT_GT(RESULT.transferTimeUs, 0.0) << "the transfer legs were not measured at all";
  EXPECT_NEAR(RESULT.totalTimeUs, LEGS, 0.25 * LEGS);
  EXPECT_NEAR(RESULT.callsPerSecond, 1e6 / RESULT.totalTimeUs, 1.0);

  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.kernelTimeUs.has_value());
  EXPECT_DOUBLE_EQ(ROW.stats.median, RESULT.totalTimeUs);
  EXPECT_DOUBLE_EQ(*ROW.kernelTimeUs, RESULT.kernelTimeUs);
}

/**
 * @test A kernel-only test records no transfer time: with no declared
 *       transfers there is no leg to time.
 */
TEST_F(PerfGpuHarnessTest, KernelOnlyRecordsNoTransferTime) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuKernelOnly") + ".NoTransfers", cfg_};
  perf.cudaWarmup(data.launch());

  const ub::PerfGpuResult RESULT = perf.cudaKernel(data.launch(), "saxpy").measure();

  EXPECT_DOUBLE_EQ(RESULT.transferTimeUs, 0.0);
  EXPECT_DOUBLE_EQ(RESULT.stats.transfers.h2dTimeUs, 0.0);
  EXPECT_DOUBLE_EQ(RESULT.stats.transfers.d2hTimeUs, 0.0);
  EXPECT_EQ(RESULT.stats.transfers.h2dBytes, 0U);
  EXPECT_DOUBLE_EQ(RESULT.totalTimeUs, RESULT.kernelTimeUs);

  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.transferTimeUs.has_value());
  EXPECT_DOUBLE_EQ(*ROW.transferTimeUs, 0.0);
}

/* ----------------------------- Speedup ----------------------------- */

/**
 * @test A CPU baseline reaches the GPU cases of the same suite, whichever
 *       object measured it.
 */
TEST_F(PerfGpuHarnessTest, SpeedupUsesTheSuiteBaseline) {
  const std::string SUITE = uniqueSuite("GpuSuiteBaseline");
  SaxpyFixtureData data;

  // A separate PerfGpuCase per test case, as one TEST per case produces.
  ub::PerfGpuCase baselineCase{SUITE + ".CpuBaseline", cfg_};
  std::vector<float> x(ELEMENTS, 1.0F);
  std::vector<float> y(ELEMENTS, 2.0F);
  const ub::PerfResult CPU = baselineCase.cpuBaseline([&] {
    for (int i = 0; i < ELEMENTS; ++i) {
      y[i] = 2.0F * x[i] + y[i];
    }
  });
  (void)ub::PerfRegistry::instance().take();

  ub::PerfGpuCase gpuCase{SUITE + ".Kernel", cfg_};
  gpuCase.cudaWarmup(data.launch());
  const ub::PerfGpuResult RESULT = gpuCase.cudaKernel(data.launch(), "saxpy").measure();

  ASSERT_GT(CPU.stats.median, 0.0);
  EXPECT_NEAR(RESULT.speedupVsCpu, CPU.stats.median / RESULT.totalTimeUs, 1e-9);

  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.speedupVsCpu.has_value());
  EXPECT_DOUBLE_EQ(*ROW.speedupVsCpu, RESULT.speedupVsCpu);
}

/** @test With no baseline anywhere in its suite, a GPU row leaves the speedup cell empty. */
TEST_F(PerfGpuHarnessTest, SpeedupCellIsEmptyWithoutBaseline) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuNoBaseline") + ".Kernel", cfg_};
  perf.cudaWarmup(data.launch());

  const ub::PerfGpuResult RESULT = perf.cudaKernel(data.launch(), "saxpy").measure();

  EXPECT_DOUBLE_EQ(RESULT.speedupVsCpu, 0.0);
  const ub::PerfRow ROW = lastRow();
  EXPECT_FALSE(ROW.speedupVsCpu.has_value())
      << "an unknown speedup must be an empty cell, not a number";
}

/**
 * @test The multi-GPU path reads the same suite baseline, so its speedup and
 *       scaling efficiency are the measured ones.
 */
TEST_F(PerfGpuHarnessTest, MultiGpuScalingUsesTheSuiteBaseline) {
  const std::string SUITE = uniqueSuite("GpuMultiBaseline");
  SaxpyFixtureData data;

  ub::PerfGpuCase baselineCase{SUITE + ".CpuBaseline", cfg_};
  std::vector<float> x(ELEMENTS, 1.0F);
  std::vector<float> y(ELEMENTS, 2.0F);
  const ub::PerfResult CPU = baselineCase.cpuBaseline([&] {
    for (int i = 0; i < ELEMENTS; ++i) {
      y[i] = 2.0F * x[i] + y[i];
    }
  });
  (void)ub::PerfRegistry::instance().take();

  ub::PerfGpuCase gpuCase{SUITE + ".MultiGpu", cfg_};
  const ub::PerfGpuCase::KernelFn LAUNCH = data.launch();
  const ub::MultiGpuResult RESULT =
      gpuCase.cudaKernelMultiGpu(1, [&LAUNCH](int, cudaStream_t s) { LAUNCH(s); }).measure();

  ASSERT_EQ(RESULT.perDevice.size(), 1U);
  ASSERT_GT(CPU.stats.median, 0.0);
  const double DEVICE_TIME = RESULT.perDevice[0].kernelTimeUs;
  EXPECT_NEAR(RESULT.totalSpeedupVsCpu, CPU.stats.median / DEVICE_TIME, 1e-9);
  ASSERT_TRUE(RESULT.aggregatedStats.multiGpu.has_value());
  EXPECT_NEAR(RESULT.aggregatedStats.multiGpu->scalingEfficiency, RESULT.totalSpeedupVsCpu, 1e-9);

  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.speedupVsCpu.has_value());
  EXPECT_DOUBLE_EQ(*ROW.speedupVsCpu, RESULT.totalSpeedupVsCpu);
}
