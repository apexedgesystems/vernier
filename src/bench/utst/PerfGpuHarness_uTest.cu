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
#include "src/bench/utst/StderrCapture.hpp"

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

  /**
   * @brief Measure a CPU baseline in a case of its own.
   * @param suite    Suite the case belongs to.
   * @param caseName Case name inside that suite.
   * @param passes   Passes over the vectors per call: the cost knob.
   * @return The baseline median in microseconds.
   */
  double recordBaseline(const std::string& suite, const std::string& caseName, int passes) {
    ub::PerfGpuCase baselineCase{suite + "." + caseName, cfg_};
    std::vector<float> x(ELEMENTS, 1.0F);
    std::vector<float> y(ELEMENTS, 2.0F);
    const ub::PerfResult CPU = baselineCase.cpuBaseline([&] {
      for (int p = 0; p < passes; ++p) {
        for (int i = 0; i < ELEMENTS; ++i) {
          y[i] = 2.0F * x[i] + y[i];
        }
      }
    });
    (void)ub::PerfRegistry::instance().take();
    return CPU.stats.median;
  }

  /** @brief What a GPU case without a baseline of its own reported. */
  struct KernelOutcome {
    double speedup{};
    bool cellSet{};
  };

  /** @brief Measure a kernel in a case of @p suite that records no baseline. */
  KernelOutcome measureKernelCase(const std::string& suite, const SaxpyFixtureData& data,
                                  const std::string& caseName = "Kernel") {
    ub::PerfGpuCase gpuCase{suite + "." + caseName, cfg_};
    gpuCase.cudaWarmup(data.launch());
    const ub::PerfGpuResult RESULT = gpuCase.cudaKernel(data.launch(), "saxpy").measure();
    const ub::PerfRow ROW = lastRow();
    return KernelOutcome{RESULT.speedupVsCpu, ROW.speedupVsCpu.has_value()};
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

  // The wall time and the leg times are medians of separate distributions, so
  // they are close but never equal, and where the clocks are free to move a
  // slow repeat moves one of them more than the other: the ratio measured over
  // 30 runs on an unpinned reference board spans 0.995 to 1.203, and a first
  // run after a rebuild has reached 2.04. The bounds are set so that only a
  // wrong formula can cross them. A wall time divided by the cycle count a
  // second time is 1/50 of the legs at the cycle count used here and 1/10,000
  // at the default, which clears the lower bound by a factor of 25; a wall
  // time that added a leg without dividing it per launch would be about the
  // cycle count times the legs, which clears the upper bound by the same
  // margin the jitter has beneath it.
  const double LEGS = RESULT.kernelTimeUs + RESULT.transferTimeUs;
  ASSERT_GT(RESULT.transferTimeUs, 0.0) << "the transfer legs were not measured at all";
  EXPECT_GT(RESULT.totalTimeUs, 0.5 * LEGS)
      << "a round trip cannot be a fraction of the legs it contains";
  EXPECT_LT(RESULT.totalTimeUs, 10.0 * LEGS)
      << "a round trip is its three legs, not a multiple of them";
  EXPECT_DOUBLE_EQ(RESULT.callsPerSecond, 1e6 / RESULT.totalTimeUs);

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

/**
 * @test A suite whose cases measured two different baselines has no baseline
 *       to share: the GPU case reports no speedup, in either order, rather
 *       than whichever baseline ran last.
 */
TEST_F(PerfGpuHarnessTest, TwoBaselinesInASuiteLeaveTheSpeedupEmpty) {
  SaxpyFixtureData data;

  const std::string CHEAP_FIRST_SUITE = uniqueSuite("GpuTwoBaselines");
  const double CHEAP = recordBaseline(CHEAP_FIRST_SUITE, "CheapBaseline", 1);
  const double EXPENSIVE = recordBaseline(CHEAP_FIRST_SUITE, "ExpensiveBaseline", 8);
  const KernelOutcome CHEAP_FIRST = measureKernelCase(CHEAP_FIRST_SUITE, data);

  const std::string CHEAP_LAST_SUITE = uniqueSuite("GpuTwoBaselines");
  (void)recordBaseline(CHEAP_LAST_SUITE, "ExpensiveBaseline", 8);
  (void)recordBaseline(CHEAP_LAST_SUITE, "CheapBaseline", 1);
  const KernelOutcome CHEAP_LAST = measureKernelCase(CHEAP_LAST_SUITE, data);

  // Precondition: the two baselines are far enough apart that a last-writer
  // rule shows up as a different number, not as noise. The expensive case does
  // eight times the work of the cheap one; the bound is a quarter of that, so
  // neither clock behaviour nor cache warmth can bring the two together.
  ASSERT_GT(EXPENSIVE, 2.0 * CHEAP) << "the two baselines must differ clearly";

  EXPECT_DOUBLE_EQ(CHEAP_FIRST.speedup, CHEAP_LAST.speedup)
      << "the speedup depends on which baseline case ran last";
  EXPECT_DOUBLE_EQ(CHEAP_FIRST.speedup, 0.0);
  EXPECT_FALSE(CHEAP_FIRST.cellSet) << "an ambiguous baseline must leave the cell empty";
  EXPECT_FALSE(CHEAP_LAST.cellSet) << "an ambiguous baseline must leave the cell empty";
}

/** @test A case's own baseline is used even where its suite's is ambiguous. */
TEST_F(PerfGpuHarnessTest, CaseBaselineWinsOverAnAmbiguousSuite) {
  SaxpyFixtureData data;
  const std::string SUITE = uniqueSuite("GpuOwnBaselineWins");
  const double CHEAP = recordBaseline(SUITE, "CheapBaseline", 1);
  const double EXPENSIVE = recordBaseline(SUITE, "ExpensiveBaseline", 8);
  // Eight times the work, asserted at twice: see the order test above.
  ASSERT_GT(EXPENSIVE, 2.0 * CHEAP) << "the two baselines must differ clearly";

  ub::PerfGpuCase gpuCase{SUITE + ".KernelWithOwnBaseline", cfg_};
  std::vector<float> x(ELEMENTS, 1.0F);
  std::vector<float> y(ELEMENTS, 2.0F);
  const ub::PerfResult OWN = gpuCase.cpuBaseline([&] {
    for (int i = 0; i < ELEMENTS; ++i) {
      y[i] = 2.0F * x[i] + y[i];
    }
  });
  (void)ub::PerfRegistry::instance().take();

  gpuCase.cudaWarmup(data.launch());
  const ub::PerfGpuResult RESULT = gpuCase.cudaKernel(data.launch(), "saxpy").measure();

  ASSERT_GT(OWN.stats.median, 0.0);
  EXPECT_NEAR(RESULT.speedupVsCpu, OWN.stats.median / RESULT.totalTimeUs, 1e-9);
  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.speedupVsCpu.has_value());
  EXPECT_DOUBLE_EQ(*ROW.speedupVsCpu, RESULT.speedupVsCpu);
}

/** @test The ambiguity is reported once for a suite, naming it, not once per case. */
TEST_F(PerfGpuHarnessTest, AmbiguousSuiteBaselineIsReportedOncePerSuite) {
  SaxpyFixtureData data;
  const std::string SUITE = uniqueSuite("GpuAmbiguousWarning");
  (void)recordBaseline(SUITE, "CheapBaseline", 1);
  (void)recordBaseline(SUITE, "ExpensiveBaseline", 8);

  std::string captured;
  {
    vernier::bench::test::StderrCapture capture;
    (void)measureKernelCase(SUITE, data, "FirstKernel");
    (void)measureKernelCase(SUITE, data, "SecondKernel");
    captured = capture.text();
  }

  std::size_t mentions = 0;
  for (std::size_t at = captured.find(SUITE); at != std::string::npos;
       at = captured.find(SUITE, at + SUITE.size())) {
    ++mentions;
  }
  EXPECT_EQ(mentions, 1U) << "stderr said:\n" << captured;
  EXPECT_NE(captured.find("cpuBaseline"), std::string::npos)
      << "the message must say what to do; stderr said:\n"
      << captured;
}

/* ----------------------------- Stability ----------------------------- */

/** @test A GPU row carries the stability verdict the console prints, not the defaults. */
TEST_F(PerfGpuHarnessTest, RowCarriesTheStabilityVerdict) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuStability") + ".Kernel", cfg_};
  perf.cudaWarmup(data.launch());

  const ub::PerfGpuResult RESULT = perf.cudaKernel(data.launch(), "saxpy").measure();

  const double THRESHOLD = ub::recommendedCVThreshold(cfg_);
  const ub::PerfRow ROW = lastRow();
  EXPECT_DOUBLE_EQ(ROW.cvThreshold, THRESHOLD);
  EXPECT_EQ(ROW.stable, RESULT.stats.cpuStats.cv < THRESHOLD);
  EXPECT_DOUBLE_EQ(ROW.stats.cv, RESULT.stats.cpuStats.cv);
}
