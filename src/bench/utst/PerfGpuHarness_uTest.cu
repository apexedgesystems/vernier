/**
 * @file PerfGpuHarness_uTest.cu
 * @brief Unit tests for what the GPU harness measures and publishes.
 *
 * These need a CUDA device: they run a small kernel through the public
 * PerfGpuCase API and check the result and the row the harness leaves in
 * PerfRegistry, and one opens the harness's private NVML helper for the device
 * as the harness does. The setup-failure tests count the streams and events
 * the harness makes, and fail one setup step, through CudaHandleRecorder.
 * Without a device every test skips.
 *
 * Each test uses a test-name prefix of its own where process-wide state is
 * involved (the per-suite CPU baseline, the probe backend's call log), so the
 * order tests run in, and repeated runs in one process, cannot change an
 * outcome. Tests that set the variables deciding CUPTI's yield restore them.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/bench/inc/CuptiCollector.hpp"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"
#include "src/bench/src/NvmlTelemetry.hpp"
#include "src/bench/utst/CudaHandleRecorder.hpp"
#include "src/bench/utst/ScopedEnv.hpp"
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

  /**
   * @brief The CUPTI launches recorded while measuring the kernel in a case of
   *        its own whose config names @p profileTool (no profiler attached).
   */
  std::size_t cuptiLaunches(const std::string& suite, const std::string& profileTool,
                            const SaxpyFixtureData& data) {
    ub::PerfConfig cfg = cfg_;
    cfg.profileTool = profileTool;
    ub::PerfGpuCase perf{suite + ".Kernel", cfg};
    perf.cudaWarmup(data.launch());
    const ub::PerfGpuResult RESULT = perf.cudaKernel(data.launch(), "saxpy").measure();
    static_cast<void>(ub::PerfRegistry::instance().take());
    return RESULT.stats.cupti.kernelLaunches;
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

  // The report itself is counted: other lines of the run name the suite's tests.
  const std::string REPORT = "suite " + SUITE + " measures a CPU baseline";
  std::size_t mentions = 0;
  for (std::size_t at = captured.find(REPORT); at != std::string::npos;
       at = captured.find(REPORT, at + REPORT.size())) {
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

/* ----------------------------- Profiler Hooks ----------------------------- */

namespace {

/** @brief What one case's hooks saw. */
struct HookLog {
  std::string calls;          ///< 'b' per before hook, 'a' per after hook, in call order
  double afterMedianUs = 0.0; ///< The median the last after hook received
};

/** @brief Hooks that log into @p log and stamp the row, as the GPU guard's hooks do. */
void installLoggingHooks(ub::PerfGpuCase& pc, HookLog& log) {
  pc.setBeforeMeasureHook([&log](const ub::PerfGpuCase&) { log.calls += 'b'; });
  pc.setAfterMeasureHook([&log](const ub::PerfGpuCase&, const ub::GpuStats& s) {
    log.calls += 'a';
    log.afterMedianUs = s.cpuStats.median;
    ub::PerfRegistry::instance().updateProfileMeta("hook-log", "hook-log-dir");
  });
}

/** @brief Hook calls the probe backend saw, per test name. */
std::map<std::string, std::string>& probeCalls() {
  static std::map<std::string, std::string> calls;
  return calls;
}

/** @brief A backend that records its hook calls: the registry path the GPU guard takes. */
class GpuHookProbe final : public ub::Profiler {
public:
  explicit GpuHookProbe(std::string testName) : testName_(std::move(testName)) {}
  std::string toolName() const noexcept override { return "gpu-hook-probe"; }
  std::string artifactDir() const noexcept override { return "probe-dir/" + testName_; }
  void beforeMeasure() override { probeCalls()[testName_] += 'b'; }
  void afterMeasure(const ub::Stats& /*s*/) override { probeCalls()[testName_] += 'a'; }

private:
  std::string testName_;
};

std::unique_ptr<ub::Profiler> makeGpuHookProbe(const ub::PerfConfig& /*cfg*/,
                                               const std::string& testName) {
  return std::make_unique<GpuHookProbe>(testName);
}

ub::EnvReport checkGpuHookProbe() {
  return ub::EnvReport{ub::EnvReport::Status::Ok, "test backend", ""};
}

} // namespace

VERNIER_REGISTER_PROFILER_BACKEND("gpu-hook-probe", makeGpuHookProbe, checkGpuHookProbe,
                                  "Registered by the GPU harness unit tests only.")

/**
 * @test A kernel measurement runs the before hook, then the after hook once its
 *       row is published, with the measured stats
 */
TEST_F(PerfGpuHarnessTest, HooksBracketAKernelMeasurement) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuHooksKernel") + ".Kernel", cfg_};
  HookLog log;
  installLoggingHooks(perf, log);
  perf.cudaWarmup(data.launch());

  const ub::PerfGpuResult RESULT = perf.cudaKernel(data.launch(), "saxpy").measure();

  EXPECT_EQ(log.calls, "ba");
  EXPECT_DOUBLE_EQ(log.afterMedianUs, RESULT.totalTimeUs);
  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.profileTool.has_value()) << "the after hook did not stamp the published row";
  EXPECT_EQ(*ROW.profileTool, "hook-log");
}

/** @test A kernel builder that is never measured fires no hook */
TEST_F(PerfGpuHarnessTest, UnmeasuredBuilderFiresNoHook) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuHooksUnmeasured") + ".Kernel", cfg_};
  HookLog log;
  installLoggingHooks(perf, log);

  {
    const ub::CudaKernelBuilder BUILDER = perf.cudaKernel(data.launch(), "saxpy");
    static_cast<void>(BUILDER);
  }

  EXPECT_EQ(log.calls, "");
}

/** @test A CPU baseline is measured inside the hooks, and its row is stamped */
TEST_F(PerfGpuHarnessTest, HooksBracketABaseline) {
  ub::PerfGpuCase perf{uniqueSuite("GpuHooksBaseline") + ".CpuBaseline", cfg_};
  HookLog log;
  installLoggingHooks(perf, log);
  std::vector<float> x(ELEMENTS, 1.0F);
  std::vector<float> y(ELEMENTS, 2.0F);

  const ub::PerfResult CPU = perf.cpuBaseline([&] {
    for (int i = 0; i < ELEMENTS; ++i) {
      y[i] = 2.0F * x[i] + y[i];
    }
  });

  EXPECT_EQ(log.calls, "ba");
  EXPECT_DOUBLE_EQ(log.afterMedianUs, CPU.stats.median);
  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.profileTool.has_value()) << "the after hook did not stamp the baseline row";
  EXPECT_EQ(*ROW.profileTool, "hook-log");
}

/**
 * @test A multi-GPU measurement is bracketed too, and its after hook gets the
 *       times of the device whose row is published
 */
TEST_F(PerfGpuHarnessTest, HooksBracketAMultiGpuMeasurement) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuHooksMulti") + ".MultiGpu", cfg_};
  HookLog log;
  installLoggingHooks(perf, log);
  const ub::PerfGpuCase::KernelFn LAUNCH = data.launch();

  const ub::MultiGpuResult RESULT =
      perf.cudaKernelMultiGpu(1, [&LAUNCH](int, cudaStream_t s) { LAUNCH(s); }).measure();

  ASSERT_EQ(RESULT.perDevice.size(), 1U);
  EXPECT_EQ(log.calls, "ba");
  EXPECT_DOUBLE_EQ(log.afterMedianUs, RESULT.perDevice[0].stats.cpuStats.median);
  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.profileTool.has_value()) << "the after hook did not stamp the published row";
  EXPECT_EQ(*ROW.profileTool, "hook-log");
}

/**
 * @test Through the GPU guard's attach path, the selected backend brackets the
 *       window and the published row names the backend and its folder
 */
TEST_F(PerfGpuHarnessTest, GuardHooksStampTheRowWithTheBackend) {
  SaxpyFixtureData data;
  const std::string NAME = uniqueSuite("GpuHooksGuard") + ".Kernel";
  cfg_.profileTool = "gpu-hook-probe";
  ub::PerfGpuCase perf{NAME, cfg_};
  ub::attachGpuProfilerHooks(perf, cfg_);
  perf.cudaWarmup(data.launch());

  static_cast<void>(perf.cudaKernel(data.launch(), "saxpy").measure());

  EXPECT_EQ(probeCalls()[NAME], "ba");
  const ub::PerfRow ROW = lastRow();
  ASSERT_TRUE(ROW.profileTool.has_value()) << "the backend's after hook did not stamp the row";
  EXPECT_EQ(*ROW.profileTool, "gpu-hook-probe");
  ASSERT_TRUE(ROW.profileDir.has_value());
  EXPECT_EQ(*ROW.profileDir, "probe-dir/" + NAME);
}

/* ----------------------------- CUPTI Yield ----------------------------- */

namespace {

using vernier::bench::test::ScopedEnv;

/** @brief Clears, for one test, every variable that decides CUPTI's yield. */
struct YieldEnvCleared {
  ScopedEnv disable{"VERNIER_DISABLE_CUPTI", nullptr};
  ScopedEnv wrap{"VERNIER_EXTERNAL_WRAP", nullptr};
  ScopedEnv nsysSession{"NSYS_PROFILING_SESSION_ID", nullptr};
  ScopedEnv ncuSession{"NV_NSIGHT_INJECTION_PORT_BASE", nullptr};
};

/** @brief The stderr line the harness prints when the collector stood down. */
constexpr const char* YIELD_LINE = "[gpu] in-process CUPTI collection disabled";

} // namespace

/**
 * @test Without a session every --profile spelling of Nsight still collects
 *       CUPTI records: a spelling starts no session
 */
TEST_F(PerfGpuHarnessTest, CuptiCollectsWithoutASessionWhateverTheSpelling) {
  const YieldEnvCleared CLEARED;
  SaxpyFixtureData data;
  const std::size_t PLAIN = cuptiLaunches(uniqueSuite("GpuCuptiPlain"), "", data);
  if (PLAIN == 0) {
    GTEST_SKIP() << "this build collects no CUPTI records";
  }
  EXPECT_EQ(PLAIN, static_cast<std::size_t>(cfg_.cycles) * cfg_.repeats);
  for (const char* tool : {"nsight", "nsys", "ncu"}) {
    EXPECT_EQ(cuptiLaunches(uniqueSuite("GpuCuptiSpelling"), tool, data), PLAIN)
        << "--profile " << tool << " without a session";
  }
}

/** @test Each kind of Nsight session stands the collector down, and says so */
TEST_F(PerfGpuHarnessTest, CuptiStandsDownUnderASession) {
  const YieldEnvCleared CLEARED;
  SaxpyFixtureData data;
  if (cuptiLaunches(uniqueSuite("GpuCuptiPlain"), "", data) == 0) {
    GTEST_SKIP() << "this build collects no CUPTI records";
  }
  const struct {
    const char* name;
    const char* value;
  } SESSIONS[] = {{"VERNIER_EXTERNAL_WRAP", "nsight"},
                  {"VERNIER_EXTERNAL_WRAP", "nsys"},
                  {"VERNIER_EXTERNAL_WRAP", "ncu"},
                  {"NSYS_PROFILING_SESSION_ID", "1017521"},
                  {"NV_NSIGHT_INJECTION_PORT_BASE", "49152"}};
  for (const auto& session : SESSIONS) {
    const ScopedEnv SET(session.name, session.value);
    std::string captured;
    std::size_t launches = 0;
    {
      vernier::bench::test::StderrCapture capture;
      launches = cuptiLaunches(uniqueSuite("GpuCuptiSession"), "", data);
      captured = capture.text();
    }
    EXPECT_EQ(launches, 0U) << session.name << "=" << session.value;
    EXPECT_NE(captured.find(YIELD_LINE), std::string::npos)
        << session.name << "=" << session.value << ", stderr said:\n"
        << captured;
  }
}

/** @test VERNIER_DISABLE_CUPTI set to a false spelling, in any case, or to nothing keeps it on */
TEST_F(PerfGpuHarnessTest, CuptiDisableZeroFalseOrEmptyKeepsCollecting) {
  const YieldEnvCleared CLEARED;
  SaxpyFixtureData data;
  const std::size_t PLAIN = cuptiLaunches(uniqueSuite("GpuCuptiPlain"), "", data);
  if (PLAIN == 0) {
    GTEST_SKIP() << "this build collects no CUPTI records";
  }
  for (const char* value : {"0", "false", "False", "no", "OFF", ""}) {
    const ScopedEnv SET("VERNIER_DISABLE_CUPTI", value);
    EXPECT_EQ(cuptiLaunches(uniqueSuite("GpuCuptiDisableOff"), "", data), PLAIN)
        << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test VERNIER_DISABLE_CUPTI=false does not keep the collector on inside a session */
TEST_F(PerfGpuHarnessTest, CuptiDisableFalseDoesNotOverrideASession) {
  const YieldEnvCleared CLEARED;
  SaxpyFixtureData data;
  if (cuptiLaunches(uniqueSuite("GpuCuptiPlain"), "", data) == 0) {
    GTEST_SKIP() << "this build collects no CUPTI records";
  }
  const ScopedEnv SESSION("NSYS_PROFILING_SESSION_ID", "1017521");
  for (const char* value : {"false", "no", "OFF"}) {
    const ScopedEnv DISABLE("VERNIER_DISABLE_CUPTI", value);
    EXPECT_EQ(cuptiLaunches(uniqueSuite("GpuCuptiFalseInSession"), "", data), 0U)
        << "VERNIER_DISABLE_CUPTI='" << value << "' in a session";
  }
}

/** @test VERNIER_DISABLE_CUPTI set to a true spelling, in any case, stands it down, and says so */
TEST_F(PerfGpuHarnessTest, CuptiDisableTrueSpellingsStandItDown) {
  const YieldEnvCleared CLEARED;
  SaxpyFixtureData data;
  if (cuptiLaunches(uniqueSuite("GpuCuptiPlain"), "", data) == 0) {
    GTEST_SKIP() << "this build collects no CUPTI records";
  }
  for (const char* value : {"1", "TRUE", "yes", "On"}) {
    const ScopedEnv DISABLE("VERNIER_DISABLE_CUPTI", value);
    std::string captured;
    std::size_t launches = 0;
    {
      vernier::bench::test::StderrCapture capture;
      launches = cuptiLaunches(uniqueSuite("GpuCuptiDisableOn"), "", data);
      captured = capture.text();
    }
    EXPECT_EQ(launches, 0U) << "VERNIER_DISABLE_CUPTI='" << value << "'";
    EXPECT_NE(captured.find(YIELD_LINE), std::string::npos) << value << ", stderr said:\n"
                                                            << captured;
  }
}

/**
 * @test Any other VERNIER_DISABLE_CUPTI value stops the GPU case before it
 *       registers with CUPTI or touches the device: a configuration error that
 *       names the value and the accepted ones; a later case collects as usual
 */
TEST_F(PerfGpuHarnessTest, InvalidCuptiSettingIsAConfigurationError) {
  const YieldEnvCleared CLEARED;
  SaxpyFixtureData data;
  const std::size_t PLAIN = cuptiLaunches(uniqueSuite("GpuCuptiPlain"), "", data);
  {
    const ScopedEnv DISABLE("VERNIER_DISABLE_CUPTI", "maybe");
    try {
      const ub::PerfGpuCase PERF{uniqueSuite("GpuCuptiInvalid") + ".Kernel", cfg_};
      FAIL() << "the case was built with VERNIER_DISABLE_CUPTI='maybe'";
    } catch (const std::invalid_argument& e) {
      const std::string WHAT = e.what();
      EXPECT_EQ(WHAT.rfind("configuration: VERNIER_DISABLE_CUPTI='maybe' is not a boolean.", 0), 0U)
          << WHAT;
      EXPECT_NE(WHAT.find("1, true, yes or on"), std::string::npos) << WHAT;
    }
  }
  EXPECT_EQ(cuptiLaunches(uniqueSuite("GpuCuptiAfterInvalid"), "", data), PLAIN)
      << "a valid case after the rejected one collects as before";
}

/**
 * @test A collector built directly applies the same decision before it
 *       registers, and an explicit forceDisabled wins over the environment
 */
TEST_F(PerfGpuHarnessTest, CollectorAppliesTheSharedDecision) {
  const YieldEnvCleared CLEARED;
  if (!vernier::bench::CuptiCollector(false).isAvailable()) {
    GTEST_SKIP() << "this build has no CUPTI collector";
  }
  {
    const ScopedEnv SET("VERNIER_DISABLE_CUPTI", "0");
    EXPECT_TRUE(vernier::bench::CuptiCollector(false).isAvailable()) << "0 must not disable";
    EXPECT_FALSE(vernier::bench::CuptiCollector(true).isAvailable()) << "forceDisabled must win";
  }
  {
    const ScopedEnv SET("VERNIER_DISABLE_CUPTI", "1");
    EXPECT_FALSE(vernier::bench::CuptiCollector(false).isAvailable());
  }
  {
    const ScopedEnv SET("NSYS_PROFILING_SESSION_ID", "1017521");
    EXPECT_FALSE(vernier::bench::CuptiCollector(false).isAvailable()) << "a session must win";
  }
  {
    const ScopedEnv SET("VERNIER_DISABLE_CUPTI", "maybe");
    EXPECT_THROW(vernier::bench::CuptiCollector(false), std::invalid_argument)
        << "an invalid value must be rejected";
    EXPECT_FALSE(vernier::bench::CuptiCollector(true).isAvailable())
        << "forceDisabled must win without reading the value";
  }
}

/* ----------------------------- CUPTI Statements ----------------------------- */

namespace {

/// The CSV columns filled from CUPTI, as the run names them when they stay empty.
constexpr const char* CUPTI_CELLS = "cuptiKernelLaunches, cuptiRegistersMedian, "
                                    "cuptiRegistersMax, cuptiStaticSmemBytes and "
                                    "cuptiDynamicSmemBytes";

/**
 * @brief Measures the kernel in two cases of this process, then writes to
 *        stderr how many lines were @p statement, how many stderr lines said
 *        the CUPTI cells stay empty, and how many rows had CUPTI cells, and
 *        exits 0.
 */
[[noreturn]] void reportCuptiStatements(const ub::PerfConfig& cfg, const std::string& statement) {
  std::size_t filledRows = 0;
  std::string captured;
  {
    SaxpyFixtureData data;
    vernier::bench::test::StderrCapture capture;
    for (int i = 0; i < 2; ++i) {
      ub::PerfGpuCase perf{uniqueSuite("GpuCuptiStatement") + ".Kernel", cfg};
      perf.cudaWarmup(data.launch());
      static_cast<void>(perf.cudaKernel(data.launch(), "saxpy").measure());
      const std::optional<ub::PerfRow> ROW = ub::PerfRegistry::instance().take();
      if (ROW.has_value() && ROW->cuptiKernelLaunches.has_value()) {
        ++filledRows;
      }
    }
    captured = capture.text();
  }
  std::size_t statements = 0;
  std::size_t emptyCuptiLines = 0;
  std::size_t from = 0;
  while (from < captured.size()) {
    std::size_t end = captured.find('\n', from);
    if (end == std::string::npos) {
      end = captured.size();
    }
    const std::string LINE = captured.substr(from, end - from);
    statements += (LINE == statement) ? 1 : 0;
    emptyCuptiLines +=
        (LINE.find(std::string(CUPTI_CELLS) + " stay empty.") != std::string::npos) ? 1 : 0;
    from = end + 1;
  }
  std::fprintf(stderr, "statements=%zu emptyCuptiLines=%zu filledRows=%zu\n", statements,
               emptyCuptiLines, filledRows);
  std::exit(0);
}

} // namespace

/**
 * @test A measured window in which CUPTI recorded no kernel launch is said for
 *       its test, naming the cells it leaves empty, and they are empty
 */
TEST_F(PerfGpuHarnessTest, AWindowWithoutAKernelRecordIsStatedForItsTest) {
  const YieldEnvCleared CLEARED;
  if (!ub::CuptiCollector(false).isAvailable()) {
    GTEST_SKIP() << "this build collects no CUPTI records";
  }
  const std::string NAME = uniqueSuite("GpuCuptiNoLaunch") + ".Kernel";
  ub::PerfGpuCase perf{NAME, cfg_};
  std::string captured;
  {
    vernier::bench::test::StderrCapture capture;
    static_cast<void>(perf.cudaKernel([](cudaStream_t) {}, "nothing").measure());
    captured = capture.text();
  }
  const ub::PerfRow ROW = lastRow();
  EXPECT_FALSE(ROW.cuptiKernelLaunches.has_value());
  EXPECT_FALSE(ROW.cuptiRegistersMedian.has_value());
  const std::string LINE = "[gpu] CUPTI recorded no kernel launch in " + NAME +
                           "'s measured window, so its " + CUPTI_CELLS + " stay empty.";
  EXPECT_NE(captured.find(LINE), std::string::npos) << "stderr said:\n" << captured;
}

/** @brief Same fixture, named so GoogleTest schedules the death test first. */
using PerfGpuHarnessDeathTest = PerfGpuHarnessTest;

/**
 * @test A collector that cannot collect is stated once per process, in the
 *       collector's words and naming the cells it leaves empty, and those cells
 *       are empty; a collector that can collect fills them and states nothing
 */
TEST_F(PerfGpuHarnessDeathTest, ACollectorThatCannotCollectIsStatedOnce) {
  const YieldEnvCleared CLEARED;
  const ub::CuptiCollector PROBE(false);
  const std::string STATEMENT =
      "[gpu] " + PROBE.unavailableReason() + ": " + CUPTI_CELLS + " stay empty.";
  const std::string EXPECTED = PROBE.isAvailable() ? "statements=0 emptyCuptiLines=0 filledRows=2"
                                                   : "statements=1 emptyCuptiLines=1 filledRows=0";

  // "threadsafe" re-executes the test binary for the child, so the
  // once-per-process state starts clear whatever ran before in this process;
  // the default style forks, which would also inherit an initialised CUDA.
  const std::string SAVED_STYLE = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(reportCuptiStatements(cfg_, STATEMENT), ::testing::ExitedWithCode(0), EXPECTED);
  GTEST_FLAG_SET(death_test_style, SAVED_STYLE);
}

/* ----------------------------- NVML Cells ----------------------------- */

namespace {

/** @brief True when @p line names @p column as a whole word. */
bool namesColumn(const std::string& line, const std::string& column) {
  const auto WORD = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; };
  for (std::size_t at = line.find(column); at != std::string::npos;
       at = line.find(column, at + 1)) {
    const std::size_t END = at + column.size();
    if ((at == 0 || !WORD(line[at - 1])) && (END == line.size() || !WORD(line[END]))) {
      return true;
    }
  }
  return false;
}

/** @brief True when @p text ends with @p tail. */
bool endsWith(const std::string& text, const std::string& tail) {
  return text.size() >= tail.size() &&
         text.compare(text.size() - tail.size(), tail.size(), tail) == 0;
}

/**
 * @brief True when @p text has a "[gpu] ... stay empty." (or "stays empty.")
 *        line naming @p column.
 */
bool statedEmpty(const std::string& text, const std::string& column) {
  std::size_t from = 0;
  while (from < text.size()) {
    std::size_t end = text.find('\n', from);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string LINE = text.substr(from, end - from);
    const bool STATEMENT = LINE.rfind("[gpu] ", 0) == 0 &&
                           (endsWith(LINE, " stay empty.") || endsWith(LINE, " stays empty."));
    if (STATEMENT && namesColumn(LINE, column)) {
      return true;
    }
    from = end + 1;
  }
  return false;
}

/**
 * @brief Measures the kernel in a case of this process and writes to stderr
 *        what is wrong with the row's NVML cells, then exits 0: a cell that is
 *        0 where 0 is not a reading, an empty cell no statement names, or a
 *        named cell that is filled.
 */
[[noreturn]] void reportNvmlCells(const ub::PerfConfig& cfg) {
  std::optional<ub::PerfRow> row;
  std::string captured;
  {
    SaxpyFixtureData data;
    vernier::bench::test::StderrCapture capture;
    ub::PerfGpuCase perf{uniqueSuite("GpuNvmlCells") + ".Kernel", cfg};
    perf.cudaWarmup(data.launch());
    static_cast<void>(perf.cudaKernel(data.launch(), "saxpy").measure());
    row = ub::PerfRegistry::instance().take();
    captured = capture.text();
  }
  std::vector<std::string> problems;
  if (!row.has_value()) {
    problems.emplace_back("no row");
  } else {
    const auto CHECK = [&](const char* column, bool filled, bool zero) {
      const bool STATED = statedEmpty(captured, column);
      if (filled && zero) {
        problems.push_back(std::string(column) + " is 0");
      }
      if (filled && STATED) {
        problems.push_back(std::string(column) + " is filled and stated empty");
      }
      if (!filled && !STATED) {
        problems.push_back(std::string(column) + " is empty and unstated");
      }
    };
    CHECK("smClockMHz", row->smClockMHz.has_value(), row->smClockMHz.value_or(1) == 0);
    CHECK("throttling", row->throttling.has_value(), false);
    CHECK("powerDrawW", row->powerDrawW.has_value(), row->powerDrawW.value_or(1.0) == 0.0);
    CHECK("powerLimitW", row->powerLimitW.has_value(), row->powerLimitW.value_or(1.0) == 0.0);
    CHECK("temperatureC", row->temperatureC.has_value(), row->temperatureC.value_or(1) == 0);
    CHECK("temperatureDeltaC", row->temperatureDeltaC.has_value(), false);
  }
  std::string summary;
  for (const std::string& p : problems) {
    summary += (summary.empty() ? "" : "; ") + p;
  }
  std::fprintf(stderr, "nvml cell problems: %s\n", summary.empty() ? "none" : summary.c_str());
  std::exit(0);
}

} // namespace

/**
 * @test A kernel row's NVML cells are readings or stated empty: none is 0
 *       where 0 is not a reading, every empty one is named by a statement of
 *       the run, and no named one is filled
 */
TEST_F(PerfGpuHarnessDeathTest, NvmlCellsAreReadingsOrStatedEmpty) {
  // A fresh process, so this run's once-per-process statements are its own.
  const std::string SAVED_STYLE = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(reportNvmlCells(cfg_), ::testing::ExitedWithCode(0), "nvml cell problems: none");
  GTEST_FLAG_SET(death_test_style, SAVED_STYLE);
}

/**
 * @test NVML, where this build has it and it initializes, finds the CUDA
 *       device by the UUID the harness looks it up by
 */
TEST_F(PerfGpuHarnessTest, NvmlFindsTheCudaDeviceByItsUuid) {
  // Device 0: the one the harness measures on unless --gpu-device says otherwise.
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  const std::string UUID = ub::nvml_telemetry::uuidText(prop.uuid.bytes);
  const ub::nvml_telemetry::Session SESSION(true, UUID);
  const std::string& REASON = SESSION.unavailableReason();
  if (COMPAT_NVML_AVAILABLE == 0 || REASON.rfind("NVML did not initialize", 0) == 0) {
    GTEST_SKIP() << REASON;
  }
  EXPECT_TRUE(SESSION.ready()) << UUID << ": " << REASON;
}

/* ----------------------------- Bandwidth and Occupancy Cells ----------------------------- */

namespace {

/// The launch shape the tests' saxpyKernel launch uses.
const dim3 GRID((ELEMENTS + BLOCK - 1) / BLOCK);
const dim3 BLOCK_DIM(BLOCK);

/** @brief The line the run writes for @p testName's measurement without a launch configuration. */
std::string noLaunchConfigLine(const std::string& testName) {
  return "[gpu] " + testName +
         " declares no launch configuration (.withLaunchConfig(grid, block)), so its occupancy "
         "stays empty.";
}

} // namespace

/**
 * @test The bandwidth cell is the declared transfers' rate, and a kernel-only
 *       row has none: a rate over no bytes is not a measurement
 */
TEST_F(PerfGpuHarnessTest, BandwidthCellOnlyWhenBytesMoved) {
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuBandwidth") + ".Kernel", cfg_};
  perf.cudaWarmup(data.launch());

  static_cast<void>(perf.cudaKernel(data.launch(), "saxpy").measure());
  const ub::PerfRow KERNEL_ONLY = lastRow();
  EXPECT_FALSE(KERNEL_ONLY.memBandwidthGBs.has_value())
      << "a kernel-only row's bandwidth cell holds " << KERNEL_ONLY.memBandwidthGBs.value_or(-1.0);

  const ub::PerfGpuResult MOVED =
      perf.cudaKernel(data.launch(), "saxpy")
          .withHostToDevice(data.hostX(), data.deviceX(), SaxpyFixtureData::bytes())
          .withDeviceToHost(data.deviceY(), data.hostY(), SaxpyFixtureData::bytes())
          .measure();
  const ub::PerfRow TRANSFERRING = lastRow();
  ASSERT_EQ(MOVED.stats.transfers.h2dBytes + MOVED.stats.transfers.d2hBytes,
            2 * SaxpyFixtureData::bytes());
  ASSERT_TRUE(TRANSFERRING.memBandwidthGBs.has_value());
  EXPECT_DOUBLE_EQ(*TRANSFERRING.memBandwidthGBs, MOVED.stats.transfers.bandwidthGBs());
}

/**
 * @test A row without a launch configuration has no occupancy cell and the run
 *       names its test; with one, the cell is the harness's estimate, the warps
 *       the shape keeps resident over the SM's maximum
 */
TEST_F(PerfGpuHarnessTest, OccupancyCellOnlyWithALaunchConfiguration) {
  SaxpyFixtureData data;
  const std::string NAME = uniqueSuite("GpuOccupancy") + ".Kernel";
  ub::PerfGpuCase perf{NAME, cfg_};
  perf.cudaWarmup(data.launch());

  std::string captured;
  {
    vernier::bench::test::StderrCapture capture;
    static_cast<void>(perf.cudaKernel(data.launch(), "saxpy").measure());
    captured = capture.text();
  }
  const ub::PerfRow WITHOUT = lastRow();
  EXPECT_FALSE(WITHOUT.occupancy.has_value())
      << "the occupancy cell holds " << WITHOUT.occupancy.value_or(-1.0);
  EXPECT_NE(captured.find(noLaunchConfigLine(NAME)), std::string::npos) << "stderr said:\n"
                                                                        << captured;

  const ub::PerfGpuResult WITH =
      perf.cudaKernel(data.launch(), "saxpy").withLaunchConfig(GRID, BLOCK_DIM).measure();
  const ub::PerfRow ROW = lastRow();
  const ub::OccupancyMetrics& OCC = WITH.stats.occupancy;
  ASSERT_EQ(OCC.blockSize, BLOCK);
  ASSERT_GT(OCC.maxWarpsPerSM, 0);
  ASSERT_TRUE(ROW.occupancy.has_value());
  EXPECT_DOUBLE_EQ(*ROW.occupancy, OCC.achievedOccupancy);
  EXPECT_DOUBLE_EQ(*ROW.occupancy, static_cast<double>(OCC.activeWarpsPerSM) / OCC.maxWarpsPerSM);
}

/**
 * @test The multi-GPU row follows the same rule: no occupancy cell without a
 *       launch configuration, with the run naming its test, and the estimate
 *       with one
 */
TEST_F(PerfGpuHarnessTest, MultiGpuOccupancyCellOnlyWithALaunchConfiguration) {
  SaxpyFixtureData data;
  const std::string NAME = uniqueSuite("GpuMultiOccupancy") + ".MultiGpu";
  ub::PerfGpuCase perf{NAME, cfg_};
  const ub::PerfGpuCase::KernelFn LAUNCH = data.launch();
  const auto ON_DEVICE = [&LAUNCH](int, cudaStream_t s) { LAUNCH(s); };

  std::string captured;
  {
    vernier::bench::test::StderrCapture capture;
    static_cast<void>(perf.cudaKernelMultiGpu(1, ON_DEVICE).measure());
    captured = capture.text();
  }
  const ub::PerfRow WITHOUT = lastRow();
  EXPECT_FALSE(WITHOUT.occupancy.has_value())
      << "the occupancy cell holds " << WITHOUT.occupancy.value_or(-1.0);
  EXPECT_NE(captured.find(noLaunchConfigLine(NAME)), std::string::npos) << "stderr said:\n"
                                                                        << captured;

  const ub::MultiGpuResult WITH =
      perf.cudaKernelMultiGpu(1, ON_DEVICE).withLaunchConfig(GRID, BLOCK_DIM).measure();
  const ub::PerfRow ROW = lastRow();
  ASSERT_EQ(WITH.perDevice.size(), 1U);
  ASSERT_TRUE(ROW.occupancy.has_value());
  EXPECT_DOUBLE_EQ(*ROW.occupancy, WITH.perDevice[0].stats.occupancy.achievedOccupancy);
}

/* ----------------------------- Device Selection ----------------------------- */

namespace {

/// What fillKernel writes at index i: FILL_BASE + i, exact in float for ELEMENTS elements.
constexpr float FILL_BASE = 1.0F;

/** @brief out[i] = base + i over n floats. */
__global__ void fillKernel(float* out, int n, float base) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    out[IDX] = base + static_cast<float>(IDX);
  }
}

/** @brief The current CUDA device, or -1 when the runtime does not say. */
int currentDevice() {
  int device = -1;
  return (cudaGetDevice(&device) == cudaSuccess) ? device : -1;
}

/**
 * @brief A zeroed buffer of ELEMENTS floats on one device, the launch that
 *        fills it and records the device current when it ran, and the check.
 */
class FillOnDevice {
public:
  explicit FillOnDevice(int device) : device_(device) {
    const int CALLER = currentDevice();
    cudaSetDevice(device_);
    cudaMalloc(&buffer_, bytes());
    cudaMemset(buffer_, 0, bytes());
    cudaSetDevice(CALLER);
  }

  ~FillOnDevice() {
    const int CALLER = currentDevice();
    cudaSetDevice(device_);
    cudaFree(buffer_);
    cudaSetDevice(CALLER);
  }

  FillOnDevice(const FillOnDevice&) = delete;
  FillOnDevice& operator=(const FillOnDevice&) = delete;

  /** @brief The measured launch: records the current device, then fills the buffer. */
  [[nodiscard]] ub::PerfGpuCase::KernelFn launch() {
    return [this](cudaStream_t s) {
      seenDevice_ = currentDevice();
      fillKernel<<<(ELEMENTS + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(buffer_, ELEMENTS, FILL_BASE);
    };
  }

  /** @brief The device that was current when the launch last ran; -2 before it ran. */
  [[nodiscard]] int seenDevice() const { return seenDevice_; }

  /**
   * @brief Index of the first element that is not FILL_BASE + its index: -1
   *        when every element is, -2 when the buffer cannot be read back.
   */
  [[nodiscard]] int firstWrong() const {
    std::vector<float> host(ELEMENTS, 0.0F);
    const int CALLER = currentDevice();
    cudaSetDevice(device_);
    const cudaError_t COPIED = cudaMemcpy(host.data(), buffer_, bytes(), cudaMemcpyDeviceToHost);
    cudaSetDevice(CALLER);
    if (COPIED != cudaSuccess) {
      return -2;
    }
    for (int i = 0; i < ELEMENTS; ++i) {
      if (host[i] != FILL_BASE + static_cast<float>(i)) {
        return i;
      }
    }
    return -1;
  }

private:
  [[nodiscard]] static std::size_t bytes() { return ELEMENTS * sizeof(float); }

  int device_;
  float* buffer_ = nullptr;
  int seenDevice_ = -2;
};

/// What a failing callback throws, so the test can tell it from the harness's own errors.
constexpr const char* CALLBACK_FAILED = "callback failed";

/**
 * @brief A measured launch that runs @p launch and throws CALLBACK_FAILED on
 *        its third call, inside the measured window, after kernels are queued.
 */
ub::PerfGpuCase::KernelFn throwingOnThirdCall(ub::PerfGpuCase::KernelFn launch, int& calls) {
  return [launch = std::move(launch), &calls](cudaStream_t s) {
    launch(s);
    if (++calls == 3) {
      throw std::runtime_error(CALLBACK_FAILED);
    }
  };
}

} // namespace

/**
 * @test withDeviceId() naming the case's own device measures there: the
 *       launch runs with that device current and fills its buffer, and the
 *       result and the row carry its id and GPU model
 */
TEST_F(PerfGpuHarnessTest, WithDeviceIdOnTheCaseDeviceMeasuresThere) {
  ub::PerfGpuCase perf{uniqueSuite("GpuDeviceOwn") + ".Kernel", cfg_};
  const int OWN = perf.gpuConfig().deviceId;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, OWN), cudaSuccess);
  FillOnDevice fill(OWN);

  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(fill.launch(), "fill").withDeviceId(OWN).measure();
  const ub::PerfRow ROW = lastRow();

  EXPECT_EQ(fill.seenDevice(), OWN);
  EXPECT_EQ(fill.firstWrong(), -1) << "the launch did not fill the buffer";
  EXPECT_EQ(currentDevice(), OWN);
  EXPECT_EQ(RESULT.deviceId, OWN);
  EXPECT_EQ(RESULT.stats.deviceInfo.name, prop.name);
  ASSERT_TRUE(ROW.deviceId.has_value());
  EXPECT_EQ(*ROW.deviceId, OWN);
  ASSERT_TRUE(ROW.gpuModel.has_value());
  EXPECT_EQ(*ROW.gpuModel, prop.name);
}

/**
 * @test withDeviceId() naming no device (the device count, or an id below -1)
 *       throws std::invalid_argument naming the id before the measurement
 *       starts: neither hook fires, no row is published and the current device
 *       is unchanged
 */
TEST_F(PerfGpuHarnessTest, WithDeviceIdNamingNoDeviceThrowsBeforeMeasuring) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  SaxpyFixtureData data;
  for (const int ID : {count, -2}) {
    ub::PerfGpuCase perf{uniqueSuite("GpuDeviceNone") + ".Kernel", cfg_};
    HookLog log;
    installLoggingHooks(perf, log);
    const int BEFORE = currentDevice();
    try {
      static_cast<void>(perf.cudaKernel(data.launch(), "saxpy").withDeviceId(ID).measure());
      ADD_FAILURE() << "withDeviceId(" << ID << ") measured";
    } catch (const std::invalid_argument& e) {
      const std::string WHAT = e.what();
      EXPECT_NE(WHAT.find("withDeviceId(" + std::to_string(ID) + ")"), std::string::npos) << WHAT;
      EXPECT_NE(WHAT.find("sees " + std::to_string(count)), std::string::npos) << WHAT;
    }
    EXPECT_EQ(log.calls, "") << "withDeviceId(" << ID << ") opened a profiler window";
    EXPECT_FALSE(ub::PerfRegistry::instance().take().has_value())
        << "withDeviceId(" << ID << ") published a row";
    EXPECT_EQ(currentDevice(), BEFORE) << "withDeviceId(" << ID << ") changed the current device";
  }
}

/**
 * @test withDeviceId() naming another device measures there: the launch runs
 *       with that device current and fills that device's buffer, the result
 *       and the row carry its id and GPU model, and afterwards the caller's
 *       device is current again and the case's own stream still works
 */
TEST_F(PerfGpuHarnessTest, WithDeviceIdOnAnotherDeviceMeasuresThere) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two CUDA devices, this machine has " << count;
  }
  ub::PerfGpuCase perf{uniqueSuite("GpuDeviceOther") + ".Kernel", cfg_};
  const int OWN = perf.gpuConfig().deviceId;
  const int OTHER = (OWN + 1) % count;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, OTHER), cudaSuccess);
  FillOnDevice fill(OTHER);

  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(fill.launch(), "fill").withDeviceId(OTHER).measure();
  const ub::PerfRow ROW = lastRow();

  EXPECT_EQ(fill.seenDevice(), OTHER);
  EXPECT_EQ(fill.firstWrong(), -1) << "the launch did not fill the other device's buffer";
  EXPECT_EQ(currentDevice(), OWN) << "the caller's device is not current again";
  EXPECT_EQ(RESULT.deviceId, OTHER);
  EXPECT_EQ(RESULT.stats.deviceInfo.name, prop.name);
  ASSERT_TRUE(ROW.deviceId.has_value());
  EXPECT_EQ(*ROW.deviceId, OTHER);
  ASSERT_TRUE(ROW.gpuModel.has_value());
  EXPECT_EQ(*ROW.gpuModel, prop.name);

  // The case's own stream still serves its own device
  SaxpyFixtureData data;
  EXPECT_NO_THROW(perf.cudaWarmup(data.launch()));
}

/**
 * @test A kernel callback that throws in the measured window: the exception
 *       reaches the caller, no row is published, and the window is closed, so
 *       the case's next measurement and a new case's count only their own
 *       launches and compute the right result
 */
TEST_F(PerfGpuHarnessTest, ThrowingCallbackLeavesTheNextWindowClean) {
  const YieldEnvCleared CLEARED;
  const bool COUNTS = ub::CuptiCollector(false).isAvailable();
  const std::size_t LAUNCHES = static_cast<std::size_t>(cfg_.cycles) * cfg_.repeats;
  FillOnDevice fill(currentDevice());
  {
    ub::PerfGpuCase perf{uniqueSuite("GpuThrowDefault") + ".Kernel", cfg_};
    int calls = 0;
    try {
      static_cast<void>(
          perf.cudaKernel(throwingOnThirdCall(fill.launch(), calls), "fill").measure());
      ADD_FAILURE() << "the callback's exception did not reach the caller";
    } catch (const std::runtime_error& e) {
      EXPECT_STREQ(e.what(), CALLBACK_FAILED);
    }
    EXPECT_EQ(calls, 3);
    EXPECT_FALSE(ub::PerfRegistry::instance().take().has_value())
        << "the failed measurement published a row";
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    const ub::PerfGpuResult NEXT = perf.cudaKernel(fill.launch(), "fill").measure();
    static_cast<void>(lastRow());
    EXPECT_EQ(fill.firstWrong(), -1);
    if (COUNTS) {
      EXPECT_EQ(NEXT.stats.cupti.kernelLaunches, LAUNCHES)
          << "the case's next window counted the failed window's launches";
    }
  }

  ub::PerfGpuCase fresh{uniqueSuite("GpuThrowDefault") + ".Fresh", cfg_};
  const ub::PerfGpuResult FRESH = fresh.cudaKernel(fill.launch(), "fill").measure();
  static_cast<void>(lastRow());
  EXPECT_EQ(fill.firstWrong(), -1);
  if (COUNTS) {
    EXPECT_EQ(FRESH.stats.cupti.kernelLaunches, LAUNCHES);
  }
}

/**
 * @test A callback that throws after withDeviceId() made the case's device
 *       current: the exception reaches the caller, no row is published, the
 *       caller's device is current again, and the case's next measurement on
 *       that device is clean and correct. With one GPU the named device is the
 *       caller's, so this covers the failure path's cleanup and a restoration
 *       to the same device, not a return from another device
 */
TEST_F(PerfGpuHarnessTest, WithDeviceIdRestoresTheCallersDeviceWhenTheCallbackThrows) {
  const YieldEnvCleared CLEARED;
  const bool COUNTS = ub::CuptiCollector(false).isAvailable();
  const std::size_t LAUNCHES = static_cast<std::size_t>(cfg_.cycles) * cfg_.repeats;
  ub::PerfGpuCase perf{uniqueSuite("GpuThrowOwn") + ".Kernel", cfg_};
  const int OWN = perf.gpuConfig().deviceId;
  const int CALLER = currentDevice();
  FillOnDevice fill(OWN);

  int calls = 0;
  try {
    static_cast<void>(perf.cudaKernel(throwingOnThirdCall(fill.launch(), calls), "fill")
                          .withDeviceId(OWN)
                          .measure());
    ADD_FAILURE() << "the callback's exception did not reach the caller";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), CALLBACK_FAILED);
  }
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(fill.seenDevice(), OWN);
  EXPECT_FALSE(ub::PerfRegistry::instance().take().has_value())
      << "the failed measurement published a row";
  EXPECT_EQ(currentDevice(), CALLER) << "the caller's device is not current again";
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  const ub::PerfGpuResult NEXT = perf.cudaKernel(fill.launch(), "fill").withDeviceId(OWN).measure();
  const ub::PerfRow ROW = lastRow();
  EXPECT_EQ(fill.firstWrong(), -1);
  EXPECT_EQ(NEXT.deviceId, OWN);
  ASSERT_TRUE(ROW.deviceId.has_value());
  EXPECT_EQ(*ROW.deviceId, OWN);
  EXPECT_EQ(currentDevice(), CALLER);
  if (COUNTS) {
    EXPECT_EQ(NEXT.stats.cupti.kernelLaunches, LAUNCHES)
        << "the case's next window counted the failed window's launches";
  }
}

/**
 * @test With two or more devices, a callback that throws after withDeviceId()
 *       made another device current: the caller's device is current again, no
 *       row is published, and the case's next measurement on that device is
 *       clean and correct
 */
TEST_F(PerfGpuHarnessTest, WithDeviceIdOnAnotherDeviceRestoresWhenTheCallbackThrows) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two CUDA devices, this machine has " << count;
  }
  const YieldEnvCleared CLEARED;
  const bool COUNTS = ub::CuptiCollector(false).isAvailable();
  const std::size_t LAUNCHES = static_cast<std::size_t>(cfg_.cycles) * cfg_.repeats;
  ub::PerfGpuCase perf{uniqueSuite("GpuThrowOther") + ".Kernel", cfg_};
  const int OWN = perf.gpuConfig().deviceId;
  const int OTHER = (OWN + 1) % count;
  FillOnDevice fill(OTHER);
  ASSERT_EQ(currentDevice(), OWN);

  int calls = 0;
  try {
    static_cast<void>(perf.cudaKernel(throwingOnThirdCall(fill.launch(), calls), "fill")
                          .withDeviceId(OTHER)
                          .measure());
    ADD_FAILURE() << "the callback's exception did not reach the caller";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), CALLBACK_FAILED);
  }
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(fill.seenDevice(), OTHER);
  EXPECT_FALSE(ub::PerfRegistry::instance().take().has_value())
      << "the failed measurement published a row";
  EXPECT_EQ(currentDevice(), OWN) << "the caller's device is not current again";

  const ub::PerfGpuResult NEXT =
      perf.cudaKernel(fill.launch(), "fill").withDeviceId(OTHER).measure();
  const ub::PerfRow ROW = lastRow();
  EXPECT_EQ(fill.firstWrong(), -1);
  EXPECT_EQ(NEXT.deviceId, OTHER);
  ASSERT_TRUE(ROW.deviceId.has_value());
  EXPECT_EQ(*ROW.deviceId, OTHER);
  EXPECT_EQ(currentDevice(), OWN);
  if (COUNTS) {
    EXPECT_EQ(NEXT.stats.cupti.kernelLaunches, LAUNCHES)
        << "the case's next window counted the failed window's launches";
  }
}

/* ----------------------------- Setup Failures ----------------------------- */

namespace {

namespace bt = vernier::bench::test;

/** @brief A step of a device's setup that a test makes fail. */
struct SetupStep {
  bt::CudaSetupFailure failure; ///< What CudaHandleRecorder makes fail
  const char* name;             ///< The step, for the test's messages
  int eventsBefore;             ///< Events the setup made before this step failed
};

/// Every step after the stream: each event, then the properties, read after both.
constexpr SetupStep SETUP_STEPS[] = {
    {bt::CudaSetupFailure::FirstEvent, "the first event", 0},
    {bt::CudaSetupFailure::SecondEvent, "the second event", 1},
    {bt::CudaSetupFailure::Properties, "the device properties", 2},
};

} // namespace

/**
 * @test A case whose device setup fails after it made its stream (at the first
 *       event, the second event, or the properties read after both): the
 *       exception reaches the caller, every stream and event the setup made is
 *       destroyed, the current device is unchanged, and a new case on the same
 *       device measures there and, when it ends, destroys all it made
 */
TEST_F(PerfGpuHarnessTest, CaseSetupThatFailsLeavesNoHandleAndANewCaseMeasures) {
  if (!bt::cudaHandleRecorderAvailable()) {
    GTEST_SKIP() << bt::cudaHandleRecorderUnavailableReason();
  }
  const std::string INJECTED = cudaGetErrorString(bt::injectedCudaError());
  for (const SetupStep& STEP : SETUP_STEPS) {
    SCOPED_TRACE(STEP.name);
    const int CALLER = currentDevice();

    bt::startCudaHandleRecording(STEP.failure);
    try {
      const ub::PerfGpuCase PERF{uniqueSuite("GpuSetupFails") + ".Kernel", cfg_};
      ADD_FAILURE() << "the case was made although " << STEP.name << " failed";
    } catch (const std::runtime_error& e) {
      EXPECT_EQ(std::string(e.what()), INJECTED);
    }
    const bt::CudaHandleCounts FAILED = bt::stopCudaHandleRecording();
    EXPECT_EQ(FAILED.streamsMade, 1) << "the recorder saw no stream made: the setup's calls "
                                        "did not reach it";
    EXPECT_EQ(FAILED.eventsMade, STEP.eventsBefore);
    EXPECT_EQ(FAILED.streamsLeft, 0) << "the failed setup left its stream";
    EXPECT_EQ(FAILED.eventsLeft, 0) << "the failed setup left its events";
    EXPECT_EQ(currentDevice(), CALLER);

    // A new case on the same device is made, measures there, and ends with
    // nothing it made left behind.
    bt::startCudaHandleRecording(bt::CudaSetupFailure::None);
    {
      ub::PerfGpuCase perf{uniqueSuite("GpuSetupRetry") + ".Kernel", cfg_};
      FillOnDevice fill(perf.gpuConfig().deviceId);
      static_cast<void>(perf.cudaKernel(fill.launch(), "fill").measure());
      static_cast<void>(lastRow());
      EXPECT_EQ(fill.seenDevice(), perf.gpuConfig().deviceId);
      EXPECT_EQ(fill.firstWrong(), -1) << "the new case's launch did not fill the buffer";
    }
    const bt::CudaHandleCounts RETRIED = bt::stopCudaHandleRecording();
    EXPECT_EQ(RETRIED.streamsMade, 1);
    EXPECT_EQ(RETRIED.eventsMade, 2);
    EXPECT_EQ(RETRIED.streamsLeft, 0) << "the new case left its stream";
    EXPECT_EQ(RETRIED.eventsLeft, 0) << "the new case left its events";
  }
}

/**
 * @test With two or more devices, withDeviceId() naming another device whose
 *       setup fails after it made its stream: the exception reaches the caller
 *       before any hook or row, every stream and event that setup made is
 *       destroyed, the caller's device is current again, and the retry makes
 *       that device's resources anew (nothing half-made was kept), measures
 *       there, and those resources end with the case
 */
TEST_F(PerfGpuHarnessTest, WithDeviceIdSetupThatFailsLeavesNoHandleAndTheRetryMeasures) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two CUDA devices, this machine has " << count;
  }
  if (!bt::cudaHandleRecorderAvailable()) {
    GTEST_SKIP() << bt::cudaHandleRecorderUnavailableReason();
  }
  const std::string INJECTED = cudaGetErrorString(bt::injectedCudaError());
  for (const SetupStep& STEP : SETUP_STEPS) {
    SCOPED_TRACE(STEP.name);
    {
      ub::PerfGpuCase perf{uniqueSuite("GpuSetupOther") + ".Kernel", cfg_};
      const int OWN = perf.gpuConfig().deviceId;
      const int OTHER = (OWN + 1) % count;
      FillOnDevice fill(OTHER);
      HookLog log;
      installLoggingHooks(perf, log);
      ASSERT_EQ(currentDevice(), OWN);

      bt::startCudaHandleRecording(STEP.failure);
      try {
        static_cast<void>(perf.cudaKernel(fill.launch(), "fill").withDeviceId(OTHER).measure());
        ADD_FAILURE() << "measured although " << STEP.name << " failed";
      } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), INJECTED);
      }
      const bt::CudaHandleCounts FAILED = bt::stopCudaHandleRecording();
      EXPECT_EQ(FAILED.streamsMade, 1) << "the recorder saw no stream made: the setup's calls "
                                          "did not reach it";
      EXPECT_EQ(FAILED.eventsMade, STEP.eventsBefore);
      EXPECT_EQ(FAILED.streamsLeft, 0) << "the failed setup left its stream";
      EXPECT_EQ(FAILED.eventsLeft, 0) << "the failed setup left its events";
      EXPECT_EQ(currentDevice(), OWN) << "the caller's device is not current again";
      EXPECT_EQ(log.calls, "") << "the failed setup opened a profiler window";
      EXPECT_FALSE(ub::PerfRegistry::instance().take().has_value())
          << "the failed setup published a row";

      bt::startCudaHandleRecording(bt::CudaSetupFailure::None);
      const ub::PerfGpuResult RESULT =
          perf.cudaKernel(fill.launch(), "fill").withDeviceId(OTHER).measure();
      static_cast<void>(lastRow());
      const bt::CudaHandleCounts KEPT = bt::cudaHandleCounts();
      EXPECT_EQ(KEPT.streamsMade, 1) << "the retry did not make the device's resources anew";
      EXPECT_EQ(KEPT.eventsMade, 2);
      EXPECT_EQ(fill.seenDevice(), OTHER);
      EXPECT_EQ(fill.firstWrong(), -1) << "the retry did not fill the other device's buffer";
      EXPECT_EQ(RESULT.deviceId, OTHER);
      EXPECT_EQ(currentDevice(), OWN);
    }
    const bt::CudaHandleCounts ENDED = bt::stopCudaHandleRecording();
    EXPECT_EQ(ENDED.streamsLeft, 0) << "the other device's stream outlived the case";
    EXPECT_EQ(ENDED.eventsLeft, 0) << "the other device's events outlived the case";
  }
}
