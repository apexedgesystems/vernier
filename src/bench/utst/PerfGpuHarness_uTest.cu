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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "src/bench/inc/CuptiCollector.hpp"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
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

/**
 * @brief Skips when no CUDA device is present; keeps the registry slot clean,
 *        and clears the profile failures a test records (ProfilerRegistry).
 */
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
    ub::ProfilerRegistry::instance().resetFailures();
  }

  void TearDown() override {
    (void)ub::PerfRegistry::instance().take();
    ub::ProfilerRegistry::instance().resetFailures();
  }

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

/* ----------------------------- Rows ----------------------------- */

/** @test A CPU baseline runs on the calling thread: its row records one thread at --threads 4 */
TEST_F(PerfGpuHarnessTest, BaselineRowRecordsOneThread) {
  ub::PerfConfig cfg = cfg_;
  cfg.threads = 4;
  ub::PerfGpuCase perf{uniqueSuite("GpuRowThreads") + ".CpuBaseline", cfg};
  std::vector<float> x(ELEMENTS, 1.0F);
  std::vector<float> y(ELEMENTS, 2.0F);

  (void)perf.cpuBaseline([&] {
    for (int i = 0; i < ELEMENTS; ++i) {
      y[i] = 2.0F * x[i] + y[i];
    }
  });

  EXPECT_EQ(lastRow().threads, 1);
}

/** @test A kernel row keeps the one host thread that drove it at --threads 4 */
TEST_F(PerfGpuHarnessTest, KernelRowRecordsOneHostThread) {
  SaxpyFixtureData data;
  ub::PerfConfig cfg = cfg_;
  cfg.threads = 4;
  ub::PerfGpuCase perf{uniqueSuite("GpuRowThreads") + ".Kernel", cfg};
  perf.cudaWarmup(data.launch());

  (void)perf.cudaKernel(data.launch(), "saxpy").measure();

  EXPECT_EQ(lastRow().threads, 1);
}

namespace {

/** @brief Split a CSV line on commas (the names written here hold none). */
std::vector<std::string> splitCells(const std::string& line) {
  std::vector<std::string> cells(1);
  for (const char C : line) {
    if (C == ',') {
      cells.emplace_back();
    } else {
      cells.back() += C;
    }
  }
  return cells;
}

/**
 * @brief Write every waiting row through the CSV listener, as GoogleTest does
 *        at a test's end, and read the file back: one column -> cell map per
 *        row. A row whose cell count is not the header's fails the test.
 */
std::vector<std::map<std::string, std::string>> writeRowsAndRead() {
  static std::atomic<int> files{0};
  const std::string PATH = "/tmp/vernier_gpu_rows_" + std::to_string(::getpid()) + "_" +
                           std::to_string(files.fetch_add(1)) + ".csv";
  {
    ub::detail::CsvListener listener(PATH, /*includeProfile=*/false, /*includeGpu=*/true);
    listener.OnTestEnd(*::testing::UnitTest::GetInstance()->current_test_info());
  }
  std::ifstream in(PATH);
  std::string line;
  std::getline(in, line);
  const std::vector<std::string> NAMES = splitCells(line);
  std::vector<std::map<std::string, std::string>> rows;
  while (std::getline(in, line)) {
    const std::vector<std::string> CELLS = splitCells(line);
    EXPECT_EQ(CELLS.size(), NAMES.size()) << "a row without the header's columns: " << line;
    std::map<std::string, std::string> row;
    for (std::size_t i = 0; i < NAMES.size() && i < CELLS.size(); ++i) {
      row[NAMES[i]] = CELLS[i];
    }
    rows.push_back(row);
  }
  std::remove(PATH.c_str());
  return rows;
}

/** @brief A number as the CSV writer formats a row's statistics. */
std::string csvNumber(double value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

} // namespace

/** @test A CPU baseline and a kernel measured by one case are written as two rows, each its own */
TEST_F(PerfGpuHarnessTest, BaselineAndKernelAreTwoRows) {
  SaxpyFixtureData data;
  const std::string CASE = uniqueSuite("GpuRows") + ".BaselineAndKernel";
  ub::PerfGpuCase perf{CASE, cfg_};
  std::vector<float> x(ELEMENTS, 1.0F);
  std::vector<float> y(ELEMENTS, 2.0F);

  const ub::PerfResult CPU = perf.cpuBaseline(
      [&] {
        for (int i = 0; i < ELEMENTS; ++i) {
          y[i] = 2.0F * x[i] + y[i];
        }
      },
      "cpu");
  perf.cudaWarmup(data.launch());
  const ub::PerfGpuResult GPU = perf.cudaKernel(data.launch(), "saxpy").measure();

  const std::vector<std::map<std::string, std::string>> ROWS = writeRowsAndRead();

  ASSERT_EQ(ROWS.size(), 2U);
  EXPECT_EQ(ROWS[0].at("test"), CASE + "/cpu");
  EXPECT_EQ(ROWS[0].at("threads"), "1");
  EXPECT_EQ(ROWS[0].at("wallMedian"), csvNumber(CPU.stats.median));
  EXPECT_EQ(ROWS[0].at("gpuModel"), "") << "the baseline row carries no GPU cells";
  EXPECT_EQ(ROWS[0].at("kernelTimeUs"), "");
  EXPECT_EQ(ROWS[1].at("test"), CASE + "/saxpy");
  EXPECT_EQ(ROWS[1].at("threads"), "1");
  EXPECT_EQ(ROWS[1].at("wallMedian"), csvNumber(GPU.stats.cpuStats.median));
  EXPECT_NE(ROWS[1].at("gpuModel"), "");
  // The kernel's speedup is over this case's own baseline, the first row.
  EXPECT_DOUBLE_EQ(GPU.speedupVsCpu, CPU.stats.median / GPU.totalTimeUs);
  EXPECT_EQ(ROWS[1].at("speedupVsCpu"), std::to_string(GPU.speedupVsCpu));
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

/* ----------------------------- Unified Memory Cells ----------------------------- */

namespace {

/// The CSV columns of the unified-memory counters.
const std::vector<std::string> UM_COLUMNS = {"umPageFaults", "umH2DMigrations", "umD2HMigrations",
                                             "umMigrationTimeUs", "umThrashing"};

/** @brief What parseGpuFlags() makes of @p flags, as a GPU benchmark's main() parses them. */
ub::PerfGpuConfig parsedGpuFlags(const std::vector<std::string>& flags) {
  std::vector<std::string> args{"bench"};
  args.insert(args.end(), flags.begin(), flags.end());
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  int argc = static_cast<int>(argv.size());
  ub::PerfGpuConfig parsed;
  ub::parseGpuFlags(parsed, &argc, argv.data());
  return parsed;
}

/** @brief A GPU config whose caller asks for unified-memory counters in code. */
ub::PerfGpuConfig programmaticCapture() {
  ub::PerfGpuConfig cfg;
  cfg.captureUnifiedMemory = true;
  return cfg;
}

/**
 * @brief Makes @p cfg the process's GPU config, the one a GPU case reads when
 *        it is built, for a scope; the config it found is restored when the
 *        scope ends.
 */
class ScopedGpuConfig {
public:
  explicit ScopedGpuConfig(const ub::PerfGpuConfig& cfg)
      : saved_(ub::detail::gpuConfigSingleton()) {
    ub::detail::gpuConfigSingleton() = cfg;
  }

  ~ScopedGpuConfig() { ub::detail::gpuConfigSingleton() = saved_; }

  ScopedGpuConfig(const ScopedGpuConfig&) = delete;
  ScopedGpuConfig& operator=(const ScopedGpuConfig&) = delete;

private:
  ub::PerfGpuConfig saved_;
};

/** @brief True when @p row carries any unified-memory field. */
bool hasUnifiedMemoryField(const ub::PerfRow& row) {
  return row.umPageFaults.has_value() || row.umH2DMigrations.has_value() ||
         row.umD2HMigrations.has_value() || row.umMigrationTimeUs.has_value() ||
         row.umThrashing.has_value();
}

/** @brief The recorded failures of the unified-memory request: unsupported, for the whole run. */
std::size_t unifiedMemoryFailures() {
  std::size_t count = 0;
  for (const ub::ProfileFailure& failure : ub::ProfilerRegistry::instance().failures()) {
    count += failure.backend == "capture-um" && failure.test.empty() &&
                     failure.result.cause == ub::ReadinessCause::UNSUPPORTED
                 ? 1
                 : 0;
  }
  return count;
}

/**
 * @brief Why this device records no unified-memory counters, in the
 *        collector's words (it is unavailable, or CUPTI refused the counters),
 *        from one window of a collector asked for them; empty when nothing
 *        refused them. A collector that records none without a reason is not
 *        a device that cannot: the tests that ask then expect the counters.
 */
std::string unifiedMemoryRefusedHere() {
  ub::CuptiCollector probe(false);
  probe.collectUnifiedMemory(true);
  probe.start();
  probe.stop();
  return probe.isAvailable() ? probe.unifiedMemoryRefusal() : probe.unavailableReason();
}

/** @brief The name CUDA gives device 0, the device these tests measure on. */
std::string deviceZeroName() {
  cudaDeviceProp prop{};
  return cudaGetDeviceProperties(&prop, 0) == cudaSuccess ? std::string(prop.name) : std::string();
}

/**
 * @brief The lines of @p text that name all five um* columns, and of those,
 *        the ones that start with @p prefix.
 */
std::pair<std::size_t, std::size_t> linesNamingTheCells(const std::string& text,
                                                        const std::string& prefix) {
  std::size_t naming = 0;
  std::size_t prefixed = 0;
  std::size_t from = 0;
  while (from < text.size()) {
    std::size_t end = text.find('\n', from);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string LINE = text.substr(from, end - from);
    bool namesEveryCell = true;
    for (const std::string& column : UM_COLUMNS) {
      namesEveryCell = namesEveryCell && namesColumn(LINE, column);
    }
    if (namesEveryCell) {
      ++naming;
      prefixed += LINE.rfind(prefix, 0) == 0 ? 1 : 0;
    }
    from = end + 1;
  }
  return {naming, prefixed};
}

/** @brief The first line of @p text that starts with @p prefix, or "none". */
std::string firstLineStarting(const std::string& text, const std::string& prefix) {
  for (std::size_t at = text.find(prefix); at != std::string::npos;
       at = text.find(prefix, at + 1)) {
    if (at == 0 || text[at - 1] == '\n') {
      return text.substr(at, text.find('\n', at) - at);
    }
  }
  return "none";
}

/// The registry's report of a request the run cannot meet.
constexpr const char* UM_REPORT = "[FAIL] Profiler 'capture-um': unsupported: ";

/**
 * @brief Runs a GPU benchmark's measurements and end in this process, under
 *        GPU config @p gpu, then exits with the run's status.
 *
 * Measures the kernel through the multi-GPU path on one device, or as two
 * kernel measurements (one with declared copies), and ends the run as a GPU
 * benchmark's main() does, with ProfilerRegistry::finishRun() after tests of
 * status @p testStatus. Writes to stderr how many lines named all five um*
 * cells while measuring and how many of those were the registry's report of
 * the request, how many rows carried a unified-memory field, how many failures
 * the request recorded, how many lines of the run's end named the request, and
 * the run's end's first line; then, on a line of its own, how many results
 * carried unified-memory counters and whether the request's report names the
 * device; and on a third, the report's first line ("none" without one).
 */
[[noreturn]] void endUnifiedMemoryRun(const ub::PerfConfig& cfg, const ub::PerfGpuConfig& gpu,
                                      bool multiGpu, int testStatus) {
  ub::ProfilerRegistry::instance().resetFailures();
  const ScopedGpuConfig GPU(gpu);
  std::size_t umRows = 0;
  std::size_t umResults = 0;
  std::string measured;
  {
    SaxpyFixtureData data;
    vernier::bench::test::StderrCapture capture;
    ub::PerfGpuCase perf{uniqueSuite("GpuUnifiedMemoryOutcome") + ".Kernel", cfg};
    perf.cudaWarmup(data.launch());
    if (multiGpu) {
      const ub::PerfGpuCase::KernelFn LAUNCH = data.launch();
      ub::MultiGpuKernelBuilder multi =
          perf.cudaKernelMultiGpu(1, [&LAUNCH](int, cudaStream_t s) { LAUNCH(s); }, "saxpy");
      umResults += multi.measure().aggregatedStats.unifiedMemory.has_value() ? 1 : 0;
    } else {
      const ub::PerfGpuResult COPIES =
          perf.cudaKernel(data.launch(), "copies")
              .withHostToDevice(data.hostX(), data.deviceX(), SaxpyFixtureData::bytes())
              .withDeviceToHost(data.deviceY(), data.hostY(), SaxpyFixtureData::bytes())
              .measure();
      const ub::PerfGpuResult KERNEL = perf.cudaKernel(data.launch(), "kernel").measure();
      umResults += (COPIES.stats.unifiedMemory.has_value() ? 1 : 0) +
                   (KERNEL.stats.unifiedMemory.has_value() ? 1 : 0);
    }
    for (const ub::PerfRow& row : ub::PerfRegistry::instance().takeAll()) {
      umRows += hasUnifiedMemoryField(row) ? 1 : 0;
    }
    measured = capture.text();
  }
  const std::size_t FAILURES = unifiedMemoryFailures();
  int status = 0;
  std::string ended;
  {
    vernier::bench::test::StderrCapture capture;
    status = ub::ProfilerRegistry::instance().finishRun(cfg, testStatus, 1);
    ended = capture.text();
  }
  const auto [umLines, umReports] = linesNamingTheCells(measured, UM_REPORT);
  const std::string REPORT = firstLineStarting(measured, UM_REPORT);
  const std::string DEVICE = deviceZeroName();
  std::fprintf(stderr,
               "umLines=%zu umReports=%zu umRows=%zu umFailures=%zu umRunEndLines=%zu "
               "runEnd=%s\n",
               umLines, umReports, umRows, FAILURES,
               linesNamingTheCells(ended, "[profile]   capture-um: unsupported: ").second,
               firstLineStarting(ended, "[profile] ").c_str());
  std::fprintf(stderr, "umResults=%zu umDevice=%d\n", umResults,
               !DEVICE.empty() && REPORT.find(UM_REPORT + DEVICE + ": ") == 0 ? 1 : 0);
  std::fprintf(stderr, "umReport=%s\n", REPORT.c_str());
  std::exit(status);
}

/** @brief N floats of managed memory, freed with their owner; null when the allocation failed. */
struct ManagedFloats {
  float* data = nullptr;
  explicit ManagedFloats(std::size_t n) {
    if (cudaMallocManaged(&data, n * sizeof(float)) != cudaSuccess) {
      data = nullptr;
    }
  }
  ~ManagedFloats() { cudaFree(data); }
  ManagedFloats(const ManagedFloats&) = delete;
  ManagedFloats& operator=(const ManagedFloats&) = delete;
};

/** @brief @p bytes rounded up to the 2 MiB blocks unified memory migrates in at most. */
constexpr std::size_t upTo2MiB(std::size_t bytes) {
  constexpr std::size_t BLOCK2M = 2U << 20U;
  return (bytes + BLOCK2M - 1) / BLOCK2M * BLOCK2M;
}

} // namespace

/**
 * @test A --capture-um measurement with declared copies of device memory
 *       records no migration: where CUPTI records the counters they are all
 *       zero (the old byte-count formula gave 64 faults for these copies) and
 *       nothing is recorded as failed; where it cannot, the result has none and
 *       the measurement records the request's failure. Its CSV row keeps the
 *       header's columns, the five um* cells zero or empty accordingly, and the
 *       timing and byte cells of the measurement. Parsing the flag records nothing
 */
TEST_F(PerfGpuHarnessTest, ExplicitCopiesAreNoMigrations) {
  const ScopedGpuConfig CAPTURE(parsedGpuFlags({"--capture-um"}));
  EXPECT_EQ(unifiedMemoryFailures(), 0U) << "parsing --capture-um recorded a failure";
  SaxpyFixtureData data;
  ub::PerfGpuCase perf{uniqueSuite("GpuCaptureUm") + ".Transfers", cfg_};
  ASSERT_TRUE(perf.gpuConfig().captureUnifiedMemory) << "--capture-um did not reach the case";
  perf.cudaWarmup(data.launch());

  ub::PerfGpuResult result;
  {
    vernier::bench::test::StderrCapture capture; // a refusal's report, checked elsewhere
    result = perf.cudaKernel(data.launch(), "saxpy")
                 .withHostToDevice(data.hostX(), data.deviceX(), SaxpyFixtureData::bytes())
                 .withDeviceToHost(data.deviceY(), data.hostY(), SaxpyFixtureData::bytes())
                 .measure();
  }

  const bool RECORDED = result.stats.unifiedMemory.has_value();
  if (RECORDED) {
    EXPECT_EQ(unifiedMemoryFailures(), 0U) << "a request CUPTI met recorded a failure";
    const ub::UnifiedMemoryProfile& UM = *result.stats.unifiedMemory;
    EXPECT_EQ(UM.pageFaults, 0U);
    EXPECT_EQ(UM.h2dMigrations, 0U);
    EXPECT_EQ(UM.d2hMigrations, 0U);
    EXPECT_EQ(UM.h2dBytes + UM.d2hBytes, 0U);
    EXPECT_DOUBLE_EQ(UM.migrationTimeUs, 0.0);
    EXPECT_EQ(UM.thrashingEvents, 0U);
  } else {
    EXPECT_EQ(unifiedMemoryFailures(), 1U) << "a request CUPTI did not meet was not recorded";
  }
  const std::vector<std::map<std::string, std::string>> ROWS = writeRowsAndRead();
  ASSERT_EQ(ROWS.size(), 1U);
  const std::map<std::string, std::string> UM_CELLS_EXPECTED =
      RECORDED ? std::map<std::string, std::string>{{"umPageFaults", "0"},
                                                    {"umH2DMigrations", "0"},
                                                    {"umD2HMigrations", "0"},
                                                    {"umMigrationTimeUs", "0.000000"},
                                                    {"umThrashing", "0"}}
               : std::map<std::string, std::string>{{"umPageFaults", ""},
                                                    {"umH2DMigrations", ""},
                                                    {"umD2HMigrations", ""},
                                                    {"umMigrationTimeUs", ""},
                                                    {"umThrashing", ""}};
  for (const std::string& column : UM_COLUMNS) {
    EXPECT_EQ(ROWS[0].at(column), UM_CELLS_EXPECTED.at(column))
        << column << (RECORDED ? ": not the observed zero" : ": a value nothing recorded");
  }
  EXPECT_EQ(ROWS[0].at("h2dBytes"), std::to_string(SaxpyFixtureData::bytes()));
  EXPECT_EQ(ROWS[0].at("d2hBytes"), std::to_string(SaxpyFixtureData::bytes()));
  EXPECT_EQ(ROWS[0].at("kernelTimeUs"), std::to_string(result.kernelTimeUs));
  EXPECT_EQ(ROWS[0].at("transferTimeUs"), std::to_string(result.transferTimeUs));
}

/**
 * @test Under --capture-um, a kernel reading a managed buffer the host wrote is
 *       seen faulting and migrating it: at least one GPU fault group, at least
 *       one host-to-device migration of at least the buffer's bytes and at most
 *       those rounded up to unified memory's 2 MiB blocks, nothing back to the
 *       host while the kernel only reads, a time for the transfers, no
 *       thrashing, and no failure recorded. Through the real CSV writer the
 *       row's five um* cells are those counts. Skipped, with CUPTI's or the
 *       collector's words, on a device where a collector asked for the
 *       counters is refused them
 */
TEST_F(PerfGpuHarnessTest, ManagedBufferCountersAreObserved) {
  const YieldEnvCleared CLEARED;
  const std::string REFUSED = unifiedMemoryRefusedHere();
  if (!REFUSED.empty()) {
    GTEST_SKIP() << REFUSED;
  }
  const ScopedGpuConfig CAPTURE(parsedGpuFlags({"--capture-um"}));
  const ManagedFloats MANAGED(ELEMENTS);
  ASSERT_NE(MANAGED.data, nullptr) << "cudaMallocManaged failed";
  for (int i = 0; i < ELEMENTS; ++i) {
    MANAGED.data[i] = 1.0F; // the host's write leaves every page on the host
  }
  SaxpyFixtureData data;
  float* x = MANAGED.data;
  float* y = data.deviceY();
  ub::PerfGpuCase perf{uniqueSuite("GpuUnifiedMemoryObserved") + ".Read", cfg_};

  ub::PerfGpuResult result;
  {
    vernier::bench::test::StderrCapture capture; // a refusal's report, if any
    result = perf.cudaKernel(
                     [x, y](cudaStream_t s) {
                       saxpyKernel<<<(ELEMENTS + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(2.0F, x, y,
                                                                                    ELEMENTS);
                     },
                     "managed_read")
                 .measure();
  }
  ASSERT_TRUE(result.stats.unifiedMemory.has_value())
      << "this device's CUPTI records the counters, and the measurement carries none";
  EXPECT_EQ(unifiedMemoryFailures(), 0U);
  const ub::UnifiedMemoryProfile& UM = *result.stats.unifiedMemory;
  const std::size_t BYTES = SaxpyFixtureData::bytes();
  EXPECT_GE(UM.pageFaults, 1U) << "the first launch touched pages that were on the host";
  EXPECT_GE(UM.h2dMigrations, 1U);
  EXPECT_GE(UM.h2dBytes, BYTES) << "less than the buffer reached the device";
  EXPECT_LE(UM.h2dBytes, upTo2MiB(BYTES)) << "more moved than the buffer's blocks hold";
  EXPECT_EQ(UM.d2hMigrations, 0U) << "pages went back to the host while the kernel only read";
  EXPECT_EQ(UM.d2hBytes, 0U);
  EXPECT_GT(UM.migrationTimeUs, 0.0) << "migrations recorded without a time";
  EXPECT_EQ(UM.thrashingEvents, 0U) << "thrashing recorded for a kernel that only read";
  const std::vector<std::map<std::string, std::string>> ROWS = writeRowsAndRead();
  ASSERT_EQ(ROWS.size(), 1U);
  EXPECT_EQ(ROWS[0].at("umPageFaults"), std::to_string(UM.pageFaults));
  EXPECT_EQ(ROWS[0].at("umH2DMigrations"), std::to_string(UM.h2dMigrations));
  EXPECT_EQ(ROWS[0].at("umD2HMigrations"), "0");
  EXPECT_EQ(ROWS[0].at("umMigrationTimeUs"), std::to_string(UM.migrationTimeUs));
  EXPECT_EQ(ROWS[0].at("umThrashing"), "0");
}

namespace {

/** @brief The GPU's global timer, in nanoseconds. */
__device__ unsigned long long globalTimerNs() {
  unsigned long long now = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

/**
 * @brief Adds one to one int per page until @p durationNs have passed on the
 *        GPU's timer: the kernel side of a host and a kernel writing the same
 *        managed pages at once, for as long as asked however fast the
 *        additions go.
 */
__global__ void contendKernel(int* pages, int stride, unsigned long long durationNs) {
  const unsigned long long START = globalTimerNs();
  while (globalTimerNs() - START < durationNs) {
    atomicAdd(&pages[threadIdx.x * stride], 1);
  }
}

} // namespace

/**
 * @test Thrashing that CUPTI records sets the row's umThrashing cell: a kernel
 *       and the host writing the same 32 managed pages at once for 20 ms, in
 *       one measured launch, give at least one thrashing record and the cell 1
 *       (CUPTI's report, not UnifiedMemoryProfile::isThrashing()'s ratio of
 *       migrations to faults). Skipped where a collector
 *       asked for the counters is refused them, or where the host cannot write
 *       managed memory while a kernel runs
 */
TEST_F(PerfGpuHarnessTest, CuptiThrashingSetsTheThrashingCell) {
  const YieldEnvCleared CLEARED;
  const std::string REFUSED = unifiedMemoryRefusedHere();
  if (!REFUSED.empty()) {
    GTEST_SKIP() << REFUSED;
  }
  int concurrent = 0;
  if (cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0) != cudaSuccess ||
      concurrent != 1) {
    GTEST_SKIP() << "the host cannot write managed memory while a kernel runs on this device";
  }
  const ScopedGpuConfig CAPTURE(parsedGpuFlags({"--capture-um"}));
  constexpr int PAGES = 32;
  constexpr int PAGE = 4096;
  constexpr int STRIDE = PAGE / static_cast<int>(sizeof(int));
  struct ManagedPages {
    int* data = nullptr;
    ManagedPages() {
      if (cudaMallocManaged(&data, PAGES * PAGE) != cudaSuccess) {
        data = nullptr;
      }
    }
    ~ManagedPages() { cudaFree(data); }
    ManagedPages(const ManagedPages&) = delete;
    ManagedPages& operator=(const ManagedPages&) = delete;
  } owned;
  ASSERT_NE(owned.data, nullptr) << "cudaMallocManaged failed";
  std::memset(owned.data, 0, PAGES * PAGE);
  int* pages = owned.data;
  ub::PerfConfig oneLaunch = cfg_;
  oneLaunch.cycles = 1;
  oneLaunch.repeats = 1;
  ub::PerfGpuCase perf{uniqueSuite("GpuUnifiedMemoryThrash") + ".Race", oneLaunch};

  ub::PerfGpuResult result;
  {
    vernier::bench::test::StderrCapture capture; // a one-sample measurement's notes
    result = perf.cudaKernel(
                     [pages](cudaStream_t s) {
                       // A race of a set length: one that ends with a set
                       // number of additions can be over before the host has
                       // written enough for CUPTI to record thrashing.
                       contendKernel<<<1, PAGES, 0, s>>>(pages, STRIDE, 20'000'000ULL);
                       // The host writes the same pages until the kernel ends
                       // (bounded at 5 s).
                       volatile int* host = pages;
                       const auto START = std::chrono::steady_clock::now();
                       while (cudaStreamQuery(s) == cudaErrorNotReady &&
                              std::chrono::steady_clock::now() - START < std::chrono::seconds(5)) {
                         for (int p = 0; p < PAGES; ++p) {
                           host[p * STRIDE + 1] = host[p * STRIDE + 1] + 1;
                         }
                       }
                     },
                     "host_and_kernel")
                 .measure();
  }
  ASSERT_TRUE(result.stats.unifiedMemory.has_value())
      << "this device's CUPTI records the counters, and the measurement carries none";
  EXPECT_GE(result.stats.unifiedMemory->thrashingEvents, 1U)
      << "a host and a kernel writing the same pages at once recorded no thrashing";
  const std::vector<std::map<std::string, std::string>> ROWS = writeRowsAndRead();
  ASSERT_EQ(ROWS.size(), 1U);
  EXPECT_EQ(ROWS[0].at("umThrashing"), "1");
}

/**
 * @test An explicit request for unified-memory counters that the run cannot
 *       meet (here because VERNIER_DISABLE_CUPTI stands the collector down; on
 *       a platform without the counters, because CUPTI refuses them), by
 *       --capture-um or by a caller setting PerfGpuConfig::captureUnifiedMemory,
 *       fails the run: the registry reports it once per process, naming the
 *       device and the reason, for kernel and multi-GPU measurements alike, with
 *       no other line naming the five cells; no result or row carries the
 *       counters; and the run's end names the request and gives a run whose
 *       tests passed status 4
 */
TEST_F(PerfGpuHarnessDeathTest, AnUnmetUnifiedMemoryRequestFailsTheRun) {
  ASSERT_TRUE(parsedGpuFlags({"--capture-um"}).captureUnifiedMemory) << "--capture-um not read";
  // The reason, as a probe window under the same variable gets it.
  const std::string WHY = [] {
    const vernier::bench::test::ScopedEnv OFF("VERNIER_DISABLE_CUPTI", "1");
    return unifiedMemoryRefusedHere();
  }();
  ASSERT_FALSE(WHY.empty()) << "a collector under VERNIER_DISABLE_CUPTI=1 gave no reason";
  const auto FAILED = ::testing::AllOf(
      ::testing::ContainsRegex("umLines=1 umReports=1 umRows=0 umFailures=1 umRunEndLines=1 "
                               "runEnd=\\[profile\\] a profile failed; the run exits with "
                               "status 4:"),
      ::testing::ContainsRegex("umResults=0 umDevice=1"),
      ::testing::HasSubstr("umReport=" + std::string(UM_REPORT) + deviceZeroName() + ": " + WHY +
                           ": "));
  // A fresh process for each run, so each run's once-per-process state is its own.
  const std::string SAVED_STYLE = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto UNMET = [this](const ub::PerfGpuConfig& gpu, bool multiGpu) {
    const vernier::bench::test::ScopedEnv OFF("VERNIER_DISABLE_CUPTI", "1");
    endUnifiedMemoryRun(cfg_, gpu, multiGpu, 0);
  };
  EXPECT_EXIT(UNMET(parsedGpuFlags({"--capture-um"}), false), ::testing::ExitedWithCode(4), FAILED);
  EXPECT_EXIT(UNMET(programmaticCapture(), false), ::testing::ExitedWithCode(4), FAILED);
  EXPECT_EXIT(UNMET(parsedGpuFlags({"--capture-um"}), true), ::testing::ExitedWithCode(4), FAILED);
  GTEST_FLAG_SET(death_test_style, SAVED_STYLE);
}

/**
 * @test Without anything standing the collector down, the request follows the
 *       device: where nothing refuses the counters (a probe window says), the
 *       run passes, nothing is reported or recorded, and every result and row
 *       carries them; where CUPTI refuses them, the run fails as an unmet
 *       request does, the report naming the device and CUPTI's answer, the one
 *       the probe window got
 */
TEST_F(PerfGpuHarnessDeathTest, TheRequestFollowsTheDevice) {
  const YieldEnvCleared CLEARED;
  const std::string WHY = unifiedMemoryRefusedHere();
  const bool RECORDS = WHY.empty();
  const std::string SAVED_STYLE = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  if (RECORDS) {
    const auto MET = [](const char* measured) {
      return ::testing::AllOf(
          ::testing::ContainsRegex(std::string("umLines=0 umReports=0 umRows=") + measured +
                                   " umFailures=0 umRunEndLines=0 runEnd=none"),
          ::testing::ContainsRegex(std::string("umResults=") + measured + " umDevice=0"));
    };
    EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--capture-um"}), false, 0),
                ::testing::ExitedWithCode(0), MET("2"));
    EXPECT_EXIT(endUnifiedMemoryRun(cfg_, programmaticCapture(), false, 0),
                ::testing::ExitedWithCode(0), MET("2"));
    EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--capture-um"}), true, 0),
                ::testing::ExitedWithCode(0), MET("1"));
  } else {
    const auto REFUSED = ::testing::AllOf(
        ::testing::ContainsRegex("umLines=1 umReports=1 umRows=0 umFailures=1 umRunEndLines=1 "
                                 "runEnd=\\[profile\\] a profile failed; the run exits with "
                                 "status 4:"),
        ::testing::ContainsRegex("umResults=0 umDevice=1"),
        ::testing::HasSubstr("umReport=" + std::string(UM_REPORT) + deviceZeroName() + ": " + WHY +
                             ": "));
    EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--capture-um"}), false, 0),
                ::testing::ExitedWithCode(4), REFUSED);
    EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--capture-um"}), true, 0),
                ::testing::ExitedWithCode(4), REFUSED);
  }
  GTEST_FLAG_SET(death_test_style, SAVED_STYLE);
}

/**
 * @test A run whose tests failed keeps their status when it also asked for
 *       unified-memory counters it cannot get; its end still reports the request
 */
TEST_F(PerfGpuHarnessDeathTest, AFailedTestKeepsItsStatusOverTheRequest) {
  const std::string SAVED_STYLE = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        const vernier::bench::test::ScopedEnv OFF("VERNIER_DISABLE_CUPTI", "1");
        endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--capture-um"}), false, 1);
      },
      ::testing::ExitedWithCode(1),
      "umLines=1 umReports=1 umRows=0 umFailures=1 umRunEndLines=1 "
      "runEnd=\\[profile\\] a profile failed; the run exits with the tests' status 1:");
  GTEST_FLAG_SET(death_test_style, SAVED_STYLE);
}

/**
 * @test Choosing the managed-memory strategy (--gpu-memory unified) asks for
 *       no unified-memory counters: like a run without GPU flags, it reports
 *       nothing, records nothing, no row carries a unified-memory field, and
 *       the run passes
 */
TEST_F(PerfGpuHarnessDeathTest, ChoosingManagedMemoryAloneFailsNothing) {
  ASSERT_EQ(parsedGpuFlags({"--gpu-memory", "unified"}).memStrategy,
            ub::PerfGpuConfig::MemoryStrategy::Unified)
      << "--gpu-memory unified not read";
  const std::string PASSED =
      "umLines=0 umReports=0 umRows=0 umFailures=0 umRunEndLines=0 runEnd=none";
  const std::string SAVED_STYLE = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--gpu-memory", "unified"}), false, 0),
              ::testing::ExitedWithCode(0), PASSED);
  EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({"--gpu-memory", "unified"}), true, 0),
              ::testing::ExitedWithCode(0), PASSED);
  EXPECT_EXIT(endUnifiedMemoryRun(cfg_, parsedGpuFlags({}), false, 0), ::testing::ExitedWithCode(0),
              PASSED);
  GTEST_FLAG_SET(death_test_style, SAVED_STYLE);
}

/**
 * @test Each --gpu-memory value, parsed as a GPU benchmark's main() parses it,
 *       reaches a case built under it: the case's gpuConfig() holds that
 *       strategy, where a test that allocates by it reads it, and no request
 *       for unified-memory counters comes with it
 */
TEST_F(PerfGpuHarnessTest, TheSelectedStrategyReachesTheCase) {
  using Strategy = ub::PerfGpuConfig::MemoryStrategy;
  const std::pair<const char*, Strategy> VALUES[] = {{"explicit", Strategy::Explicit},
                                                     {"unified", Strategy::Unified},
                                                     {"pinned", Strategy::Pinned},
                                                     {"mapped", Strategy::Mapped}};
  for (const auto& [VALUE, STRATEGY] : VALUES) {
    const ScopedGpuConfig GPU(parsedGpuFlags({"--gpu-memory", VALUE}));
    const ub::PerfGpuCase perf{uniqueSuite("GpuSelectedStrategy") + ".Case", cfg_};
    EXPECT_EQ(perf.gpuConfig().memStrategy, STRATEGY) << VALUE;
    EXPECT_FALSE(perf.gpuConfig().captureUnifiedMemory) << VALUE;
  }
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

namespace {

/** @brief The GPU config's stream priority rule set for a scope, the old one back after. */
class StreamPriorityRule {
public:
  explicit StreamPriorityRule(bool highPriority)
      : saved_(ub::detail::getGlobalGpuConfig().useHighPriorityStream) {
    ub::detail::gpuConfigSingleton().useHighPriorityStream = highPriority;
  }

  ~StreamPriorityRule() { ub::detail::gpuConfigSingleton().useHighPriorityStream = saved_; }

  StreamPriorityRule(const StreamPriorityRule&) = delete;
  StreamPriorityRule& operator=(const StreamPriorityRule&) = delete;

private:
  bool saved_;
};

} // namespace

/**
 * @test The recorder forwards each of its six calls to the symbol the
 *       harness's call is linked to. Once the harness has run all six (a case
 *       with a high-priority stream and one with a plain one, each made and
 *       ended while recording), each call's definition is exported under a
 *       symbol, its forward looked that same symbol up, this process resolves
 *       it to the recorder, and the definition the forward found is that
 *       symbol in another file. Its lines name the toolkit the run exercised
 */
TEST_F(PerfGpuHarnessTest, CudaHandleRecorderForwardsTheSymbolTheHarnessCalls) {
  if (!bt::cudaHandleRecorderAvailable()) {
    GTEST_SKIP() << bt::cudaHandleRecorderUnavailableReason();
  }
  for (const bool HIGH_PRIORITY : {true, false}) {
    SCOPED_TRACE(HIGH_PRIORITY ? "a high-priority stream" : "a plain stream");
    const StreamPriorityRule RULE(HIGH_PRIORITY);
    bt::startCudaHandleRecording(bt::CudaSetupFailure::None);
    {
      const ub::PerfGpuCase PERF{uniqueSuite("GpuRecorderForward") + ".Case", cfg_};
    }
    const bt::CudaHandleCounts MADE = bt::stopCudaHandleRecording();
    EXPECT_EQ(MADE.streamsMade, 1);
    EXPECT_EQ(MADE.eventsMade, 2);
    EXPECT_EQ(MADE.streamsLeft, 0);
    EXPECT_EQ(MADE.eventsLeft, 0);
  }

  const std::vector<bt::CudaForwardedCall> CALLS = bt::cudaHandleRecorderForwarding();
  ASSERT_EQ(CALLS.size(), 6U);
  for (const bt::CudaForwardedCall& CALL : CALLS) {
    SCOPED_TRACE(CALL.api);
    std::printf("[recorder] CUDART_VERSION %d: %s is %s, forwarded to %s in %s\n", CUDART_VERSION,
                CALL.api.c_str(), CALL.definedAs.c_str(), CALL.forwardedTo.c_str(),
                CALL.forwardFile.c_str());
    EXPECT_TRUE(CALL.forwarded) << "the call never ran";
    EXPECT_EQ(CALL.symbol, CALL.definedAs)
        << "the forward looks up another symbol than the one the recorder defines";
    EXPECT_TRUE(CALL.resolvedHere) << "this process resolves " << CALL.symbol << " elsewhere";
    EXPECT_EQ(CALL.forwardedTo, CALL.symbol) << "the forward found another symbol";
    EXPECT_FALSE(CALL.forwardFile.empty());
    EXPECT_NE(CALL.forwardFile, CALL.definedIn)
        << "the forward found the recorder's own definition";
  }
}
