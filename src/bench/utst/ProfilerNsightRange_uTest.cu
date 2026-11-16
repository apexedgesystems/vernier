/**
 * @file ProfilerNsightRange_uTest.cu
 * @brief The Nsight backend's NVTX range opens when a measurement starts and
 *        closes when it ends, on every path a measurement takes.
 *
 * Under --profile nsight the guard attaches the Nsight backend, which pushes an
 * NVTX range named after the test when a measurement starts and pops it when
 * the measurement ends, so an Nsight Systems timeline shows each measured
 * window under its test's name. These tests record the ranges the way nsys
 * receives them: NVTX_INJECTION64_PATH names NvtxRangeRecorder, a stand-in NVTX
 * tool this binary links, before anything calls NVTX. Each test measures
 * through the guard's attach path and checks that one range named after the
 * test was pushed after the warmup, held every measured call, and was popped,
 * on the test's thread, before the measurement returned: not when the case or
 * the process ended. A measurement that fails pops its range before its
 * exception reaches the caller, and the case's next measurement pushes a range
 * of its own.
 *
 * Notes:
 *  - NSYS_PROFILING_SESSION_ID, which nsys exports to the process it starts,
 *    and fake nsys and ncu first on PATH let the registry create the backend
 *    on any machine, as in an nsys session; the backend never runs them.
 *  - "Inside the range" is judged by order, not by a thread's nesting: each
 *    call notes how many pushes and pops the recorder had received when it
 *    ran, because the multi-GPU path launches on a worker thread while the
 *    range belongs to the test's.
 *  - The GPU cases need a CUDA device and skip without one; the CPU case does
 *    not.
 *  - Every case skips in a process another NVTX tool already watches
 *    (NVTX_INJECTION64_PATH naming another library, as under nsys), quoting the
 *    variable: that tool receives the ranges instead.
 *  - A build without the toolkit's nvtx3 headers emits no range at all: there
 *    vernier_nvtx_enable() defines COMPAT_NVTX_AVAILABLE=0, so Nvtx.hpp's
 *    VERNIER_NVTX_USABLE is 0 and the backend's push and pop compile to
 *    nothing. This test is built only where the headers are.
 */

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"
#include "src/bench/utst/NvtxRangeRecorder.hpp"
#include "src/bench/utst/ScopedEnv.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ub = vernier::bench;

using vernier::bench::test::clearNvtxRangeEvents;
using vernier::bench::test::NvtxRangeEvent;
using vernier::bench::test::nvtxRangeEventCount;
using vernier::bench::test::nvtxRangeEvents;
using vernier::bench::test::nvtxRecorderModules;
using vernier::bench::test::ScopedEnv;

namespace {

/* ----------------------------- Constants ----------------------------- */

constexpr int ELEMENTS = 1 << 16;
constexpr int BLOCK = 256;
constexpr int CYCLES = 5;
constexpr int REPEATS = 3;
constexpr int WARMUP = 1;

/// The recorder library this binary links, which NVTX_INJECTION64_PATH names.
constexpr const char* RECORDER = VERNIER_NVTX_RECORDER_PATH;

/* ----------------------------- Helpers ----------------------------- */

/** @brief y = a * y over ELEMENTS floats: work for a launch to do. */
__global__ void scaleKernel(float a, float* y, int n) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    y[IDX] = a * y[IDX];
  }
}

/** @brief A distinct case name per call, so repeated runs in one process never share one. */
std::string uniqueName(const char* suite) {
  static std::atomic<int> counter{0};
  return std::string(suite) + std::to_string(counter.fetch_add(1)) + ".Case";
}

/** @brief Whether the recorder watches this process, and what was named instead. */
struct RecorderInjection {
  bool installed = false; ///< NVTX_INJECTION64_PATH names the recorder
  std::string otherTool;  ///< The value that named another tool instead
};

/**
 * @brief Names the recorder in NVTX_INJECTION64_PATH, once per process, unless
 * another tool is named there already. nvtx3 reads the variable when a module
 * first calls NVTX, so every fixture calls this before anything else.
 */
const RecorderInjection& recorderInjection() {
  static const RecorderInjection STATE = [] {
    RecorderInjection state;
    const char* named = std::getenv("NVTX_INJECTION64_PATH");
    if (named != nullptr && *named != '\0' && std::string(named) != RECORDER) {
      state.otherTool = named;
      return state;
    }
    ::setenv("NVTX_INJECTION64_PATH", RECORDER, 1);
    state.installed = true;
    return state;
  }();
  return STATE;
}

/** @brief The events on one line, "push A; pop", for comparisons and messages. */
std::string describe(const std::vector<NvtxRangeEvent>& events) {
  std::string line;
  for (const NvtxRangeEvent& event : events) {
    line += line.empty() ? "" : "; ";
    line += event.push ? "push " + event.name : "pop";
  }
  return line;
}

/** @brief True when every event came from the calling thread. */
bool allOnThisThread(const std::vector<NvtxRangeEvent>& events) {
  for (const NvtxRangeEvent& event : events) {
    if (event.where != std::this_thread::get_id()) {
      return false;
    }
  }
  return true;
}

/**
 * @brief The checks every single-measurement case makes. @p seen holds, per
 * call in call order, how many events the recorder had when the call ran: the
 * first @p warmupCalls none (outside the range), the next @p measuredCalls
 * exactly the push (inside it). @p atReturn is the count when the measurement
 * returned: the push and the pop.
 */
void expectOneRangeAround(const std::string& name, const std::vector<std::size_t>& seen,
                          std::size_t warmupCalls, std::size_t measuredCalls,
                          std::size_t atReturn) {
  const std::vector<NvtxRangeEvent> EVENTS = nvtxRangeEvents();
  ASSERT_GE(nvtxRecorderModules(), 1) << "NVTX delivered nothing to the recorder";
  EXPECT_EQ(describe(EVENTS), "push " + name + "; pop");
  EXPECT_TRUE(allOnThisThread(EVENTS)) << "the range was pushed or popped on another thread";
  EXPECT_EQ(atReturn, 2U) << "the range was still open when the measurement returned";
  ASSERT_EQ(seen.size(), warmupCalls + measuredCalls);
  for (std::size_t i = 0; i < warmupCalls; ++i) {
    EXPECT_EQ(seen[i], 0U) << "warmup call " << i << " ran inside the range";
  }
  for (std::size_t i = warmupCalls; i < seen.size(); ++i) {
    EXPECT_EQ(seen[i], 1U) << "measured call " << (i - warmupCalls) << " ran outside the range";
  }
}

/** @brief What a failed measurement left, read where its caller catches the error. */
struct Failure {
  std::string what;                   ///< The std::runtime_error's message; empty if none
  std::vector<NvtxRangeEvent> events; ///< The recorder's events when it was caught
};

/**
 * @brief Runs @p measure, which should fail, and reads the error and the
 *        recorder's events in the catch block: the first place the caller can
 *        do anything after the failure.
 */
template <typename Measure> Failure failureOf(Measure measure) {
  Failure failure;
  try {
    measure();
  } catch (const std::runtime_error& e) {
    failure.events = nvtxRangeEvents();
    failure.what = e.what();
  }
  return failure;
}

/**
 * @brief The checks on a failed measurement of the case @p name: the error
 * reached the caller, the range was pushed and popped on this thread before
 * it did, and no row was published.
 */
void expectClosedBeforeTheCaller(const std::string& name, const Failure& failure) {
  ASSERT_FALSE(failure.what.empty()) << "the measurement did not fail";
  EXPECT_EQ(describe(failure.events), "push " + name + "; pop")
      << "the range was still open when the caller caught the error";
  EXPECT_TRUE(allOnThisThread(failure.events))
      << "the range was pushed or popped on another thread";
  EXPECT_FALSE(ub::PerfRegistry::instance().take().has_value())
      << "the failed measurement published a row";
}

/**
 * @brief The checks on the measurement a case makes after a failed one: a
 * range of its own, pushed after the failed window's pop, holding every
 * measured call and popped before it returned. @p seen and @p atReturn are as
 * for expectOneRangeAround(), two events later.
 */
void expectARangeOfItsOwnAfter(const std::string& name, const std::vector<std::size_t>& seen,
                               std::size_t warmupCalls, std::size_t measuredCalls,
                               std::size_t atReturn) {
  const std::vector<NvtxRangeEvent> EVENTS = nvtxRangeEvents();
  EXPECT_EQ(describe(EVENTS), "push " + name + "; pop; push " + name + "; pop")
      << "the measurement after the failed one has no range of its own";
  EXPECT_TRUE(allOnThisThread(EVENTS)) << "a range was pushed or popped on another thread";
  EXPECT_EQ(atReturn, 4U) << "the range was still open when the measurement returned";
  std::vector<std::size_t> inside(warmupCalls, 2U);
  inside.resize(warmupCalls + measuredCalls, 3U);
  EXPECT_EQ(seen, inside) << "events seen by each call: 2 before the range, 3 inside it";
}

/** @brief One launch the runtime refuses: a block of more threads than any device takes. */
void refusedLaunch(cudaStream_t stream) { scaleKernel<<<1, 2048, 0, stream>>>(1.0F, nullptr, 0); }

/** @brief The runtime's answer to refusedLaunch(), from a launch of the test's own. */
cudaError_t refusalError() {
  refusedLaunch(nullptr);
  return cudaGetLastError();
}

/**
 * @brief A private directory holding fake nsys and ncu, which let the registry
 * create the Nsight backend, and the backend's artifact root.
 */
class FakeNsightTools {
public:
  /** @param name Makes the directory this test's own: one per test name and process. */
  explicit FakeNsightTools(const std::string& name)
      : dir_((std::filesystem::temp_directory_path() /
              ("vernier_nsight_range_" + name + "_" + std::to_string(::getpid())))
                 .string()) {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    if (!std::filesystem::create_directories(dir_, ec)) {
      dir_.clear();
      return;
    }
    for (const char* tool : {"nsys", "ncu"}) {
      const std::string TOOL_PATH = dir_ + "/" + tool;
      std::ofstream(TOOL_PATH) << "#!/bin/sh\nexit 0\n";
      ::chmod(TOOL_PATH.c_str(), 0755);
    }
  }
  ~FakeNsightTools() {
    std::error_code ec;
    if (!dir_.empty()) {
      std::filesystem::remove_all(dir_, ec);
    }
  }
  FakeNsightTools(const FakeNsightTools&) = delete;
  FakeNsightTools& operator=(const FakeNsightTools&) = delete;

  [[nodiscard]] bool ok() const { return !dir_.empty(); }
  [[nodiscard]] const std::string& dir() const { return dir_; }

private:
  std::string dir_;
};

/** @brief One device vector for a kernel to scale, freed however the test leaves. */
class DeviceVector {
public:
  DeviceVector() {
    if (cudaMalloc(&data_, ELEMENTS * sizeof(float)) != cudaSuccess) {
      data_ = nullptr;
    }
  }
  ~DeviceVector() { cudaFree(data_); }
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;

  [[nodiscard]] bool ok() const { return data_ != nullptr; }

  /** @brief One launch on @p stream. */
  void launch(cudaStream_t stream) const {
    scaleKernel<<<(ELEMENTS + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(1.0F, data_, ELEMENTS);
  }

private:
  float* data_ = nullptr;
};

} // namespace

/* ----------------------------- Fixtures ----------------------------- */

/**
 * @brief --profile nsight in an nsys session's environment, fake tools first on
 * PATH, artifacts in a private root.
 */
class NsightRangeTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!recorderInjection().installed) {
      GTEST_SKIP() << "another NVTX tool watches this process (NVTX_INJECTION64_PATH="
                   << recorderInjection().otherTool << ") and receives its ranges";
    }
    tools_.emplace(::testing::UnitTest::GetInstance()->current_test_info()->name());
    ASSERT_TRUE(tools_->ok()) << "could not create the fake tool directory";
    const char* oldPath = std::getenv("PATH");
    path_.emplace("PATH", tools_->dir() + ":" + (oldPath != nullptr ? oldPath : ""));
    // A request for nsight collects only in an nsys session; this is the
    // variable nsys exports to the process it starts.
    session_.emplace("NSYS_PROFILING_SESSION_ID", "nsight-range-test");

    cfg_.profileTool = "nsight";
    cfg_.artifactRoot = tools_->dir() + "/artifacts";
    cfg_.cycles = CYCLES;
    cfg_.repeats = REPEATS;
    cfg_.warmup = WARMUP;
    (void)ub::PerfRegistry::instance().take();
    clearNvtxRangeEvents();
  }

  void TearDown() override {
    (void)ub::PerfRegistry::instance().take();
    session_.reset();
    path_.reset();
    tools_.reset();
  }

  ub::PerfConfig cfg_{};
  std::optional<FakeNsightTools> tools_;
  std::optional<ScopedEnv> path_;
  std::optional<ScopedEnv> session_;
};

/** @brief The same, for cases that measure on a CUDA device: skips without one. */
class NsightRangeGpuTest : public NsightRangeTest {
protected:
  void SetUp() override {
    NsightRangeTest::SetUp();
    if (IsSkipped() || HasFatalFailure()) {
      return;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      GTEST_SKIP() << "no CUDA device available";
    }
    ub::ensureBenchGpuAbi();
  }
};

/* ----------------------------- API Tests ----------------------------- */

/**
 * @test A kernel measurement's range is pushed after the warmup, holds every
 *       measured launch, and is popped before measure() returns
 */
TEST_F(NsightRangeGpuTest, KernelMeasurementIsBracketed) {
  const std::string NAME = uniqueName("NsightRangeKernel");
  const DeviceVector DATA;
  ASSERT_TRUE(DATA.ok()) << "device allocation failed";
  ub::PerfGpuCase perf{NAME, cfg_};
  ub::attachGpuProfilerHooks(perf, cfg_);

  std::vector<std::size_t> seen;
  const ub::PerfGpuCase::KernelFn LAUNCH = [&](cudaStream_t s) {
    seen.push_back(nvtxRangeEventCount());
    DATA.launch(s);
  };
  perf.cudaWarmup(LAUNCH);
  const std::size_t WARMUP_LAUNCHES = seen.size();
  (void)perf.cudaKernel(LAUNCH, "scale").measure();
  const std::size_t AT_RETURN = nvtxRangeEventCount();

  expectOneRangeAround(NAME, seen, WARMUP_LAUNCHES, CYCLES * REPEATS, AT_RETURN);
}

/**
 * @test A CPU baseline's range is pushed after the baseline's warmup, holds
 *       every measured call, and is popped before cpuBaseline() returns
 */
TEST_F(NsightRangeGpuTest, BaselineIsBracketed) {
  const std::string NAME = uniqueName("NsightRangeBaseline");
  ub::PerfGpuCase perf{NAME, cfg_};
  ub::attachGpuProfilerHooks(perf, cfg_);

  std::vector<std::size_t> seen;
  (void)perf.cpuBaseline([&] { seen.push_back(nvtxRangeEventCount()); });
  const std::size_t AT_RETURN = nvtxRangeEventCount();

  // The baseline warms up with WARMUP rounds of CYCLES calls.
  expectOneRangeAround(NAME, seen, WARMUP * CYCLES, CYCLES * REPEATS, AT_RETURN);
}

/**
 * @test A multi-GPU measurement's range, pushed and popped on the test's
 *       thread, holds every launch its device thread makes
 */
TEST_F(NsightRangeGpuTest, MultiGpuMeasurementIsBracketed) {
  const std::string NAME = uniqueName("NsightRangeMultiGpu");
  const DeviceVector DATA;
  ASSERT_TRUE(DATA.ok()) << "device allocation failed";
  ub::PerfGpuCase perf{NAME, cfg_};
  ub::attachGpuProfilerHooks(perf, cfg_);

  // Written by the device's thread; read here after measure() has joined it.
  std::vector<std::size_t> seen;
  (void)perf
      .cudaKernelMultiGpu(1,
                          [&](int, cudaStream_t s) {
                            seen.push_back(nvtxRangeEventCount());
                            DATA.launch(s);
                          })
      .measure();
  const std::size_t AT_RETURN = nvtxRangeEventCount();

  expectOneRangeAround(NAME, seen, 0, CYCLES * REPEATS, AT_RETURN);
}

/**
 * @test A case that measures twice gets two ranges, the first popped before
 *       the second measurement starts
 */
TEST_F(NsightRangeGpuTest, EachMeasurementGetsItsOwnRange) {
  const std::string NAME = uniqueName("NsightRangeTwice");
  const DeviceVector DATA;
  ASSERT_TRUE(DATA.ok()) << "device allocation failed";
  ub::PerfGpuCase perf{NAME, cfg_};
  ub::attachGpuProfilerHooks(perf, cfg_);

  std::vector<std::size_t> baselineSeen;
  (void)perf.cpuBaseline([&] { baselineSeen.push_back(nvtxRangeEventCount()); });
  const std::size_t AFTER_BASELINE = nvtxRangeEventCount();

  std::vector<std::size_t> kernelSeen;
  const ub::PerfGpuCase::KernelFn LAUNCH = [&](cudaStream_t s) {
    kernelSeen.push_back(nvtxRangeEventCount());
    DATA.launch(s);
  };
  perf.cudaWarmup(LAUNCH);
  const std::size_t WARMUP_LAUNCHES = kernelSeen.size();
  (void)perf.cudaKernel(LAUNCH, "scale").measure();
  const std::size_t AFTER_KERNEL = nvtxRangeEventCount();

  const std::vector<NvtxRangeEvent> EVENTS = nvtxRangeEvents();
  EXPECT_EQ(describe(EVENTS), "push " + NAME + "; pop; push " + NAME + "; pop");
  EXPECT_TRUE(allOnThisThread(EVENTS)) << "a range was pushed or popped on another thread";
  EXPECT_EQ(AFTER_BASELINE, 2U) << "the baseline's range was still open when it returned";
  EXPECT_EQ(AFTER_KERNEL, 4U) << "the kernel's range was still open when it returned";

  const std::size_t BASELINE_WARMUP = static_cast<std::size_t>(WARMUP * CYCLES);
  ASSERT_EQ(baselineSeen.size(), BASELINE_WARMUP + CYCLES * REPEATS);
  for (std::size_t i = 0; i < baselineSeen.size(); ++i) {
    EXPECT_EQ(baselineSeen[i], i < BASELINE_WARMUP ? 0U : 1U) << "baseline call " << i;
  }
  ASSERT_EQ(kernelSeen.size(), WARMUP_LAUNCHES + CYCLES * REPEATS);
  for (std::size_t i = 0; i < kernelSeen.size(); ++i) {
    EXPECT_EQ(kernelSeen[i], i < WARMUP_LAUNCHES ? 2U : 3U) << "kernel launch " << i;
  }
}

/**
 * @test A measurement that fails pops its range before the exception reaches
 *       the caller, while the case lives on: a refused launch, a refused
 *       launch on a device's thread, and a baseline whose callable throws
 *       inside the range. The caller gets the measurement's own error, no row
 *       is published, and the case's next measurement has a range of its own
 */
TEST_F(NsightRangeGpuTest, AFailedMeasurementPopsItsRangeBeforeTheCallerRecovers) {
  const cudaError_t REFUSAL = refusalError();
  ASSERT_NE(REFUSAL, cudaSuccess) << "the runtime accepted a block of 2048 threads";
  const std::string BASELINE_FAILED = "the baseline's callable threw inside its range";
  const DeviceVector DATA;
  ASSERT_TRUE(DATA.ok()) << "device allocation failed";
  struct Path {
    const char* name;
    std::function<void(ub::PerfGpuCase&)> measure;
    bool refused;          ///< The error names the test and the runtime's refusal
    const char* alsoNamed; ///< What else the error names
  };
  const Path PATHS[] = {
      {"a refused launch",
       [](ub::PerfGpuCase& perf) { (void)perf.cudaKernel(refusedLaunch, "refused").measure(); },
       true, ""},
      {"a refused launch on a device's thread",
       [](ub::PerfGpuCase& perf) {
         (void)perf.cudaKernelMultiGpu(1, [](int, cudaStream_t s) { refusedLaunch(s); }).measure();
       },
       true, "device 0"},
      {"a baseline whose callable throws inside the range",
       [&BASELINE_FAILED](ub::PerfGpuCase& perf) {
         // The warmup runs before the push: only a measured call throws.
         (void)perf.cpuBaseline([&] {
           if (nvtxRangeEventCount() == 1) {
             throw std::runtime_error(BASELINE_FAILED);
           }
         });
       },
       false, ""},
  };
  for (const Path& path : PATHS) {
    SCOPED_TRACE(path.name);
    clearNvtxRangeEvents();
    (void)ub::PerfRegistry::instance().take();
    const std::string NAME = uniqueName("NsightRangeFailed");
    ub::PerfGpuCase perf{NAME, cfg_};
    ub::attachGpuProfilerHooks(perf, cfg_);

    const Failure FAILED = failureOf([&] { path.measure(perf); });
    expectClosedBeforeTheCaller(NAME, FAILED);
    if (path.refused) {
      EXPECT_NE(FAILED.what.find(NAME), std::string::npos) << "the caller got: " << FAILED.what;
      EXPECT_NE(FAILED.what.find(cudaGetErrorName(REFUSAL)), std::string::npos) << FAILED.what;
      EXPECT_NE(FAILED.what.find(path.alsoNamed), std::string::npos) << FAILED.what;
    } else {
      EXPECT_EQ(FAILED.what, BASELINE_FAILED);
    }

    // The case lives on, and its next measurement is bracketed by its own range.
    std::vector<std::size_t> seen;
    const ub::PerfGpuCase::KernelFn LAUNCH = [&](cudaStream_t s) {
      seen.push_back(nvtxRangeEventCount());
      DATA.launch(s);
    };
    perf.cudaWarmup(LAUNCH);
    const std::size_t WARMUP_LAUNCHES = seen.size();
    (void)perf.cudaKernel(LAUNCH, "scale").measure();
    const std::size_t AT_RETURN = nvtxRangeEventCount();

    expectARangeOfItsOwnAfter(NAME, seen, WARMUP_LAUNCHES, CYCLES * REPEATS, AT_RETURN);
  }
}

/**
 * @test A CPU measurement's range is pushed after the warmup, holds every
 *       measured call, and is popped before throughputLoop() returns
 */
TEST_F(NsightRangeTest, CpuMeasurementIsBracketed) {
  const std::string NAME = uniqueName("NsightRangeCpu");
  ub::PerfCase perf{NAME, cfg_};
  ub::attachProfilerHooks(perf, cfg_);

  std::vector<std::size_t> seen;
  const auto CALL = [&] { seen.push_back(nvtxRangeEventCount()); };
  perf.warmup(CALL);
  (void)perf.throughputLoop(CALL, "call");
  const std::size_t AT_RETURN = nvtxRangeEventCount();

  expectOneRangeAround(NAME, seen, WARMUP, CYCLES * REPEATS, AT_RETURN);
}

/**
 * @test A CPU measurement whose callback throws pops its range before the
 *       exception reaches the caller, while the case lives on: the caller
 *       gets the callback's error, no row is published, and the case's next
 *       measurement has a range of its own
 */
TEST_F(NsightRangeTest, AFailedCpuMeasurementPopsItsRangeBeforeTheCallerRecovers) {
  const std::string NAME = uniqueName("NsightRangeCpuFailed");
  const std::string CALLBACK_FAILED = "the callback threw inside its range";
  ub::PerfCase perf{NAME, cfg_};
  ub::attachProfilerHooks(perf, cfg_);

  const Failure FAILED = failureOf(
      [&] { (void)perf.measured([&] { throw std::runtime_error(CALLBACK_FAILED); }, "failing"); });
  expectClosedBeforeTheCaller(NAME, FAILED);
  EXPECT_EQ(FAILED.what, CALLBACK_FAILED);

  std::vector<std::size_t> seen;
  const auto CALL = [&] { seen.push_back(nvtxRangeEventCount()); };
  perf.warmup(CALL);
  (void)perf.throughputLoop(CALL, "call");
  const std::size_t AT_RETURN = nvtxRangeEventCount();

  expectARangeOfItsOwnAfter(NAME, seen, WARMUP, CYCLES * REPEATS, AT_RETURN);
}
