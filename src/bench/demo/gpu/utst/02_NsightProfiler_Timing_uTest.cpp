/**
 * @file 02_NsightProfiler_Timing_uTest.cpp
 * @brief Demo 02's guard timing releases its stream and events however it leaves.
 *
 * Built against the tracked fake runtime in examples/saxpy/utst/fake_cuda
 * instead of CUDA, so it runs on any machine: it compiles the real
 * 02_NsightProfiler_Timing.cpp, makes one step fail (an acquisition, the launch
 * check, or the host allocation of the sample buffer, by the scalar operator new
 * family this binary replaces), and checks that nothing acquired before the
 * failure is still held afterwards. The launch does nothing; the fake's event
 * pair reports a set elapsed time.
 */

#include "src/bench/demo/gpu/02_NsightProfiler_Timing.hpp"

#include <cuda_runtime.h> // the tracked fake, first on this target's include path
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <new>

namespace ubd = vernier::bench::demo;

/* ----------------------------- Host allocation fault ----------------------------- */

namespace {

/// The next throwing operator new throws std::bad_alloc, once.
bool g_failNextNew = false;

/** @brief Arms the allocation fault once the third acquisition (the stop event) is held. */
void failNextNewAfterThird(int acquisitions) {
  if (acquisitions == 3) {
    g_failNextNew = true;
  }
}

} // namespace

// Memory from one form of operator new may be released by any delete of its
// family, so the whole scalar family is replaced together, on malloc and free:
// throwing and nothrow new, plain, sized and nothrow delete (GoogleTest itself
// allocates through nothrow new and frees through sized delete). Nothing then
// crosses between these and a runtime's own forms, a sanitizer's included.
// Array and aligned forms pair only within their own families and stay the
// runtime's. The allocation keeps the standard behaviour: retry through the
// new-handler while one is installed, then std::bad_alloc, or a null pointer
// from the nothrow form.
void* operator new(std::size_t size) {
  if (g_failNextNew) {
    g_failNextNew = false;
    throw std::bad_alloc();
  }
  const std::size_t BYTES = size != 0 ? size : 1;
  for (;;) {
    if (void* p = std::malloc(BYTES)) {
      return p;
    }
    const std::new_handler HANDLER = std::get_new_handler();
    if (HANDLER == nullptr) {
      throw std::bad_alloc();
    }
    HANDLER();
  }
}

void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

// Out of line: inlined into a caller, free() would meet a pointer the compiler
// knows came from operator new, and it warns about the pair.
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }

[[gnu::noinline]] void operator delete(void* p, std::size_t /*size*/) noexcept { std::free(p); }

[[gnu::noinline]] void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}

/* ----------------------------- Launch stand-in ----------------------------- */

namespace {

/// Launches made since the last reset, and the block size of the last one.
int g_launches = 0;
int g_lastThreadsPerBlock = 0;

} // namespace

namespace vernier {
namespace bench {
namespace demo {

/** @brief The device launch is not part of this test: count it, do nothing. */
void launchSaxpy(float /*a*/, const float* /*dX*/, float* /*dY*/, std::size_t /*n*/,
                 int threadsPerBlock, void* /*stream*/) {
  ++g_launches;
  g_lastThreadsPerBlock = threadsPerBlock;
}

} // namespace demo
} // namespace bench
} // namespace vernier

namespace {

/* ----------------------------- Helpers ----------------------------- */

constexpr int CALLS = 5;   ///< Launches per sample, as demo 02's guard uses
constexpr int SAMPLES = 5; ///< Samples, as demo 02's guard uses

/** @brief Resources acquired and not yet released. */
std::size_t live() { return fake_cuda::state().live.size(); }

/** @brief The guard timing on stand-in buffers, 256 threads per block. */
double timeIt() {
  static float x[8] = {};
  static float y[8] = {};
  return ubd::medianLaunchUs(2.5F, x, y, 8, 256, CALLS, SAMPLES);
}

} // namespace

/* ----------------------------- Fixture ----------------------------- */

/** @brief Every test starts and ends with nothing acquired and no fault armed. */
class NsightProfilerTiming : public ::testing::Test {
protected:
  void SetUp() override {
    fake_cuda::reset();
    g_failNextNew = false;
    g_launches = 0;
    g_lastThreadsPerBlock = 0;
  }
  void TearDown() override {
    g_failNextNew = false;
    fake_cuda::reset();
  }
};

/** @brief One of the three acquisitions (stream, start event, stop event) fails. */
class NsightProfilerTimingAcquisition : public NsightProfilerTiming,
                                        public ::testing::WithParamInterface<int> {};

/* ----------------------------- Tests ----------------------------- */

/** @test An ordinary run times every batch and releases the stream and both events. */
TEST_F(NsightProfilerTiming, OrdinaryRunReleasesEverything) {
  fake_cuda::state().elapsedMs = 2.0F; // 2 ms per batch of 5 launches

  const double US = timeIt();

  EXPECT_DOUBLE_EQ(US, 2.0 * 1000.0 / CALLS);
  EXPECT_EQ(fake_cuda::state().acquisitions, 3) << "one stream, two events";
  EXPECT_EQ(g_launches, 1 + SAMPLES * CALLS) << "one untimed launch, then the timed batches";
  EXPECT_EQ(g_lastThreadsPerBlock, 256);
  EXPECT_EQ(fake_cuda::state().eventRecords, 2 * SAMPLES) << "an event pair per batch";
  EXPECT_EQ(live(), 0U);
}

/** @test Whichever acquisition fails, the timing reports failure and holds nothing. */
TEST_P(NsightProfilerTimingAcquisition, FailedAcquisitionReleasesEarlierOnes) {
  fake_cuda::state().failAcquisition = GetParam();

  EXPECT_LT(timeIt(), 0.0);
  EXPECT_EQ(g_launches, 0) << "nothing launched without a stream and both events";
  EXPECT_EQ(live(), 0U) << "resources still held after acquisition " << GetParam() << " failed";
}

INSTANTIATE_TEST_SUITE_P(StreamStartStop, NsightProfilerTimingAcquisition,
                         ::testing::Values(1, 2, 3));

/** @test A launch error reported after the batches: failure, nothing held. */
TEST_F(NsightProfilerTiming, LaunchErrorReleasesEverything) {
  fake_cuda::state().failNextLaunchCheck = true;

  EXPECT_LT(timeIt(), 0.0);
  EXPECT_EQ(live(), 0U);
}

/** @test The sample buffer's allocation throws after all three acquisitions: nothing held. */
TEST_F(NsightProfilerTiming, HostAllocationFaultReleasesEverything) {
  fake_cuda::state().afterAcquisition = failNextNewAfterThird;

  EXPECT_THROW(timeIt(), std::bad_alloc);
  EXPECT_EQ(fake_cuda::state().acquisitions, 3) << "the fault came after the stop event";
  EXPECT_FALSE(g_failNextNew) << "the armed fault was not reached";
  EXPECT_EQ(live(), 0U) << "the stream or an event outlived the exception";
}
