/**
 * @file SaxpyDriving_uTest.cpp
 * @brief How each GPU version drives the device, call by call.
 *
 * Built against the tracked fake runtime in fake_cuda/ instead of CUDA, so it
 * runs on any machine: it compiles the real SaxpyGpu.cpp and counts what each
 * call acquires, releases, copies and launches. These are the facts a profiler
 * reads off the two versions: Nsight Systems counts G0's cudaMalloc and
 * cudaFree calls (two of each per call, none for G1 after construction), and
 * Nsight Compute shows each version's launch shape (one thread per block for
 * G0, 256 for G1). What the kernel computes is the device tests' concern
 * (Saxpy_uTest.cu); here the launch is recorded and does nothing.
 */

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"

#include <cuda_runtime.h> // the tracked fake, first on this target's include path
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace ubd = vernier::bench::demo;

/* ----------------------------- Launch stand-in ----------------------------- */

namespace saxpy_driving {

/** @brief One recorded launch. */
struct Launch {
  std::size_t n;       ///< Elements the launch covers
  int threadsPerBlock; ///< Block size it was given
};

/** @brief Every launch since the last reset. */
inline std::vector<Launch>& launches() {
  static std::vector<Launch> recorded;
  return recorded;
}

} // namespace saxpy_driving

namespace vernier {
namespace bench {
namespace demo {

/** @brief Record the launch shape; the device work is not part of this test. */
void launchSaxpy(float /*a*/, const float* /*dX*/, float* /*dY*/, std::size_t n,
                 int threadsPerBlock, void* /*stream*/) {
  saxpy_driving::launches().push_back(saxpy_driving::Launch{n, threadsPerBlock});
}

} // namespace demo
} // namespace bench
} // namespace vernier

using saxpy_driving::launches;

namespace {

/* ----------------------------- Constants ----------------------------- */

/// Elements per call: any size works on the fake; a small one keeps it cheap.
constexpr std::size_t ELEMENTS = 1000;
/// Calls per test, so per-call counts are told apart from one-off ones.
constexpr int CALLS = 3;

} // namespace

/* ----------------------------- Fixtures ----------------------------- */

/** @brief Every test starts from nothing acquired and nothing launched. */
class SaxpyDriving : public ::testing::Test {
protected:
  void SetUp() override {
    fake_cuda::reset();
    launches().clear();
  }
  void TearDown() override {
    fake_cuda::reset();
    launches().clear();
  }

  std::vector<float> x_ = std::vector<float>(ELEMENTS, 1.0F);
  std::vector<float> y_ = std::vector<float>(ELEMENTS, 2.0F);
};

/* ----------------------------- G0 ----------------------------- */

/** @test G0 allocates two device buffers on every call and frees both before it returns */
TEST_F(SaxpyDriving, G0AllocatesAndFreesTwoBuffersEveryCall) {
  for (int call = 1; call <= CALLS; ++call) {
    ubd::saxpyG0(2.0F, x_, y_);
    EXPECT_EQ(fake_cuda::state().acquisitions, 2 * call) << "after call " << call;
    EXPECT_EQ(fake_cuda::state().live.size(), 0U) << "buffers still held after call " << call;
  }
}

/** @test G0 copies three times per call: x and y in, y out */
TEST_F(SaxpyDriving, G0CopiesThreeTimesEveryCall) {
  for (int call = 1; call <= CALLS; ++call) {
    ubd::saxpyG0(2.0F, x_, y_);
  }
  EXPECT_EQ(fake_cuda::state().copies, 3 * CALLS);
}

/** @test G0 launches once per call with one thread per block over every element */
TEST_F(SaxpyDriving, G0LaunchesOneThreadPerBlock) {
  for (int call = 1; call <= CALLS; ++call) {
    ubd::saxpyG0(2.0F, x_, y_);
  }
  ASSERT_EQ(launches().size(), static_cast<std::size_t>(CALLS));
  for (const auto& launch : launches()) {
    EXPECT_EQ(launch.threadsPerBlock, 1);
    EXPECT_EQ(launch.n, ELEMENTS);
  }
}

/* ----------------------------- G1 ----------------------------- */

/** @test G1 acquires everything once, when it is built, and nothing per call */
TEST_F(SaxpyDriving, G1AcquiresOnlyWhenBuilt) {
  ubd::SaxpyG1 runner(ELEMENTS);
  const int BUILT = fake_cuda::state().acquisitions;
  EXPECT_EQ(BUILT, 5) << "two device buffers, two pinned buffers, one stream";

  for (int call = 1; call <= CALLS; ++call) {
    runner.apply(2.0F, x_, y_);
    EXPECT_EQ(fake_cuda::state().acquisitions, BUILT) << "call " << call << " acquired";
    EXPECT_EQ(fake_cuda::state().live.size(), 5U) << "call " << call << " released";
  }
}

/** @test G1 copies three times per call: x and y in, y out */
TEST_F(SaxpyDriving, G1CopiesThreeTimesEveryCall) {
  ubd::SaxpyG1 runner(ELEMENTS);
  for (int call = 1; call <= CALLS; ++call) {
    runner.apply(2.0F, x_, y_);
  }
  EXPECT_EQ(fake_cuda::state().copies, 3 * CALLS);
}

/** @test G1 launches once per call with 256 threads per block over every element */
TEST_F(SaxpyDriving, G1LaunchesTwoHundredFiftySixThreadsPerBlock) {
  ubd::SaxpyG1 runner(ELEMENTS);
  for (int call = 1; call <= CALLS; ++call) {
    runner.apply(2.0F, x_, y_);
  }
  ASSERT_EQ(launches().size(), static_cast<std::size_t>(CALLS));
  for (const auto& launch : launches()) {
    EXPECT_EQ(launch.threadsPerBlock, 256);
    EXPECT_EQ(launch.n, ELEMENTS);
  }
}
