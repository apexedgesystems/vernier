/**
 * @file SaxpyOwnership_uTest.cpp
 * @brief The GPU versions release everything they acquired, however they leave.
 *
 * Built against the tracked fake runtime in fake_cuda/ instead of CUDA, so it
 * runs on any machine: it compiles the real SaxpyGpu.cpp, makes one
 * acquisition, copy or launch check fail, and checks that nothing acquired
 * before the failure is still held afterwards. What the kernel computes is the
 * device tests' concern (Saxpy_uTest.cu); here the launch does nothing.
 */

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"

#include <cuda_runtime.h> // the tracked fake, first on this target's include path
#include <gtest/gtest.h>

#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ubd = vernier::bench::demo;

/* ----------------------------- Launch stand-in ----------------------------- */

namespace vernier {
namespace bench {
namespace demo {

/** @brief The device launch is not part of this test: record nothing, do nothing. */
void launchSaxpy(float /*a*/, const float* /*dX*/, float* /*dY*/, std::size_t /*n*/,
                 int /*threadsPerBlock*/, void* /*stream*/) {}

} // namespace demo
} // namespace bench
} // namespace vernier

/* ----------------------------- Test parameter ----------------------------- */

namespace saxpy_ownership {

/** @brief Where a call is made to fail, and what the exception must name. */
struct Failure {
  int acquisition;     ///< Acquisition to fail (0: none)
  int copy;            ///< Copy to fail (0: none)
  bool launchCheck;    ///< Make the check after the launch fail
  const char* message; ///< The start of the exception's message
};

/** @brief Name the failure in test output instead of printing its bytes. */
void PrintTo(const Failure& f, std::ostream* os) { *os << "failure at \"" << f.message << "\""; }

} // namespace saxpy_ownership

using saxpy_ownership::Failure;

namespace {

/* ----------------------------- Helpers ----------------------------- */

/** @brief Resources acquired and not yet released. */
std::size_t live() { return fake_cuda::state().live.size(); }

/** @brief Arm the fake for @p f. */
void inject(const Failure& f) {
  fake_cuda::State& s = fake_cuda::state();
  s.failAcquisition = f.acquisition;
  s.failCopy = f.copy;
  s.failNextLaunchCheck = f.launchCheck;
}

/** @brief True when @p text starts with @p prefix. */
bool startsWith(const std::string& text, const char* prefix) { return text.rfind(prefix, 0) == 0; }

} // namespace

/* ----------------------------- Fixtures ----------------------------- */

/** @brief Every test starts and ends with nothing acquired. */
class SaxpyOwnership : public ::testing::Test {
protected:
  void SetUp() override { fake_cuda::reset(); }
  void TearDown() override { fake_cuda::reset(); }
};

/** @brief One G0 failure point per instantiation. */
class SaxpyG0Failure : public SaxpyOwnership, public ::testing::WithParamInterface<Failure> {};

/** @brief One G1 constructor acquisition to fail per instantiation. */
class SaxpyG1ConstructionFailure : public SaxpyOwnership,
                                   public ::testing::WithParamInterface<Failure> {};

/* ----------------------------- G0 ----------------------------- */

/** @test G0 allocates two buffers per call and frees both when it returns. */
TEST_F(SaxpyOwnership, G0ReleasesBothBuffersOnSuccess) {
  std::vector<float> x(8, 1.0F);
  std::vector<float> y(8, 2.0F);

  ubd::saxpyG0(2.0F, x, y);

  EXPECT_EQ(fake_cuda::state().acquisitions, 2);
  EXPECT_EQ(live(), 0U);
}

/** @test Whichever step of G0 fails, the buffers it had allocated are freed. */
TEST_P(SaxpyG0Failure, ReleasesWhatItAcquired) {
  std::vector<float> x(8, 1.0F);
  std::vector<float> y(8, 2.0F);
  inject(GetParam());

  try {
    ubd::saxpyG0(2.0F, x, y);
    FAIL() << "expected the injected failure to throw";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(startsWith(e.what(), GetParam().message)) << "threw: " << e.what();
  }
  EXPECT_EQ(live(), 0U) << "resources still held after the failure";
}

INSTANTIATE_TEST_SUITE_P(
    EachStep, SaxpyG0Failure,
    ::testing::Values(Failure{1, 0, false, "cudaMalloc x"}, Failure{2, 0, false, "cudaMalloc y"},
                      Failure{0, 1, false, "HtoD x"}, Failure{0, 2, false, "HtoD y"},
                      Failure{0, 0, true, "launch"}, Failure{0, 3, false, "DtoH y"}));

/* ----------------------------- G1 ----------------------------- */

/** @test G1 holds five resources while it lives and none once it is gone. */
TEST_F(SaxpyOwnership, G1ReleasesEverythingWhenDestroyed) {
  {
    const ubd::SaxpyG1 RUNNER(8);
    EXPECT_EQ(live(), 5U) << "two device buffers, two pinned buffers, one stream";
  }
  EXPECT_EQ(live(), 0U);
}

/** @test Whichever acquisition of the G1 constructor fails, the earlier ones are released. */
TEST_P(SaxpyG1ConstructionFailure, ReleasesEarlierAcquisitions) {
  inject(GetParam());

  try {
    const ubd::SaxpyG1 RUNNER(8);
    FAIL() << "expected the injected failure to throw";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(startsWith(e.what(), GetParam().message)) << "threw: " << e.what();
  }
  EXPECT_EQ(live(), 0U) << "resources still held after the failed construction";
}

INSTANTIATE_TEST_SUITE_P(EachAcquisition, SaxpyG1ConstructionFailure,
                         ::testing::Values(Failure{1, 0, false, "cudaMalloc x"},
                                           Failure{2, 0, false, "cudaMalloc y"},
                                           Failure{3, 0, false, "pinned x"},
                                           Failure{4, 0, false, "pinned y"},
                                           Failure{5, 0, false, "stream"}));

/** @test A failed apply() keeps G1 whole, and destroying it then releases everything. */
TEST_F(SaxpyOwnership, G1FailedApplyStillReleasesOnDestruction) {
  {
    ubd::SaxpyG1 runner(8);
    std::vector<float> x(8, 1.0F);
    std::vector<float> y(8, 2.0F);
    fake_cuda::state().failCopy = fake_cuda::state().copies + 1;

    EXPECT_THROW(runner.apply(2.0F, x, y), std::runtime_error);
    EXPECT_EQ(live(), 5U);
  }
  EXPECT_EQ(live(), 0U);
}
