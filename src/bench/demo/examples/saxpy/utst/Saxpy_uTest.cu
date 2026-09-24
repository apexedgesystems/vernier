/**
 * @file Saxpy_uTest.cu
 * @brief Every SAXPY version answers what the CPU loop answers.
 *
 * The three versions are only comparable as timings if they compute the same
 * thing, so this holds all of them to the CPU result at sizes that are not
 * multiples of a block: one element, a prime, and either side of 256.
 *
 * The GPU versions need a device and skip without one; the CPU loop's own
 * check runs anywhere the target is built.
 */

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ubd = vernier::bench::demo;

namespace {

/** @brief The tolerance both GPU versions are held to, per element. */
constexpr float RELATIVE_TOLERANCE = 1e-6F;

/** @brief A deterministic, non-trivial input of @p n elements. */
std::vector<float> ramp(std::size_t n, float scale) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = scale * static_cast<float>((i * 2654435761U) % 1000) / 1000.0F;
  }
  return v;
}

/** @brief True when the process can run a kernel. */
bool hasCudaDevice() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

} // namespace

/* ----------------------------- Fixtures ----------------------------- */

/** @brief One length per instantiation; the GPU versions skip without a device. */
class SaxpyVersions : public ::testing::TestWithParam<std::size_t> {
protected:
  void SetUp() override {
    if (!hasCudaDevice()) {
      GTEST_SKIP() << "no CUDA device available";
    }
  }
};

/** @brief One length per instantiation; the CPU loop needs no device. */
class SaxpyCpuLoop : public ::testing::TestWithParam<std::size_t> {};

/* ----------------------------- Tests ----------------------------- */

/**
 * @test The naive port and the properly driven version both reproduce the CPU
 *       loop, element for element.
 *
 * The device may contract the multiply and the add into one instruction, which
 * rounds once instead of twice, so the comparison allows one part in a million
 * of the reference value rather than demanding equality.
 */
TEST_P(SaxpyVersions, GpuVersionsMatchTheCpuLoop) {
  const std::size_t N = GetParam();
  const float A = 2.5F;
  const std::vector<float> X = ramp(N, 3.0F);
  const std::vector<float> Y0 = ramp(N, -2.0F);

  std::vector<float> reference = Y0;
  ubd::saxpyCpu(A, X, reference);

  std::vector<float> g0 = Y0;
  ubd::saxpyG0(A, X, g0);

  std::vector<float> g1 = Y0;
  ubd::SaxpyG1 runner(N);
  runner.apply(A, X, g1);

  ASSERT_EQ(g0.size(), N);
  ASSERT_EQ(g1.size(), N);
  for (std::size_t i = 0; i < N; ++i) {
    const float TOLERANCE = RELATIVE_TOLERANCE * std::max(1.0F, std::fabs(reference[i]));
    ASSERT_NEAR(g0[i], reference[i], TOLERANCE) << "G0 differs at element " << i;
    ASSERT_NEAR(g1[i], reference[i], TOLERANCE) << "G1 differs at element " << i;
  }
}

/** @test The CPU loop computes a*x + y, so the reference is itself checked. */
TEST_P(SaxpyCpuLoop, ComputesSaxpy) {
  const std::size_t N = GetParam();
  const float A = 2.5F;
  const std::vector<float> X = ramp(N, 3.0F);
  const std::vector<float> Y0 = ramp(N, -2.0F);

  std::vector<float> result = Y0;
  ubd::saxpyCpu(A, X, result);

  for (std::size_t i = 0; i < N; ++i) {
    ASSERT_FLOAT_EQ(result[i], A * X[i] + Y0[i]) << "element " << i;
  }
}

/** @brief Lengths either side of a block, a prime, and one element. */
const auto LENGTHS =
    ::testing::Values(std::size_t{1}, std::size_t{7}, std::size_t{255}, std::size_t{256},
                      std::size_t{257}, std::size_t{1000}, std::size_t{100003});

INSTANTIATE_TEST_SUITE_P(Lengths, SaxpyVersions, LENGTHS);
INSTANTIATE_TEST_SUITE_P(Lengths, SaxpyCpuLoop, LENGTHS);
