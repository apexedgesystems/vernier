/**
 * @file 03_SharedMemoryOpt_uTest.cu
 * @brief Demo 03's three transpose kernels answer what a CPU transpose
 *        answers, and declare the shared memory the demo hands the harness.
 *
 * The kernels are comparable as timings and as counter readings only if they
 * compute the same thing, so every one of them is held to a CPU transpose at
 * sizes that are not multiples of a tile, on either side of one, and at the
 * demo's own. The bytes test reads each kernel's static shared memory from
 * the runtime and holds it to the constant the demo passes to
 * withLaunchConfig(), so the harness's occupancy estimate accounts for what
 * the kernel really declares. Both need a device and skip without one.
 */

#include "src/bench/demo/gpu/03_SharedMemoryOpt_Transpose.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace sm = vernier::bench::demo::shared_memory_demo;

namespace {

/// A kernel of the demo's shape: input, output, one side of the matrix.
using TransposeKernel = void (*)(const float*, float*, int);

/// The three kernels, by name, for one loop per test.
struct NamedKernel {
  const char* name;
  TransposeKernel kernel;
};

constexpr NamedKernel KERNELS[] = {{"transposeNaive", sm::transposeNaive},
                                   {"transposeSharedConflict", sm::transposeSharedConflict},
                                   {"transposeSharedPadded", sm::transposeSharedPadded}};

/** @brief A deterministic, non-trivial input of @p n elements. */
std::vector<float> ramp(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<float>((i * 2654435761U) % 1000) / 8.0F;
  }
  return v;
}

/** @brief The transpose of a dim x dim matrix, on the CPU. */
std::vector<float> transposeCpu(const std::vector<float>& input, int dim) {
  std::vector<float> output(input.size());
  for (int y = 0; y < dim; ++y) {
    for (int x = 0; x < dim; ++x) {
      output[static_cast<std::size_t>(x) * dim + y] = input[static_cast<std::size_t>(y) * dim + x];
    }
  }
  return output;
}

/** @brief The first index where @p a and @p b differ; -1 when they do not. */
long firstMismatch(const std::vector<float>& a, const std::vector<float>& b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return static_cast<long>(i);
    }
  }
  return -1;
}

/** @brief True when the process can run a kernel. */
bool hasCudaDevice() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

/**
 * @brief Run @p kernel once over a dim x dim @p input on the device and
 *        return its output; empty when a CUDA call fails.
 */
std::vector<float> transposeOnDevice(TransposeKernel kernel, const std::vector<float>& input,
                                     int dim) {
  const std::size_t BYTES = input.size() * sizeof(float);
  float* dIn = nullptr;
  float* dOut = nullptr;
  std::vector<float> output;
  if (cudaMalloc(&dIn, BYTES) == cudaSuccess && cudaMalloc(&dOut, BYTES) == cudaSuccess &&
      cudaMemcpy(dIn, input.data(), BYTES, cudaMemcpyHostToDevice) == cudaSuccess) {
    kernel<<<sm::transposeGrid(dim), sm::transposeBlock()>>>(dIn, dOut, dim);
    output.resize(input.size());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(output.data(), dOut, BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) {
      output.clear();
    }
  }
  cudaFree(dOut);
  cudaFree(dIn);
  return output;
}

} // namespace

/* ----------------------------- Fixtures ----------------------------- */

/** @brief One matrix size per instantiation; skips without a device. */
class TransposeVersions : public ::testing::TestWithParam<int> {
protected:
  void SetUp() override {
    if (!hasCudaDevice()) {
      GTEST_SKIP() << "no CUDA device available";
    }
  }
};

/* ----------------------------- Tests ----------------------------- */

/**
 * @test All three kernels reproduce the CPU transpose, element for element,
 *       at a size that is a single element, ones that are not a multiple of
 *       a tile, the two sides of one tile, and the demo's own.
 */
TEST_P(TransposeVersions, MatchTheCpuTranspose) {
  const int DIM = GetParam();
  const std::size_t N = static_cast<std::size_t>(DIM) * DIM;
  const std::vector<float> INPUT = ramp(N);
  const std::vector<float> EXPECTED = transposeCpu(INPUT, DIM);

  for (const NamedKernel& K : KERNELS) {
    const std::vector<float> OUTPUT = transposeOnDevice(K.kernel, INPUT, DIM);
    ASSERT_EQ(OUTPUT.size(), N) << K.name << " did not run at dim " << DIM;
    const long AT = firstMismatch(OUTPUT, EXPECTED);
    EXPECT_EQ(AT, -1) << K.name << " differs from the CPU transpose at dim " << DIM << ", element "
                      << AT << ": " << OUTPUT[static_cast<std::size_t>(AT)] << " against "
                      << EXPECTED[static_cast<std::size_t>(AT)];
  }
}

/** @brief A single element, sizes off a multiple of a tile, either side of one, the demo's. */
INSTANTIATE_TEST_SUITE_P(Sizes, TransposeVersions,
                         ::testing::Values(1, 7, 31, 32, 33, 100, sm::MATRIX_DIM));

/**
 * @test The static shared memory each kernel declares is the number of bytes
 *       the demo hands the harness for it: none for the naive kernel, one
 *       tile for the conflicting one, one padded tile for the padded one.
 */
TEST(TransposeSharedBytes, MatchTheKernels) {
  if (!hasCudaDevice()) {
    GTEST_SKIP() << "no CUDA device available";
  }
  cudaFuncAttributes naive{};
  cudaFuncAttributes conflict{};
  cudaFuncAttributes padded{};
  ASSERT_EQ(cudaFuncGetAttributes(&naive, sm::transposeNaive), cudaSuccess);
  ASSERT_EQ(cudaFuncGetAttributes(&conflict, sm::transposeSharedConflict), cudaSuccess);
  ASSERT_EQ(cudaFuncGetAttributes(&padded, sm::transposeSharedPadded), cudaSuccess);

  EXPECT_EQ(naive.sharedSizeBytes, 0U) << "transposeNaive declares shared memory";
  EXPECT_EQ(conflict.sharedSizeBytes, sm::TILE_BYTES)
      << "transposeSharedConflict's tile is not the size the demo reports";
  EXPECT_EQ(padded.sharedSizeBytes, sm::TILE_PADDED_BYTES)
      << "transposeSharedPadded's tile is not the size the demo reports";
}
