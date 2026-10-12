/**
 * @file 03_SharedMemoryOpt_Demo.cu
 * @brief Demo 12: a matrix transpose three ways, and the shared-memory bank
 *        conflicts Nsight Compute counts in one of them.
 *
 * Three kernels transpose the same 1024 x 1024 matrix of floats
 * (03_SharedMemoryOpt_Transpose.cu):
 *  1. transposeNaive, through global memory alone: every warp's writes
 *     stride through the output a column at a time;
 *  2. transposeSharedConflict, through a 32 x 32 tile in shared memory: the
 *     global reads and writes are both coalesced, but every warp's reads of
 *     one tile column land in one shared-memory bank;
 *  3. transposeSharedPadded, the same tile with its rows padded by one
 *     float, so those reads land in 32 banks.
 *
 * Each test measures one kernel, writes one CSV row, and checks the
 * transpose the kernel produced. The harness is told the static shared
 * memory each tiled kernel declares, which its occupancy estimate accounts
 * for; on the reference rig the 1,024-thread block, not the tile, limits
 * occupancy, so the three rows read the same figure.
 *
 * The counter the walkthrough reads, bank conflicts per launch from Nsight
 * Compute, is checked beside the demo by TestDemoBankConflicts
 * (utst/03_SharedMemoryOpt_BankConflicts_uTest.cpp), so this source holds
 * only what it teaches.
 *
 * Usage:
 *   @code{.sh}
 *   # Measure all three, write the CSV the walkthrough reads
 *   ./BenchDemo_Gpu_03_SharedMemoryOpt --repeats 10 --csv shared_memory_opt.csv
 *
 *   # Count the bank conflicts of every launch (root on Jetson; ncu replays
 *   # each launch, so keep them few)
 *   sudo env PATH="$PATH" ncu --target-processes all \
 *     --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
 *     ./BenchDemo_Gpu_03_SharedMemoryOpt --cycles 3 --repeats 1
 *   @endcode
 *
 * @see docs/12_SHARED_MEMORY_OPT.md for the walkthrough this demo backs
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "src/bench/demo/gpu/03_SharedMemoryOpt_Transpose.cuh"
#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace sm = vernier::bench::demo::shared_memory_demo;

namespace {

/* ----------------------------- Constants ----------------------------- */

constexpr std::size_t N = static_cast<std::size_t>(sm::MATRIX_DIM) * sm::MATRIX_DIM; ///< Elements
constexpr std::size_t BYTES = N * sizeof(float); ///< One matrix, in bytes

/* ----------------------------- Helpers ----------------------------- */

/**
 * @brief The input: element i holds i, exact in a float up to 2^24, so a
 *        transpose and a copy differ everywhere off the diagonal.
 */
std::vector<float> rampMatrix() {
  std::vector<float> m(N);
  for (std::size_t i = 0; i < N; ++i) {
    m[i] = static_cast<float>(i);
  }
  return m;
}

/** @brief How many elements of @p output are not the transpose of @p input. */
std::size_t transposeMismatches(const std::vector<float>& input, const std::vector<float>& output,
                                int dim) {
  std::size_t mismatches = 0;
  for (int y = 0; y < dim; ++y) {
    for (int x = 0; x < dim; ++x) {
      const std::size_t IN = static_cast<std::size_t>(y) * dim + x;
      const std::size_t OUT = static_cast<std::size_t>(x) * dim + y;
      if (output[OUT] != input[IN]) {
        ++mismatches;
      }
    }
  }
  return mismatches;
}

/** @brief The input and output matrices on the device, freed however the test leaves. */
class DeviceMatrices {
public:
  DeviceMatrices() {
    if (cudaMalloc(&input_, BYTES) != cudaSuccess) {
      input_ = nullptr;
    }
    if (cudaMalloc(&output_, BYTES) != cudaSuccess) {
      output_ = nullptr;
    }
  }

  ~DeviceMatrices() {
    cudaFree(output_);
    cudaFree(input_);
  }

  DeviceMatrices(const DeviceMatrices&) = delete;
  DeviceMatrices& operator=(const DeviceMatrices&) = delete;

  [[nodiscard]] bool ok() const { return input_ != nullptr && output_ != nullptr; }
  [[nodiscard]] const float* input() const { return input_; }
  [[nodiscard]] float* output() const { return output_; }

  /** @brief Copy @p host into the input matrix. */
  [[nodiscard]] bool upload(const std::vector<float>& host) const {
    return cudaMemcpy(input_, host.data(), BYTES, cudaMemcpyHostToDevice) == cudaSuccess;
  }

  /** @brief Copy the output matrix into @p host. */
  [[nodiscard]] bool download(std::vector<float>& host) const {
    return cudaMemcpy(host.data(), output_, BYTES, cudaMemcpyDeviceToHost) == cudaSuccess;
  }

private:
  float* input_ = nullptr;
  float* output_ = nullptr;
};

} // namespace

/**
 * @test The transpose through global memory alone. Each warp reads 32
 *       consecutive floats of a row and writes them a whole row apart, down
 *       a column of the output.
 */
PERF_GPU_TEST(SharedMemoryOpt, NaiveGlobalMemory) {
  PERF_GPU_GUARD(perf);

  DeviceMatrices device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  const std::vector<float> INPUT = rampMatrix();
  ASSERT_TRUE(device.upload(INPUT)) << "device upload failed";

  const dim3 GRID = sm::transposeGrid(sm::MATRIX_DIM);
  const dim3 BLOCK = sm::transposeBlock();
  const auto LAUNCH = [&](cudaStream_t s) {
    sm::transposeNaive<<<GRID, BLOCK, 0, s>>>(device.input(), device.output(), sm::MATRIX_DIM);
  };
  perf.cudaWarmup(LAUNCH);
  perf.cudaKernel(LAUNCH, "transpose_naive").withLaunchConfig(GRID, BLOCK).measure();

  // Every launch wrote the transpose.
  std::vector<float> output(N);
  ASSERT_TRUE(device.download(output)) << "device download failed";
  EXPECT_EQ(transposeMismatches(INPUT, output, sm::MATRIX_DIM), 0U)
      << "the kernel did not transpose the input";
}

/**
 * @test The transpose through a shared-memory tile. The global reads and
 *       writes are both coalesced; every warp's reads of one tile column
 *       land in one bank, so each is served 32 times over.
 */
PERF_GPU_TEST(SharedMemoryOpt, SharedWithBankConflicts) {
  PERF_GPU_GUARD(perf);

  DeviceMatrices device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  const std::vector<float> INPUT = rampMatrix();
  ASSERT_TRUE(device.upload(INPUT)) << "device upload failed";

  const dim3 GRID = sm::transposeGrid(sm::MATRIX_DIM);
  const dim3 BLOCK = sm::transposeBlock();
  const auto LAUNCH = [&](cudaStream_t s) {
    sm::transposeSharedConflict<<<GRID, BLOCK, 0, s>>>(device.input(), device.output(),
                                                       sm::MATRIX_DIM);
  };
  perf.cudaWarmup(LAUNCH);
  // The tile is static shared memory, declared in the kernel; the harness is
  // told its size so its occupancy estimate can account for it.
  perf.cudaKernel(LAUNCH, "transpose_shared_conflict")
      .withLaunchConfig(GRID, BLOCK, sm::TILE_BYTES)
      .measure();

  std::vector<float> output(N);
  ASSERT_TRUE(device.download(output)) << "device download failed";
  EXPECT_EQ(transposeMismatches(INPUT, output, sm::MATRIX_DIM), 0U)
      << "the kernel did not transpose the input";
}

/**
 * @test The same transpose through a tile padded by one float per row, so
 *       every warp's reads of one tile column land in 32 banks and each is
 *       served once.
 */
PERF_GPU_TEST(SharedMemoryOpt, SharedPadded) {
  PERF_GPU_GUARD(perf);

  DeviceMatrices device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  const std::vector<float> INPUT = rampMatrix();
  ASSERT_TRUE(device.upload(INPUT)) << "device upload failed";

  const dim3 GRID = sm::transposeGrid(sm::MATRIX_DIM);
  const dim3 BLOCK = sm::transposeBlock();
  const auto LAUNCH = [&](cudaStream_t s) {
    sm::transposeSharedPadded<<<GRID, BLOCK, 0, s>>>(device.input(), device.output(),
                                                     sm::MATRIX_DIM);
  };
  perf.cudaWarmup(LAUNCH);
  perf.cudaKernel(LAUNCH, "transpose_shared_padded")
      .withLaunchConfig(GRID, BLOCK, sm::TILE_PADDED_BYTES)
      .measure();

  std::vector<float> output(N);
  ASSERT_TRUE(device.download(output)) << "device download failed";
  EXPECT_EQ(transposeMismatches(INPUT, output, sm::MATRIX_DIM), 0U)
      << "the kernel did not transpose the input";
}

/* ----------------------------- Main ----------------------------- */

PERF_GPU_MAIN()
