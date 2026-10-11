/**
 * @file GpuUnifiedMemory_pTest.cu
 * @brief Kernels on managed (Unified) memory, timed and their results checked
 *
 * The data of these kernels lives in cudaMallocManaged() memory: the host
 * writes and reads it through the same pointer the kernel uses, and no
 * transfer is declared to the harness. Each test checks every element a kernel
 * wrote, from the host.
 *
 * What this measures, and what it does not:
 *  - Kernel time, as for any kernel: the median over the repeats of each
 *    repeat's time per launch. A wait for pages to reach the GPU inside the
 *    timed interval counts in that repeat's sample; averaging over the
 *    launches and taking the median can dilute or hide a one-time wait, but
 *    neither isolates it nor guarantees that it is left out.
 *  - No unified-memory counters. The harness reads no page-fault or migration
 *    record: GpuStats::unifiedMemory stays empty and the CSV's um* cells with
 *    it, with or without --capture-um, which only adds a line saying so.
 *
 * Usage:
 *   @code{.sh}
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuUnifiedMemory.*"
 *   @endcode
 *
 * @see PerfGpuCase
 * @see UnifiedMemoryProfile
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

constexpr int N = 256 * 1024;
constexpr size_t BYTES = N * sizeof(float);
const dim3 BLOCK(256);
const dim3 GRID((N + 255) / 256);

/** @brief out[i] = 2 * in[i] + 1: reads one buffer, writes another. */
__global__ void scaleKernel(const float* in, float* out, int n) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    out[IDX] = 2.0f * in[IDX] + 1.0f;
  }
}

/** @brief data[i] = value + i, whatever data held before. */
__global__ void writeIndexKernel(float* data, float value, int n) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    data[IDX] = value + static_cast<float>(IDX);
  }
}

/** @brief The value scaleKernel writes for in[i] = i. */
float scaledIndex(int i) { return 2.0f * static_cast<float>(i) + 1.0f; }

/**
 * @brief The first index at which @p data differs from @p expected(i), or -1.
 *        Every value compared is an integer below 2^24, exact in a float.
 */
template <typename Expected> int firstMismatch(const float* data, Expected expected) {
  for (int i = 0; i < N; ++i) {
    if (data[i] != expected(i)) {
      return i;
    }
  }
  return -1;
}

} // anonymous namespace

/**
 * @brief A kernel reads managed memory the host wrote and writes managed
 *        memory the host reads
 *
 * @test KernelOnManagedMemory
 *
 * Validates:
 *  - The kernel is timed with no transfer declared
 *  - Every element it wrote is right when the host reads it
 *  - The harness reports no unified-memory counters
 */
PERF_GPU_TEST(GpuUnifiedMemory, KernelOnManagedMemory) {
  PERF_GPU_GUARD(perf);

  float* um_in = nullptr;
  float* um_out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&um_in, BYTES), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&um_out, BYTES), cudaSuccess);
  for (int i = 0; i < N; ++i) {
    um_in[i] = static_cast<float>(i);
  }

  const auto LAUNCH = [&](cudaStream_t s) { scaleKernel<<<GRID, BLOCK, 0, s>>>(um_in, um_out, N); };
  perf.cudaWarmup(LAUNCH);
  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(LAUNCH, "managed_scale").withLaunchConfig(GRID, BLOCK).measure();

  // The host may read managed memory once the device has finished with it.
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(firstMismatch(um_out, scaledIndex), -1) << "first wrong element";
  EXPECT_FALSE(RESULT.stats.unifiedMemory.has_value())
      << "the harness reported unified-memory counters it does not collect";

  cudaFree(um_in);
  cudaFree(um_out);
}

/**
 * @brief The host and the kernel take turns on one managed buffer
 *
 * @test HostAndKernelTakeTurns
 *
 * Validates:
 *  - A kernel pass after the host wrote the buffer, and one after the host
 *    read it, are each timed
 *  - The host reads what each pass wrote
 *
 * A wait for the buffer's pages after a host access falls inside the next
 * pass's timed launches; the pass median can dilute or hide it but does not
 * isolate it. The pass medians are printed, not compared: what a host access
 * costs the next launch can differ from one GPU to another, and where the GPU
 * shares the host's memory.
 */
PERF_GPU_TEST(GpuUnifiedMemory, HostAndKernelTakeTurns) {
  PERF_GPU_GUARD(perf);

  float* um_buf = nullptr;
  ASSERT_EQ(cudaMallocManaged(&um_buf, BYTES), cudaSuccess);
  perf.cudaWarmup(
      [&](cudaStream_t s) { writeIndexKernel<<<GRID, BLOCK, 0, s>>>(um_buf, 0.0f, N); });

  for (int i = 0; i < N; ++i) {
    um_buf[i] = -1.0f;
  }
  const ub::PerfGpuResult AFTER_WRITE =
      perf.cudaKernel(
              [&](cudaStream_t s) { writeIndexKernel<<<GRID, BLOCK, 0, s>>>(um_buf, 2.0f, N); },
              "after_host_write")
          .withLaunchConfig(GRID, BLOCK)
          .measure();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(firstMismatch(um_buf, [](int i) { return 2.0f + static_cast<float>(i); }), -1)
      << "first wrong element after the first pass";

  const ub::PerfGpuResult AFTER_READ =
      perf.cudaKernel(
              [&](cudaStream_t s) { writeIndexKernel<<<GRID, BLOCK, 0, s>>>(um_buf, 3.0f, N); },
              "after_host_read")
          .withLaunchConfig(GRID, BLOCK)
          .measure();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(firstMismatch(um_buf, [](int i) { return 3.0f + static_cast<float>(i); }), -1)
      << "first wrong element after the second pass";

  std::printf("[GpuUnifiedMemory.HostAndKernelTakeTurns] kernel medians: after a host write "
              "%.3f us, after a host read %.3f us\n",
              AFTER_WRITE.kernelTimeUs, AFTER_READ.kernelTimeUs);

  cudaFree(um_buf);
}

/**
 * @brief The same kernel on managed memory and on device memory
 *
 * @test ManagedAgainstDeviceMemory
 *
 * Validates:
 *  - Both versions are timed with the same launch shape
 *  - Both write the right result
 *
 * The ratio of the two medians is printed, not required: it is this
 * machine's, and can differ on another GPU or driver, and where the GPU
 * shares the host's memory.
 */
PERF_GPU_TEST(GpuUnifiedMemory, ManagedAgainstDeviceMemory) {
  PERF_GPU_GUARD(perf);

  std::vector<float> h_in(N);
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }
  float* um_in = nullptr;
  float* um_data = nullptr;
  float* d_in = nullptr;
  float* d_out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&um_in, BYTES), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&um_data, BYTES), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_in, BYTES), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, BYTES), cudaSuccess);
  for (int i = 0; i < N; ++i) {
    um_in[i] = h_in[i];
  }
  ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice), cudaSuccess);

  const auto MANAGED = [&](cudaStream_t s) {
    scaleKernel<<<GRID, BLOCK, 0, s>>>(um_in, um_data, N);
  };
  const auto DEVICE = [&](cudaStream_t s) { scaleKernel<<<GRID, BLOCK, 0, s>>>(d_in, d_out, N); };
  perf.cudaWarmup(MANAGED);
  perf.cudaWarmup(DEVICE);
  const ub::PerfGpuResult ON_MANAGED =
      perf.cudaKernel(MANAGED, "managed_memory").withLaunchConfig(GRID, BLOCK).measure();
  const ub::PerfGpuResult ON_DEVICE =
      perf.cudaKernel(DEVICE, "device_memory").withLaunchConfig(GRID, BLOCK).measure();

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> h_out(N);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(firstMismatch(um_data, scaledIndex), -1) << "first wrong element on managed memory";
  EXPECT_EQ(firstMismatch(h_out.data(), scaledIndex), -1) << "first wrong element on device memory";
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(um_in);

  std::printf("[GpuUnifiedMemory.ManagedAgainstDeviceMemory] kernel medians: managed %.3f us, "
              "device %.3f us, managed/device %.2f\n",
              ON_MANAGED.kernelTimeUs, ON_DEVICE.kernelTimeUs,
              ON_MANAGED.kernelTimeUs / ON_DEVICE.kernelTimeUs);

  cudaFree(um_data);
}

// Note: PERF_MAIN() is defined in MatMul_pTest.cu for this test binary
