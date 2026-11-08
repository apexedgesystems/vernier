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
 *  - Unified-memory counters only under --capture-um: where CUPTI records
 *    them, each measurement's GpuStats::unifiedMemory, and its row's um*
 *    cells, hold CUPTI's counters over its window, and CountersFollowTheBuffer
 *    checks them against what it does to a managed buffer. Where CUPTI cannot
 *    record them, the tests still run and their rows are written, but the run
 *    reports the request as failed and exits with status 4, and
 *    CountersFollowTheBuffer is skipped with that reason.
 *
 * Usage:
 *   @code{.sh}
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuUnifiedMemory.*"
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuUnifiedMemory.*" --capture-um
 *   @endcode
 *
 * @see PerfGpuCase
 * @see UnifiedMemoryProfile
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"
#include "src/bench/ptst/gpu/UnifiedMemoryRefusal.hpp"

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

/** @brief Frees a CUDA allocation, managed or device, when its owner goes. */
struct CudaFree {
  void operator()(float* p) const noexcept { cudaFree(p); }
};

/** @brief N floats of CUDA memory, freed however the test ends. */
using CudaFloats = std::unique_ptr<float[], CudaFree>;

/** @brief N floats of managed memory, owned from the start; empty if the allocation fails. */
CudaFloats managedFloats() {
  float* p = nullptr;
  return CudaFloats(cudaMallocManaged(&p, BYTES) == cudaSuccess ? p : nullptr);
}

/** @brief N floats of device memory, owned from the start; empty if the allocation fails. */
CudaFloats deviceFloats() {
  float* p = nullptr;
  return CudaFloats(cudaMalloc(&p, BYTES) == cudaSuccess ? p : nullptr);
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
 *  - Without --capture-um, the result carries no unified-memory counters
 */
PERF_GPU_TEST(GpuUnifiedMemory, KernelOnManagedMemory) {
  PERF_GPU_GUARD(perf);

  const CudaFloats um_in = managedFloats();
  const CudaFloats um_out = managedFloats();
  ASSERT_TRUE(um_in && um_out) << "cudaMallocManaged failed";
  for (int i = 0; i < N; ++i) {
    um_in[i] = static_cast<float>(i);
  }

  const auto LAUNCH = [&](cudaStream_t s) {
    scaleKernel<<<GRID, BLOCK, 0, s>>>(um_in.get(), um_out.get(), N);
  };
  perf.cudaWarmup(LAUNCH);
  const ub::PerfGpuResult RESULT =
      perf.cudaKernel(LAUNCH, "managed_scale").withLaunchConfig(GRID, BLOCK).measure();

  // The host may read managed memory once the device has finished with it.
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(firstMismatch(um_out.get(), scaledIndex), -1) << "first wrong element";
  if (!perf.gpuConfig().captureUnifiedMemory) { // CountersFollowTheBuffer checks them when asked
    EXPECT_FALSE(RESULT.stats.unifiedMemory.has_value())
        << "unified-memory counters in a run that did not ask for them";
  }
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

  const CudaFloats um_buf = managedFloats();
  ASSERT_TRUE(um_buf) << "cudaMallocManaged failed";
  float* data = um_buf.get();
  perf.cudaWarmup([&](cudaStream_t s) { writeIndexKernel<<<GRID, BLOCK, 0, s>>>(data, 0.0f, N); });

  for (int i = 0; i < N; ++i) {
    data[i] = -1.0f;
  }
  const ub::PerfGpuResult AFTER_WRITE =
      perf.cudaKernel(
              [&](cudaStream_t s) { writeIndexKernel<<<GRID, BLOCK, 0, s>>>(data, 2.0f, N); },
              "after_host_write")
          .withLaunchConfig(GRID, BLOCK)
          .measure();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(firstMismatch(data, [](int i) { return 2.0f + static_cast<float>(i); }), -1)
      << "first wrong element after the first pass";

  const ub::PerfGpuResult AFTER_READ =
      perf.cudaKernel(
              [&](cudaStream_t s) { writeIndexKernel<<<GRID, BLOCK, 0, s>>>(data, 3.0f, N); },
              "after_host_read")
          .withLaunchConfig(GRID, BLOCK)
          .measure();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(firstMismatch(data, [](int i) { return 3.0f + static_cast<float>(i); }), -1)
      << "first wrong element after the second pass";

  std::printf("[GpuUnifiedMemory.HostAndKernelTakeTurns] kernel medians: after a host write "
              "%.3f us, after a host read %.3f us\n",
              AFTER_WRITE.kernelTimeUs, AFTER_READ.kernelTimeUs);
}

/**
 * @brief Under --capture-um, the unified-memory counters follow what the test
 *        does to a managed buffer
 *
 * @test CountersFollowTheBuffer
 *
 * Measured on a case of its own with one launch per measurement, so that each
 * measured window holds exactly one step:
 *  - a host write, then a kernel read: the buffer moves to the device
 *  - a kernel write, then the host's read of it: the buffer moves back
 *
 * Validates, under --capture-um:
 *  - The first step migrates at least the buffer's bytes to the device and
 *    nothing back; the second migrates at least the buffer's bytes back and
 *    nothing to the device
 *  - Every element the kernel and the host read is right
 *
 * Skipped without --capture-um, and, with that reason, where the run reported
 * the request as unsupported (CUPTI cannot record the counters there; the run
 * then exits with status 4). Its two measurements are not bracketed by a
 * --profile profiler.
 */
PERF_GPU_TEST(GpuUnifiedMemory, CountersFollowTheBuffer) {
  PERF_GPU_GUARD(perf);
  if (!perf.gpuConfig().captureUnifiedMemory) {
    GTEST_SKIP() << "the unified-memory counters are collected only under --capture-um";
  }
  const CudaFloats um_buf = managedFloats();
  const CudaFloats d_out = deviceFloats();
  ASSERT_TRUE(um_buf && d_out) << "a CUDA allocation failed";
  float* data = um_buf.get();
  float* out = d_out.get();

  // One launch per measurement: each measured window holds one step.
  ub::PerfConfig oneLaunch = perf.cpuConfig();
  oneLaunch.cycles = 1;
  oneLaunch.repeats = 1;
  ub::PerfGpuCase steps{perf.testName(), oneLaunch};

  // Step 1: the host writes the buffer (its pages are then the host's), then
  // a kernel reads it.
  for (int i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i);
  }
  const ub::PerfGpuResult TO_DEVICE =
      steps
          .cudaKernel([&](cudaStream_t s) { scaleKernel<<<GRID, BLOCK, 0, s>>>(data, out, N); },
                      "host_write_kernel_read")
          .withLaunchConfig(GRID, BLOCK)
          .measure();

  // Step 2: a kernel writes the buffer, then the host reads it, in one window.
  int hostReadWrong = -2; // -2 until the host has read it
  const ub::PerfGpuResult TO_HOST =
      steps
          .cudaKernel(
              [&](cudaStream_t s) {
                writeIndexKernel<<<GRID, BLOCK, 0, s>>>(data, 4.0f, N);
                if (cudaStreamSynchronize(s) == cudaSuccess) {
                  hostReadWrong =
                      firstMismatch(data, [](int i) { return 4.0f + static_cast<float>(i); });
                }
              },
              "kernel_write_host_read")
          .withLaunchConfig(GRID, BLOCK)
          .measure();

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> h_out(N);
  ASSERT_EQ(cudaMemcpy(h_out.data(), out, BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(firstMismatch(h_out.data(), scaledIndex), -1) << "first wrong element the kernel read";
  EXPECT_EQ(hostReadWrong, -1) << "first wrong element the host read (-2: it never read)";

  if (!TO_DEVICE.stats.unifiedMemory.has_value() || !TO_HOST.stats.unifiedMemory.has_value()) {
    const std::string WHY = ub::test::unifiedMemoryRefusal();
    ASSERT_FALSE(WHY.empty()) << "no unified-memory counters, and the request was not unsupported";
    GTEST_SKIP() << WHY;
  }
  const ub::UnifiedMemoryProfile& IN = *TO_DEVICE.stats.unifiedMemory;
  const ub::UnifiedMemoryProfile& BACK = *TO_HOST.stats.unifiedMemory;
  std::printf("[GpuUnifiedMemory.CountersFollowTheBuffer] host write, kernel read: %zu "
              "migrations, %zu bytes to the device, %zu back, %zu page faults; kernel write, "
              "host read: %zu migrations, %zu bytes to the host, %zu to the device, %zu page "
              "faults\n",
              IN.h2dMigrations, IN.h2dBytes, IN.d2hMigrations, IN.pageFaults, BACK.d2hMigrations,
              BACK.d2hBytes, BACK.h2dMigrations, BACK.pageFaults);
  EXPECT_GE(IN.h2dMigrations, 1U) << "the kernel read the host's pages without a migration";
  EXPECT_GE(IN.h2dBytes, BYTES) << "less than the buffer reached the device";
  EXPECT_EQ(IN.d2hMigrations, 0U) << "pages went back to the host while the kernel only read";
  EXPECT_GE(BACK.d2hMigrations, 1U) << "the host read the kernel's pages without a migration";
  EXPECT_GE(BACK.d2hBytes, BYTES) << "less than the buffer came back to the host";
  EXPECT_EQ(BACK.h2dMigrations, 0U) << "pages went to the device the kernel already held";
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
  const CudaFloats um_in = managedFloats();
  const CudaFloats um_out = managedFloats();
  const CudaFloats d_in = deviceFloats();
  const CudaFloats d_out = deviceFloats();
  ASSERT_TRUE(um_in && um_out && d_in && d_out) << "a CUDA allocation failed";
  for (int i = 0; i < N; ++i) {
    um_in[i] = h_in[i];
  }
  ASSERT_EQ(cudaMemcpy(d_in.get(), h_in.data(), BYTES, cudaMemcpyHostToDevice), cudaSuccess);

  const auto MANAGED = [&](cudaStream_t s) {
    scaleKernel<<<GRID, BLOCK, 0, s>>>(um_in.get(), um_out.get(), N);
  };
  const auto DEVICE = [&](cudaStream_t s) {
    scaleKernel<<<GRID, BLOCK, 0, s>>>(d_in.get(), d_out.get(), N);
  };
  perf.cudaWarmup(MANAGED);
  perf.cudaWarmup(DEVICE);
  const ub::PerfGpuResult ON_MANAGED =
      perf.cudaKernel(MANAGED, "managed_memory").withLaunchConfig(GRID, BLOCK).measure();
  const ub::PerfGpuResult ON_DEVICE =
      perf.cudaKernel(DEVICE, "device_memory").withLaunchConfig(GRID, BLOCK).measure();

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> h_out(N);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out.get(), BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(firstMismatch(um_out.get(), scaledIndex), -1)
      << "first wrong element on managed memory";
  EXPECT_EQ(firstMismatch(h_out.data(), scaledIndex), -1) << "first wrong element on device memory";

  std::printf("[GpuUnifiedMemory.ManagedAgainstDeviceMemory] kernel medians: managed %.3f us, "
              "device %.3f us, managed/device %.2f\n",
              ON_MANAGED.kernelTimeUs, ON_DEVICE.kernelTimeUs,
              ON_MANAGED.kernelTimeUs / ON_DEVICE.kernelTimeUs);
}
