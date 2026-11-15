/**
 * @file GpuCollectorCost_pTest.cu
 * @brief What the in-process CUPTI collector leaves on timed launches:
 *        measurements to compare across runs, collector on against stood down
 *
 * The harness times no launch while CUPTI records kernels: its cupti* cells
 * come from an untimed batch after the timed repeats, which TestBenchGpu
 * checks by the activity kinds the collector switches, without a clock. What
 * the collector can still leave on a timed launch is host time, which reaches
 * only kernels shorter than the host's interval between two launches, and
 * what a launch costs the host depends on the core the launching thread runs
 * on. These cases measure; they judge nothing:
 *
 *  - ShortKernel: a 16 x 256 store, a kernel shorter than that interval, the
 *    first measurement of its process. The collector on (the default) against
 *    VERNIER_DISABLE_CUPTI=1: what the collector's registration with CUPTI
 *    leaves on a launch.
 *  - ShortKernelAfterAWindow: the same store measured twice; the figure is
 *    the second measurement, after the first one's CUPTI batch. On against
 *    off: what a stopped CUPTI window leaves on every later launch of the
 *    process.
 *  - GpuBoundKernelAfterAWindow: a SAXPY over 1 Mi floats, longer than the
 *    interval, measured twice in the same way: whether that host time reaches
 *    a kernel the GPU paces.
 *  - MultiGpuShortKernel: the store through a one-device multi-GPU
 *    measurement. With --capture-um against without, the collector on: what
 *    the unified-memory counters' window leaves on a launch.
 *
 * Each case prints its figure, the CPU the launching thread started and ended
 * on and, for a kernel measurement, the CUPTI batch's launches (0: the
 * collector stood down). Compare runs made the same way: each a process of
 * its own, pinned to one CPU (a figure taken on a performance core against
 * one taken on an efficiency core compares the cores), the two settings
 * alternating, over several rounds, as below; the GPU guide gives the
 * laptop's figures per core type.
 *
 * Usage:
 *   @code{.sh}
 *   P=./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST
 *   for r in 1 2 3 4 5; do
 *     for off in 0 1; do
 *       VERNIER_DISABLE_CUPTI=$off taskset -c 2 "$P" \
 *           --gtest_filter=GpuCollectorCost.ShortKernel --cycles 200 --repeats 20
 *     done
 *   done
 *   # The counters' window: the same, with and without --capture-um
 *   taskset -c 2 "$P" --gtest_filter=GpuCollectorCost.MultiGpuShortKernel \
 *       --cycles 200 --repeats 20 --capture-um
 *   @endcode
 *
 * @see PerfGpuCase
 * @see CuptiCollector
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <sched.h>

#include <cstddef>
#include <cstdio>
#include <string>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"

namespace ub = vernier::bench;

namespace {

/// The short kernel's blocks and threads: 16 x 256 stores.
constexpr int STORE_BLOCKS = 16;
constexpr int BLOCK = 256;

/// The GPU-bound kernel's elements: a SAXPY over 1 Mi floats.
constexpr int LONG_ELEMENTS = 1 << 20;

/** @brief Writes 1 to each of its threads' elements of @p out. */
__global__ void storeKernel(float* out) { out[blockIdx.x * blockDim.x + threadIdx.x] = 1.0F; }

/** @brief y = a * x + y over @p n floats. */
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    y[IDX] = a * x[IDX] + y[IDX];
  }
}

/** @brief Device floats, freed however the case leaves. */
struct DeviceFloats {
  float* data = nullptr;
  explicit DeviceFloats(int count) {
    if (cudaMalloc(&data, static_cast<std::size_t>(count) * sizeof(float)) != cudaSuccess) {
      data = nullptr;
    }
  }
  ~DeviceFloats() { cudaFree(data); }
  DeviceFloats(const DeviceFloats&) = delete;
  DeviceFloats& operator=(const DeviceFloats&) = delete;
};

/**
 * @brief The CPU the launching thread was on at its first call and at its
 *        latest 64th: noted on those calls only, so that a timed launch pays
 *        an increment, not a CPU query. Read after measure() returns, which
 *        has joined any thread that launched.
 */
struct LaunchCpus {
  long calls = 0;
  int first = -1;
  int latest = -1;

  /** @brief Starts over, for the measurement whose CPUs are reported. */
  void reset() { *this = LaunchCpus{}; }

  void note() {
    ++calls;
    if (calls == 1 || calls % 64 == 0) {
      latest = sched_getcpu();
      if (calls == 1) {
        first = latest;
      }
    }
  }
};

/**
 * @brief Prints one figure: the per-launch time, the launching thread's CPUs
 *        and, for a kernel measurement, its CUPTI batch's launches (0 with the
 *        collector stood down; a multi-GPU measurement has no batch).
 */
void report(const char* name, const char* what, double us, const LaunchCpus& cpus,
            const std::string& batch) {
  std::printf("[GpuCollectorCost.%s] %s: %.3f us a launch, launching thread on CPU %d to %d%s\n",
              name, what, us, cpus.first, cpus.latest, batch.c_str());
}

/** @brief ", CUPTI batch of N launches" for a kernel measurement's @p result. */
std::string batchOf(const ub::PerfGpuResult& result) {
  return ", CUPTI batch of " + std::to_string(result.stats.cupti.kernelLaunches) + " launches";
}

} // namespace

/**
 * @brief A kernel shorter than the launch interval, the first measurement of
 *        its process
 * @test ShortKernel
 */
PERF_GPU_TEST(GpuCollectorCost, ShortKernel) {
  UB_PERF_GPU_GUARD(perf);
  const DeviceFloats OUT(STORE_BLOCKS * BLOCK);
  ASSERT_NE(OUT.data, nullptr);
  float* out = OUT.data;
  LaunchCpus cpus;
  const auto LAUNCH = [out, &cpus](cudaStream_t s) {
    cpus.note();
    storeKernel<<<STORE_BLOCKS, BLOCK, 0, s>>>(out);
  };
  perf.cudaWarmup(LAUNCH);
  cpus.reset();
  const ub::PerfGpuResult RESULT = perf.cudaKernel(LAUNCH, "store").measure();
  report("ShortKernel", "16 x 256 store", RESULT.kernelTimeUs, cpus, batchOf(RESULT));
}

/**
 * @brief The same kernel measured again, after its first measurement's
 *        CUPTI batch
 * @test ShortKernelAfterAWindow
 */
PERF_GPU_TEST(GpuCollectorCost, ShortKernelAfterAWindow) {
  UB_PERF_GPU_GUARD(perf);
  const DeviceFloats OUT(STORE_BLOCKS * BLOCK);
  ASSERT_NE(OUT.data, nullptr);
  float* out = OUT.data;
  LaunchCpus cpus;
  const auto LAUNCH = [out, &cpus](cudaStream_t s) {
    cpus.note();
    storeKernel<<<STORE_BLOCKS, BLOCK, 0, s>>>(out);
  };
  perf.cudaWarmup(LAUNCH);
  static_cast<void>(perf.cudaKernel(LAUNCH, "store, first").measure());
  cpus.reset();
  const ub::PerfGpuResult RESULT = perf.cudaKernel(LAUNCH, "store, after a window").measure();
  report("ShortKernelAfterAWindow", "16 x 256 store after a window", RESULT.kernelTimeUs, cpus,
         batchOf(RESULT));
}

/**
 * @brief A kernel longer than the launch interval, measured after a window
 * @test GpuBoundKernelAfterAWindow
 */
PERF_GPU_TEST(GpuCollectorCost, GpuBoundKernelAfterAWindow) {
  UB_PERF_GPU_GUARD(perf);
  const DeviceFloats X(LONG_ELEMENTS);
  const DeviceFloats Y(LONG_ELEMENTS);
  ASSERT_NE(X.data, nullptr);
  ASSERT_NE(Y.data, nullptr);
  const float* x = X.data;
  float* y = Y.data;
  LaunchCpus cpus;
  const auto LAUNCH = [x, y, &cpus](cudaStream_t s) {
    cpus.note();
    saxpyKernel<<<(LONG_ELEMENTS + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(2.0F, x, y, LONG_ELEMENTS);
  };
  perf.cudaWarmup(LAUNCH);
  static_cast<void>(perf.cudaKernel(LAUNCH, "saxpy, first").measure());
  cpus.reset();
  const ub::PerfGpuResult RESULT = perf.cudaKernel(LAUNCH, "saxpy, after a window").measure();
  report("GpuBoundKernelAfterAWindow", "SAXPY over 1 Mi floats after a window", RESULT.kernelTimeUs,
         cpus, batchOf(RESULT));
}

/**
 * @brief The short kernel through a one-device multi-GPU measurement
 * @test MultiGpuShortKernel
 */
PERF_GPU_TEST(GpuCollectorCost, MultiGpuShortKernel) {
  UB_PERF_GPU_GUARD(perf);
  const DeviceFloats OUT(STORE_BLOCKS * BLOCK);
  ASSERT_NE(OUT.data, nullptr);
  float* out = OUT.data;
  LaunchCpus cpus;
  const ub::MultiGpuResult RESULT = perf.cudaKernelMultiGpu(
                                            1,
                                            [out, &cpus](int, cudaStream_t s) {
                                              cpus.note();
                                              storeKernel<<<STORE_BLOCKS, BLOCK, 0, s>>>(out);
                                            },
                                            "store")
                                        .measure();
  ASSERT_EQ(RESULT.perDevice.size(), 1U);
  report("MultiGpuShortKernel",
         perf.gpuConfig().captureUnifiedMemory ? "store, counters asked for" : "store",
         RESULT.perDevice.front().kernelTimeUs, cpus, "");
}
