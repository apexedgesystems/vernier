# GPU Benchmarking Guide

**Namespace:** `vernier::bench`
**Platform:** Linux + NVIDIA CUDA (and AMD ROCm where noted)
**C++ Standard:** C++20 (C++23 used when available)

Complete guide to GPU/CUDA performance benchmarking. This guide assumes you're familiar with CUDA programming basics and covers benchmarking CUDA kernels, measuring GPU performance, and optimizing GPU workloads.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Understanding GPU Metrics](#understanding-gpu-metrics)
- [Writing GPU Benchmarks](#writing-gpu-benchmarks)
- [Launch Configuration](#launch-configuration)
- [CPU vs GPU Comparison](#cpu-vs-gpu-comparison)
- [Multi-GPU Benchmarking](#multi-gpu-benchmarking)
- [Memory Strategies](#memory-strategies)
- [Profiling GPU Code](#profiling-gpu-code)
- [Result Validation](#result-validation)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Your First GPU Benchmark

```cpp
#include <gtest/gtest.h>
#include "Perf.hpp"
#include "PerfGpu.hpp"

namespace ub = vernier::bench;

// Simple vector addition kernel
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

PERF_GPU_TEST(VectorAdd, Basic) {
  // Create GPU benchmark harness
  PERF_GPU_GUARD(perf);

  // Setup
  const int N = 1024 * 1024;  // 1M elements
  const size_t SIZE = N * sizeof(float);

  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  const int THREADS = 256;
  const int BLOCKS = (N + THREADS - 1) / THREADS;

  dim3 block(THREADS);
  dim3 grid(BLOCKS);

  // Inputs first: the kernel reads both. They are queued on the case's
  // stream, the one the warmup launches on, so the warmup starts only after
  // both copies finish (see Launch on the Stream You Are Given). The copies
  // declared below repeat this in every measured repeat.
  cudaMemcpyAsync(d_a, h_a.data(), SIZE, cudaMemcpyHostToDevice, perf.stream());
  cudaMemcpyAsync(d_b, h_b.data(), SIZE, cudaMemcpyHostToDevice, perf.stream());

  // Warmup: Critical for GPU (JIT compilation + cache priming)
  perf.cudaWarmup([&](cudaStream_t s) {
    vectorAdd<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
  });

  // Measure kernel + transfers (with descriptive label)
  auto result = perf.cudaKernel([&](cudaStream_t s) {
    vectorAdd<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
  }, "vector_add")  // Label helps identify in CSV
    .withHostToDevice(h_a.data(), d_a, SIZE)
    .withHostToDevice(h_b.data(), d_b, SIZE)
    .withDeviceToHost(d_c, h_c.data(), SIZE)
    .withLaunchConfig(grid, block)  // Enables the occupancy estimate
    .measure();

  // The harness prints its own result line; these pick fields from `result`
  std::printf("Kernel time: %.3f ms\n", result.kernelTimeUs / 1000.0);
  std::printf("Transfer time: %.3f ms\n", result.transferTimeUs / 1000.0);
  std::printf("Occupancy (launch-shape estimate): %.1f%%\n",
              result.stats.occupancy.achievedOccupancy * 100);

  // Validate: the copy back holds a + b, and the repeats agreed. The 12 MiB
  // of copies dominate this round trip, so a calls-per-second requirement
  // is one you set for your own machine.
  EXPECT_FLOAT_EQ(h_c[0], 3.0f);
  EXPECT_FLOAT_EQ(h_c[N - 1], 3.0f);
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

PERF_GPU_MAIN()
```

`PERF_GPU_MAIN()` is the entry point of a GPU benchmark: it parses the GPU
flags (`--gpu-device`, `--gpu-memory`, `--gpu-warmup`, `--min-speedup`,
`--capture-um`) as well as the common ones, and marks the run as a GPU run so
that the CSV carries the GPU columns whatever the tests are named.
`PERF_MAIN()` does neither: the GPU flags are passed through unread, and the
CSV gets its GPU columns only if a test name happens to contain `Gpu`, `GPU`
or `CUDA`.

### Build and Run

```bash
# Build (CUDA automatically detected), in your project as for a CPU benchmark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run
./build/VectorAdd_GPU_PTEST --csv results.csv

# Quick mode during development
./build/VectorAdd_GPU_PTEST --quick

# With an Nsight Systems timeline: nsys records only a process it starts,
# so bench run starts the benchmark under it
bench run ./build/VectorAdd_GPU_PTEST --profile nsight -- --gtest_filter="*Basic"

# With Nsight Compute kernel analysis, under ncu (--profile-args replay for the
# replay metrics, or a set, sections and metrics: --profile-args '--set full')
bench run ./build/VectorAdd_GPU_PTEST --profile ncu --cycles 3 --repeats 1 \
  -- --gtest_filter="*Basic"

# Specific GPU device
./build/VectorAdd_GPU_PTEST --gpu-device 1 --csv results.csv
```

---

## Understanding GPU Metrics

### What Gets Measured?

Each repeat runs the declared host-to-device copies, `--cycles` launches of the
kernel and the declared device-to-host copies, each timed with CUDA events on
the case's stream. A result holds medians over the repeats, each taken on its
own:

**Timing:**

- **kernelTimeUs** - One launch: the median, over the repeats, of the
  launches' event time divided by their count
- **transferTimeUs** - The median host-to-device time plus the median
  device-to-host time
- **totalTimeUs** - One round trip: the median, over the repeats, of the
  host-to-device time plus one launch plus the device-to-host time; host work
  between them is not in it

Because each is a median of its own, `totalTimeUs` is not `kernelTimeUs` plus
`transferTimeUs`, and when the copy times vary between repeats the copies'
medians can add up to more than the round trip
([Transfer Overhead](#transfer-overhead)).

**Performance:**

- **callsPerSecond** - Round trips per second (1e6 / totalTimeUs)
- **speedupVsCpu** - The suite's CPU baseline time over totalTimeUs; empty in
  the CSV when the suite has no baseline
- **memBandwidthGBs** - The copy rate in GB/s: the declared copies' bytes over
  `transferTimeUs`, the sum of the two copy medians (see
  [Memory Bandwidth](#memory-bandwidth)); empty for a test that declares no
  copies; the kernel's own memory traffic is not measured

**Resource Utilization:**

- **achievedOccupancy** (the CSV's `occupancy`) - An estimate of the occupancy
  the launch shape allows, not a measurement; empty without
  `.withLaunchConfig()` (see [Occupancy](#occupancy))
- **smClockMHz** - The SM clock NVML reports at the end of the measured window
- **throttling** - Whether the SM clock fell by more than 10% between NVML's
  readings at the window's start and end; empty unless NVML reported both and
  the maximum SM clock

A value the run could not read stays an empty CSV cell, and the run says why on
stderr: once per process for what holds for the whole run (no CUPTI in the
build, NVML not reporting a reading) and for a window too short for
`powerDrawW` (under 1.5 s), and for the test when its measured window or its
CUPTI batch lost it.

### PerfGpuResult Structure

```cpp
struct PerfGpuResult {
  GpuStats stats;             // Comprehensive GPU metrics (see below)
  double kernelTimeUs;        // CUDA event kernel time
  double transferTimeUs;      // H2D + D2H transfer time
  double totalTimeUs;         // One round trip: copies in, one launch, copies out
  double callsPerSecond;      // 1e6 / totalTimeUs
  double speedupVsCpu;        // Set by cpuBaseline() comparison
  std::string label;          // Kernel label from cudaKernel()
  int deviceId;               // CUDA device ID
};
```

**Accessing results:**

```cpp
auto result = perf.cudaKernel([&](cudaStream_t s) {
  myKernel<<<grid, block, 0, s>>>(d_data, N);
}, "my_kernel")
  .withLaunchConfig(grid, block)
  .measure();

// Timing
std::printf("Kernel: %.2f ms\n", result.kernelTimeUs / 1000.0);
std::printf("Transfers: %.2f ms\n", result.transferTimeUs / 1000.0);
std::printf("Total: %.2f ms\n", result.totalTimeUs / 1000.0);

// Throughput
std::printf("Throughput: %.0f ops/sec\n", result.callsPerSecond);

// GPU metrics (always present in GpuStats; 0 where the run could not read them)
std::printf("Occupancy (estimate): %.1f%%\n", result.stats.occupancy.achievedOccupancy * 100);
std::printf("SM Clock: %d MHz\n", result.stats.clocks.smClockMHzEnd);
// isThrottling() reads 0 for a reading NVML did not report: ask it only
// when the start, end and maximum SM clocks are all there
const auto& clocks = result.stats.clocks;
if (clocks.smClockMHzStart > 0 && clocks.smClockMHzEnd > 0 && clocks.boostClockMHz > 0 &&
    clocks.isThrottling()) {
  std::printf("Warning: the SM clock fell more than 10%% during the measurement\n");
}
std::printf("Copy rate: %.2f GB/s\n", result.stats.transfers.bandwidthGBs());
```

### GpuStats Structure

```cpp
struct GpuStats {
  Stats cpuStats{};                                 // Per-repeat round-trip times
  GpuDeviceInfo deviceInfo{};                       // GPU model, compute capability
  MemoryTransferProfile transfers{};                // H2D/D2H bytes and timing
  OccupancyMetrics occupancy{};                     // Launch shape and its occupancy estimate
  ClockSpeedProfile clocks{};                       // NVML SM/memory clocks at start and end
  PowerThermalProfile powerThermal{};               // NVML power and temperature at start and end
  CuptiKernelStats cupti{};                         // CUPTI launch, register and shared-memory counts
  double kernelTimeMedianUs{};                      // Kernel execution time
  double transferTimeMedianUs{};                    // Transfer time
  double totalTimeMedianUs{};                       // Total time
  std::optional<MultiGpuMetrics> multiGpu{};        // Only in cudaKernelMultiGpu()
  std::optional<P2PTransferProfile> p2pProfile{};   // Only with P2P transfers
  std::optional<UnifiedMemoryProfile> unifiedMemory{}; // Only under --capture-um
};
```

**Field availability:**

- **transfers, occupancy, clocks, powerThermal, cupti**: Always present; a
  value the run could not read is 0 here and an empty cell in the CSV (see
  [What Gets Measured?](#what-gets-measured))
- **multiGpu**: Only when using `cudaKernelMultiGpu()`
- **p2pProfile**: Only when P2P transfers are measured
- **unifiedMemory**: Only when the measurement asked for unified-memory
  counters (`--capture-um`) and CUPTI recorded them completely (see
  [Unified-Memory Counters](#unified-memory-counters))

### Key Metrics Explained

#### Occupancy

**What it is:** An estimate of the share of an SM's warp slots the launch
shape lets the kernel fill, which the harness computes when a test declares
`.withLaunchConfig()`: from the block size, the SM's thread and block limits
and the dynamic shared memory the launch declares.

**Formula:** `min(1, warps per block x blocks per SM / max warps per SM)`,
with warps per block the block's threads over 32 rounded up, max warps per SM
the SM's thread limit over 32, and blocks per SM the smallest of the SM's
thread limit over the block's threads, the SM's block limit, and the SM's
shared memory over the declared dynamic shared memory, each rounded down (the
last only when the launch declares some).

What it leaves out can make it overstate what the launch shape allows:

- The kernel's registers and static shared memory.
- Partial warps: a block whose size is not a multiple of 32 still takes
  whole warps, but the thread limit is applied to its threads, so the
  estimate can count more such blocks than fit; it then reads 1.0. On the
  laptop's RTX 5000 Ada (1,536 threads, 48 warps per SM), blocks of 129, 200
  and 300 threads read 1.0 where the CUDA runtime's calculation gives 0.94,
  0.88 and 0.83 (45, 42 and 40 warps).
- The shared memory the runtime reserves for each block, 1 KiB there: a
  256-thread block declaring 33,792 bytes reads 0.5 (3 blocks) where the
  runtime gives 0.33 (2 blocks).

Nor does it count the grid: one block and thousands of blocks of the same
shape get the same figure, although one block runs on a single SM and
thousands spread over all of them.

**Interpretation:** the figure estimates the most the launch shape allows; it
is not a measure of how busy the GPU was, so it grades neither the kernel nor
the GPU's use. A low figure says the block size or the declared shared memory
holds the warps an SM can take below its maximum; a high one does not rule
out a lower limit, for the reasons above. The CUDA runtime's
`cudaOccupancyMaxActiveBlocksPerMultiprocessor()` counts those three (see
[Understanding Occupancy](#understanding-occupancy)); for the occupancy a
kernel achieves, and whether occupancy is what limits it, measure with a
kernel profiler ([Profiling GPU Code](#profiling-gpu-code)).

#### Transfer Overhead

**What it is:** How the copies compare with the round trip: `transferTimeUs`
over `totalTimeUs`.

**Formula:** `(transferTimeUs / totalTimeUs) * 100%`

The two are medians taken separately (see
[What Gets Measured?](#what-gets-measured)): the two copy medians and the
round-trip median can each come from a different repeat, so the ratio is not
the share of any one round trip, and it can pass 100%. A quick-start run on a
laptop GPU reported
`transferTimeUs` 2,082.048 against `totalTimeUs` 2,064.5, a ratio of 100.9%.
`result.stats.transfers.transferOverheadPct(result.kernelTimeUs)` divides the
copy medians by their sum with the launch median instead, so it stays within
100% (99.6% for that run); its three medians are taken separately too, so it
is not the share of one round trip either.

**Interpretation:**

- Only while the copy times hold steady from repeat to repeat is the ratio
  close to the copies' share of a round trip (the median one). Then near 100%
  says the copies decide `totalTimeUs` and the kernel hardly shows in it, and
  a small ratio says the launch takes most of it.
- When the copy times vary, the medians cannot tell which part takes most of
  a round trip, and the ratio can point the wrong way. Three repeats with
  host-to-device copies of 100, 1 and 1 us, device-to-host copies of 1, 100
  and 1 us and launches of 50, 50 and 1 us give a `transferTimeUs` of 2 and a
  `totalTimeUs` of 151, a ratio of 1.3%, although the copies take two thirds
  of every round trip.
- The result keeps no copy time per repeat, and steady round trips do not
  show steady copies. An Nsight Systems timeline does: it records every copy
  and launch, and `bench run --profile nsight` writes
  `cuda_gpu_mem_time_sum.txt` with each copy direction's count, median,
  minimum, maximum and spread
  ([walkthrough 11](../demo/docs/11_NSIGHT_PROFILER.md)).

**How to reduce:**

```cpp
// Bad: Many small transfers
for (int i = 0; i < 1000; ++i) {
  cudaMemcpy(d_data + i, h_data + i, sizeof(float), cudaMemcpyHostToDevice);
}

// Good: One large transfer
cudaMemcpy(d_data, h_data, 1000 * sizeof(float), cudaMemcpyHostToDevice);

// Better: Keep data on GPU between kernel calls
// Or: Use Unified Memory for infrequent access
```

#### Memory Bandwidth

**What it is:** The rate of the host-device copies a test declares
(`memBandwidthGBs`). The kernel's own reads and writes of device memory are
not measured.

**Formula:** `(h2dBytes + d2hBytes) / (h2dTimeUs + d2hTimeUs) / 1000`: the
bytes the test declares for one repeat over the two copy medians, whose sum is
`transferTimeUs`, are bytes per microsecond, and 1,000 bytes per microsecond
are 1 GB/s (10^9 bytes per second). For the quick-start run above, 12,582,912
bytes over 2,082.048 us are 6,043.5 bytes per microsecond, 6.04 GB/s.

**Interpretation:**

- Compare it with the link the copies cross: PCIe on a discrete GPU, the
  memory a Jetson's CPU and GPU share. The GPU's device-memory peak is not the
  comparison.
- Pageable host memory copies more slowly than pinned memory
  (`cudaMallocHost`)
- A test that declares no copies has no copy rate: its cell is empty

---

## Writing GPU Benchmarks

### Test Macros

```cpp
PERF_GPU_TEST(Suite, Name)           // General GPU test
PERF_GPU_THROUGHPUT(Suite, Name)     // Throughput-focused test
PERF_GPU_LATENCY(Suite, Name)        // Latency-focused test
PERF_GPU_COMPARISON(Suite, Name)     // CPU vs GPU comparison
PERF_GPU_BANDWIDTH(Suite, Name)      // Memory bandwidth test
PERF_GPU_SCALING(Suite, Name)        // Multi-GPU scaling test

PERF_GPU_GUARD(varName)           // Create scoped PerfGpuCase
PERF_GPU_MAIN()                      // Main function with GPU flag parsing
```

### Basic Kernel Test

```cpp
PERF_GPU_TEST(MyKernel, Basic) {
  PERF_GPU_GUARD(perf);

  // Setup
  const int N = 1024 * 1024;
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  });

  // Measure
  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "my_kernel")  // Label parameter
    .withLaunchConfig(grid, block)
    .measure();

  // Validate
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  cudaFree(d_data);
}
```

### Launch on the Stream You Are Given

The callable gets the stream to launch on, and the harness times that stream
with CUDA events, so queue the measured work there. The case's stream is a
blocking stream (with `PerfGpuConfig::useHighPriorityStream`, the default, and
without it): work that lands on the legacy default stream, such as a launch
without a stream, `cudaMemset`, or a library call left on its default stream,
is ordered with it and timed too. Work on any other stream is not timed unless
the callable joins it back to the stream it was given (records an event there
and has the given stream wait for it), and work on the per-thread default
stream, in code built with `--default-stream per-thread`, is not ordered with
the case's stream at all: the events do not time it.

### Comparing Implementations

```cpp
PERF_GPU_TEST(MyAlgorithm, CompareVersions) {
  PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  std::printf("\n=== Comparing Kernel Versions ===\n");

  // Test baseline version
  perf.cudaWarmup([&](cudaStream_t s) {
    naiveKernel<<<grid, block, 0, s>>>(d_data, N);
  });

  auto baseline = perf.cudaKernel([&](cudaStream_t s) {
    naiveKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "naive")  // Unique label for CSV
    .withLaunchConfig(grid, block)
    .measure();

  std::printf("Naive:     %.3f ms, occupancy estimate: %.1f%%\n",
              baseline.kernelTimeUs / 1000.0,
              baseline.stats.occupancy.achievedOccupancy * 100);

  // Test optimized version
  perf.cudaWarmup([&](cudaStream_t s) {
    optimizedKernel<<<grid, block, 0, s>>>(d_data, N);
  });

  auto optimized = perf.cudaKernel([&](cudaStream_t s) {
    optimizedKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "optimized")  // Different label
    .withLaunchConfig(grid, block)
    .measure();

  std::printf("Optimized: %.3f ms, occupancy estimate: %.1f%%\n",
              optimized.kernelTimeUs / 1000.0,
              optimized.stats.occupancy.achievedOccupancy * 100);

  // Calculate improvement
  double speedup = baseline.kernelTimeUs / optimized.kernelTimeUs;
  std::printf("\nSpeedup: %.2fx\n", speedup);

  // Validate
  EXPECT_GT(speedup, 1.1) << "Optimization should be >10% faster";

  cudaFree(d_data);
}
```

---

## Launch Configuration

### Choosing Thread Count

Launch configuration (grid/block dimensions) significantly impacts performance.

**Common thread counts:**

- **128**: Small kernels, high register usage
- **256**: **Default choice** - good balance for most kernels
- **512**: Large kernels, low register usage
- **1024**: Maximum, rarely optimal

**How to choose:**

1. Start with 256 threads/block
2. Measure the other sizes on the same workload
   ([Block Size Sweep](#block-size-sweep))
3. Keep the size with the shortest measured time: the occupancy estimate in
   the results says what each launch shape allows, not which one runs fastest

### Block Size Sweep

```cpp
PERF_GPU_TEST(MyKernel, BlockSizeSweep) {
  PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  std::printf("\n%-12s %-12s %-12s %-12s\n",
              "Threads", "Time(ms)", "Occ. est. %", "Throughput");
  std::printf("%s\n", std::string(50, '-').c_str());

  for (int threads : {128, 256, 512, 1024}) {
    int blocks = (N + threads - 1) / threads;
    dim3 block(threads);
    dim3 grid(blocks);

    perf.cudaWarmup([&](cudaStream_t s) {
      myKernel<<<grid, block, 0, s>>>(d_data, N);
    });

    auto result = perf.cudaKernel([&](cudaStream_t s) {
      myKernel<<<grid, block, 0, s>>>(d_data, N);
    }, "threads_" + std::to_string(threads))
      .withLaunchConfig(grid, block)
      .measure();

    // The launch-shape estimate (see Occupancy), not a measurement
    double occupancy = result.stats.occupancy.achievedOccupancy * 100;

    std::printf("%-12d %-12.3f %-12.1f %-12.0f\n",
                threads,
                result.kernelTimeUs / 1000.0,
                occupancy,
                result.callsPerSecond);
  }

  cudaFree(d_data);
}
```

**Output of one run** on the laptop RTX 5000 Ada (`--cycles 200 --repeats 5`,
a stand-in `myKernel` doing 256 multiply-adds per element; the harness's own
result line before each row is left out):

```
Threads      Time(ms)     Occ. est. %  Throughput
--------------------------------------------------
128          0.022        100.0        45716
256          0.022        100.0        46408
512          0.021        100.0        46833
1024         0.024        66.7         41544
```

An SM of this GPU holds 1,536 threads (48 warps) and at most 24 blocks:
blocks of 128, 256 and 512 threads fill it (12, 6 and 3 blocks), so the
estimate is 100% for each, and one 1024-thread block, 32 of the 48 warps,
gives 66.7%. The estimate follows from the launch shape alone, whatever the
kernel, so it does not say which size runs fastest; the times do.

### Understanding Occupancy

**Occupancy** = Active warps / Maximum possible warps per SM

**What limits occupancy:**

- **Registers per thread** - More registers = fewer active threads
- **Shared memory per block** - More shared mem = fewer active blocks
- **Thread block size** - Too small/large reduces occupancy

The harness's estimate counts the block size and the dynamic shared memory a
launch declares; it leaves out the kernel's registers and static shared
memory, among other things (see [Occupancy](#occupancy)). The CSV's CUPTI
columns give the registers per thread and the static and dynamic shared
memory of the kernel's launches in the untimed batch after the timed repeats
([CUPTI in-process kernel metrics](#cupti-in-process-kernel-metrics)), and
the CUDA runtime's `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` gives
how many blocks of a kernel an SM can hold at once for a block size and
dynamic shared memory, counting the kernel's registers and static shared
memory as well.

No occupancy figure is right for every kernel: higher occupancy is not always
faster, and the launch shape worth keeping is the one with the best measured
time for the workload. For the occupancy a kernel achieves, and whether
occupancy limits it, measure with a kernel profiler
([Profiling GPU Code](#profiling-gpu-code)).

---

## CPU vs GPU Comparison

### Measuring Speedup

Use `cpuBaseline()` to measure GPU speedup over CPU:

```cpp
PERF_GPU_TEST(MyAlgorithm, CpuVsGpu) {
  PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  std::vector<float> h_data(N, 1.0f);
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  std::printf("\n=== CPU vs GPU Comparison ===\n");

  // Measure CPU baseline (must be first!)
  auto cpuResult = perf.cpuBaseline([&] {
    for (int i = 0; i < N; ++i) {
      h_data[i] = processOnCpu(h_data[i]);
    }
  }, "cpu_version");

  std::printf("CPU time: %.2f ms\n", cpuResult.stats.median / 1000.0);

  // Measure GPU version
  dim3 block(256);
  dim3 grid((N + 255) / 256);

  perf.cudaWarmup([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  });

  auto gpuResult = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "gpu_version")
    .withHostToDevice(h_data.data(), d_data, N * sizeof(float))
    .withDeviceToHost(d_data, h_data.data(), N * sizeof(float))
    .withLaunchConfig(grid, block)
    .measure();

  std::printf("GPU time: %.2f ms (kernel: %.2f ms, transfers: %.2f ms)\n",
              gpuResult.totalTimeUs / 1000.0,
              gpuResult.kernelTimeUs / 1000.0,
              gpuResult.transferTimeUs / 1000.0);

  // Speedup automatically calculated
  std::printf("\nSpeedup: %.2fx\n", gpuResult.speedupVsCpu);

  // Validate speedup
  EXPECT_GT(gpuResult.speedupVsCpu, 5.0)
    << "GPU should be at least 5x faster than CPU for this workload";

  cudaFree(d_data);
}
```

**How speedup is calculated:**

```
speedupVsCpu = cpuTime / gpuTotalTime
```

**In the CSV** (`--csv results.csv`) the two measurements are the test's two
rows, named by their labels. The columns that compare them, picked by name
with a CSV reader, which keeps a quoted test name (a label may hold a comma)
in one column:

```bash
python3 -c '
import csv, sys
path, *cols = sys.argv[1:]
out = csv.writer(sys.stdout)
out.writerow(cols)
for row in csv.DictReader(open(path, newline="")):
    if None in row or None in row.values():
        sys.exit(f"{path}: a row whose cells do not match the header")
    out.writerow([row[c] for c in cols])
' results.csv test wallMedian kernelTimeUs transferTimeUs speedupVsCpu
```

From one run on the laptop RTX 5000 Ada (`--cycles 10 --repeats 5`, with
stand-ins for `processOnCpu()` and `myKernel` that each do 64 multiply-adds
per element):

```
test,wallMedian,kernelTimeUs,transferTimeUs,speedupVsCpu
MyAlgorithm.CpuVsGpu/cpu_version,11080.3,,,
MyAlgorithm.CpuVsGpu/gpu_version,1194.94,12.016000,1186.623991,9.272702
```

The CPU row leaves the GPU columns empty. `speedupVsCpu` is the CPU row's
`wallMedian` over the GPU row's, the round trip with its copies
(11080.3 / 1194.94 = 9.27); over `kernelTimeUs` alone the ratio would be 922.
The API reference's [CSV Schema](API_REFERENCE.md#csv-schema) describes every
column.

**When to use CPU comparison:**

- Justifying GPU port (is it worth the effort?)
- Reporting speedup to stakeholders
- Identifying whether workload is GPU-suitable
- Validating optimization efforts

---

## Multi-GPU Benchmarking

### Basic Multi-GPU Test

Test performance across multiple GPUs:

```cpp
PERF_GPU_TEST(MyKernel, MultiGpu) {
  PERF_GPU_GUARD(perf);

  const int DEVICE_COUNT = 4;
  const int N = 1024 * 1024;

  // Allocate per-device memory
  std::vector<float*> d_data(DEVICE_COUNT);
  for (int i = 0; i < DEVICE_COUNT; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&d_data[i], N * sizeof(float));
  }

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  // Benchmark across all GPUs
  auto result = perf.cudaKernelMultiGpu(DEVICE_COUNT,
    [&](int dev, cudaStream_t s) {
      // This lambda runs once per device (in parallel)
      myKernel<<<grid, block, 0, s>>>(d_data[dev], N);
    }, "multi_gpu_kernel")
    .withLaunchConfig(grid, block)
    .withP2PAccess()  // Enable peer-to-peer transfers
    .measure();

  std::printf("\n=== Multi-GPU Results ===\n");

  // Per-device results
  for (int i = 0; i < DEVICE_COUNT; ++i) {
    std::printf("GPU %d: %.2f ms\n", i,
                result.perDevice[i].kernelTimeUs / 1000.0);
  }

  // Aggregated metrics: the devices against each other
  std::printf("\nLoad imbalance: %.2f (slowest over fastest device)\n",
              result.aggregatedStats.multiGpu->loadImbalance);

  // Check that no device lags the others
  EXPECT_LT(result.aggregatedStats.multiGpu->loadImbalance, 1.2)
    << "Uneven work across " << DEVICE_COUNT << " GPUs";

  // Cleanup
  for (int i = 0; i < DEVICE_COUNT; ++i) {
    cudaSetDevice(i);
    cudaFree(d_data[i]);
  }
}
```

### Understanding Scaling Efficiency

Scaling efficiency measures how well your code uses multiple GPUs. The harness
does not measure it: it needs the time one device takes on the whole job, and
a multi-GPU measurement times each device on its own share only, so
`MultiGpuMetrics::scalingEfficiency` stays 0 and the CSV's `multiGpuEfficiency`
cell stays empty. Time the whole job on one device in a case of its own and
compare its median with the multi-GPU row's `wallMedian`:

```
scaling efficiency = single_gpu_time / (num_devices * multi_gpu_time) * 100%
```

A multi-GPU row's `speedupVsCpu` is the CPU baseline over the row's own
`wallMedian` (the first device's median), as a kernel row's is over its own;
`MultiGpuResult::totalSpeedupVsCpu` is the devices' speedups summed.

**Interpretation:**

- **100%**: Perfect linear scaling (4 GPUs = 4x faster)
- **80-95%**: Excellent scaling (typical for compute-bound)
- **50-80%**: Fair (communication overhead present)
- **<50%**: Poor (investigate bottlenecks)

**What affects scaling:**

- **Compute/communication ratio** - More compute = better scaling
- **P2P bandwidth** - NVLink >> PCIe
- **Load balance** - Uneven work distribution hurts scaling
- **Synchronization overhead** - Frequent sync reduces scaling

### P2P Bandwidth Testing

Test peer-to-peer transfer bandwidth between GPUs:

```cpp
PERF_GPU_TEST(P2P, Bandwidth) {
  PERF_GPU_GUARD(perf);

  const size_t TEST_SIZE = 100 * 1024 * 1024;  // 100MB

  auto result = perf.cudaKernelMultiGpu(2,
    [&](int dev, cudaStream_t s) {
      // Empty kernel (we're measuring P2P only)
    }, "p2p_test")
    .withP2PAccess()
    .measureP2PBandwidth(0, 1, TEST_SIZE)  // GPU 0 -> GPU 1
    .measure();

  double p2pBandwidth = result.aggregatedStats.multiGpu->p2pBandwidthGBs;

  std::printf("P2P Bandwidth (GPU 0 -> GPU 1): %.2f GB/s\n", p2pBandwidth);

  // Expected bandwidth:
  // NVLink (A100/H100): 50-300 GB/s
  // PCIe 4.0: ~12 GB/s
  // PCIe 3.0: ~10 GB/s

  EXPECT_GT(p2pBandwidth, 10.0) << "P2P bandwidth unusually low";
}
```

### Multi-GPU CSV Schema

Additional columns when using multi-GPU:

| Column               | Type   | Description                          |
| -------------------- | ------ | ------------------------------------ |
| `deviceCount`        | int    | Number of GPUs used                  |
| `multiGpuEfficiency` | double | Not measured: always empty           |
| `p2pBandwidthGBs`    | double | Peer-to-peer bandwidth (if measured) |

---

## Memory Strategies

### Explicit Memory (Default)

Traditional CUDA memory management:

```cpp
// Explicit allocation and transfers
float* d_data;
cudaMalloc(&d_data, SIZE);
cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
kernel<<<...>>>(d_data);
cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

**Pros:**

- Explicit control over transfers
- Predictable performance
- Best for high-frequency access

**Cons:**

- More code
- Manual memory management

### Unified Memory

Managed memory gives the CPU and the GPU one pointer, and CUDA makes the data
available where it is used:

```cpp
float* data;
cudaMallocManaged(&data, SIZE);
kernel<<<...>>>(data);
cudaDeviceSynchronize();  // before the host reads data
cudaFree(data);
```

**Measuring it.** The harness times a kernel on managed memory as it times any
kernel. It does not allocate or move a test's memory, so the test chooses the
strategy: allocate with `cudaMallocManaged()` and declare no transfer
(`withHostToDevice()` and `withDeviceToHost()` time explicit copies, which
managed data does not need). Kernel time is the median over the repeats of
each repeat's time per launch. A wait for pages to reach the GPU inside the
timed interval counts in its repeat's sample: averaging over the launches and
taking the median can dilute or hide a one-time wait, but neither isolates it
nor guarantees that it is left out. Under `--capture-um` each measurement also
says whether pages moved in its window, in which direction and for how long
([Unified-Memory Counters](#unified-memory-counters)); a warmup that touches
the data moves its pages before the window, so the measurement after it shows
no migration.

### Unified-Memory Counters

`--capture-um` asks for unified-memory counters, as does a test that sets
`PerfGpuConfig::captureUnifiedMemory`. The harness's in-process CUPTI
collector then records CUPTI's unified-memory counter activity over each
measurement's window, from before its timed repeats to after them (the
declared copies and the launches, not the warmup), and the result's
`GpuStats::unifiedMemory` and its CSV row carry what the records add up to:

| Column              | From CUPTI's records in the window                                            |
| ------------------- | ----------------------------------------------------------------------------- |
| `umPageFaults`      | the GPU page-fault groups, plus one for each CPU page fault                   |
| `umH2DMigrations`   | the host-to-device transfers (their bytes: `unifiedMemory->h2dBytes`)         |
| `umD2HMigrations`   | the device-to-host transfers (their bytes: `unifiedMemory->d2hBytes`)         |
| `umMigrationTimeUs` | the transfers' durations, summed: transfers that overlap each count their own |
| `umThrashing`       | 1 when CUPTI recorded thrashing in the window, otherwise 0                    |

A migration is one transfer record. The driver moves a buffer in transfers
of its own choosing, so their number varies from run to run for the same
access while the bytes do not. A window that migrates nothing has zeros, which
are observed: copies of device memory migrate nothing, and neither does a
kernel on managed pages already on the GPU. The counters are the process's,
so a multi-GPU measurement, which records them over its devices' launches,
counts every device's. `isThrashing()` and `migrationOverheadPct()` are
arithmetic over the counts, not CUPTI's report. The counts reach the result
and the CSV; the run prints no line for them, as for the other GPU cells.

`GpuUnifiedMemory.CountersFollowTheBuffer`
([`GpuUnifiedMemory_pTest.cu`](../ptst/gpu/GpuUnifiedMemory_pTest.cu)) checks
them against what it does to a 1 MiB managed buffer: a host write then a
kernel read moves at least the buffer to the device and nothing back, a
kernel write then a host read at least the buffer back and nothing to the
device. On an RTX 5000 Ada laptop GPU each step moved exactly 1048576 bytes,
to the device in a number of transfers that changed from run to run and back
in 10 every time, and an Nsight Systems trace of the same steps gave the same
bytes and directions.

**Where CUPTI cannot record them.** On the Jetson AGX Thor CUPTI refuses the
configuration, and under an Nsight Systems or Nsight Compute session, with
`VERNIER_DISABLE_CUPTI=1` or in a build without CUPTI the collector records
nothing. There the request fails as a `--profile` request that cannot collect
does: the tests still run, timed as without it, and their rows are written
with the five cells empty, but the first measurement reports the request with
the device and the reason, and a run whose tests pass exits with status 4
([troubleshooting](TROUBLESHOOTING.md#--profile--failed-and-exit-status-4)).
On the Thor:

```text
[FAIL] Profiler 'capture-um': unsupported: NVIDIA Thor: CUPTI refused to configure the unified-memory counters (CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED): umPageFaults, umH2DMigrations, umD2HMigrations, umMigrationTimeUs and umThrashing stay empty
   Run without --capture-um (or leave PerfGpuConfig::captureUnifiedMemory false); the timing and the CSV do not change.
   The run will fail (exit status 4 if the tests pass).
```

and, at the end of the run:

```text
[profile] a profile failed; the run exits with status 4:
[profile]   capture-um: unsupported: NVIDIA Thor: CUPTI refused to configure the unified-memory counters (CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED): umPageFaults, umH2DMigrations, umD2HMigrations, umMigrationTimeUs and umThrashing stay empty
```

A measurement whose records are incomplete (CUPTI dropped some, for instance)
fails the request for its test in the same way, naming the problem, and its
cells stay empty. A failed test keeps its own status.

**What the request costs.** Switching the counters on before the timed
repeats, and flushing and switching them off after them, adds about 45 ms to
each measurement on the laptop GPU, outside the timed interval. While the
repeats run, CUPTI records every migration and fault and no kernel record
(the CUPTI cells come from a batch after the repeats); on that GPU the
sample's kernel medians with and without the request agreed within their
run-to-run spread, and a 16 x 256 store through a one-device multi-GPU
measurement read 1.76 against 1.85 us a launch with the request and without
on a performance core, and 2.80 against 2.90 us on an efficiency core
(medians of 20 alternating runs each, every run pinned to one CPU), but
compare timings from runs made the same way.

`--gpu-memory unified` asks for no counters. It selects managed memory for
the tests that allocate by the selected strategy, such as
`GpuMemoryStrategies.SelectedStrategy`, whose row under
`--gpu-memory unified --capture-um` counts its input's move to the GPU; the
harness changes no test's allocation
([GPU Memory Strategy Selection](ADVANCED_GUIDE.md#gpu-memory-strategy-selection)).

**When Unified Memory is good:**

- Large datasets exceeding GPU memory
- Data accessed infrequently
- Prototyping/ease of development
- Sparse access patterns

**When explicit is better:**

- Data accessed frequently
- Predictable access patterns
- Need maximum performance
- Production optimization

**Example comparison:** the same kernel on managed and on device memory, in
one test. Each measurement is its own CSV row,
`MemoryStrategy.Compare/managed_memory` and
`MemoryStrategy.Compare/device_memory`:

```cpp
PERF_GPU_TEST(MemoryStrategy, Compare) {
  PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  const size_t BYTES = N * sizeof(float);
  std::vector<float> h_data(N, 1.0f);

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  float* um_data;  // the host fills it through the same pointer
  float* d_data;   // filled with an explicit copy
  cudaMallocManaged(&um_data, BYTES);
  cudaMalloc(&d_data, BYTES);
  std::copy(h_data.begin(), h_data.end(), um_data);
  cudaMemcpy(d_data, h_data.data(), BYTES, cudaMemcpyHostToDevice);

  auto managed = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(um_data, N);
  }, "managed_memory")
    .withLaunchConfig(grid, block)
    .measure();

  auto device = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "device_memory")
    .withLaunchConfig(grid, block)
    .measure();

  std::printf("managed/device kernel time: %.2f\n",
              managed.kernelTimeUs / device.kernelTimeUs);

  cudaFree(um_data);
  cudaFree(d_data);
}
```

The ratio is this machine's, and can differ with another GPU or driver, and
where the GPU shares the host's memory. `GpuUnifiedMemory` in
`src/bench/ptst/gpu/GpuUnifiedMemory_pTest.cu` runs this comparison with every
result checked, and times kernels after the host wrote and after it read a
managed buffer.

---

## Profiling GPU Code

### Using Nsight Compute

NVIDIA Nsight Compute provides detailed kernel profiling:

```bash
# Profile a kernel's run (default Systems mode -> nsys timeline): bench run
# starts the binary under nsys
bench run ./MyKernel_GPU_PTEST --profile nsight -- --gtest_filter="*MyKernel"
# Generates: bench-out/MyKernel_GPU_PTEST.nsight/profile.nsys-rep (+ the
# nsys stats summaries)

# Kernel deep-dive (Compute replay) -> .ncu-rep instead, under ncu
bench run ./MyKernel_GPU_PTEST --profile nsight --profile-args replay \
  --cycles 3 --repeats 1 -- --gtest_filter="*MyKernel"
# Generates: bench-out/MyKernel_GPU_PTEST.nsight/kernel_replay.ncu-rep

# Analyze with Nsight UI
ncu-ui bench-out/MyKernel_GPU_PTEST.nsight/kernel_replay.ncu-rep

# Or command-line report
ncu --import bench-out/MyKernel_GPU_PTEST.nsight/kernel_replay.ncu-rep --csv --print-summary per-kernel
```

**What the replay collects:**

- Throughput of the SMs, L1 and L2, and of DRAM where the GPU reports it (the
  Jetson AGX Thor does not)
- Achieved occupancy
- Warp execution efficiency and branch efficiency
- Registers per thread
- Instruction mix (ALU, FMA and load/store pipe utilization)

For roofline analysis and occupancy limiters, name ncu's sections instead:
`--profile-args '--section SpeedOfLight_RooflineChart --section Occupancy'`,
or a set such as `--profile-args '--set full'` (see
[tools/README.md](../../../tools/README.md#run---execute-benchmark-binary)).

**No timing under Nsight Compute.** A process ncu profiles, under
`bench run --profile ncu`, nsight's compute route or an ncu command typed by
hand, is instrumented: ncu's work adds to what host timers and CUDA events
measure, whatever metrics it collects (whether it also replays a launch
depends on those metrics). A `PERF_TEST` case whose work launches kernels is
slowed as a `PERF_GPU_TEST` case is, so under ncu neither harness times a
measurement:

- every row leaves its timing cells empty: `wallMedian` through `wallCV`,
  `callsPerSecond` and `stable`; a GPU row also leaves the values computed
  from them empty: `kernelTimeUs`, `transferTimeUs`, `speedupVsCpu`,
  `memBandwidthGBs`, `multiGpuEfficiency` and `p2pBandwidthGBs`;
- the run says why once on stderr, each test's console line reads
  `[Suite.Case]  not timed: Nsight Compute (ncu) profiles this process`, and
  the end-of-run table shows `NOT TIMED` for such a row;
- the other cells stay: the configuration, the metadata, `gpuModel`,
  `computeCapability`, the bytes moved, the occupancy estimate, the NVML
  readings and the device columns.

The result a test reads in its own code is unchanged. ncu's own measurements
are in its report. Time the code in a run without ncu. A run under Nsight
Systems (nsys) is not an ncu session: its rows stay timed. `bench doctor
<binary> --profile ncu` says before the run that no row of it is timed.
`bench summary` and `bench compare` refuse a row left untimed under ncu,
saying that its timing cells are empty as in a run under ncu, and `bench run`
refuses `--analyze` for a run under ncu before it starts.

### GPU Profiler Integration

`PERF_GPU_GUARD` auto-attaches profiler hooks; whichever backend is
selected by `--profile X` fires around the measured window. No manual
`attachGpuProfilerHooks` call required. A measurement that ends by an
exception (a refused launch, a device's thread that failed, a callable that
throws) calls the backend's `abortMeasure()` at once, with nothing analyzed or
published. Only a backend that implements it ends what it started this way,
gperf and Nsight today: gperf stops its capture and Nsight pops the window's
NVTX range, so the case and other cases then measure under a capture or range
of their own.

```cpp
PERF_GPU_TEST(MyKernel, WithProfiling) {
  PERF_GPU_GUARD(perf);

  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "profiled_kernel")
    .withLaunchConfig(grid, block)
    .measure();
}
```

### GPU-side backends

The GPU profilers are `nsight` (or `nsys`), `ncu`, `compute-sanitizer` and
`rocprof`, the legacy integration that is not validated on AMD hardware
(walkthrough 18). The advanced guide's
[Available Profilers](ADVANCED_GUIDE.md#available-profilers) gives what each
records, needs, writes and costs.

Each of these tools records a process only when it starts it. Under the tool,
the backend stays passive and reports the tool's output as the artifact; run
without it, the benchmark runs its tests, fails with exit status 4 when they
pass, and prints the command that would start it under the tool.
`bench run --profile <name>` starts it that way (for every name here but
`rocprof`).

```bash
# Timeline profile, started by hand: nsys writes run.nsys-rep (bench run
# --profile nsight also writes the nsys stats summaries beside it). The
# in-process CUPTI collector stands down inside the session by itself (see
# "CUPTI in-process kernel metrics" below).
nsys profile -o run.nsys-rep -t cuda,nvtx \
    ./MyGpuTest --profile nsight --gtest_filter='*'

# GPU memory error detection: --error-exitcode makes a run in which the
# tool finds an error exit 5, so that a job fails on it
compute-sanitizer --tool=memcheck --error-exitcode 5 --log-file run.sanitizer.log \
    ./MyGpuTest --profile compute-sanitizer --cycles 5

# AMD GPU (HIP-built binary): the legacy rocprof integration, not validated
# on AMD hardware
rocprof --stats -o run.csv \
    ./MyHipTest --profile rocprof --profile-args stats
```

Whether nsys can trace a benchmark depends on the toolkit it was built with,
the CUDA driver it runs on and the nsys version together. One combination has
been traced: nsys 2026.3.1 in the CUDA 13.1 dev image recorded every kernel
launch of a quick-start run built with CUDA 13.1 (25,011: 25,000 measured, 11
warmup) on a laptop with driver 580.178.04, which reports CUDA 13.0. Inside the
image the benchmark ran on the image's own CUDA 13.1 user-mode driver (libcuda
590.44.01, CUDA's forward-compatibility package), and `nvidia-smi` there
reports CUDA 13.1. The in-process CUPTI columns were filled in the same setup;
`ncu` was not run in it.

NVIDIA documents what a newer toolkit can do on an older driver:
[minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
runs a CUDA 13.x application on a driver of 580 or later, without the features
that need a newer driver and without device code the driver would have to
compile from PTX, and the
[Nsight Systems installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
gives its CUDA requirements and warns that under forward compatibility some
profiling features may be unavailable. Other combinations, a CUDA 13.1 binary
on a 13.0 user-mode driver among them, have not been tried.

### NVTX annotation API

`BENCH_NVTX_SCOPE("name")` pushes a labeled range that an Nsight tool records
with the run. Under `--profile nsight` (the profile `bench run --profile nsight`
passes the benchmark), the Nsight backend also marks each measured window with
a range named after the test, so even an unannotated benchmark shows its tests
on the nsys timeline. A run without an Nsight profile gets no such range.

```cpp
#include "Nvtx.hpp"

void processFrame() {
  {
    BENCH_NVTX_SCOPE("phase_decode");
    decode();
  }
  {
    BENCH_NVTX_SCOPE("phase_render");
    render();
  }
}
```

The macros emit ranges only where two conditions hold as the code that uses
them compiles: `COMPAT_NVTX_AVAILABLE` is nonzero, and `<nvtx3/nvToolsExt.h>`
can be included (`Nvtx.hpp`). Otherwise they compile to nothing, so the same
code builds and runs without CUDA. For code that links `vernier::bench`,
vernier's configuration sets the first. With the CUDA toolkit found (a GPU
build, the default) and CMake 3.25 or newer, which defines the toolkit's
`CUDA::nvtx3` target, `vernier::bench` hands `COMPAT_NVTX_AVAILABLE=1` and
that target, which adds the toolkit's include directories, to the code that
links it. CMake defines the target without checking that the header is there,
so the header stays a condition of its own. Without the target (CMake 3.24,
no toolkit, or `-DVERNIER_BUILD_GPU=OFF`) `vernier::bench` hands on
`COMPAT_NVTX_AVAILABLE=0`, and the macros compile to nothing even where the
header can be included. An installed vernier keeps the value of the build it
came from, which says nothing about whether the code using it can include the
header. The range `--profile nsight` puts around each test depends on both
conditions in vernier's own build. Code that includes `Nvtx.hpp` without
vernier's targets, where `COMPAT_NVTX_AVAILABLE` is undefined, gets ranges
whenever the header can be included.

### CUPTI in-process kernel metrics

CUPTI's Activity API fills per-launch register count, static + dynamic shared
memory, and launch count into the GPU CSV section on every GPU run, with no
`--profile` flag, except where the collector cannot or must not run: a build
without CUPTI, `VERNIER_DISABLE_CUPTI=1`, an nsys or ncu session (below),
CUPTI refusing the collector, or a CUPTI window of the process that did not
stop recording kernels or flush its records, which ends collection for the
rest of the process. There the cells stay empty and the run says why,
as it does for a test whose batch's records are not known to be complete. The
columns:

| Column                  | Meaning                                                 |
| ----------------------- | ------------------------------------------------------- |
| `cuptiKernelLaunches`   | Number of kernel launches in the untimed batch (below)  |
| `cuptiRegistersMedian`  | Median registers per thread across the batch's launches |
| `cuptiRegistersMax`     | Max registers per thread observed                       |
| `cuptiStaticSmemBytes`  | Median static `__shared__` allocation                   |
| `cuptiDynamicSmemBytes` | Median dynamic shared memory passed at launch           |

**Where CUPTI collects.** No timed launch runs while CUPTI records kernels:
its record keeping adds host time to every launch made while it records,
which a kernel shorter than the host's interval between two launches
carries in full; on the laptop a 16 x 256 store took about 0.6 us more a
launch with the launching thread on a performance core (1.35x) and about
2 us more on an efficiency core (1.8x). After a kernel measurement's timed
repeats the harness calls the measured callable
`cycles` times more, at most 1000, untimed, with CUPTI recording kernels, and
the five cells describe those launches: `cuptiKernelLaunches` counts the
batch's launches (20 for a test at `--cycles 20`, 40 if each call launches
two kernels), not the timed ones. The callable runs `cycles x repeats` times
timed and those calls more, so a callable with side effects sees them all. No
batch runs where the collector does not collect (an nsys or ncu session,
`VERNIER_DISABLE_CUPTI=1`, a build without CUPTI, CUPTI refusing the
collector), so a tool that holds CUPTI sees no extra launch. What the batch
adds to one measurement on the laptop's RTX 5000 Ada, at 1000 cycles: a few
milliseconds for a 16 x 256 store (a microsecond kernel; 4.2 ms with the
launching thread on a performance core and 3.3 ms on an efficiency core,
medians of five rounds that spread by several milliseconds) and 1.0 s for a
1 ms kernel, whose 1000 calls run once more; less in proportion at fewer
cycles.

**What CUPTI still leaves on the timed launches.** Host time, which reaches
only kernels shorter than the host's launch interval, and how much depends
on the core the launching thread runs on. The collector's registration with
CUPTI, at a case's construction, leaves none the measurements could tell
from their run-to-run spread: with the collector on and stood down, a
16 x 256 store at 200 cycles read 1.84 against 1.80 us a launch on a
performance core and 2.73 against 2.43 us on an efficiency core (medians of
20 alternating runs; the differences per pair of runs spread from -0.7 to
+1.1 us), and over 100 fresh processes on each core the median difference
between the two was 0.02 us or less. Once a CUPTI batch has run in the
process, a launch takes more, because a stopped CUPTI window leaves host cost
on every later launch of the process: about 0.3 us more on a performance core
(1.98 against 1.72 us) and about 0.9 us more on an efficiency core (3.48
against 2.57 us). Kernels longer than the launch interval (walkthroughs 10
and 19's SAXPY over 1 Mi floats, a 1 ms spin) read within 1.5% of their time
with the collector stood down on either core (medians over the rounds), as
long as the host keeps the GPU fed: on a busy core the launch interval grows,
and so does the set of kernels that host time reaches. Every figure here is a
median of runs pinned to one CPU, a process each, the settings alternating;
`GpuCollectorCost`
([`GpuCollectorCost_pTest.cu`](../ptst/gpu/GpuCollectorCost_pTest.cu)) takes
them on another machine. For the shortest kernels, compare runs made
the same way, on the same kind of core, or set `VERNIER_DISABLE_CUPTI=1`.

Removes the need to spawn `ncu` as an external process for the metrics it
captures (which is fragile inside container PID namespaces). For the
metrics CUPTI Activity doesn't expose (achieved occupancy, warp efficiency,
cache hit rates), use `bench run --profile ncu --profile-args replay`, or name
the sections and metrics ncu collects (`--profile-args '--set full'`). Under `--capture-um` the same collector also records
CUPTI's unified-memory counters over each measured window, in a window of the
counters alone ([Unified-Memory Counters](#unified-memory-counters)).

CUPTI allows a single client per process, so this in-process collector and an
nsys or ncu session cannot both hold it. The collector stands down by itself
when one of them started the process (`bench run` says so, and a session typed
by hand is recognized from the variables each tool exports), and its cells stay
empty. `VERNIER_DISABLE_CUPTI=1` turns it off in any run.

---

## Result Validation

### Using GPU Validation Macros

```cpp
PERF_GPU_TEST(MyKernel, Validated) {
  PERF_GPU_GUARD(perf);

  // ... setup and measurement ...

  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "my_kernel")
    .withLaunchConfig(grid, block)
    .measure();

  // 1. Validate result stability
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  // 2. No occupancy threshold: the estimate says what the launch shape
  //    allows, not how busy the GPU was or what limits the kernel (see
  //    Occupancy). A test tuned for a launch shape can require the estimate
  //    that shape gave, so that a changed shape shows

  // 3. No SM-clock drop. isThrottling() compares NVML's SM-clock readings
  //    at the window's start and end: a drop says the run's conditions
  //    changed, not why (see Troubleshooting). A reading NVML did not
  //    report is 0, so the check needs the start, end and maximum clocks
  const auto& clocks = result.stats.clocks;
  if (clocks.smClockMHzStart > 0 && clocks.smClockMHzEnd > 0 && clocks.boostClockMHz > 0) {
    EXPECT_FALSE(clocks.isThrottling())
      << "SM clock fell more than 10% during the measurement - find out why";
  }

  // 4. No copy-rate check: this test declares no copies, so bandwidthGBs()
  //    is 0. A test that declares copies can require a floor measured for
  //    those copies on its own system's link (see Memory Bandwidth)

  // 5. Validate speedup (if CPU baseline measured)
  if (result.speedupVsCpu > 0) {
    EXPECT_GT(result.speedupVsCpu, 1.0)
      << "GPU should be faster than CPU";
  }

  // 6. No transfer check either: with no copies declared, transferTimeUs
  //    is 0. A test that declares copies can hold transferTimeUs to a
  //    budget set for its workload; its ratio to totalTimeUs is not a share
  //    of one round trip (see Transfer Overhead)
}
```

### Common Validation Patterns

| Check            | Macro                      | Typical Threshold     | Purpose           |
| ---------------- | -------------------------- | --------------------- | ----------------- |
| Result stability | `EXPECT_STABLE_CV_GPU`     | Auto-adaptive         | Detect jitter     |
| Occupancy est.   | `EXPECT_GE(occupancy, X)`  | Tuned shape's value   | Launch shape kept |
| Copy rate        | `EXPECT_GT(bandwidth, X)`  | Measured for the link | Declared copies   |
| No SM-clock drop | `EXPECT_FALSE(throttling)` | With all 3 readings   | Flag a clock drop |
| Speedup          | `EXPECT_GT(speedup, X)`    | Problem-dependent     | GPU worthwhile    |
| Copy time        | `EXPECT_LT(transferUs, X)` | Budget for the copies | Copies in budget  |

---

## Best Practices

### 1. Always Warmup GPU Kernels

**Why:** First kernel launch includes JIT compilation (~50-200ms overhead).

```cpp
// ALWAYS do this before measuring
perf.cudaWarmup([&](cudaStream_t s) {
  myKernel<<<grid, block, 0, s>>>(d_data, N);
});
```

### 2. Use Descriptive Labels

**Why:** Distinguish kernels in CSV and analysis tools.

```cpp
// Bad - all get same "cuda_kernel" label
auto result1 = perf.cudaKernel(...).measure();
auto result2 = perf.cudaKernel(...).measure();

// Good - unique labels
auto baseline = perf.cudaKernel(..., "naive").measure();
auto optimized = perf.cudaKernel(..., "optimized_v2").measure();
```

### 3. Always Specify Launch Configuration

**Why:** The harness computes the occupancy estimate only for a declared
launch configuration; without one, the CSV's `occupancy` cell stays empty.

```cpp
// Missing occupancy data
auto result = perf.cudaKernel(...).measure();

// Includes occupancy in results
auto result = perf.cudaKernel(...)
  .withLaunchConfig(grid, block)  // Add this!
  .measure();
```

### 4. Test Multiple Block Sizes

**Why:** Optimal block size varies by kernel and GPU architecture.

```cpp
for (int threads : {128, 256, 512, 1024}) {
  // Test each configuration
}
```

### 5. Measure CPU Baseline When Relevant

**Why:** Justifies GPU port and shows speedup.

```cpp
auto cpuResult = perf.cpuBaseline(...);  // Do this first
auto gpuResult = perf.cudaKernel(...).measure();  // Then GPU
// gpuResult.speedupVsCpu now populated
```

### 6. Validate Results

**Why:** Catch performance regressions and instability.

```cpp
EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());
const auto& clocks = result.stats.clocks;  // 0 where NVML gave no reading
if (clocks.smClockMHzStart > 0 && clocks.smClockMHzEnd > 0 && clocks.boostClockMHz > 0) {
  EXPECT_FALSE(clocks.isThrottling());
}
```

### 7. Use Quick Mode During Development

```bash
# Fast iteration
./test --quick --csv dev.csv

# Production benchmarks
./test --cycles 100000 --repeats 30 --csv prod.csv
```

### 8. Profile Selectively

**Why:** a profiler costs time, from none seen under nsys to about a second
a launch under ncu (the cost column of
[Available Profilers](ADVANCED_GUIDE.md#available-profilers)) - use it for
specific investigations.

```bash
# Profile just one test: nsys records only a process it starts, so bench run
# starts the benchmark under it
bench run ./test --profile nsight -- --gtest_filter="*MyKernel.Optimized"
```

---

## Troubleshooting

### High CV% (Unstable Results)

**Symptoms:**

- CV% > 10%
- Large variation between runs

**Causes and solutions:**

1. **GPU frequency scaling:**

   ```bash
   # Check current clock state
   bench gpu-env

   # Lock GPU clocks for stable benchmarks (auto-detects max frequency)
   bench gpu-lock lock

   # Or wrap a benchmark run (auto-resets on exit)
   bench gpu-lock lock -- ./bin/ptests/BenchmarkGPU_PTEST --csv results.csv

   # Reset when done
   bench gpu-lock reset
   ```

2. **The SM clock fell during the run:**

   `isThrottling()` (the CSV's `throttling`) compares two SM-clock readings,
   NVML's at the start and the end of the measured window. It reads no
   temperature and no reason for the change, so a drop is a signal to
   investigate, not a diagnosis of heat, and no drop does not show the clock
   held between the two readings. The CSV cell is filled only when NVML
   reported the SM clock at both ends and the maximum SM clock; otherwise it
   stays empty and the run names the readings NVML did not report. The
   public `isThrottling()` reads `result.stats.clocks`, where such a reading
   is 0: it returns false when the start or the maximum is missing, and true
   when only the end is (a fall to 0), so read it only when
   `smClockMHzStart`, `smClockMHzEnd` and `boostClockMHz` are all above 0.
   Snapshots before and after the run record the temperature, power, clocks
   and the clock-event reasons `nvidia-smi` reports at each moment:

   ```bash
   # Snapshot before and after: what changed around the run
   bench gpu-monitor snapshot -o before.json
   ./bin/ptests/BenchmarkGPU_PTEST --csv results.csv
   bench gpu-monitor snapshot -o after.json
   bench gpu-monitor diff before.json after.json
   ```

   ```cpp
   const auto& clocks = result.stats.clocks;
   if (clocks.smClockMHzStart > 0 && clocks.smClockMHzEnd > 0 && clocks.boostClockMHz > 0 &&
       clocks.isThrottling()) {
     // The SM clock fell more than 10% over the window: compare the
     // snapshots and the row's temperature and power cells before trusting
     // the timing
   }
   ```

3. **Too few warmup iterations:**

   ```bash
   ./test --gpu-warmup 20  # Increase from default 10
   ```

4. **Background GPU processes:**

   ```bash
   # Check GPU state including memory usage and utilization
   bench gpu-monitor snapshot
   # Or use nvidia-smi directly
   nvidia-smi
   ```

### Low Occupancy

**Symptoms:**

- A low occupancy estimate (the CSV's `occupancy`)
- A kernel profiler measures a lower achieved occupancy than the launch shape
  allows

The estimate counts only the block size and the dynamic shared memory a
launch declares, so a low estimate comes from one of those (causes 2 and 3);
registers and static shared memory lower what a kernel can reach without
changing the estimate. Neither figure is a target of its own: the launch
shape to keep is the one with the best measured time
([Understanding Occupancy](#understanding-occupancy)).

**Causes and solutions:**

1. **Too many registers:**

   - Simplify kernel logic
   - Use fewer local variables
   - Check with: the CSV's `cuptiRegistersMedian` and `cuptiRegistersMax`

2. **Too much shared memory:**

   - Reduce `__shared__` allocations
   - Tile data differently
   - Check with: the CSV's `cuptiStaticSmemBytes` and `cuptiDynamicSmemBytes`

3. **Block size too small:**
   - Try larger block sizes (256, 512)
   - Test multiple configurations

### Poor Speedup vs CPU

**Symptoms:**

- speedupVsCpu < 5x
- GPU slower than expected

**Causes:**

1. **The copies take most of the median round trip:**

   `speedupVsCpu` divides by `totalTimeUs`, the median round trip, which
   includes the declared copies. A `transferTimeUs / totalTimeUs` near 100%
   shows that the copies take most of the median round trip only while the
   copy times hold steady; an Nsight Systems timeline shows each copy's time
   ([Transfer Overhead](#transfer-overhead)). If the copies dominate:

   - Keep data on GPU longer
   - Or use Unified Memory
   - Or overlap transfers with compute

2. **Not enough parallelism:**

   - Need at least 10,000+ threads for good GPU utilization
   - Consider if workload is GPU-suitable

3. **Memory-bound kernel:**
   - Measure the kernel's memory throughput with a kernel profiler
     ([Profiling GPU Code](#profiling-gpu-code)); `memBandwidthGBs` is the
     host-device copy rate, not the kernel's memory traffic
   - Optimize memory access patterns

### CUDA Errors

**Common errors:**

```cpp
// Always check CUDA errors
cudaError_t err = cudaMalloc(&d_data, SIZE);
if (err != cudaSuccess) {
  std::fprintf(stderr, "cudaMalloc failed: %s\n",
               cudaGetErrorString(err));
}

// Or use error-checking wrapper
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

CUDA_CHECK(cudaMalloc(&d_data, SIZE));
```

**A launch the runtime refuses fails the measurement.** A launch with a
block larger than the device allows, or more shared memory than it has, does
not run, and the runtime reports it only through `cudaGetLastError()`. The
harness reads it after each batch of launches: `measure()` and `cudaWarmup()`
throw `std::runtime_error` naming the test and the error, and no row is
published. A measurement also refuses to start while an earlier CUDA call's
error is waiting in `cudaGetLastError()`, and names it; read that error where
the call was made. A multi-GPU measurement rethrows the first device's error
once every device's thread has ended. A measurement and `cudaWarmup()` make the
case's device current while they launch, and the caller's device current
again afterwards.

---

## See Also

- **[CLI Tools README](../../../tools/README.md)** - Full `bench` tool reference (`gpu-env`, `gpu-lock`, `gpu-monitor`)
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking guide
- **[Advanced Guide](ADVANCED_GUIDE.md)** - Deep dives and advanced patterns
- **[Demos](../demo/docs/)** - Interactive GPU demos (10-12) with step-by-step walkthroughs
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Main README](../../../README.md)** - Framework overview
