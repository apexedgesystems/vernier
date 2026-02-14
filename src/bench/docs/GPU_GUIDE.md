# GPU Benchmarking Guide

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
  UB_PERF_GPU_GUARD(perf);

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
    .withLaunchConfig(grid, block)  // Enables occupancy tracking
    .measure();

  // Results printed automatically
  std::printf("Kernel time: %.3f ms\n", result.kernelTimeUs / 1000.0);
  std::printf("Transfer time: %.3f ms\n", result.transferTimeUs / 1000.0);
  std::printf("Occupancy: %.1f%%\n",
              result.stats.occupancy.achievedOccupancy * 100);

  // Validate
  EXPECT_GT(result.callsPerSecond, 1000);
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

PERF_MAIN()
```

### Build and Run

```bash
# Build (CUDA automatically detected)
cmake -B build -S . && cmake --build build

# Run
./build/native-linux-debug/bin/ptests/VectorAdd_GPU_PTEST --csv results.csv

# Quick mode during development
./VectorAdd_GPU_PTEST --quick

# With Nsight profiling
./VectorAdd_GPU_PTEST --profile nsight --gtest_filter="*Basic"

# Specific GPU device
./VectorAdd_GPU_PTEST --gpu-device 1 --csv results.csv
```

---

## Understanding GPU Metrics

### What Gets Measured?

GPU benchmarks measure several key metrics automatically:

**Timing:**

- **kernelTimeUs** - Pure kernel execution (CUDA events)
- **transferTimeUs** - H2D + D2H combined
- **totalTimeUs** - Kernel + transfers + overhead

**Performance:**

- **callsPerSecond** - Throughput (1e6 / totalTimeUs)
- **speedupVsCpu** - GPU vs CPU baseline comparison
- **memBandwidthGBs** - Achieved memory bandwidth

**Resource Utilization:**

- **achievedOccupancy** - Fraction of GPU utilized [0-1]
- **smClockMHz** - GPU frequency during test
- **throttling** - Whether GPU thermally throttled

### PerfGpuResult Structure

```cpp
struct PerfGpuResult {
  GpuStats stats;             // Comprehensive GPU metrics (see below)
  double kernelTimeUs;        // CUDA event kernel time
  double transferTimeUs;      // H2D + D2H transfer time
  double totalTimeUs;         // Kernel + transfers
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

// GPU metrics (always populated in GpuStats)
std::printf("Occupancy: %.1f%%\n", result.stats.occupancy.achievedOccupancy * 100);
std::printf("SM Clock: %d MHz\n", result.stats.clocks.smClockMHzEnd);
if (result.stats.clocks.isThrottling()) {
  std::printf("Warning: GPU is thermally throttling!\n");
}
std::printf("Bandwidth: %.2f GB/s\n", result.stats.transfers.bandwidthGBs());
```

### GpuStats Structure

```cpp
struct GpuStats {
  Stats cpuStats{};                                 // CPU timing stats
  GpuDeviceInfo deviceInfo{};                       // GPU model, compute capability
  MemoryTransferProfile transfers{};                // H2D/D2H bytes and timing
  OccupancyMetrics occupancy{};                     // Warps, block size, achieved occupancy
  ClockSpeedProfile clocks{};                       // SM/memory clocks, throttling detection
  double kernelTimeMedianUs{};                      // Kernel execution time
  double transferTimeMedianUs{};                    // Transfer time
  double totalTimeMedianUs{};                       // Total time
  std::optional<MultiGpuMetrics> multiGpu{};        // Only in cudaKernelMultiGpu()
  std::optional<P2PTransferProfile> p2pProfile{};   // Only with P2P transfers
  std::optional<UnifiedMemoryProfile> unifiedMemory{}; // Only with managed memory
};
```

**Field availability:**

- **transfers, occupancy, clocks**: Always populated (values may be zero/default)
- **multiGpu**: Only when using `cudaKernelMultiGpu()`
- **p2pProfile**: Only when P2P transfers are measured
- **unifiedMemory**: Only when using managed memory

### Key Metrics Explained

#### Occupancy

**What it is:** Fraction of GPU compute resources actively used.

**Formula:** `Active warps / Max possible warps per SM`

**Interpretation:**

- **>75%** = Excellent, GPU well-utilized
- **50-75%** = Good, some optimization possible
- **25-50%** = Fair, likely resource-constrained
- **<25%** = Poor, need to investigate

**Important:** High occupancy doesn't always mean fast!

- Memory-bound kernels: 25% occupancy can be optimal
- Compute-bound kernels: aim for >50%

#### Transfer Overhead

**What it is:** Percentage of time spent on memory transfers.

**Formula:** `(transferTime / totalTime) * 100%`

**Interpretation:**

- **<10%** = Excellent, compute-bound kernel
- **10-30%** = Good, transfers are minor cost
- **30-50%** = Fair, consider reducing transfers
- **>50%** = Poor, transfer-bound workload

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

**What it is:** Data transfer rate achieved by kernel.

**Formula:** `(bytes_read + bytes_written) / kernel_time`

**Interpretation:**

- Compare to theoretical peak (e.g., A100 = 1555 GB/s, RTX 4090 = 1008 GB/s)
- **>80% of peak** = Memory-bound, excellent utilization
- **50-80% of peak** = Good bandwidth usage
- **<50% of peak** = Compute-bound or inefficient access

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

UB_PERF_GPU_GUARD(varName)           // Create scoped PerfGpuCase
PERF_MAIN()                          // Main function with GPU support
```

### Basic Kernel Test

```cpp
PERF_GPU_TEST(MyKernel, Basic) {
  UB_PERF_GPU_GUARD(perf);

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

### Comparing Implementations

```cpp
PERF_GPU_TEST(MyAlgorithm, CompareVersions) {
  UB_PERF_GPU_GUARD(perf);

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

  std::printf("Naive:     %.3f ms, Occupancy: %.1f%%\n",
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

  std::printf("Optimized: %.3f ms, Occupancy: %.1f%%\n",
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
2. Check occupancy in results
3. If occupancy <50%, try other sizes
4. Benchmark multiple configurations

### Block Size Sweep

```cpp
PERF_GPU_TEST(MyKernel, BlockSizeSweep) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  std::printf("\n%-12s %-12s %-12s %-12s\n",
              "Threads", "Time(ms)", "Occupancy", "Throughput");
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

    double occupancy = result.stats.occupancy.achievedOccupancy * 100;

    std::printf("%-12d %-12.3f %-12.1f%% %-12.0f\n",
                threads,
                result.kernelTimeUs / 1000.0,
                occupancy,
                result.callsPerSecond);
  }

  cudaFree(d_data);
}
```

**Typical output:**

```
Threads      Time(ms)     Occupancy    Throughput
--------------------------------------------------
128          15.234       32.5%        65600
256          12.456       67.8%        80300     <- Best
512          13.890       45.2%        72000
1024         18.123       28.9%        55200
```

### Understanding Occupancy

**Occupancy** = Active warps / Maximum possible warps per SM

**What limits occupancy:**

- **Registers per thread** - More registers = fewer active threads
- **Shared memory per block** - More shared mem = fewer active blocks
- **Thread block size** - Too small/large reduces occupancy

**Occupancy targets:**

- **>66%**: Excellent (compute-bound kernels)
- **33-66%**: Good (typical)
- **<33%**: Fair (may be resource-constrained)

**Important notes:**

- High occupancy != always faster
- Memory-bound kernels: 25-50% occupancy can be optimal
- Compute-bound kernels: aim for >50% occupancy
- Focus on wall-clock time, not just occupancy

**Use NVIDIA's occupancy calculator:**

```bash
# Command-line tool
occupancy_calculator --sm 80 --threads 256 --registers 32 --shared 4096

# Or use Excel/spreadsheet version from CUDA toolkit
```

---

## CPU vs GPU Comparison

### Measuring Speedup

Use `cpuBaseline()` to measure GPU speedup over CPU:

```cpp
PERF_GPU_TEST(MyAlgorithm, CpuVsGpu) {
  UB_PERF_GPU_GUARD(perf);

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

**CSV output:**

```
test,cpuMedianUs,gpuTotalUs,speedupVsCpu
MyAlgorithm.CpuVsGpu/cpu_version,1234.5,,,
MyAlgorithm.CpuVsGpu/gpu_version,,156.7,7.87
```

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
  UB_PERF_GPU_GUARD(perf);

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

  // Aggregated metrics
  std::printf("\nScaling efficiency: %.1f%%\n",
              result.aggregatedStats.multiGpu->scalingEfficiency * 100);
  std::printf("Total speedup: %.2fx vs single GPU\n",
              result.totalSpeedupVsCpu);

  // Check scaling efficiency
  EXPECT_GT(result.aggregatedStats.multiGpu->scalingEfficiency, 0.8)
    << "Poor scaling with " << DEVICE_COUNT << " GPUs";

  // Cleanup
  for (int i = 0; i < DEVICE_COUNT; ++i) {
    cudaSetDevice(i);
    cudaFree(d_data[i]);
  }
}
```

### Understanding Scaling Efficiency

Scaling efficiency measures how well your code utilizes multiple GPUs:

```
scalingEfficiency = (speedup / num_devices) * 100%

where speedup = single_gpu_time / multi_gpu_time
```

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
  UB_PERF_GPU_GUARD(perf);

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
| `multiGpuEfficiency` | double | Scaling efficiency [0-1]             |
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

Automatic memory migration between CPU and GPU:

```cpp
// Unified Memory - same pointer on CPU and GPU
float* data;
cudaMallocManaged(&data, SIZE);
kernel<<<...>>>(data);  // Automatic migration!
cudaFree(data);
```

**Testing with Unified Memory:**

```bash
# Run with UM
./MyKernel_GPU_PTEST --gpu-memory unified --csv um_results.csv

# Compare strategies
./MyKernel_GPU_PTEST --gpu-memory explicit --csv explicit.csv
./MyKernel_GPU_PTEST --gpu-memory unified --csv unified.csv
bench compare explicit.csv unified.csv
```

### Understanding UM Metrics

Unified Memory CSV columns:

| Column              | Meaning             | Good Value    | Bad Value     |
| ------------------- | ------------------- | ------------- | ------------- |
| `umPageFaults`      | GPU page faults     | <1000         | >10000        |
| `umH2DMigrations`   | Host->device moves  | Low           | High          |
| `umD2HMigrations`   | Device->host moves  | Low           | High          |
| `umMigrationTimeUs` | Migration overhead  | <10% of total | >50% of total |
| `umThrashing`       | Memory ping-ponging | false         | true          |

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

**Example comparison:**

```cpp
PERF_GPU_TEST(MemoryStrategy, Compare) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 1024 * 1024;
  std::vector<float> h_data(N, 1.0f);

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  // Test explicit memory
  float* d_data;
  cudaMalloc(&d_data, N * sizeof(float));

  auto explicit_result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "explicit_memory")
    .withHostToDevice(h_data.data(), d_data, N * sizeof(float))
    .withDeviceToHost(d_data, h_data.data(), N * sizeof(float))
    .withLaunchConfig(grid, block)
    .measure();

  std::printf("Explicit:  kernel=%.2f ms, transfers=%.2f ms\n",
              explicit_result.kernelTimeUs / 1000.0,
              explicit_result.transferTimeUs / 1000.0);

  cudaFree(d_data);

  // Test unified memory (if enabled with --gpu-memory unified)
  // Results will show umPageFaults, umMigrationTimeUs, etc.
}
```

**Typical output:**

```
Explicit:  kernel=123.4 ms, transfers=45.6 ms
Unified:   kernel=234.5 ms, umMigrationTimeUs=156.7 ms, umPageFaults=12450

Analysis: Explicit 2x faster due to UM page fault overhead
```

---

## Profiling GPU Code

### Using Nsight Compute

NVIDIA Nsight Compute provides detailed kernel profiling:

```bash
# Profile specific kernel
./MyKernel_GPU_PTEST --profile nsight --gtest_filter="*MyKernel"

# Generates: nsight-MyTest.MyKernel-TIMESTAMP.ncu-rep

# Analyze with Nsight UI
ncu-ui nsight-MyTest.MyKernel-*.ncu-rep

# Or command-line report
ncu --csv --print-summary per-kernel nsight-*.ncu-rep
```

**What Nsight shows:**

- Roofline analysis (compute vs memory bound)
- Warp execution efficiency
- Memory throughput breakdown
- Instruction mix
- Occupancy limiters

### GPU Profiler Integration

```cpp
PERF_GPU_TEST(MyKernel, WithProfiling) {
  UB_PERF_GPU_GUARD(perf);
  // Profiling automatically enabled if --profile flag passed

  // Setup and measurement as usual
  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "profiled_kernel")
    .withLaunchConfig(grid, block)
    .measure();

  // Profiler data saved automatically
  // Check CSV for profile_tool and profile_dir columns
}
```

---

## Result Validation

### Using GPU Validation Macros

```cpp
PERF_GPU_TEST(MyKernel, Validated) {
  UB_PERF_GPU_GUARD(perf);

  // ... setup and measurement ...

  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, N);
  }, "my_kernel")
    .withLaunchConfig(grid, block)
    .measure();

  // 1. Validate result stability
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  // 2. Validate occupancy
  EXPECT_GT(result.stats.occupancy.achievedOccupancy, 0.5)
    << "Low occupancy (<50%) may indicate resource constraints";

  // 3. Validate no thermal throttling
  EXPECT_FALSE(result.stats.clocks.isThrottling())
    << "GPU is thermally throttling - results may be unstable";

  // 4. Validate bandwidth utilization
  double bw = result.stats.transfers.bandwidthGBs();
  EXPECT_GT(bw, 100.0) << "Memory bandwidth suspiciously low";

  // 5. Validate speedup (if CPU baseline measured)
  if (result.speedupVsCpu > 0) {
    EXPECT_GT(result.speedupVsCpu, 1.0)
      << "GPU should be faster than CPU";
  }

  // 6. Validate transfer overhead
  double transferPct = (result.transferTimeUs / result.totalTimeUs) * 100;
  EXPECT_LT(transferPct, 50.0)
    << "Transfer overhead >50% - consider keeping data on GPU";
}
```

### Common Validation Patterns

| Check             | Macro                       | Typical Threshold     | Purpose            |
| ----------------- | --------------------------- | --------------------- | ------------------ |
| Result stability  | `EXPECT_STABLE_CV_GPU`      | Auto-adaptive         | Detect jitter      |
| Occupancy         | `EXPECT_GT(occupancy, X)`   | >0.5 compute-bound    | Resource usage     |
| Bandwidth         | `EXPECT_GT(bandwidth, X)`   | >100 GB/s modern GPUs | Memory efficiency  |
| No throttling     | `EXPECT_FALSE(throttling)`  | Always                | Thermal stability  |
| Speedup           | `EXPECT_GT(speedup, X)`     | Problem-dependent     | GPU worthwhile     |
| Transfer overhead | `EXPECT_LT(transferPct, X)` | <50%                  | Not transfer-bound |

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

**Why:** Enables occupancy tracking and validation.

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
EXPECT_GT(result.stats.occupancy.achievedOccupancy, 0.5);
EXPECT_FALSE(result.stats.clocks.isThrottling());
```

### 7. Use Quick Mode During Development

```bash
# Fast iteration
./test --quick --csv dev.csv

# Production benchmarks
./test --cycles 100000 --repeats 30 --csv prod.csv
```

### 8. Profile Selectively

**Why:** Profiling adds overhead - use for specific investigations.

```bash
# Profile just one test
./test --profile nsight --gtest_filter="*MyKernel.Optimized"
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
   # Lock GPU clocks (requires root)
   sudo nvidia-smi -lgc 1410  # Lock to max clock
   ```

2. **Thermal throttling:**

   ```cpp
   if (result.stats.clocks.isThrottling()) {
     // Improve cooling, reduce load, or increase fan speed
   }
   ```

3. **Too few warmup iterations:**

   ```bash
   ./test --gpu-warmup 20  # Increase from default 10
   ```

4. **Background GPU processes:**

   ```bash
   nvidia-smi  # Check for other processes
   # Kill interfering processes
   ```

### Low Occupancy

**Symptoms:**

- Occupancy < 50%
- Kernel runs slowly

**Causes and solutions:**

1. **Too many registers:**

   - Simplify kernel logic
   - Use fewer local variables
   - Check with: `cuobjdump -sass kernel.o | grep regs`

2. **Too much shared memory:**

   - Reduce `__shared__` allocations
   - Tile data differently
   - Check with occupancy calculator

3. **Block size too small:**
   - Try larger block sizes (256, 512)
   - Test multiple configurations

### Poor Speedup vs CPU

**Symptoms:**

- speedupVsCpu < 5x
- GPU slower than expected

**Causes:**

1. **Transfer overhead dominant:**

   ```cpp
   double transferPct = (transferTimeUs / totalTimeUs) * 100;
   if (transferPct > 50%) {
     // Keep data on GPU longer
     // Or use Unified Memory
     // Or overlap transfers with compute
   }
   ```

2. **Not enough parallelism:**

   - Need at least 10,000+ threads for good GPU utilization
   - Consider if workload is GPU-suitable

3. **Memory-bound kernel:**
   - Check bandwidth utilization
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

---

## See Also

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking guide
- **[Advanced Guide](ADVANCED_GUIDE.md)** - Deep dives and advanced patterns
- **[Demos](../demo/docs/)** - Interactive GPU demos (10-12) with step-by-step walkthroughs
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Main README](../../../README.md)** - Framework overview
