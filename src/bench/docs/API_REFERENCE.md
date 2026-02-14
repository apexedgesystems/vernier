# API Reference

Complete API documentation for the benchmarking framework. This reference covers all CPU and GPU APIs, profiling integration, and configuration options.

---

## Table of Contents

- [Quick Start](#quick-start)
- [CPU API](#cpu-api)
- [GPU API](#gpu-api)
- [Profiler API](#profiler-api)
- [CSV Schema](#csv-schema)
- [Configuration Flags](#configuration-flags)
- [Helper Functions](#helper-functions)

---

## Quick Start

```cpp
#include <gtest/gtest.h>
#include "Perf.hpp"

namespace ub = vernier::bench;

PERF_TEST(MyComponent, Throughput) {
  UB_PERF_GUARD(perf);  // Creates PerfCase with config from command-line

  // Setup test data
  std::vector<int> data(10000, 42);

  // Warmup
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = processData(data);
      (void)result;
    }
  });

  // Measure
  volatile int sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += processData(data);
  }, "throughput");

  // Validate
  EXPECT_GT(result.callsPerSecond, 10000) << "Throughput too low";
  EXPECT_STABLE_CV_CPU(result, perf.config());
}

PERF_MAIN()  // Standard main() with CSV export
```

**Note:** See [demo/](../demo/) for complete working examples with step-by-step walkthroughs.

---

## CPU API

### Test Macros

```cpp
PERF_TEST(SuiteName, TestName)           // General performance test
PERF_THROUGHPUT(SuiteName, TestName)     // Throughput measurement
PERF_LATENCY(SuiteName, TestName)        // Latency distribution
PERF_CONTENTION(SuiteName, TestName)     // Multi-threaded test

UB_PERF_GUARD(varName)                   // Create scoped PerfCase
PERF_MAIN()                              // Main function with CSV export
```

### PerfCase

Core CPU benchmark harness.

**Constructor:**

```cpp
PerfCase(std::string testName, PerfConfig cfg);
```

**Key methods:**

```cpp
// Warmup (run before measurements to prime caches)
void warmup(std::function<void()> fn);

// Measurement functions
PerfResult throughputLoop(std::function<void()> op,
                         std::string label = "throughput",
                         std::optional<MemoryProfile> memProfile = std::nullopt);

PerfResult measured(std::function<void()> fn,
                   std::string label = "measured");

PerfResult contentionRun(const std::function<void()>& worker,
                        std::string label = "contention");

// Hooks (optional)
void setBeforeMeasureHook(std::function<void(const PerfCase&)> hook);
void setAfterMeasureHook(std::function<void(const PerfCase&, const Stats&)> hook);

// Accessors
int cycles() const noexcept;
int repeats() const noexcept;
int threads() const noexcept;
int warmup() const noexcept;
const PerfConfig& config() const noexcept;
const std::string& testName() const noexcept;
```

**Method semantics:**

**`throughputLoop(op, label, memProfile)`**

- Framework calls `op()` exactly `cycles()` times
- Use for: Per-operation measurements
- MemoryProfile: Specify per-operation bytes

**`measured(fn, label)`**

- Framework calls `fn()` once (you do the loop inside)
- Use for: Batch measurements, custom loop logic
- Your lambda should perform exactly `cycles()` operations

**`contentionRun(worker, label)`**

- Framework spawns `threads()` workers, each calling `worker()` in a loop
- Use for: Multi-threaded scalability testing
- Each worker performs `cycles()` operations

### PerfResult

```cpp
struct PerfResult {
  Stats stats{};                            // Statistical summary
  double callsPerSecond{};                  // Throughput (1e6 / median_us)
  std::string label;                        // Section label
  std::optional<MemoryProfile> memoryProfile{}; // Stored input profile (if provided)
  bool stable{true};                        // CV% below adaptive threshold
  double cvThreshold{0.05};                 // Threshold used (from recommendedCVThreshold)
};
```

**Note:** `memoryProfile` contains the input profile you passed (if any). Framework automatically prints bandwidth analysis - you don't need to access this field directly.

### Stats

```cpp
struct Stats {
  double median;   // 50th percentile (us per call)
  double p10;      // 10th percentile
  double p90;      // 90th percentile
  double min;      // Minimum latency
  double max;      // Maximum latency
  double mean;     // Arithmetic mean
  double stddev;   // Standard deviation
  double cv;       // Coefficient of variation (stddev/mean)
};
```

**CV% interpretation:**

- `< 0.05` (5%): Excellent stability
- `0.05 - 0.10` (5-10%): Good
- `> 0.10` (>10%): High jitter - investigate

### PerfConfig

```cpp
struct PerfConfig {
  int cycles = 10000;               // Operations per repeat
  int repeats = 10;                 // Samples collected
  int warmup = 1;                   // Warmup repeats (0 = auto-scale)
  int threads = 1;                  // Worker threads
  int msgBytes = 64;                // Payload size (bytes)
  bool console = false;             // Echo to console (when relevant)
  bool nonBlocking = false;         // Non-blocking mode (when relevant)
  std::string minLevel = "INFO";    // DEBUG|INFO|WARNING|ERROR|FATAL
  std::optional<std::string> csv{}; // CSV output path

  // Profiling knobs (default off)
  std::string profileTool;             // "perf", "gperf", "bpftrace", "rapl", "callgrind"
  std::string profileArgs;             // Verbatim pass-through to the tool
  std::vector<std::string> bpfScripts; // Curated script names: "offcpu", "syslat", "bio"
  std::string artifactRoot;            // Profiler artifact directory
  int profileFrequency = 10000;        // Sampling Hz for CPU profilers
  bool profileAnalyze = false;         // Auto-run analysis after profiling

  bool quickMode = false;              // Apply reduced cycles/repeats
};
```

### MemoryProfile

```cpp
struct MemoryProfile {
  size_t bytesRead = 0;      // Input data size per operation (bytes)
  size_t bytesWritten = 0;   // Output data size per operation (bytes)
  size_t bytesAllocated = 0; // Heap allocations per operation (optional)

  // Helper methods
  [[nodiscard]] double bandwidthMBs(double durationUs) const;
  [[nodiscard]] double efficiency(double durationUs, double peakMBs) const;
};
```

**Usage with `throughputLoop` (per-operation bytes):**

```cpp
// Specify bytes accessed PER OPERATION
ub::MemoryProfile memProfile{
  .bytesRead = 1024,     // Each call reads 1KB
  .bytesWritten = 1024,  // Each call writes 1KB
  .bytesAllocated = 0
};

auto result = perf.throughputLoop([&]{
  processData();  // Framework calls this cycles() times
}, "throughput", memProfile);

// Framework automatically prints:
// - Memory bandwidth (MB/s)
// - Estimated efficiency (% of theoretical peak)
// - Hints (CPU-bound vs memory-bound)
```

**Currently supported by:** `throughputLoop()` only

**Output example:**

```
Memory bandwidth: 9403.3 MB/s (1.0 MB read, 1.0 MB written per call)
Estimated efficiency: 78.4% of theoretical peak (~12000 MB/s)
Hint: High bandwidth utilization -> Memory-bound (consider memory layout)
```

**Interpretation:**

- **Efficiency < 10%:** CPU-bound (algorithm optimization needed)
- **Efficiency > 50%:** Memory-bound (memory layout optimization needed)

### Profiler Integration

```cpp
void attachProfilerHooks(PerfCase& perf, const PerfConfig& cfg);
```

**Supported profilers:**

- `perf` - Linux perf_events (CPU profiling, hardware counters)
- `gperf` - gperftools (CPU/heap profiling)
- `bpftrace` - BPF tracing (off-CPU, syscalls)
- `rapl` - Intel RAPL (energy measurement)

**Example:**

```cpp
PERF_TEST(MyComponent, Throughput) {
  UB_PERF_GUARD(perf);
  ub::attachProfilerHooks(perf, ub::detail::getPerfConfig());

  perf.warmup([&]{ /* ... */ });
  auto result = perf.throughputLoop([&]{ /* ... */ }, "op");
}
```

**Command line:**

```bash
# CPU profiling with perf
./MyComponent_PTEST --profile perf --gtest_filter="*Throughput"

# Heap profiling with gperftools
./MyComponent_PTEST --profile gperf --gtest_filter="*Throughput"

# Energy measurement with RAPL
./MyComponent_PTEST --profile rapl --csv results.csv
```

### Validation Macros

```cpp
// Adaptive CV% validation (threshold based on payload size)
// Pass entire config object (not just msgBytes)
EXPECT_STABLE_CV_CPU(result, perf.config());

// Manual threshold
EXPECT_LT(result.stats.cv, 0.10) << "High jitter";
```

**Macro details:**

```cpp
EXPECT_STABLE_CV_CPU(result, cfg)
```

- **Parameters:**

  - `result`: `PerfResult` from `measured()` or `throughputLoop()`
  - `cfg`: `PerfConfig` object (must have `.msgBytes` and `.quickMode` members)

- **Examples:**

  ```cpp
  EXPECT_STABLE_CV_CPU(result, perf.config());  // Correct
  EXPECT_STABLE_CV_CPU(result, getCfg());       // Correct (if getCfg() returns PerfConfig)
  ```

- **Automatic thresholds:**
  - <64B: 20% (tiny payloads)
  - <256B: 10% (small payloads)
  - <1KB: 5% (medium payloads)
  - > =1KB: 3% (large payloads)
  - Quick mode: +50% relaxation (capped at 30%)

---

## GPU API

### Test Macros

```cpp
PERF_GPU_TEST(SuiteName, TestName)       // General GPU test
PERF_GPU_TEST(SuiteName, TestName)     // Kernel-focused test

UB_PERF_GPU_GUARD(varName)               // Create scoped GpuPerfCase
PERF_GPU_MAIN()                          // Main with GPU support
```

### GpuPerfCase

Core GPU benchmark harness.

**Key methods:**

```cpp
// Kernel builder (fluent interface)
GpuKernelBuilder cudaKernel(
  void (*kernel)(...),    // Kernel function pointer
  dim3 grid,              // Grid dimensions
  dim3 block,             // Block dimensions
  size_t sharedMem = 0    // Shared memory per block
);

// Memory transfer
GpuTransferBuilder transfer(
  void* dst, void* src, size_t bytes,
  cudaMemcpyKind kind
);

// Unified Memory testing
GpuUnifiedMemoryBuilder unifiedMemory(
  void* ptr, size_t bytes
);

// Accessors
int cycles() const noexcept;
int repeats() const noexcept;
int gpuWarmup() const noexcept;
const PerfConfig& cpuConfig() const noexcept;
const GpuPerfConfig& gpuConfig() const noexcept;
```

### GpuKernelBuilder

Fluent interface for kernel configuration and measurement.

```cpp
class GpuKernelBuilder {
public:
  // Configuration (optional)
  GpuKernelBuilder& withHostInput(const void* host, size_t bytes);
  GpuKernelBuilder& withDeviceInput(void* device, size_t bytes);
  GpuKernelBuilder& withHostOutput(void* host, size_t bytes);
  GpuKernelBuilder& withDeviceOutput(void* device, size_t bytes);
  GpuKernelBuilder& withPinnedAlloc();

  // Measurement
  GpuPerfResult measure(std::string label = "kernel");
};
```

**Usage pattern:**

```cpp
PERF_GPU_TEST(MyKernel, Throughput) {
  UB_PERF_GPU_GUARD(perf);

  // Setup
  std::vector<float> h_input(N);
  std::vector<float> h_output(N);

  // Measure kernel performance
  auto result = perf.cudaKernel(myKernel, grid, block)
    .withHostInput(h_input.data(), h_input.size() * sizeof(float))
    .withHostOutput(h_output.data(), h_output.size() * sizeof(float))
    .measure("myKernel");

  // Validate
  EXPECT_GT(result.stats.kernelGpuThroughput, 1e9) << "GPU throughput too low";
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());
}
```

### GpuPerfResult

```cpp
struct GpuPerfResult {
  GpuStats stats;           // Comprehensive GPU statistics
  std::string label;        // Kernel label
};

struct GpuStats {
  // CPU-side timing
  CpuTimingStats cpuStats;  // Wall-clock measurements

  // GPU-side timing
  double kernelTimeUs;      // CUDA event kernel time
  double transferTimeUs;    // H2D + D2H transfer time

  // Throughput
  double kernelGpuThroughput;  // Ops/sec based on GPU time
  double totalCpuThroughput;   // Ops/sec based on CPU time

  // Memory bandwidth
  double h2dBandwidthGBs;   // Host-to-device bandwidth
  double d2hBandwidthGBs;   // Device-to-host bandwidth

  // GPU metrics
  double achievedOccupancy; // Actual occupancy (0.0-1.0)
  double smClockMHz;        // SM clock frequency
  std::string gpuModel;     // Device name
  std::string computeCapability; // e.g., "8.6"
};
```

### Validation Macros

```cpp
// GPU CV% validation
EXPECT_STABLE_CV_GPU(gpuResult, perf.cpuConfig());

// Occupancy validation
EXPECT_GT(gpuResult.stats.achievedOccupancy, 0.5)
  << "Low occupancy - kernel may be register/shared memory bound";

// Bandwidth validation
EXPECT_GT(gpuResult.stats.h2dBandwidthGBs, 10.0)
  << "Low transfer bandwidth - check PCIe link";
```

---

## Profiler API

### Common Interface

All profilers implement:

```cpp
class Profiler {
public:
  virtual ~Profiler() = default;
  virtual std::string toolName() const noexcept = 0;
  virtual std::string artifactDir() const noexcept = 0;
  virtual void beforeMeasure() {}
  virtual void afterMeasure(const Stats& s) {}

  // Factory: returns a concrete profiler or no-op based on cfg.profileTool
  static std::unique_ptr<Profiler> make(const PerfConfig& cfg, const std::string& testName);
};
```

### ProfilerPerf

Linux `perf` integration for CPU profiling.

**Features:**

- Hardware performance counters
- Call graph sampling
- Cache miss analysis
- Branch prediction analysis

**Usage:**

```bash
./MyComponent_PTEST --profile perf --gtest_filter="*Throughput"
# Generates: perf-MyComponent.Throughput-TIMESTAMP.data
```

**Analysis:**

```bash
perf report -i perf-MyComponent.Throughput-*.data
perf annotate -i perf-MyComponent.Throughput-*.data
```

### ProfilerGperf

Google Performance Tools integration.

**Features:**

- CPU profiling (sampling)
- Heap profiling
- Contention profiling

**Usage:**

```bash
# CPU profiling
./MyComponent_PTEST --profile gperf --gtest_filter="*Throughput"

# Heap profiling
HEAPPROFILE=/tmp/heap ./MyComponent_PTEST --profile gperf
```

**Analysis:**

```bash
google-pprof --text ./MyComponent_PTEST cpu-profile.prof
google-pprof --pdf ./MyComponent_PTEST cpu-profile.prof > profile.pdf
```

### ProfilerBpftrace

BPF-based kernel tracing.

**Features:**

- Off-CPU analysis
- Syscall tracing
- Custom probe points

**Usage:**

```bash
# Requires root or CAP_BPF capability
sudo ./MyComponent_PTEST --profile bpftrace --profile-args "fsync_latency.bt"
```

### ProfilerRAPL

Intel RAPL energy measurement.

**Features:**

- Package energy consumption
- DRAM energy consumption
- Power efficiency analysis

**Usage:**

```bash
# Requires root for /dev/cpu/*/msr access
sudo ./MyComponent_PTEST --profile rapl --csv results.csv
```

### ProfilerNsight (GPU)

NVIDIA Nsight Compute integration.

**Features:**

- Kernel profiling
- Memory bandwidth analysis
- Occupancy analysis
- Roofline model

**Usage:**

```bash
# Requires NVIDIA driver with profiling support
./MyGpuTest_PTEST --profile nsight --gtest_filter="*Kernel"
```

---

## CSV Schema

### CPU Columns

| Column           | Type   | Description                  |
| ---------------- | ------ | ---------------------------- |
| `test`           | string | Suite.TestName               |
| `cycles`         | int    | Operations per repeat        |
| `repeats`        | int    | Number of repeats            |
| `warmup`         | int    | Warmup iterations used       |
| `threads`        | int    | Worker thread count          |
| `msgBytes`       | int    | Payload size (bytes)         |
| `console`        | bool   | Console output enabled       |
| `nonBlocking`    | bool   | Non-blocking mode enabled    |
| `minLevel`       | string | Minimum log level            |
| `wallMedian`     | double | Median latency (us)          |
| `wallP10`        | double | 10th percentile (us)         |
| `wallP90`        | double | 90th percentile (us)         |
| `wallMin`        | double | Minimum latency (us)         |
| `wallMax`        | double | Maximum latency (us)         |
| `wallMean`       | double | Mean latency (us)            |
| `wallStddev`     | double | Standard deviation (us)      |
| `wallCV`         | double | Coefficient of variation     |
| `callsPerSecond` | double | Throughput (ops/sec)         |
| `stable`         | bool   | CV below adaptive threshold  |
| `cvThreshold`    | double | Adaptive CV threshold used   |
| `profileTool`    | string | Profiler used (if any)       |
| `profileDir`     | string | Profiler artifact path       |
| `timestamp`      | string | ISO 8601 timestamp           |
| `gitHash`        | string | Short commit hash            |
| `hostname`       | string | Machine hostname             |
| `platform`       | string | Architecture (x86_64, arm64) |

### GPU Columns

Additional columns for GPU tests:

| Column              | Type   | Description                 |
| ------------------- | ------ | --------------------------- |
| `gpuModel`          | string | GPU device name             |
| `computeCapability` | string | CUDA compute capability     |
| `kernelTimeUs`      | double | GPU kernel time (us)        |
| `transferTimeUs`    | double | H2D + D2H time (us)         |
| `h2dBytes`          | int64  | Host-to-device bytes        |
| `d2hBytes`          | int64  | Device-to-host bytes        |
| `speedupVsCpu`      | double | GPU/CPU speedup ratio       |
| `memBandwidthGBs`   | double | Memory bandwidth (GB/s)     |
| `occupancy`         | double | Kernel occupancy [0.0-1.0]  |
| `sm_clock_MHz`      | double | SM clock frequency          |
| `throttling`        | bool   | Thermal throttling detected |

### Multi-GPU Columns

| Column                 | Type   | Description            |
| ---------------------- | ------ | ---------------------- |
| `device_id`            | int    | Primary CUDA device ID |
| `device_count`         | int    | Number of GPUs used    |
| `multi_gpu_efficiency` | double | Parallel efficiency    |
| `p2p_bandwidth_GBs`    | double | Peer-to-peer bandwidth |

### Unified Memory Columns

| Column                 | Type   | Description               |
| ---------------------- | ------ | ------------------------- |
| `um_page_faults`       | int64  | GPU page faults           |
| `um_h2d_migrations`    | int64  | Host-to-device migrations |
| `um_d2h_migrations`    | int64  | Device-to-host migrations |
| `um_migration_time_us` | double | Total migration overhead  |
| `um_thrashing`         | bool   | Thrashing detected        |

---

## Configuration Flags

### Core Flags

| Flag            | Type | Default | Description                           |
| --------------- | ---- | ------- | ------------------------------------- |
| `--cycles N`    | int  | 10000   | Operations per repeat                 |
| `--repeats N`   | int  | 10      | Number of measurement repeats         |
| `--warmup N`    | int  | 1       | Warmup iterations (0 = auto)          |
| `--threads N`   | int  | 1       | Worker thread count                   |
| `--msg-bytes N` | int  | 64      | Payload size (bytes)                  |
| `--quick`       | bool | false   | Fast iteration (fewer cycles/repeats) |

### Output Flags

| Flag              | Type   | Default  | Description                        |
| ----------------- | ------ | -------- | ---------------------------------- |
| `--csv PATH`      | string | -        | CSV output path                    |
| `--console`       | bool   | false    | Echo to console (when relevant)    |
| `--nonblocking`   | bool   | false    | Non-blocking mode (when relevant)  |
| `--min-level STR` | string | `"INFO"` | DEBUG\|INFO\|WARNING\|ERROR\|FATAL |

### Profiling Flags

| Flag                    | Type   | Default | Description                                      |
| ----------------------- | ------ | ------- | ------------------------------------------------ |
| `--profile TOOL`        | string | -       | Profiler: perf\|gperf\|bpftrace\|rapl\|callgrind |
| `--profile-args ARGS`   | string | -       | Profiler-specific arguments                      |
| `--artifact-root DIR`   | string | .       | Profiler output directory                        |
| `--profile-frequency N` | int    | 10000   | Sampling Hz for CPU profilers                    |
| `--profile-analyze`     | bool   | false   | Auto-run analysis after profiling                |
| `--bpf LIST`            | string | -       | BPF scripts (comma-separated): offcpu,syslat,bio |

### GPU-Specific Flags

| Flag                  | Type   | Default  | Description                                        |
| --------------------- | ------ | -------- | -------------------------------------------------- |
| `--gpu-device N`      | int    | 0        | CUDA device ID                                     |
| `--gpu-warmup N`      | int    | 10       | GPU warmup iterations                              |
| `--gpu-memory MODE`   | string | explicit | Memory strategy: explicit\|unified\|pinned\|mapped |
| `--min-speedup FLOAT` | double | 1.0      | Minimum expected GPU speedup                       |

### GoogleTest Integration

| Flag                     | Description                              |
| ------------------------ | ---------------------------------------- |
| `--gtest_filter=PATTERN` | Run specific tests (wildcards: `*`, `?`) |
| `--gtest_list_tests`     | List available tests                     |
| `--gtest_repeat=N`       | Repeat test suite N times                |

**Examples:**

```bash
# Run all tests with CSV export
./MyComponent_PTEST --csv results.csv

# Quick iteration during development
./MyComponent_PTEST --quick --gtest_filter="*Throughput"

# Multi-threaded test with profiling
./MyComponent_PTEST --threads 8 --profile perf --gtest_filter="*Contention"

# GPU test with specific device
./MyGpuTest_PTEST --gpu-device 1 --gpu-memory unified --csv gpu_results.csv

# Production CI run
./MyComponent_PTEST --cycles 100000 --repeats 30 --csv ci_results.csv
```

---

## Helper Functions

### Global Config Access

```cpp
const PerfConfig& vernier::bench::detail::getPerfConfig();
```

Access global configuration parsed from command-line flags. Used for profiler integration and custom validation.

### Metadata Capture

```cpp
// Capture all metadata
std::tuple<std::string, std::string, std::string, std::string>
captureMetadata(bool cacheMetadata = true);

// Individual functions
std::string captureTimestamp();   // ISO 8601 format
std::string captureGitHash();     // Short commit hash
std::string captureHostname();    // Machine name
std::string capturePlatform();    // Architecture (x86_64, arm64, etc.)
```

**Usage:**

```cpp
auto [timestamp, gitHash, hostname, platform] = captureMetadata();
std::printf("Test run: %s on %s (%s) [%s]\n",
            timestamp.c_str(), hostname.c_str(),
            platform.c_str(), gitHash.c_str());
```

### Timing Utilities

```cpp
// Get current time in microseconds
double nowUs();
```

High-precision wall-clock timing using `std::chrono::high_resolution_clock`.

### Statistics

```cpp
// Compute all statistics from raw samples
Stats summarize(std::vector<double>& samples);
```

**Note:** Input vector is modified (sorted for percentile calculation).

### Preventing Compiler Optimization

**Use the `volatile` keyword pattern:**

```cpp
// Force compiler to compute result
int sum = computeSum(data);
volatile int result = sum;  // Prevents dead code elimination

// OR: Accumulate into volatile
volatile int sink = 0;
auto result = perf.throughputLoop([&] {
  sink += computeValue();
}, "operation");
```

**Why volatile?**

- Simple and portable
- Works with all compilers
- Clear intent to code reviewers
- Template-friendly (no function calls needed)

---

## Thread Safety

- **PerfCase:** Not thread-safe (create one per test)
- **GpuPerfCase:** Not thread-safe (create one per test)
- **contentionRun:** Thread-safe (uses internal synchronization)
- **PerfRegistry:** Thread-safe (uses mutex for CSV updates)

**Best practices:**

```cpp
// CORRECT: One PerfCase per test
PERF_TEST(MyComponent, Test1) {
  UB_PERF_GUARD(perf);  // Independent instance
  // ...
}

PERF_TEST(MyComponent, Test2) {
  UB_PERF_GUARD(perf);  // Independent instance
  // ...
}

// INCORRECT: Sharing PerfCase across threads
PerfCase globalPerf{"Test", cfg};  // Don't do this!
```

---

## Performance Considerations

### Measurement Overhead

| Operation         | Overhead   | Impact                        |
| ----------------- | ---------- | ----------------------------- |
| Wall-clock timing | ~100-200ns | Negligible for >1us workloads |
| CUDA event timing | ~1-2us     | Negligible for >10us kernels  |
| CSV write         | ~1ms       | Per test (not per sample)     |

**Guideline:** Overhead is negligible when workload >100x measurement cost.

### Memory Usage

- Each repeat stores one `double` (8 bytes)
- Default 10 repeats = 80 bytes per test
- GPU tests add transfer arrays and device memory
- CSV export: ~200 bytes per row

### Warmup Impact

| Platform | First Run | After Warmup | Notes                       |
| -------- | --------- | ------------ | --------------------------- |
| CPU      | 10-100us  | ~stable      | Caches, branch predictor    |
| GPU      | 50-200ms  | ~stable      | JIT compilation, GPU wakeup |

**Auto-scaling warmup (when `warmup=0`):**

- Small workloads (<1000 cycles): 5 warmup iterations
- Medium workloads (<10000 cycles): 3 warmup iterations
- Large workloads (>=10000 cycles): 1 warmup iteration

---

## Common Patterns

### Basic Throughput Measurement

```cpp
PERF_TEST(MyComponent, Throughput) {
  UB_PERF_GUARD(perf);

  std::vector<int> data(10000);
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = process(data);
      (void)result;
    }
  });

  volatile int sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += process(data);
  }, "throughput");

  EXPECT_GT(result.callsPerSecond, 10000);
  EXPECT_STABLE_CV_CPU(result, perf.config());
}
```

### A/B Comparison

```cpp
PERF_TEST(MyComponent, CompareAlgorithms) {
  UB_PERF_GUARD(perf);

  // Test baseline
  volatile int sink = 0;
  auto baseline = perf.throughputLoop([&] {
    sink += algorithmA(data);
  }, "baseline");

  // Test optimized
  auto optimized = perf.throughputLoop([&] {
    sink += algorithmB(data);
  }, "optimized");

  double speedup = baseline.stats.median / optimized.stats.median;
  std::printf("Speedup: %.2fx\n", speedup);

  EXPECT_GT(speedup, 1.1) << "Optimization <10% improvement";
}
```

### Memory Bandwidth Analysis

```cpp
PERF_TEST(Memory, Bandwidth) {
  UB_PERF_GUARD(perf);

  const size_t SIZE = 1024 * 1024;
  std::vector<uint8_t> data(SIZE);

  ub::MemoryProfile memProfile{
    .bytesRead = SIZE,
    .bytesWritten = 0,
    .bytesAllocated = 0
  };

  volatile uint64_t sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += processData(data.data(), SIZE);
  }, "bandwidth", memProfile);

  // Framework prints bandwidth automatically
  (void)sink;
}
```

### Multi-threaded Scalability

```cpp
PERF_TEST(SharedResource, Scalability) {
  UB_PERF_GUARD(perf);

  std::atomic<int64_t> counter{0};

  auto result = perf.contentionRun([&](int threadId) {
    for (int i = 0; i < perf.cycles(); ++i) {
      counter.fetch_add(1);
    }
  }, "contention");

  double efficiency = result.callsPerSecond /
                     (perf.threads() * expectedSingleThreadThroughput);

  std::printf("Parallel efficiency: %.1f%%\n", efficiency * 100);
}
```

---

## See Also

- **[CPU Guide](CPU_GUIDE.md)** - Complete CPU benchmarking guide
- **[GPU Guide](GPU_GUIDE.md)** - Complete GPU benchmarking guide
- **[Demos](../demo/docs/)** - Interactive demos with step-by-step walkthroughs
- **CLI Tools:** `tools/README.md` - CLI tools reference
- **[CI/CD Integration](CI_CD_INTEGRATION.md)** - Automated regression detection
- **[Docker Setup](DOCKER_SETUP.md)** - Container configuration
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Main README](../../../README.md)** - Framework overview
