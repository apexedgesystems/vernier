# API Reference

**Namespace:** `vernier::bench`
**Platform:** Linux-only
**C++ Standard:** C++20 (C++23 used when available)

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
  PERF_GUARD(perf);  // Creates PerfCase with config from command-line

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

PERF_GUARD(varName)                   // Create scoped PerfCase
PERF_MAIN()                              // Main function with CSV export
```

`PERF_MAIN()` and `PERF_GPU_MAIN()` return the tests' status, or 4
(`BENCH_PROFILE_FAILED_EXIT_CODE`) when the tests passed and the requested
`--profile` failed (or, in a GPU benchmark, `--capture-um` asked for
unified-memory counters that CUPTI could not record), after a report of each
failure. A benchmark with its own
`main()` gets the same status from `ProfilerRegistry::finishRun()`; the
[advanced guide](ADVANCED_GUIDE.md#perf_main-macro) shows the call.

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
void setAbortMeasureHook(std::function<void(const PerfCase&)> hook); // runs instead of the after hook when the callback throws

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
- If `fn()` throws, the abort hook runs in place of the after hook, nothing
  is published, and the exception reaches you unchanged; the case can measure
  again. With `attachProfilerHooks()`, the abort hook calls the profiler's
  `abortMeasure()`, which does nothing by default; the gperf profiler stops
  its CPU and heap captures there at once, without analysis, and the Nsight
  profiler pops the NVTX range it pushed, so this case and others can capture
  again

**`contentionRun(worker, label)`**

- Framework starts `threads()` threads together; each calls `worker()`
  `cycles()` times per repeat
- Use for: Multi-threaded scalability testing
- `worker()` is one operation: the framework supplies the loop and the
  threads

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
  std::vector<std::string> bpfScripts; // Script names/paths: "fsync_latency", "write_latency"
  std::string artifactRoot;            // Profiler artifact directory
  int profileFrequency = 0;            // gperf CPU sampling rate asked for (Hz, 1 to 4000); 0: gperftools' default
  bool profileAnalyze = false;         // Auto-run analysis after profiling
  int profileTestTimeoutSecs = -1;     // Watchdog seconds per measured()/throughputLoop() loop under --profile: -1 not given (300 under --profile), 0 off; contentionRun() and GPU runs are not covered

  bool quickMode = false;              // Apply reduced cycles/repeats

  // Append-only tail: libbench shares this layout, new members go last
  int targetTimeUs = 0;                // >0: auto-size cycles so one repeat spans ~this (us)
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
PerfCase makePerfCaseWithProfiler(std::string testName, const PerfConfig& cfg);
```

`--profile` profiles the cases that create a profiler: `PERF_GUARD` does it
for its case, `attachProfilerHooks()` for a `PerfCase` you construct, and
`makePerfCaseWithProfiler()` constructs the case and attaches it (on the GPU,
`PERF_GPU_GUARD` and `attachGpuProfilerHooks()`). A bare `PerfCase`, or one
built with `PERF_GUARD_NOPROFILE`, gets none, so the backend does nothing
around its measured phase;
a tool that runs the whole process still records it. The request is checked
when the profiler is created: one that cannot run fails the run (exit status
4 when the tests pass), and a run in which no case created a profiler ends
with a notice that nothing was profiled. See the
[advanced guide](ADVANCED_GUIDE.md#attachprofilerhooks-function).

**Supported profilers** (each self-registers via the backend registry):

- CPU: `perf`, `gperf`, `callgrind`, `bpftrace`, `rapl`, `massif`,
  `memcheck`, `helgrind`, `offcpu`, `heaptrack`, `jemalloc`
- GPU: `nsight` (or `nsys`), `ncu`, `compute-sanitizer`, `rocprof` (not
  validated on AMD hardware)

The advanced guide's
[Available Profilers](ADVANCED_GUIDE.md#available-profilers) gives what each
records, needs, writes and costs.

Adjacent in-process instrumentation:

- **CUPTI** - per-launch register count, static + dynamic shared memory,
  launch count populate the GPU CSV section on every `PERF_GPU_*` test
  (no `--profile` flag required).
- **NVTX** - `BENCH_NVTX_SCOPE("name") / BENCH_NVTX_MARK("name")` macros
  label the nsys timeline; `NsightProfiler` auto-emits a range named after
  the test around each measured window.

**Example:**

```cpp
PERF_TEST(MyComponent, Throughput) {
  PERF_GUARD(perf);  // creates the case and attaches the --profile profiler

  perf.warmup([&]{ /* ... */ });
  auto result = perf.throughputLoop([&]{ /* ... */ }, "op");
}
```

**Command line:**

```bash
# CPU profiling with perf
./MyComponent_PTEST --profile perf --gtest_filter="*Throughput"

# CPU sampling with gperftools
./MyComponent_PTEST --profile gperf --gtest_filter="*Throughput"

# Energy measurement with RAPL (read access to /dev/cpu/0/msr needed)
./MyComponent_PTEST --profile rapl
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
PERF_GPU_THROUGHPUT(SuiteName, TestName) // Throughput-focused test
PERF_GPU_LATENCY(SuiteName, TestName)    // Latency-focused test
PERF_GPU_COMPARISON(SuiteName, TestName) // CPU vs GPU comparison
PERF_GPU_BANDWIDTH(SuiteName, TestName)  // Memory bandwidth test
PERF_GPU_SCALING(SuiteName, TestName)    // Multi-GPU scaling test

PERF_GPU_GUARD(varName)               // Create scoped PerfGpuCase
PERF_GPU_MAIN()                          // Main with GPU support
```

### PerfGpuCase

Core GPU benchmark harness.

**Key methods:**

```cpp
// CPU baseline (call before the GPU kernel to populate speedupVsCpu)
PerfResult cpuBaseline(CpuFn fn, std::string label = "cpu_baseline");

// Kernel builder (fluent interface). KernelFn = std::function<void(cudaStream_t)>.
CudaKernelBuilder cudaKernel(KernelFn kernel, std::string label = "cuda_kernel");

// Multi-GPU builder. MultiGpuKernelFn = std::function<void(int deviceId, cudaStream_t)>.
MultiGpuKernelBuilder cudaKernelMultiGpu(int deviceCount, MultiGpuKernelFn kernel,
                                         std::string label = "multi_gpu_kernel");

// Warmup the kernel before measuring
void cudaWarmup(KernelFn kernel);

// Accessors
int cycles() const noexcept;
int repeats() const noexcept;
const PerfConfig& cpuConfig() const noexcept;
const PerfGpuConfig& gpuConfig() const noexcept;
const std::string& testName() const noexcept;
cudaStream_t stream() const noexcept;
```

### CudaKernelBuilder

Fluent interface for kernel configuration and measurement.

```cpp
class CudaKernelBuilder {
public:
  // Configuration (optional)
  CudaKernelBuilder& withHostToDevice(const void* src, void* dst, size_t bytes);
  CudaKernelBuilder& withDeviceToHost(const void* src, void* dst, size_t bytes);
  CudaKernelBuilder& withLaunchConfig(dim3 grid, dim3 block, size_t sharedMemBytes = 0);
  CudaKernelBuilder& withDeviceId(int deviceId);

  // Measurement (label is set on cudaKernel(), not here)
  PerfGpuResult measure();
};
```

**Usage pattern:**

```cpp
PERF_GPU_TEST(MyKernel, Throughput) {
  PERF_GPU_GUARD(perf);

  // Setup
  std::vector<float> h_input(N);
  std::vector<float> h_output(N);
  float *d_input, *d_output;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output, N * sizeof(float));

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  perf.cudaWarmup([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
  });

  // Measure kernel performance
  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
  }, "myKernel")
    .withHostToDevice(h_input.data(), d_input, N * sizeof(float))
    .withDeviceToHost(d_output, h_output.data(), N * sizeof(float))
    .withLaunchConfig(grid, block)
    .measure();

  // Validate
  EXPECT_GT(result.callsPerSecond, 1000) << "GPU throughput too low";
  EXPECT_STABLE_CV_GPU(result, perf.cpuConfig());

  cudaFree(d_input);
  cudaFree(d_output);
}
```

### PerfGpuResult

```cpp
struct PerfGpuResult {
  GpuStats stats{};         // Comprehensive GPU statistics (see GpuStats below)
  double kernelTimeUs{};    // CUDA event kernel time
  double transferTimeUs{};  // H2D + D2H transfer time
  double totalTimeUs{};     // One round trip: copies in, one launch, copies out
  double callsPerSecond{};  // 1e6 / totalTimeUs
  double speedupVsCpu{};    // Set by cpuBaseline() comparison
  std::string label;        // Kernel label from cudaKernel()
  int deviceId{-1};         // CUDA device ID
};

struct GpuStats {
  Stats cpuStats{};                    // Per-repeat round-trip times
  GpuDeviceInfo deviceInfo{};          // GPU model, compute capability
  MemoryTransferProfile transfers{};   // H2D/D2H bytes and bandwidthGBs()
  OccupancyMetrics occupancy{};        // Launch shape and its occupancy estimate
  ClockSpeedProfile clocks{};          // NVML SM/memory clocks at start and end
  PowerThermalProfile powerThermal{};  // NVML power and temperature at start and end
  CuptiKernelStats cupti{};            // In-process per-launch kernel metrics
  double kernelTimeMedianUs{};
  double transferTimeMedianUs{};
  double totalTimeMedianUs{};
  std::optional<MultiGpuMetrics> multiGpu{};        // Only in cudaKernelMultiGpu()
  std::optional<P2PTransferProfile> p2pProfile{};   // Only with P2P transfers
  std::optional<UnifiedMemoryProfile> unifiedMemory{}; // Only under --capture-um
};
```

### Validation Macros

```cpp
// GPU CV% validation
EXPECT_STABLE_CV_GPU(gpuResult, perf.cpuConfig());

// Occupancy: the estimate says what the launch shape allows, not how busy
// the GPU was. A test tuned for a launch shape can require the estimate that
// shape gave, so that a changed shape shows
EXPECT_GE(gpuResult.stats.occupancy.achievedOccupancy, TUNED_SHAPE_ESTIMATE);

// Copy rate: the declared copies' bytes over their time, against a floor
// measured for those copies on this system's link
EXPECT_GT(gpuResult.stats.transfers.bandwidthGBs(), MEASURED_FLOOR_GBS);

// No SM-clock drop: isThrottling() compares NVML's readings, and a reading
// NVML did not report is 0, so ask it only when the start, end and maximum
// SM clocks are all there
const auto& clocks = gpuResult.stats.clocks;
if (clocks.smClockMHzStart > 0 && clocks.smClockMHzEnd > 0 && clocks.boostClockMHz > 0) {
  EXPECT_FALSE(clocks.isThrottling());
}
```

The GPU guide's [Result Validation](GPU_GUIDE.md#result-validation) gives
what each check can and cannot say.

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
  virtual void abortMeasure() noexcept {} // the measured window threw; no-op (gperf stops its captures, Nsight pops its range)

  // Factory: returns a concrete profiler or no-op based on cfg.profileTool
  static std::unique_ptr<Profiler> make(const PerfConfig& cfg, const std::string& testName);
};
```

### ProfilerPerf

Linux `perf` integration for CPU profiling. The backend attaches perf to the
benchmark process for the measured phase only; `--profile-args` picks the mode:

| `--profile-args`     | Runs                                                                                          | Writes, in `<Test>.perf/`      |
| -------------------- | --------------------------------------------------------------------------------------------- | ------------------------------ |
| none, or other flags | `perf stat` on cpu-cycles, instructions, branches, branch-misses and cache-misses, plus flags | `stat.txt`                     |
| `record [flags]`     | `perf record [flags]`                                                                         | `perf.data`, `record.err.txt`  |
| `mem`                | `perf mem record`                                                                             | `perf.mem.data`, `mem.err.txt` |
| `c2c`                | `perf c2c record`                                                                             | `perf.c2c.data`, `c2c.err.txt` |

**Usage:**

```bash
./MyComponent_PTEST --profile perf --gtest_filter="*Throughput"
# Writes: MyComponent.Throughput.perf/stat.txt

./MyComponent_PTEST --profile perf --profile-args "record -g" --target-time 250ms \
    --gtest_filter="*Throughput"
# Writes: MyComponent.Throughput.perf/perf.data
```

The measured phase starts once perf answers a `ping` on its `--control` fifo,
which perf does from its main loop with its counters on, so the phase is
profiled from its first iteration however long perf takes to start
(`perf record` takes longer than `perf stat`). The wait is bounded at 5 s: a
perf that has not answered by then is stopped and the request fails. A perf
without `--control`, which the doctor's perf row reports, and `perf mem` start
after a fixed 200 ms instead. A measured phase of a few milliseconds holds few
samples, which `--target-time` (or more `--cycles`) avoids.

**Analysis** (of a `record` run):

```bash
perf report -i MyComponent.Throughput.perf/perf.data
perf annotate -i MyComponent.Throughput.perf/perf.data
```

### ProfilerGperf

Google Performance Tools integration.

**Features:**

- CPU profiling (sampling)
- Heap profiling (`--profile-args heap`, or `both` for CPU + heap), in a build
  with `-DVERNIER_LINK_TCMALLOC=ON`: the heap profiler is part of tcmalloc,
  which replaces the allocator of the whole process

**Usage:**

```bash
# CPU profiling (default) -> MyComponent.Throughput.gperf/cpu.prof
./MyComponent_PTEST --profile gperf --gtest_filter="*Throughput"

# Heap profiling -> MyComponent.Throughput.gperf/heap.<N>.heap
./MyComponent_PTEST --profile gperf --profile-args heap
```

**Analysis:**

```bash
google-pprof --text ./MyComponent_PTEST MyComponent.Throughput.gperf/cpu.prof
google-pprof --pdf ./MyComponent_PTEST MyComponent.Throughput.gperf/cpu.prof > profile.pdf
```

**Sampling rate:** gperftools samples 100 times a second by default and reads
its rate once, from the `CPUPROFILE_FREQUENCY` environment variable, when the
process starts
([gperftools CPU profiler](https://gperftools.github.io/gperftools/cpuprofile.html)).
Ask for another rate with `--profile-frequency`, a whole number of Hz from 1
to 4000, the most gperftools samples. `bench run` sets the variable before it
starts the benchmark; run directly, the benchmark needs the variable set to
the same value before it starts, and otherwise fails without profiling,
printing the command that works. The variable set alone is the rate too, and
is held to the same rule: any other value fails the run, since gperftools
would read it its own way without saying so.

```bash
# Period 4000 us: 250 samples a second of CPU time
bench run ./MyComponent_PTEST --profile gperf --profile-frequency 250 -- --gtest_filter="*Throughput" --target-time 200ms
CPUPROFILE_FREQUENCY=250 ./MyComponent_PTEST --profile gperf --profile-frequency 250 --gtest_filter="*Throughput" --target-time 200ms
```

Each finished profile records the period gperftools used, `1000000 / rate`
microseconds rounded down (3000 Hz records 333 us), and is checked: a profile
whose period is not the one expected fails the run with status 4 and is kept.
With a rate asked for or set, each case prints what its profile records:
`[gperf] MyComponent.Throughput: cpu.prof records a period of 4000 us (250 Hz requested), <n> samples`.
The rate taken can still be lower than the rate configured: on the
[Pi 4 rig](rigs/RIG_PI4.md), whose kernel ticks 250 times a second, 1000 Hz
gave about 249 samples a second, and on an x86_64 machine whose kernel ticks
1000 times, 4000 Hz gave about 1000.

Only gperf takes a rate: a request that gives one to another profiler fails
before anything is collected, in the doctor and the run alike, and one with
no profiler (no `--profile`, or `--profile cupti`) stops the benchmark with
status 2. A backend's own factory function called directly (for example
`makePerfProfiler()`), rather than through the profiler guard,
`attachProfilerHooks()` or `makePerfCaseWithProfiler()`, does not check the
rate and ignores it.

### ProfilerBpftrace

BPF-based kernel tracing.

**Features:**

- Syscall tracing
- Custom probe points
- Off-CPU time has a backend of its own, `offcpu`

**Usage:**

```bash
# bpftrace runs only as root: run the benchmark as root, or set BENCH_SUDO=1
# with a sudoers grant for bpftrace and kill, which runs only the tracer as root
BENCH_SUDO=1 ./MyComponent_PTEST --profile bpftrace --bpf fsync_latency
```

### ProfilerRAPL

Intel RAPL energy measurement: package 0's energy-status counter (MSR 0x611),
read through `/dev/cpu/0/msr` before and after each measured window. No DRAM
counter is read; the package counter covers every core, cache and the
integrated graphics, whatever runs on them.

**Writes:** `<Test>.rapl/energy.txt`, one numbered section per measured window
(in the order of the case's CSV rows): the window's energy, duration, average
power and energy per operation (average power x median time per call), with
the counters. A summary naming the file follows each window.

**Decides** from an actual read of MSR 0x606 and MSR 0x611, for the doctor and
the run alike; a refusal names its cause (`denied`, `missing`, `unsupported`,
`unusable`) and fails the run with exit status 4. It takes no mode words, and
`--profile-analyze` fails the run after the capture (there is no analysis).
`RAPLProfiler(cfg, test)` and `makeRAPLProfiler()` decide the same way, and
`checkRaplRequest()` is the decision itself.

**Usage:**

```bash
# Needs read access to /dev/cpu/0/msr: root, or CAP_SYS_RAWIO together with
# read permission on the device
sudo ./MyComponent_PTEST --profile rapl
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
# nsys and ncu record a process only when they start it: bench run starts the
# benchmark under nsys (under ncu for --profile ncu or --profile-args compute)
bench run ./MyGpuTest_PTEST --profile nsight -- --gtest_filter="*Kernel"
```

Run directly with `--profile nsight`, without a tool around it, the run fails
(exit status 4 when the tests pass) and prints the command that captures it.

---

## CSV Schema

This is the one description of the file `--csv` writes; the other guides link
here.

**The file.** One header line, then one row per published measurement, its
columns in four groups, always in this order:

1. the 22 base columns, in every file;
2. `profileTool` and `profileDir`, only in a run given `--profile`;
3. the four metadata columns, in every file;
4. the 29 GPU columns, in a GPU benchmark's file: one whose binary ran
   `PERF_GPU_MAIN()`, or has a test whose name holds `Gpu`, `GPU` or `CUDA`.

Every row has as many cells as the header; a value a row does not have is an
empty cell (a CPU row in a GPU benchmark's file leaves the GPU columns empty).
A test name holding a comma, a double quote or a line break is written in
double quotes, its quotes doubled (RFC 4180), so read the file with a CSV
reader that honours the quotes, not by splitting lines at commas. `bench` and
`bench-plot` need the `test`, `wallMedian`, `wallCV` and `callsPerSecond`
columns.

**Row names.** A test that publishes one measurement names its row with its
case name, `Suite.Test`. When it publishes several, each row is
`Suite.Test/<label>`, with the label the measurement was given; a label that
is empty, or repeats within the test, also gets `#1`, `#2` and so on in the
order the measurements completed.

**The `wall*` columns.** Each repeat gives one sample: its elapsed time over
the operations it ran (`cycles`, or `cycles` times `threads` in
`contentionRun()`), in microseconds. `wallMedian` to `wallStddev` describe the
`repeats` samples, not single calls, and `wallCV` is their standard deviation
over their mean. In a GPU row each sample is one repeat's round trip: its
copies in, one launch and its copies out (the GPU guide's
[What Gets Measured?](GPU_GUIDE.md#what-gets-measured)).

### Base Columns

| Column                   | Type   | Description                                                                                                                                      |
| ------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test`                   | string | The row's name (above)                                                                                                                           |
| `cycles`                 | int    | Operations per repeat, per thread in `contentionRun()`; under `--target-time`, the count `throughputLoop()` or `contentionRun()` calibrated      |
| `repeats`                | int    | Repeats, one sample each                                                                                                                         |
| `warmup`                 | int    | Times `warmup()` ran its function (`--warmup`; when that is 0, 5, 3 or 1 by `cycles`), 1 when the test did not call it; in a GPU row, `--warmup` |
| `threads`                | int    | Threads `contentionRun()` started; 1 for `measured()`, `throughputLoop()` and GPU rows                                                           |
| `msgBytes`               | int    | The configuration's `msgBytes` (`--msg-bytes`, default 64), which sets `cvThreshold`                                                             |
| `console`                | 0/1    | The configuration's `console` (`--console`)                                                                                                      |
| `nonBlocking`            | 0/1    | The configuration's `nonBlocking` (`--nonblocking`)                                                                                              |
| `minLevel`               | string | The configuration's `minLevel` (`--min-level`, default `INFO`)                                                                                   |
| `wallMedian`             | double | Median sample (us per operation)                                                                                                                 |
| `wallP10`, `wallP90`     | double | 10th and 90th percentiles of the samples (us)                                                                                                    |
| `wallP99`, `wallP999`    | double | 99th and 99.9th percentiles of the samples (us)                                                                                                  |
| `wallMin`, `wallMax`     | double | Smallest and largest sample (us)                                                                                                                 |
| `wallMean`, `wallStddev` | double | Mean and standard deviation of the samples (us)                                                                                                  |
| `wallCV`                 | double | `wallStddev` over `wallMean`, a fraction                                                                                                         |
| `callsPerSecond`         | double | 1e6 / `wallMedian`                                                                                                                               |
| `stable`                 | 0/1    | 1 when `wallCV` is below `cvThreshold`                                                                                                           |
| `cvThreshold`            | double | 0.20 for `msgBytes` under 64, 0.10 under 256, 0.05 under 1024, else 0.03; under `--quick` 1.5 times that, at most 0.30                           |

### Profile Columns

Only in a run given `--profile`:

| Column        | Type   | Description                                                  |
| ------------- | ------ | ------------------------------------------------------------ |
| `profileTool` | string | The backend's name (`perf`, `memcheck`, ...)                 |
| `profileDir`  | string | The case's artifact folder; empty where the backend has none |

### Metadata Columns

| Column      | Type   | Description                                                                                                 |
| ----------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| `timestamp` | string | When the row was built, ISO 8601 in UTC (`2026-11-14T19:27:31Z`)                                            |
| `gitHash`   | string | `git describe --always --dirty --tags`, run in the benchmark's working directory; `unknown` when that fails |
| `hostname`  | string | The machine's host name                                                                                     |
| `platform`  | string | The build's architecture: `x86_64`, `aarch64`, `arm`, `x86`, `riscv` or `unknown`                           |

### GPU Columns

What each holds in a single-device measurement's row; the GPU guide defines
them ([Understanding GPU Metrics](GPU_GUIDE.md#understanding-gpu-metrics)),
and [walkthrough 19](../demo/docs/19_CUPTI_KERNEL_METRICS.md) reads them in
a run:

| Column              | Type   | Description                                                                                                                     |
| ------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `gpuModel`          | string | The device's name                                                                                                               |
| `computeCapability` | string | Major.minor (`8.9`)                                                                                                             |
| `kernelTimeUs`      | double | One launch: the median, over the repeats, of the launches' time over their count (us)                                           |
| `transferTimeUs`    | double | The median host-to-device time plus the median device-to-host time (us); 0 when the test declares no copies                     |
| `h2dBytes`          | int64  | Bytes the test declares copying to the device each repeat                                                                       |
| `d2hBytes`          | int64  | Bytes the test declares copying back each repeat                                                                                |
| `speedupVsCpu`      | double | The case's or its suite's CPU baseline (`cpuBaseline()`) over the median round trip; empty without one                          |
| `memBandwidthGBs`   | double | The declared copies' bytes over `transferTimeUs`, in GB/s; empty without copies                                                 |
| `occupancy`         | double | An estimate of the occupancy the launch shape allows [0-1], not a measurement; empty without `.withLaunchConfig()`              |
| `smClockMHz`        | int    | The SM clock NVML read at the window's end (MHz)                                                                                |
| `throttling`        | 0/1    | 1 when the SM clock fell more than 10% from the window's start to its end; empty unless NVML read both and the maximum SM clock |

### Power and Thermal Columns

| Column              | Type   | Description                                              |
| ------------------- | ------ | -------------------------------------------------------- |
| `powerDrawW`        | double | NVML's power at the end of a window of 1.5 s or more (W) |
| `powerLimitW`       | double | The power limit (W)                                      |
| `temperatureC`      | int    | GPU temperature at the window's end (C)                  |
| `temperatureDeltaC` | int    | Temperature at the end minus at the start (C)            |

Source: NVML, at the measured window's start and end. A cell stays empty when
NVML did not report the readings it needs (on the Jetson AGX Thor it reports
none of them), and the run names the missing readings on stderr. `powerDrawW`
is the endpoint reading: NVML's power reading taken at the window's end, which
NVIDIA's NVML reference describes as a one-second average on Ampere (except
GA100) and newer GPUs and an instantaneous reading on GA100 and older; neither
is an average over the measured window. Vernier publishes it only for a window
of at least 1.5 s, a rule of its own: a shorter window's cell is empty and the
run says why.

### CUPTI In-Process Kernel Metrics

| Column                  | Type | Description                                                  |
| ----------------------- | ---- | ------------------------------------------------------------ |
| `cuptiKernelLaunches`   | int  | Kernel launches in the untimed batch after the timed repeats |
| `cuptiRegistersMedian`  | int  | Median registers per thread across launches                  |
| `cuptiRegistersMax`     | int  | Max registers per thread observed                            |
| `cuptiStaticSmemBytes`  | int  | Median static `__shared__` allocation per launch             |
| `cuptiDynamicSmemBytes` | int  | Median dynamic shared memory passed at launch                |

Source: CUPTI's Activity API, in every GPU run; no `--profile` flag is
needed. After a kernel measurement's timed repeats, the harness calls the
measured callable `cycles` times more, at most 1000, untimed, with CUPTI
recording kernels, and the five cells describe those launches; no timed launch
runs while CUPTI records kernels. The cells stay empty where the collector
cannot or must not run (a build without CUPTI, `VERNIER_DISABLE_CUPTI=1`, an
nsys or ncu session) and for a measured window whose records are not known to
be complete, and the run says why on stderr
([CUPTI in-process kernel metrics](GPU_GUIDE.md#cupti-in-process-kernel-metrics)).

### Multi-GPU Columns

| Column               | Type   | Description                |
| -------------------- | ------ | -------------------------- |
| `deviceId`           | int    | Primary CUDA device ID     |
| `deviceCount`        | int    | Number of GPUs used        |
| `multiGpuEfficiency` | double | Not measured: always empty |
| `p2pBandwidthGBs`    | double | Peer-to-peer bandwidth     |

### Unified Memory Columns

| Column              | Type   | Description                                     |
| ------------------- | ------ | ----------------------------------------------- |
| `umPageFaults`      | int64  | GPU page-fault groups plus CPU page faults      |
| `umH2DMigrations`   | int64  | Host-to-device transfers of managed memory      |
| `umD2HMigrations`   | int64  | Device-to-host transfers of managed memory      |
| `umMigrationTimeUs` | double | The transfers' durations, summed (us)           |
| `umThrashing`       | bool   | CUPTI recorded thrashing in the measured window |

These cells are filled only under `--capture-um` (or
`PerfGpuConfig::captureUnifiedMemory`), from CUPTI's unified-memory counters
over each measurement's window, on a GPU whose CUPTI records them; zeros mean
nothing migrated. Without the request they are empty. Where CUPTI cannot
record them, they stay empty, the run reports the request as unsupported, and
a run whose tests pass exits with status 4. See the GPU guide's
[Unified-Memory Counters](GPU_GUIDE.md#unified-memory-counters).

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

| Flag              | Type   | Default  | Description                                 |
| ----------------- | ------ | -------- | ------------------------------------------- |
| `--csv PATH`      | string | -        | CSV output path ([CSV Schema](#csv-schema)) |
| `--console`       | bool   | false    | Echo to console (when relevant)             |
| `--nonblocking`   | bool   | false    | Non-blocking mode (when relevant)           |
| `--min-level STR` | string | `"INFO"` | DEBUG\|INFO\|WARNING\|ERROR\|FATAL          |

### Profiling Flags

| Flag                     | Type   | Default | Description                                                                                                                                                                                                                                 |
| ------------------------ | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--profile TOOL`         | string | -       | Profiler: perf\|gperf\|bpftrace\|rapl\|callgrind\|massif\|memcheck\|helgrind\|offcpu\|heaptrack\|jemalloc\|nsight\|ncu\|compute-sanitizer\|rocprof; `nsys` is another name for `nsight`                                                     |
| `--profile-args ARGS`    | string | -       | The profile's mode, read by its backend (and by `bench run` for the tools it wraps; see [tools/README.md](../../../tools/README.md#run---execute-benchmark-binary))                                                                         |
| `--profile-output-dir`   | path   | -       | Where backend artifacts land (alias of `--artifact-root`)                                                                                                                                                                                   |
| `--profile-test-timeout` | int    | 300     | Watchdog seconds for each measured loop (`measured()`, `throughputLoop()`) under `--profile`; 0 turns it off, and a value that is not a whole number from 0 exits 2. `contentionRun()` and GPU measurements are not covered                 |
| `--target-time DUR`      | string | -       | Auto-size cycles so one repeat spans ~DUR (`500us`, `100ms`, `2s`; bare number = ms). Calibrates from a timed batch of calls, doubled until ~1 ms; floor of one cycle. `throughputLoop`/`contentionRun` only; `measured()` keeps `--cycles` |
| `--profile-check`        | flag   | -       | Print binary readiness + per-backend env doctor, then exit                                                                                                                                                                                  |
| `--profile-check-json`   | flag   | -       | Machine-readable twin of `--profile-check`: one JSON document (readiness rows + backend rows), then exit. Consumed by `bench doctor --json` / `--require`                                                                                   |
| `--artifact-root DIR`    | string | .       | Profiler output directory                                                                                                                                                                                                                   |
| `--profile-frequency N`  | int    | 0       | gperf CPU sampling rate, 1 to 4000 Hz (0: gperftools' 100): `bench run` sets it before the benchmark starts (run directly, set `CPUPROFILE_FREQUENCY=N` too); checked in each profile, see [ProfilerGperf](#profilergperf)                  |
| `--profile-analyze`      | bool   | false   | Run the requested profile's analysis after profiling: gperf's pprof report in the benchmark; for callgrind, `bench run --profile-analyze` annotates the profile after valgrind has written it                                               |
| `--bpf LIST`             | string | -       | Scripts to run, comma-separated: a name is `<name>.bt` in the `--bpf-scripts` directory, a name with a `/` a path from the working directory, `.bt` optional; e.g. `write_latency`, `./my_script.bt`                                        |
| `--bpf-scripts DIR`      | string | -       | Directory `--bpf` looks script names up in: sets `PERF_BPF_SCRIPTS` in the process environment, over an inherited value. Default, or an empty value: `src/bench/bpf/` of the source tree the library was built from                         |

### GPU-Specific Flags

| Flag                  | Type   | Default  | Description                                                                                                                                            |
| --------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--gpu-device N`      | int    | 0        | CUDA device ID                                                                                                                                         |
| `--gpu-warmup N`      | int    | 10       | GPU warmup iterations                                                                                                                                  |
| `--gpu-memory MODE`   | string | explicit | Memory strategy for the tests that read it (`gpuConfig().memStrategy`): explicit\|unified\|pinned\|mapped; any other value stops the run with status 2 |
| `--min-speedup FLOAT` | double | 1.0      | Minimum expected GPU speedup                                                                                                                           |
| `--capture-um`        | bool   | false    | Unified-memory counters, where CUPTI records them                                                                                                      |

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

# GPU test on device 1; the tests that read the memory strategy allocate managed memory
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
// Capture all metadata (cached: git, host name and platform once per
// process; the timestamp each call)
std::tuple<std::string, std::string, std::string, std::string>
captureMetadata(bool cacheMetadata = true);

// Individual functions
std::string captureTimestamp();   // ISO 8601, UTC
std::string captureGitHash();     // git describe --always --dirty --tags, or "unknown"
std::string captureHostname();    // Machine name
std::string capturePlatform();    // Architecture (x86_64, aarch64, arm, x86, riscv, unknown)
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
- **PerfGpuCase:** Not thread-safe (create one per test)
- **contentionRun:** calls `worker` from `threads()` threads at once and
  synchronizes only their start and their end; the worker keeps its own
  shared state safe (atomics or locks)
- **PerfRegistry:** Thread-safe: a mutex guards the rows measurements publish
  and the listeners take at each test's end

**Best practices:**

```cpp
// CORRECT: One PerfCase per test
PERF_TEST(MyComponent, Test1) {
  PERF_GUARD(perf);  // Independent instance
  // ...
}

PERF_TEST(MyComponent, Test2) {
  PERF_GUARD(perf);  // Independent instance
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
  PERF_GUARD(perf);

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
  PERF_GUARD(perf);

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
  PERF_GUARD(perf);

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
PERF_CONTENTION(SharedResource, Scalability) {
  PERF_GUARD(perf);

  std::atomic<std::int64_t> counter{0};

  // The worker takes no arguments and makes one call; contentionRun()
  // supplies the loop and the threads
  auto result = perf.contentionRun([&] { counter.fetch_add(1); }, "contention");

  // Run it with --threads 1 and with --threads N: the ratio of the two
  // callsPerSecond values is the scaling
  std::printf("%d threads: %.0f calls/s\n", perf.threads(), result.callsPerSecond);
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
