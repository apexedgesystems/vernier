# Advanced Benchmarking Guide

Complete guide to advanced features, patterns, and best practices for the benchmarking framework.

---

## Table of Contents

- [Memory Profiling Deep Dive](#memory-profiling-deep-dive)
- [Parameterized Tests](#parameterized-tests)
- [CSV Schema Reference](#csv-schema-reference)
- [Profiler Integration](#profiler-integration)
- [Quick Iteration Mode](#quick-iteration-mode)
- [Custom Configuration](#custom-configuration)
- [GPU Advanced Topics](#gpu-advanced-topics)

---

## Memory Profiling Deep Dive

### Understanding MemoryProfile

`MemoryProfile` tracks memory operations to calculate bandwidth and identify CPU-bound vs memory-bound code.

```cpp
struct MemoryProfile {
  size_t bytesRead;       // Input data read from memory
  size_t bytesWritten;    // Output data written to memory
  size_t bytesAllocated;  // Heap allocations during operation

  // Calculate total bandwidth (MB/s) -- read + write only (allocations excluded)
  double bandwidthMBs(double durationUs) const {
    const double totalBytes = bytesRead + bytesWritten;
    const double durationS = durationUs / 1e6;
    return (totalBytes / durationS) / 1e6;  // MB/s
  }

  // Calculate efficiency vs theoretical peak
  double efficiency(double durationUs, double peakMBs) const {
    return (bandwidthMBs(durationUs) / peakMBs) * 100.0;
  }
};
```

### Field Explanations

**`bytesRead`** - Total bytes loaded from memory into CPU/cache:

```cpp
// Reading array
bytesRead = sizeof(data);

// Reading struct members
bytesRead = sizeof(input.field1) + sizeof(input.field2);

// Reading via pointer dereference
bytesRead = input_size_in_bytes;
```

**`bytesWritten`** - Total bytes stored from CPU/cache to memory:

```cpp
// Writing array
bytesWritten = sizeof(output);

// Writing struct members
bytesWritten = sizeof(result.x) + sizeof(result.y);

// Modifying in-place
bytesWritten = bytesRead;  // Read-modify-write
```

**`bytesAllocated`** - Heap allocations during operation:

```cpp
// No allocations (typical for performance tests)
bytesAllocated = 0;

// Testing allocation overhead
bytesAllocated = malloc_size * allocation_count;

// Vector resize
bytesAllocated = vec.capacity() * sizeof(T);
```

### When to Use Each Field

| Scenario                | bytesRead     | bytesWritten   | bytesAllocated |
| ----------------------- | ------------- | -------------- | -------------- |
| Read-only scan          | Size of input | 0              | 0              |
| Write-only fill         | 0             | Size of output | 0              |
| Transform (read->write) | Input size    | Output size    | 0              |
| In-place modify         | Data size     | Data size      | 0              |
| With allocation         | Input size    | Output size    | Alloc size     |

### Bandwidth vs Throughput

**Throughput:** How much data your algorithm processes

```cpp
// Process 1MB of payload data
throughput = 1 MB / time = 1000 MB/s
```

**Bandwidth:** Total memory system utilization

```cpp
// But you read 1MB AND write 1MB
bandwidth = (1 MB read + 1 MB write) / time = 2000 MB/s
```

**Key insight:** Bandwidth can be much higher than throughput!

### Complete Example

```cpp
PERF_TEST(Codec, Encode) {
  UB_PERF_GUARD(perf);
  ub::attachProfilerHooks(perf, ub::detail::getPerfConfig());

  const size_t INPUT_SIZE = 1024;   // 1KB input
  const size_t OUTPUT_SIZE = 2048;  // 2KB output (with framing overhead)

  std::vector<uint8_t> input(INPUT_SIZE);
  std::vector<uint8_t> output(OUTPUT_SIZE);

  // Define memory profile
  ub::MemoryProfile memProfile{
    .bytesRead = INPUT_SIZE,      // Read entire input
    .bytesWritten = OUTPUT_SIZE,  // Write entire output
    .bytesAllocated = 0           // No heap allocations
  };

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = encode(input.data(), INPUT_SIZE,
                                   output.data(), OUTPUT_SIZE);
      (void)result;
    }
  });

  volatile int sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += encode(input.data(), INPUT_SIZE, output.data(), OUTPUT_SIZE);
  }, "encode", memProfile);

  // Output shows:
  // - Throughput: 1000 MB/s of input processed
  // - Bandwidth: 3000 MB/s total memory traffic (1MB read + 2MB write)
  // - Efficiency: 25% of DDR4 peak (~12 GB/s)

  (void)sink;
}
```

### When to Omit MemoryProfile

**Omit when:**

- Quick development iteration
- Testing pure compute (no memory I/O)
- Bandwidth isn't relevant to your optimization

```cpp
// Simple version - just get throughput
auto result = perf.throughputLoop([&] {
  work();
}, "simple");
// CSV: wallMedian, callsPerSecond
```

**Include when:**

- Analyzing memory-bound code
- Need bandwidth in reports
- Identifying cache effects
- Production benchmarks

```cpp
// Detailed version - with bandwidth analysis
ub::MemoryProfile mp{INPUT_SIZE, OUTPUT_SIZE, 0};
auto result = perf.throughputLoop([&] {
  work();
}, "detailed", mp);
// CSV: wallMedian, callsPerSecond, memBandwidthMBs, bytesRead, bytesWritten
```

**Performance impact:** None - it's just metadata for CSV export.

---

## Parameterized Tests

### WARNING: CRITICAL: PERF_TEST_P Does Not Exist

**There is no `PERF_TEST_P` macro!** For parameterized tests, use standard GoogleTest `TEST_P`.

**This will NOT compile:**

```cpp
PERF_TEST_P(MyTest, Case) {  // ERROR: PERF_TEST_P doesn't exist!
  const int size = GetParam();
  // ...
}
```

**Correct approach:**

```cpp
class MyTest : public ::testing::TestWithParam<int> {
protected:
  const ub::PerfConfig& getCfg() {
    return ub::detail::getPerfConfig();
  }
};

TEST_P(MyTest, Case) {  // Use standard GoogleTest TEST_P
  const int size = GetParam();

  ub::PerfConfig cfg = getCfg();
  cfg.msgBytes = size;

  std::string testName = ::testing::UnitTest::GetInstance()
                           ->current_test_info()->test_suite_name() +
                         std::string(".") +
                         ::testing::UnitTest::GetInstance()
                           ->current_test_info()->name();

  ub::PerfCase perf{testName, cfg};
  ub::attachProfilerHooks(perf, cfg);

  // Test implementation...
}

INSTANTIATE_TEST_SUITE_P(
  Params,
  MyTest,
  ::testing::Values(64, 256, 1024)
);
```

### Standard Pattern with UB_PERF_GUARD

For simple tests without parameters:

```cpp
PERF_TEST(MyComponent, Basic) {
  UB_PERF_GUARD(perf);  // Gets "MyComponent.Basic" automatically
  ub::attachProfilerHooks(perf, ub::detail::getPerfConfig());

  // Test code...
}
```

### Custom Configuration Pattern

For parameter sweeps, manually construct `PerfCase` with custom config:

```cpp
PERF_TEST(MyComponent, PayloadSweep) {
  for (int payloadSize : {64, 256, 1024, 4096}) {
    // Get default config
    ub::PerfConfig cfg = ub::detail::getPerfConfig();

    // Override specific field
    cfg.msgBytes = payloadSize;

    // Create custom PerfCase (can't use UB_PERF_GUARD here!)
    std::string testName = "MyComponent.PayloadSweep/" + std::to_string(payloadSize);
    ub::PerfCase perf{testName, cfg};
    ub::attachProfilerHooks(perf, cfg);

    // Setup test data
    std::vector<uint8_t> data(payloadSize);

    // Run test
    perf.warmup([&] {
      for (int i = 0; i < perf.cycles(); ++i) {
        volatile auto result = process(data.data(), payloadSize);
        (void)result;
      }
    });

    volatile uint64_t sink = 0;
    auto result = perf.throughputLoop([&] {
      sink += process(data.data(), payloadSize);
    }, "throughput");

    // Each iteration gets separate CSV row with different msgBytes
    (void)sink;
  }
}
```

### Why Custom Config?

**`UB_PERF_GUARD(perf)` limitations:**

- Gets test name from GoogleTest automatically
- Uses global config (can't override per-parameter)
- All parameter values would have same `msgBytes` in CSV

**Manual `PerfCase` constructor:**

- Set custom test name with parameter value
- Override config fields per parameter
- Each parameter gets unique CSV row

### GoogleTest Parameterized Tests

For more complex parameter combinations, use standard GoogleTest fixtures:

```cpp
class PayloadTest : public ::testing::TestWithParam<int> {
protected:
  const ub::PerfConfig& getCfg() {
    return ub::detail::getPerfConfig();
  }
};

TEST_P(PayloadTest, Encode) {  // Use TEST_P, not PERF_TEST_P
  const int payloadSize = GetParam();

  // Custom config with parameter
  ub::PerfConfig cfg = ub::detail::getPerfConfig();
  cfg.msgBytes = payloadSize;

  // Custom name with parameter
  std::string testName = ::testing::UnitTest::GetInstance()
                           ->current_test_info()->test_suite_name() +
                         std::string(".") +
                         ::testing::UnitTest::GetInstance()
                           ->current_test_info()->name();

  ub::PerfCase perf{testName, cfg};
  ub::attachProfilerHooks(perf, cfg);

  // Test implementation...
}

INSTANTIATE_TEST_SUITE_P(
  PayloadSizes,
  PayloadTest,
  ::testing::Values(64, 256, 1024, 4096)
);
```

**CSV output:**

```
test,msgBytes,wallMedian,callsPerSecond
PayloadTest/PayloadSizes.Encode/64,64,0.123,8130000
PayloadTest/PayloadSizes.Encode/256,256,0.456,2192000
PayloadTest/PayloadSizes.Encode/1024,1024,1.234,810000
PayloadTest/PayloadSizes.Encode/4096,4096,4.567,219000
```

### Key Points

1. **Use `UB_PERF_GUARD`** for simple tests with no parameters
2. **Use manual constructor** for parameter sweeps
3. **Always set `cfg.msgBytes`** to actual payload size
4. **Include parameter in test name** for CSV clarity
5. **Call `attachProfilerHooks`** with your custom config

---

## CSV Schema Reference

### Complete CPU Test Schema

All CPU performance tests produce CSV with these columns:

#### Base Columns (Always Present)

| Column        | Type   | Description              | Example                  |
| ------------- | ------ | ------------------------ | ------------------------ |
| `test`        | string | Test name (Suite.Name)   | "MyComponent.Throughput" |
| `cycles`      | int    | Operations per repeat    | 10000                    |
| `repeats`     | int    | Samples collected        | 10                       |
| `warmup`      | int    | Warmup iterations        | 1000                     |
| `threads`     | int    | Thread count             | 1                        |
| `msgBytes`    | int    | Payload size (bytes)     | 1024                     |
| `console`     | bool   | Console output enabled   | true                     |
| `nonBlocking` | bool   | Non-blocking I/O mode    | false                    |
| `minLevel`    | int    | Minimum log level        | 0                        |
| `medianUs`    | double | Median per-call time     | 0.543                    |
| `p10Us`       | double | 10th percentile          | 0.512                    |
| `p90Us`       | double | 90th percentile          | 0.587                    |
| `minUs`       | double | Minimum time             | 0.498                    |
| `maxUs`       | double | Maximum time             | 0.623                    |
| `meanUs`      | double | Mean time                | 0.546                    |
| `stddevUs`    | double | Standard deviation       | 0.023                    |
| `cv`          | double | Coefficient of variation | 0.042                    |
| `callsPerSec` | double | Throughput               | 1843317.0                |

#### Metadata Columns

| Column        | Type   | Description        | Example                |
| ------------- | ------ | ------------------ | ---------------------- |
| `profileTool` | string | Profiler used      | "perf"                 |
| `profileDir`  | string | Artifact directory | "./perf-data"          |
| `timestamp`   | string | ISO 8601 timestamp | "2024-11-02T10:30:45Z" |
| `gitHash`     | string | Git commit (short) | "a3f9c2b"              |
| `hostname`    | string | Machine name       | "perf-test-01"         |
| `platform`    | string | Architecture       | "x86_64"               |

**Note:** Memory bandwidth analysis from `MemoryProfile` is printed to the console
during test execution but is not exported to CSV columns. The CSV captures timing
and throughput metrics. For bandwidth data, parse the console output or add custom
columns via the `PerfRegistry`.

#### GPU-Specific Columns

See GPU_GUIDE.md for complete GPU column schema.

### Using CSV Data

**Plot scaling curves:**

```bash
bench-plot plot results.csv --x-axis msgBytes --y-axis callsPerSec
```

**Compare implementations:**

```bash
bench compare baseline.csv optimized.csv
```

**Statistical analysis:**

```bash
bench compare baseline.csv candidate.csv --threshold 5
```

**Filter by test:**

```bash
grep "MyComponent.Encode" results.csv > encode_only.csv
```

---

## Profiler Integration

### attachProfilerHooks() Function

```cpp
void attachProfilerHooks(PerfCase& perf, const PerfConfig& cfg);
```

**Purpose:**
Enables command-line profilers for this test case. Must be called to activate `--profile` flag.

**When to call:**

- Right after `UB_PERF_GUARD(perf)` in every test
- Right after manual `PerfCase` constructor for parameterized tests

**What it does:**

1. Checks if user passed `--profile` flag
2. Creates appropriate profiler backend (perf, gperf, etc.)
3. Attaches before/after hooks to PerfCase
4. Enables automatic data collection
5. Adds profiler columns to CSV output

**What happens if omitted:**

- Test runs normally
- `--profile` flag is silently ignored for this test
- No profiler data collected
- Missing profiler columns in CSV

**Example:**

```cpp
PERF_TEST(MyComponent, Throughput) {
  UB_PERF_GUARD(perf);
  ub::attachProfilerHooks(perf, ub::detail::getPerfConfig());  // Required!

  // If user runs: ./test --profile perf
  // This test will collect perf data

  // Without attachProfilerHooks(), --profile is silently ignored
}
```

**Best practice:** Always call it - zero overhead if `--profile` not used.

### Available Profilers

| Profiler    | Purpose                       | Requirements                           | Overhead | Output                                                                            |
| ----------- | ----------------------------- | -------------------------------------- | -------- | --------------------------------------------------------------------------------- |
| `perf`      | CPU performance counters      | Linux, `kernel.perf_event_paranoid=-1` | ~5%      | `cpuCycles`, `instructions`, `ipc`, `l1dMisses`, `llcMisses`, `branchMispredicts` |
| `gperf`     | CPU/heap profiling            | gperftools installed                   | ~10%     | Profile files (`.prof`)                                                           |
| `bpftrace`  | Kernel tracing                | Linux, BPF support, root               | <1%      | Custom trace files (`.bt`)                                                        |
| `rapl`      | Energy consumption            | Linux, Intel CPU, root                 | <1%      | `energyJoules`, `powerWatts`, `energyPerOp`                                       |
| `callgrind` | Deterministic instruction cnt | Valgrind installed                     | 20-50x   | Callgrind annotation files (`.callgrind`)                                         |

### Using perf

```bash
# Enable perf (requires root, one-time setup)
sudo sysctl -w kernel.perf_event_paranoid=-1

# Run test with perf profiling
./MyComponent_PTEST --profile perf --csv results.csv

# Analyze results
perf report -i perf-MyComponent.Throughput-*.data
perf annotate -i perf-MyComponent.Throughput-*.data
```

**CSV columns added:**

- `cpuCycles` - Total CPU cycles
- `instructions` - Instructions executed
- `ipc` - Instructions per cycle
- `l1dMisses` - L1 data cache misses
- `llcMisses` - Last-level cache misses
- `branchMispredicts` - Branch mispredictions

### Using RAPL

```bash
# Run test with RAPL (requires root)
sudo ./MyComponent_PTEST --profile rapl --csv results.csv

# Results include energy consumption
cat results.csv | grep MyComponent
```

**CSV columns added:**

- `energyJoules` - Total energy consumed (J)
- `powerWatts` - Average power (W)
- `energyPerOp` - Energy per operation (uJ)

### Custom Profilers

See API_REFERENCE.md for `Profiler` base class interface to implement custom profilers.

---

## Quick Iteration Mode

### What --quick Does

The `--quick` flag reduces test duration for fast feedback during development:

```bash
./MyComponent_PTEST --quick --csv results.csv
```

**Default values:**

```cpp
// Normal mode (production)
cycles = 10000;
repeats = 10;
warmup = 1;

// Quick mode (if not explicitly overridden)
cycles = 5000;   // 2x fewer operations
repeats = 5;     // 2x fewer samples
warmup = 2;      // Extra warmup to stabilize
```

**Performance impact:**

- 30-100x faster than full run
- Less statistical confidence (higher CV%)
- May miss outliers/edge cases
- Same CSV format (compatible with analysis tools)

### When to Use Quick Mode

**GOOD: Use --quick for:**

- Active development (tight iteration loop)
- Debugging test failures
- Sanity checking builds
- Rapid experimentation
- Local development

**BAD: Don't use --quick for:**

- Production benchmarks
- CI/CD regression testing
- Publishing results
- Comparing implementations
- Performance validation

### Development Workflow

```bash
# Phase 1: Development (fast feedback)
make
./test --quick --csv dev.csv
# Make changes...
make
./test --quick --csv dev2.csv
bench compare dev.csv dev2.csv

# Phase 2: Validation (more samples)
./test --cycles 50000 --repeats 20 --csv validate.csv

# Phase 3: Production (full characterization)
./test --cycles 100000 --repeats 30 --csv production.csv
```

### Quick Mode and CV%

Quick mode may show higher coefficient of variation:

| Mode       | Typical CV% | Acceptable?        |
| ---------- | ----------- | ------------------ |
| Production | <5%         | Good               |
| Normal     | <10%        | Good               |
| Quick      | <15%        | Acceptable for dev |

Use `EXPECT_STABLE_CV_CPU` - it automatically relaxes thresholds in quick mode.

---

## Custom Configuration

### PERF_MAIN() Macro

```cpp
PERF_MAIN()  // Replaces int main()
```

**Purpose:**
Provides complete main() function with:

- GoogleTest initialization
- Command-line parsing (`--csv`, `--profile`, `--quick`, etc.)
- PerfConfig singleton setup
- CSV export after all tests
- Profiler lifecycle management
- Proper exit codes for CI/CD

**Usage:**
Place at end of test file:

```cpp
#include <gtest/gtest.h>
#include "Perf.hpp"

namespace ub = vernier::bench;

PERF_TEST(MyComponent, Test1) {
  // ...
}

PERF_TEST(MyComponent, Test2) {
  // ...
}

PERF_MAIN()  // That's it - no custom main() needed!
```

**What it expands to:**

```cpp
int main(int argc, char** argv) {
  // 1. Parse performance flags
  auto& cfg = vernier::bench::detail::perfConfigSingleton();
  vernier::bench::parsePerfFlags(cfg, &argc, argv);

  // 2. Register global config for CSV export
  vernier::bench::setGlobalPerfConfig(&cfg);

  // 3. Install CSV listener
  vernier::bench::installPerfEventListener(cfg);

  // 4. Initialize GoogleTest
  ::testing::InitGoogleTest(&argc, argv);

  // 5. Run all tests
  return RUN_ALL_TESTS();
}
```

**When NOT to use:**

- Custom main() logic needed
- Embedding tests in larger application
- Multiple test binaries with shared setup

In those cases, manually implement the steps above.

### Accessing Global Config

```cpp
// Get read-only config in tests
const ub::PerfConfig& cfg = ub::detail::getPerfConfig();

// Check flags
if (cfg.quickMode) {
  // Adjust test behavior
}

// Use values
std::printf("Running with %d cycles\n", cfg.cycles);
```

### Command-Line Flag Reference

**Core flags:**

```bash
--cycles N        # Operations per repeat (default: 10000)
--repeats N       # Samples collected (default: 10)
--warmup N        # Warmup iterations (default: 1, 0=auto)
--threads N       # Worker threads (default: 1)
--msg-bytes N     # Payload size (default: 64)
--quick           # Fast mode (reduced cycles/repeats)
```

**Output flags:**

```bash
--csv PATH        # CSV output file
--console         # Echo to console (default: false)
--nonblocking     # Non-blocking mode
--min-level STR   # Minimum log level (default: INFO)
```

**Profiling flags:**

```bash
--profile TOOL         # Profiler: perf|gperf|bpftrace|rapl|callgrind
--profile-args ARGS    # Profiler-specific arguments
--artifact-root DIR    # Output directory (default: .)
--profile-frequency N  # Sampling Hz for CPU profilers (default: 10000)
--profile-analyze      # Auto-run analysis after profiling
--bpf LIST             # BPF scripts (comma-separated): offcpu,syslat,bio
```

**GPU flags:**

```bash
--gpu-device N      # CUDA device ID (default: 0)
--gpu-warmup N      # GPU warmup iterations (default: 10)
--gpu-memory MODE   # Memory strategy: explicit|unified|pinned
--min-speedup F     # Minimum expected speedup vs CPU
```

**Example combinations:**

```bash
# Development
./test --quick --csv dev.csv

# Production CI
./test --cycles 100000 --repeats 30 --csv ci_results.csv

# With profiling
./test --profile perf --gtest_filter="*Encode*"

# Multi-threaded
./test --threads 8 --csv mt_results.csv

# GPU test
./test --gpu-device 0 --gpu-memory unified --csv gpu_results.csv
```

---

## GPU Advanced Topics

### GPU Parameterized Tests

GPU tests follow the same pattern as CPU but use `UB_PERF_GPU_GUARD`:

```cpp
class GpuPayloadTest : public ::testing::TestWithParam<int> {
protected:
  const ub::PerfConfig& getCfg() {
    return ub::detail::getPerfConfig();
  }
};

TEST_P(GpuPayloadTest, Kernel) {
  const int arraySize = GetParam();

  ub::PerfConfig cfg = getCfg();
  cfg.msgBytes = arraySize * sizeof(float);

  std::string testName = ::testing::UnitTest::GetInstance()
                           ->current_test_info()->test_suite_name() +
                         std::string(".") +
                         ::testing::UnitTest::GetInstance()
                           ->current_test_info()->name();

  // Note: PerfGpuCase uses PerfConfig, not custom config override
  ub::PerfGpuCase perf{testName, ub::detail::getPerfConfig()};

  // Allocate arrays
  std::vector<float> h_data(arraySize);
  float* d_data;
  cudaMalloc(&d_data, arraySize * sizeof(float));

  // Warmup
  perf.cudaWarmup([&] {
    myKernel<<<grid, block>>>(d_data, arraySize);
  });

  // Measure
  auto result = perf.cudaKernel(myKernel, "kernel")
    .withLaunchConfig(grid, block)
    .withHostToDevice(h_data.data(), d_data, arraySize * sizeof(float))
    .withDeviceToHost(d_data, h_data.data(), arraySize * sizeof(float))
    .measure();

  cudaFree(d_data);
}

INSTANTIATE_TEST_SUITE_P(
  ArraySizes,
  GpuPayloadTest,
  ::testing::Values(1024, 4096, 16384, 65536)
);
```

### GPU Memory Strategy Selection

Control memory management strategy via command-line:

```bash
# Explicit malloc/memcpy (default)
./test --gpu-memory explicit --csv results.csv

# Unified memory (automatic migration)
./test --gpu-memory unified --csv results.csv

# Pinned host memory (faster transfers)
./test --gpu-memory pinned --csv results.csv

# Mapped memory (zero-copy)
./test --gpu-memory mapped --csv results.csv
```

**Strategy comparison:**

| Strategy   | Transfer Speed         | Use When                      |
| ---------- | ---------------------- | ----------------------------- |
| `explicit` | Fast (PCIe bandwidth)  | Default, full control         |
| `unified`  | Variable (page faults) | Large datasets, ease of use   |
| `pinned`   | Fastest                | Frequent H2D/D2H transfers    |
| `mapped`   | Slowest (no copy)      | Infrequent access, small data |

### Multi-GPU Parameterized Tests

Test scaling across multiple GPUs:

```cpp
class MultiGpuTest : public ::testing::TestWithParam<int> {};

TEST_P(MultiGpuTest, Scaling) {
  const int deviceCount = GetParam();

  UB_PERF_GPU_GUARD(perf);

  // Allocate per-device data
  std::vector<float*> d_data(deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&d_data[i], SIZE * sizeof(float));
  }

  // Measure multi-GPU performance
  auto result = perf.cudaKernelMultiGpu(deviceCount,
    [&](int dev, cudaStream_t stream) {
      myKernel<<<grid, block, 0, stream>>>(d_data[dev], SIZE);
    }, "multi-gpu")
    .withLaunchConfig(grid, block)
    .withP2PAccess()
    .measure();

  // Check scaling efficiency
  EXPECT_GT(result.aggregatedStats.multiGpu->scalingEfficiency, 0.8)
    << "Poor scaling with " << deviceCount << " GPUs";

  // Cleanup
  for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaFree(d_data[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
  DeviceCounts,
  MultiGpuTest,
  ::testing::Values(1, 2, 4, 8)
);
```

### GPU CSV Schema

Complete GPU-specific CSV columns:

#### GPU Performance Columns

| Column              | Type   | Description                 | Example                 |
| ------------------- | ------ | --------------------------- | ----------------------- |
| `gpuModel`          | string | GPU device name             | "NVIDIA A100-SXM4-40GB" |
| `computeCapability` | string | CUDA compute capability     | "8.0"                   |
| `kernelTimeUs`      | double | GPU kernel execution time   | 123.45                  |
| `transferTimeUs`    | double | H2D + D2H transfer time     | 45.67                   |
| `h2dBytes`          | int64  | Host-to-device bytes        | 4194304                 |
| `d2hBytes`          | int64  | Device-to-host bytes        | 4194304                 |
| `speedupVsCpu`      | double | GPU vs CPU speedup          | 15.3                    |
| `memBandwidthGBs`   | double | Memory bandwidth            | 850.2                   |
| `achievedOccupancy` | double | Kernel occupancy [0-1]      | 0.82                    |
| `smClockMHz`        | double | SM clock frequency          | 1410.0                  |
| `throttling`        | bool   | Thermal throttling detected | false                   |

#### Multi-GPU Columns

| Column               | Type   | Description               | Present When    |
| -------------------- | ------ | ------------------------- | --------------- |
| `deviceId`           | int    | Primary device ID         | Multi-GPU tests |
| `deviceCount`        | int    | Number of GPUs used       | Multi-GPU tests |
| `multiGpuEfficiency` | double | Parallel efficiency [0-1] | Multi-GPU tests |
| `p2pBandwidthGBs`    | double | Peer-to-peer bandwidth    | P2P enabled     |

#### Unified Memory Columns

| Column              | Type   | Description               | Present When   |
| ------------------- | ------ | ------------------------- | -------------- |
| `umPageFaults`      | int64  | GPU page faults           | Unified memory |
| `umH2DMigrations`   | int64  | Host->device migrations   | Unified memory |
| `umD2HMigrations`   | int64  | Device->host migrations   | Unified memory |
| `umMigrationTimeUs` | double | Total migration overhead  | Unified memory |
| `umThrashing`       | bool   | Memory thrashing detected | Unified memory |

### GPU Profiler Integration

GPU-specific profilers:

| Profiler | Purpose                        | Requirements          | Output           |
| -------- | ------------------------------ | --------------------- | ---------------- |
| `nsight` | Comprehensive kernel profiling | NVIDIA Nsight Compute | `.ncu-rep` files |

**Using Nsight Compute:**

```bash
# Profile specific kernel
./test --profile nsight --gtest_filter="*MyKernel"

# Generates: nsight-MyTest.MyKernel-TIMESTAMP.ncu-rep

# Analyze with Nsight UI
ncu-ui nsight-MyTest.MyKernel-*.ncu-rep
```

### GPU Best Practices

1. **Always warmup GPU kernels** - First launch includes JIT compilation

   ```cpp
   perf.cudaWarmup([&] {
     myKernel<<<grid, block>>>(d_data);
   });
   ```

2. **Use appropriate launch configuration** - Check occupancy

   ```cpp
   auto result = perf.cudaKernel(myKernel, "kernel")
     .withLaunchConfig(grid, block, sharedMemBytes)
     .measure();

   EXPECT_GT(result.stats.achievedOccupancy, 0.5)
     << "Low occupancy - increase threads or reduce resources";
   ```

3. **Profile with multiple block sizes** - Find optimal configuration

   ```cpp
   for (int blockSize : {128, 256, 512, 1024}) {
     dim3 block(blockSize);
     dim3 grid((N + blockSize - 1) / blockSize);

     auto result = perf.cudaKernel(myKernel, "kernel")
       .withLaunchConfig(grid, block)
       .measure();
   }
   ```

4. **Measure transfer overhead separately** - Identify bottlenecks

   ```cpp
   // Just transfers (no kernel)
   auto transfer = perf.cudaKernel(emptyKernel, "transfer-only")
     .withHostToDevice(h_in, d_in, SIZE)
     .withDeviceToHost(d_out, h_out, SIZE)
     .measure();

   // Full pipeline
   auto full = perf.cudaKernel(myKernel, "full")
     .withHostToDevice(h_in, d_in, SIZE)
     .withDeviceToHost(d_out, h_out, SIZE)
     .measure();

   double kernelOnlyUs = full.stats.kernelTimeUs - transfer.stats.transferTimeUs;
   ```

---

## Best Practices Summary

### CPU Benchmarking

1. **Always call `attachProfilerHooks()`** after `UB_PERF_GUARD` or `PerfCase` constructor
2. **Use `MemoryProfile`** for memory-bound code analysis
3. **Set `cfg.msgBytes`** to actual payload size in parameterized tests
4. **Use `--quick`** during development, full config for production
5. **Check CV%** with `EXPECT_STABLE_CV_CPU` for result stability

### GPU Benchmarking

6. **Always warmup GPU kernels** - First launch includes JIT compilation
7. **Check occupancy** - Target >50% for compute-bound kernels
8. **Profile multiple block sizes** - Find optimal configuration
9. **Measure transfers separately** - Identify bottlenecks
10. **Use appropriate memory strategy** - Test explicit vs unified vs pinned

### General

11. **Call `PERF_MAIN()`** to handle all boilerplate
12. **Check CSV schema** to understand available metrics
13. **Use TEST_P for parameterized tests** - PERF_TEST_P doesn't exist!

---

## See Also

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking guide
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking guide
- **[Demos](../demo/docs/)** - Interactive demos with step-by-step walkthroughs
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
