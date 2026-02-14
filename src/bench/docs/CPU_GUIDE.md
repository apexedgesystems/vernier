# CPU Benchmarking Guide

Complete guide to CPU performance benchmarking with the framework. This guide covers everything from basic measurements to advanced profiling techniques.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Understanding the Basics](#understanding-the-basics)
- [Writing Benchmarks](#writing-benchmarks)
- [Profiling and Analysis](#profiling-and-analysis)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Your First Benchmark

```cpp
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Perf.hpp"

namespace ub = vernier::bench;

PERF_TEST(MyComponent, BasicThroughput) {
  // Create benchmark harness (auto-reads command-line flags)
  UB_PERF_GUARD(perf);

  // Setup (not measured)
  std::vector<int> data(1000);
  std::iota(data.begin(), data.end(), 0);

  // Warmup: Prime caches and JIT compilers
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      std::sort(data.begin(), data.end());
    }
  });

  // Measure: Run many times and collect statistics
  auto result = perf.throughputLoop([&] {
    std::sort(data.begin(), data.end());
  }, "sort");

  // Validate: Check performance meets expectations
  EXPECT_GT(result.callsPerSecond, 10000)
      << "Sort throughput below threshold";

  // Check stability (low jitter)
  EXPECT_STABLE_CV_CPU(result, perf.config());
}

// One-line main() with CSV export
PERF_MAIN()
```

### Build and Run

```bash
# Build
cmake -B build -S . && cmake --build build

# Run all tests
./build/native-linux-debug/bin/ptests/MyComponent_PTEST --csv results.csv

# Quick mode for fast iteration
./MyComponent_PTEST --quick

# Run specific test
./MyComponent_PTEST --gtest_filter="*BasicThroughput"

# With profiling
./MyComponent_PTEST --profile perf --gtest_filter="*BasicThroughput"
```

---

## Understanding the Basics

### What Gets Measured?

The framework measures **wall-clock time** for each operation using high-precision timers:

```cpp
auto result = perf.throughputLoop([&] {
  doWork();  // This code is timed
}, "work");
```

**Key metrics collected:**

- **Median latency** (us) - Middle value, robust to outliers
- **p10/p90 percentiles** - Best and worst typical performance
- **Standard deviation** - Measure of jitter/variance
- **Coefficient of variation (CV%)** - Normalized variance (stddev/mean)
- **Throughput** - Operations per second (1e6 / median_us)

### Why These Metrics Matter

**Median vs Mean:**

- **Median** is better for performance work - not skewed by outliers
- A few slow runs don't artificially inflate "typical" performance
- More stable across repeated test runs

**Percentiles (p10, p90):**

- **p10** = best 10% of runs (optimistic case)
- **p90** = worst 10% excluded (realistic worst case)
- Useful for understanding performance distribution
- Important for latency-sensitive applications

**Coefficient of Variation (CV%):**

- **CV% = stddev / mean** - normalized measure of jitter
- Low CV% (<5%): Very stable, repeatable results
- High CV% (>10%): Investigate - may indicate:
  - Background interference
  - Cache/TLB effects
  - Branch misprediction
  - Thermal throttling
  - Measurement overhead dominating

### Throughput vs Latency Modes

**throughputLoop - Per-operation measurement:**

```cpp
auto result = perf.throughputLoop([&] {
  doOneOperation();  // Framework calls this cycles() times
}, "per-op");
```

Use when:

- Measuring cost of a single operation
- Operation is very fast (<1us)
- Want per-call latency breakdown

**measured - Batch measurement:**

```cpp
auto result = perf.measured([&] {
  for (int i = 0; i < perf.cycles(); ++i) {
    doOneOperation();  // You do the loop
  }
}, "batch");
```

Use when:

- Custom loop logic needed
- Amortizing measurement overhead
- Batch processing pattern

---

## Writing Benchmarks

### Basic Throughput Test

Measure how many operations per second your code can perform:

```cpp
PERF_TEST(MyComponent, Throughput) {
  UB_PERF_GUARD(perf);

  // Setup: Prepare test data
  const int N = 10000;
  std::vector<int> data(N);
  std::iota(data.begin(), data.end(), 1);

  MyComponent component;

  // Warmup: Prime caches
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = component.processData(data);
      (void)result;
    }
  });

  // Measure: Framework times each call
  volatile int sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += component.processData(data);
  }, "throughput");

  // Results printed automatically, but available:
  std::printf("Median: %.3f us/call\n", result.stats.median);
  std::printf("Throughput: %.0f calls/sec\n", result.callsPerSecond);

  // Validate performance
  EXPECT_GT(result.callsPerSecond, 10000) << "Throughput too low";
  EXPECT_STABLE_CV_CPU(result, perf.config());
}
```

### Latency Distribution Test

Focus on latency percentiles for latency-sensitive code:

```cpp
PERF_LATENCY(Network, RequestLatency) {
  UB_PERF_GUARD(perf);

  NetworkClient client;

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile auto response = client.sendRequest();
      (void)response;
    }
  });

  auto result = perf.throughputLoop([&] {
    volatile auto response = client.sendRequest();
    (void)response;
  }, "request");

  // Focus on latency distribution
  std::printf("\nLatency Distribution:\n");
  std::printf("  p10: %.3f us (best case)\n", result.stats.p10);
  std::printf("  p50: %.3f us (median)\n", result.stats.median);
  std::printf("  p90: %.3f us (worst case)\n", result.stats.p90);
  std::printf("  p90/p50 ratio: %.2f\n", result.stats.p90 / result.stats.median);

  // Validate latency requirements
  EXPECT_LT(result.stats.median, 100.0) << "Median latency > 100us";
  EXPECT_LT(result.stats.p90, 200.0) << "p90 latency > 200us";
  EXPECT_LT(result.stats.cv, 0.15) << "High latency jitter";
}
```

### Preventing Compiler Optimizations

**Problem:** Compilers eliminate "dead" code that doesn't affect program output.

**Solution:** Use the `volatile` keyword to force computation:

```cpp
// BAD: Compiler might optimize away the entire loop
int sum = 0;
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
// sum is never used - compiler may remove the loop entirely!

// GOOD: Use volatile to prevent optimization
int sum = 0;
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
volatile int result = sum;  // Tells compiler: "sum must be computed"
(void)result;  // Suppresses unused variable warning

// ALSO GOOD: Accumulate into volatile variable
volatile int sink = 0;
auto result = perf.throughputLoop([&] {
  sink += computeValue();
}, "compute");
```

**Pattern for all benchmarks:**

```cpp
// Pattern 1: Single result
{
  int result = expensiveComputation();
  volatile int _sink = result;
  (void)_sink;
}

// Pattern 2: Accumulation (preferred for throughput tests)
{
  volatile int sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += computeValue();
  }, "compute");
}

// Pattern 3: Object result
{
  std::string result = processData();
  volatile auto _sink = result;
  (void)_sink;
}
```

**Why volatile works:**

- Portable across all compilers
- Simple and clear intent
- No function call overhead
- Works with any type

### Multi-Threaded Testing

Measure contention and scalability with multiple threads:

```cpp
PERF_CONTENTION(SharedCounter, Increment) {
  UB_PERF_GUARD(perf);

  std::atomic<int64_t> counter{0};

  // Warmup (single-threaded)
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  });

  // Measure with N threads (from --threads flag)
  auto result = perf.contentionRun([&] {
    // Each thread independently runs this code
    for (int i = 0; i < perf.cycles(); ++i) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  }, "contention");

  std::printf("\nScalability Results:\n");
  std::printf("  Threads: %d\n", perf.threads());
  std::printf("  Total throughput: %.0f ops/sec\n", result.callsPerSecond);
  std::printf("  Per-thread throughput: %.0f ops/sec\n",
              result.callsPerSecond / perf.threads());

  // Validate
  EXPECT_EQ(counter.load(),
            perf.cycles() * perf.threads() * perf.repeats());
}
```

**Run with different thread counts:**

```bash
./MyComponent_PTEST --threads 1 --csv t1.csv
./MyComponent_PTEST --threads 2 --csv t2.csv
./MyComponent_PTEST --threads 4 --csv t4.csv
./MyComponent_PTEST --threads 8 --csv t8.csv

# Analyze scaling
bench compare t1.csv t8.csv
```

### Memory Bandwidth Analysis

Understand if your code is CPU-bound or memory-bound:

```cpp
PERF_TEST(Memory, StreamingCopy) {
  UB_PERF_GUARD(perf);

  const size_t SIZE = 1024 * 1024;  // 1 MB
  std::vector<uint8_t> src(SIZE);
  std::vector<uint8_t> dst(SIZE);

  // Initialize source data
  std::iota(src.begin(), src.end(), 0);

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      std::memcpy(dst.data(), src.data(), SIZE);
    }
  });

  // Specify memory profile for bandwidth analysis
  // Note: Specify per-operation bytes (framework calls op cycles() times)
  ub::MemoryProfile memProfile{
    .bytesRead = SIZE,      // Each call reads 1MB
    .bytesWritten = SIZE,   // Each call writes 1MB
    .bytesAllocated = 0     // No heap allocations
  };

  auto result = perf.throughputLoop([&] {
    std::memcpy(dst.data(), src.data(), SIZE);
  }, "memcpy", memProfile);

  // Framework automatically prints:
  // - Memory bandwidth (MB/s)
  // - Bandwidth utilization percentage
  // - CPU-bound vs memory-bound analysis
}
```

**Output example:**

```
Memory bandwidth: 9403.3 MB/s (1.0 MB read, 1.0 MB written per call)
Estimated efficiency: 78.4% of theoretical peak (~12000 MB/s)
Hint: High bandwidth utilization -> Memory-bound (consider memory layout)
```

**Interpretation:**

- **Efficiency < 10%:** CPU-bound - optimize algorithms
- **Efficiency 10-50%:** Mixed - both matter
- **Efficiency > 50%:** Memory-bound - optimize data layout

**What to optimize when memory-bound:**

- Reduce memory footprint (cache blocking)
- Improve spatial locality (array-of-structs vs struct-of-arrays)
- Use SIMD for parallel loads/stores
- Prefetch data before use

### Comparing Implementations (A/B Testing)

Compare baseline vs optimized versions:

```cpp
PERF_TEST(Algorithm, CompareVersions) {
  UB_PERF_GUARD(perf);

  std::vector<int> data(10000);
  std::iota(data.begin(), data.end(), 1);

  std::printf("\n=== Algorithm Comparison ===\n");

  // Test baseline implementation
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = algorithmV1(data);
      (void)result;
    }
  });

  volatile int sink = 0;
  auto baseline = perf.throughputLoop([&] {
    sink += algorithmV1(data);
  }, "v1-baseline");

  std::printf("\nBaseline (v1):\n");
  std::printf("  Latency: %.3f us\n", baseline.stats.median);
  std::printf("  Throughput: %.0f ops/sec\n", baseline.callsPerSecond);

  // Test optimized implementation
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = algorithmV2(data);
      (void)result;
    }
  });

  auto optimized = perf.throughputLoop([&] {
    sink += algorithmV2(data);
  }, "v2-optimized");

  std::printf("\nOptimized (v2):\n");
  std::printf("  Latency: %.3f us\n", optimized.stats.median);
  std::printf("  Throughput: %.0f ops/sec\n", optimized.callsPerSecond);

  // Calculate improvement
  double speedup = baseline.stats.median / optimized.stats.median;
  double improvement = ((baseline.stats.median - optimized.stats.median)
                        / baseline.stats.median) * 100.0;

  std::printf("\nImprovement:\n");
  std::printf("  Speedup: %.2fx faster\n", speedup);
  std::printf("  Latency reduction: %.1f%%\n", improvement);

  // Validate
  EXPECT_LT(optimized.stats.median, baseline.stats.median)
      << "Optimized version should be faster";

  EXPECT_GT(speedup, 1.1)
      << "Optimization should be at least 10% faster";

  (void)sink;
}
```

### Payload Scaling Tests

Understand how performance scales with input size:

```cpp
PERF_TEST(Algorithm, PayloadScaling) {
  UB_PERF_GUARD(perf);

  struct TestCase {
    std::string name;
    size_t size;
  };

  std::vector<TestCase> testCases = {
    {"8B",    8},
    {"64B",   64},
    {"256B",  256},
    {"1KB",   1024},
    {"4KB",   4096},
    {"16KB",  16384},
    {"64KB",  65536}
  };

  std::printf("\n%-8s %-12s %-15s %-15s\n",
              "Size", "Latency(us)", "Throughput", "Bandwidth");
  std::printf("%s\n", std::string(50, '-').c_str());

  for (const auto& test : testCases) {
    std::vector<uint8_t> data(test.size);

    perf.warmup([&] {
      for (int i = 0; i < perf.cycles(); ++i) {
        volatile auto result = processData(data.data(), data.size());
        (void)result;
      }
    });

    ub::MemoryProfile memProfile{
      .bytesRead = test.size,
      .bytesWritten = 0,
      .bytesAllocated = 0
    };

    volatile uint64_t sink = 0;
    auto result = perf.throughputLoop([&] {
      sink += processData(data.data(), data.size());
    }, test.name, memProfile);

    double bandwidthMBs = memProfile.bandwidthMBs(result.stats.median);

    std::printf("%-8s %-12.3f %-15.0f %-15.1f MB/s\n",
                test.name.c_str(),
                result.stats.median,
                result.callsPerSecond,
                bandwidthMBs);
  }

  // Look for scaling patterns:
  // - Linear: Good algorithmic scaling
  // - Sub-linear: Cache effects or bandwidth limits
  // - Super-linear: May indicate measurement issues
}
```

---

## Profiling and Analysis

### CPU Profiling with perf

Find hotspots and optimization opportunities:

```bash
# Profile with hardware counters
./MyComponent_PTEST --profile perf --gtest_filter="*Throughput"

# Generates: perf-MyComponent.Throughput-TIMESTAMP.data

# View report
perf report -i perf-MyComponent.Throughput-*.data

# Annotate source
perf annotate -i perf-MyComponent.Throughput-*.data

# Export for analysis
perf script -i perf-MyComponent.Throughput-*.data > profile.txt
```

**Common patterns to look for:**

- Hot loops consuming >50% of CPU time
- Unexpected function calls in tight loops
- Cache misses (L1, L2, L3)
- Branch mispredictions

### Heap Profiling with gperftools

Identify memory allocation bottlenecks:

```bash
# CPU profiling
./MyComponent_PTEST --profile gperf --gtest_filter="*Throughput"

# Heap profiling (requires HEAPPROFILE environment variable)
HEAPPROFILE=/tmp/heap ./MyComponent_PTEST --profile gperf

# Analyze CPU profile
google-pprof --text ./MyComponent_PTEST cpu-profile.prof
google-pprof --pdf ./MyComponent_PTEST cpu-profile.prof > profile.pdf

# Analyze heap profile
google-pprof --text ./MyComponent_PTEST /tmp/heap.0001.heap
google-pprof --pdf ./MyComponent_PTEST /tmp/heap.0001.heap > heap.pdf
```

### System-wide Tracing with bpftrace

Trace syscalls and kernel interactions:

```bash
# Requires root or CAP_BPF capability
sudo ./MyComponent_PTEST --profile bpftrace \
  --profile-args "bpf/fsync_latency.bt" \
  --gtest_filter="*IO"

# Custom bpftrace script
sudo ./MyComponent_PTEST --profile bpftrace \
  --profile-args "custom_trace.bt"
```

**Example bpftrace script (fsync_latency.bt):**

```bpftrace
#!/usr/bin/env bpftrace

// Track fsync() latency
tracepoint:syscalls:sys_enter_fsync {
  @start[tid] = nsecs;
}

tracepoint:syscalls:sys_exit_fsync {
  $delta = nsecs - @start[tid];
  @latency_us = hist($delta / 1000);
  delete(@start[tid]);
}
```

### Energy Measurement with RAPL

Measure power consumption (Intel CPUs):

```bash
# Requires root for /dev/cpu/*/msr access
sudo ./MyComponent_PTEST --profile rapl --csv energy_results.csv

# Results include power consumption in CSV
# Columns: package_joules, dram_joules, duration_seconds
```

---

## Advanced Techniques

### Cache Hierarchy Analysis

Test performance at different cache levels:

```cpp
PERF_TEST(Cache, HierarchyTraversal) {
  UB_PERF_GUARD(perf);

  struct CacheTest {
    const char* name;
    size_t size;
    const char* cacheLevel;
  };

  std::vector<CacheTest> tests = {
    {"4KB",    4*1024,       "L1 (32KB)"},
    {"16KB",   16*1024,      "L1 (32KB)"},
    {"64KB",   64*1024,      "L2 (256KB)"},
    {"256KB",  256*1024,     "L2 (256KB)"},
    {"1MB",    1024*1024,    "L3 (8MB)"},
    {"16MB",   16*1024*1024, "RAM"}
  };

  std::printf("\n%-8s %-12s %-15s %-20s\n",
              "Size", "Cache", "Latency(us)", "Bandwidth(MB/s)");
  std::printf("%s\n", std::string(60, '-').c_str());

  for (const auto& test : tests) {
    std::vector<uint8_t> data(test.size);

    perf.warmup([&] {
      for (int i = 0; i < perf.cycles(); ++i) {
        volatile uint8_t sum = 0;
        for (size_t j = 0; j < data.size(); j += 64) {  // 64B stride
          sum += data[j];
        }
        (void)sum;
      }
    });

    ub::MemoryProfile memProfile{
      .bytesRead = test.size,
      .bytesWritten = 0,
      .bytesAllocated = 0
    };

    volatile uint8_t sink = 0;
    auto result = perf.throughputLoop([&] {
      uint8_t sum = 0;
      for (size_t j = 0; j < data.size(); j += 64) {
        sum += data[j];
      }
      sink = sum;
    }, test.name, memProfile);

    double bw = memProfile.bandwidthMBs(result.stats.median);

    std::printf("%-8s %-12s %-15.3f %-20.1f\n",
                test.name, test.cacheLevel,
                result.stats.median, bw);
  }

  // Expected pattern:
  // - L1: ~300-400 GB/s bandwidth, <1ns latency
  // - L2: ~100-200 GB/s bandwidth, ~3ns latency
  // - L3: ~50-100 GB/s bandwidth, ~10-15ns latency
  // - RAM: ~10-30 GB/s bandwidth, ~50-100ns latency
}
```

### Branch Prediction Analysis

Test branch predictor effectiveness:

```cpp
PERF_TEST(BranchPrediction, PredictableVsRandom) {
  UB_PERF_GUARD(perf);

  const int N = 10000;
  std::vector<int> sortedData(N);
  std::vector<int> randomData(N);

  std::iota(sortedData.begin(), sortedData.end(), 0);
  std::iota(randomData.begin(), randomData.end(), 0);
  std::shuffle(randomData.begin(), randomData.end(),
               std::mt19937{std::random_device{}()});

  // Test 1: Predictable branches
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int sum = 0;
      for (int val : sortedData) {
        if (val < N/2) {  // Predictable: first half always true
          sum += val;
        }
      }
      (void)sum;
    }
  });

  volatile int sink = 0;
  auto predictable = perf.throughputLoop([&] {
    int sum = 0;
    for (int val : sortedData) {
      if (val < N/2) {
        sum += val;
      }
    }
    sink = sum;
  }, "predictable");

  // Test 2: Random branches
  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int sum = 0;
      for (int val : randomData) {
        if (val < N/2) {  // Unpredictable: 50/50 random
          sum += val;
        }
      }
      (void)sum;
    }
  });

  auto random = perf.throughputLoop([&] {
    int sum = 0;
    for (int val : randomData) {
      if (val < N/2) {
        sum += val;
      }
    }
    sink = sum;
  }, "random");

  double penalty = (random.stats.median / predictable.stats.median - 1.0) * 100;

  std::printf("\nBranch Prediction Impact:\n");
  std::printf("  Predictable: %.3f us\n", predictable.stats.median);
  std::printf("  Random: %.3f us\n", random.stats.median);
  std::printf("  Misprediction penalty: %.1f%%\n", penalty);

  // Typical penalty: 20-50% for random branches
  EXPECT_GT(penalty, 10.0) << "Expected misprediction penalty";

  (void)sink;
}
```

### Allocation Overhead Testing

Measure cost of dynamic memory allocation:

```cpp
PERF_TEST(Allocation, Overhead) {
  UB_PERF_GUARD(perf);

  struct AllocationTest {
    const char* name;
    size_t size;
  };

  std::vector<AllocationTest> tests = {
    {"16B",   16},
    {"64B",   64},
    {"256B",  256},
    {"1KB",   1024},
    {"4KB",   4096},
    {"1MB",   1024*1024}
  };

  std::printf("\n%-8s %-15s %-15s\n",
              "Size", "Alloc+Free(us)", "Cost(ns)");
  std::printf("%s\n", std::string(40, '-').c_str());

  for (const auto& test : tests) {
    perf.warmup([&] {
      for (int i = 0; i < perf.cycles(); ++i) {
        void* ptr = std::malloc(test.size);
        std::free(ptr);
      }
    });

    volatile void* sink = nullptr;
    auto result = perf.throughputLoop([&] {
      void* ptr = std::malloc(test.size);
      sink = ptr;
      std::free(ptr);
    }, test.name);

    double costNs = result.stats.median * 1000.0;  // us to ns

    std::printf("%-8s %-15.3f %-15.0f\n",
                test.name,
                result.stats.median,
                costNs);

    (void)sink;
  }

  // Typical costs:
  // - Small allocations (<256B): 50-100ns via tcmalloc/jemalloc
  // - Large allocations (>1MB): 1-10us (syscalls involved)
}
```

---

## Best Practices

### 1. Always Use Warmup

**Why:** First few runs are slower due to:

- Cold CPU caches
- Cold TLB
- Branch predictor training
- JIT compilation (if applicable)

```cpp
// GOOD: Warmup before measurement
perf.warmup([&] {
  for (int i = 0; i < perf.cycles(); ++i) {
    doWork();
  }
});

auto result = perf.throughputLoop([&] { doWork(); }, "work");
```

**Auto-scaling warmup:**

- Set `--warmup 0` to let framework decide
- Small workloads (<1000 cycles): 5 warmup iterations
- Medium workloads (<10000 cycles): 3 warmup iterations
- Large workloads (>=10000 cycles): 1 warmup iteration

### 2. Use Volatile to Prevent Optimization

**Always prevent dead code elimination:**

```cpp
// BAD: Compiler may optimize away
int result = compute();

// GOOD: Force computation
int result = compute();
volatile int _sink = result;
(void)_sink;

// BEST: Accumulate in benchmark loop
volatile int sink = 0;
auto result = perf.throughputLoop([&] {
  sink += compute();
}, "compute");
```

### 3. Check CV% for Stability

**Low CV% means reliable results:**

```cpp
auto result = perf.throughputLoop([&] { work(); }, "test");

// Automatic validation with adaptive thresholds
EXPECT_STABLE_CV_CPU(result, perf.config());

// Or manual check
EXPECT_LT(result.stats.cv, 0.10) << "High jitter detected";
```

**If CV% is high (>10%):**

- Use `taskset` to pin to specific cores
- Close background applications
- Increase warmup iterations
- Increase sample count (--repeats 30)
- Check for thermal throttling

### 4. Use Quick Mode During Development

**Fast iteration for development:**

```bash
# Quick mode: Fewer cycles and repeats
./MyComponent_PTEST --quick --gtest_filter="*MyTest"

# Equivalent to:
./MyComponent_PTEST --cycles 1000 --repeats 3
```

**Production mode for CI:**

```bash
# More samples for stable results
./MyComponent_PTEST --cycles 100000 --repeats 30 --csv ci_results.csv
```

### 5. Compare Against Baseline

**Track performance over time:**

```bash
# Capture baseline
git checkout main
./MyComponent_PTEST --csv baseline.csv

# Test optimization
git checkout feature/optimization
./MyComponent_PTEST --csv optimized.csv

# Compare
bench compare baseline.csv optimized.csv
```

### 6. Use Profilers to Find Hotspots

**Don't guess - measure:**

```bash
# Find hotspots
./MyComponent_PTEST --profile perf --gtest_filter="*Slow"

# Analyze
perf report -i perf-*.data

# Focus optimization on top functions only
```

### 7. Test Multiple Payload Sizes

**Performance characteristics change with scale:**

```cpp
// Test scaling behavior
for (size_t size : {64, 256, 1024, 4096, 16384}) {
  auto result = testWithPayload(size);
  // Look for inflection points (cache boundaries)
}
```

### 8. Document Your Assumptions

**Make requirements explicit:**

```cpp
PERF_TEST(MyComponent, Throughput) {
  // ...

  // Document requirements clearly
  EXPECT_GT(result.callsPerSecond, 10000)
      << "Requirement: Must handle 10k ops/sec minimum";

  EXPECT_LT(result.stats.p90, 100.0)
      << "SLA: 90th percentile latency < 100us";
}
```

---

## Troubleshooting

### High CV% (Unstable Results)

**Symptoms:**

- CV% > 10%
- Large difference between min and max
- Results vary between runs

**Causes and solutions:**

1. **CPU frequency scaling:**

   ```bash
   # Disable CPU frequency scaling
   sudo cpupower frequency-set --governor performance
   ```

2. **Background processes:**

   ```bash
   # Pin to specific cores (avoid core 0 which handles interrupts)
   taskset -c 2-9 ./MyComponent_PTEST --csv results.csv
   ```

3. **Thermal throttling:**

   ```bash
   # Monitor CPU temperature
   watch -n 1 sensors

   # Add cooling time between tests
   ./MyComponent_PTEST --warmup 10
   ```

4. **Too few samples:**

   ```bash
   # Increase repeats for statistical confidence
   ./MyComponent_PTEST --repeats 30
   ```

5. **Workload too small:**
   - Measurement overhead dominates
   - Use more cycles or batch operations

### Suspiciously Fast Results

**If results are too good to be true, they probably are:**

1. **Compiler optimized away the work:**

   ```cpp
   // Add volatile to force computation
   volatile int sink = 0;
   auto result = perf.throughputLoop([&] {
     sink += compute();
   }, "compute");
   ```

2. **Workload is cached:**

   ```cpp
   // Ensure data is large enough to evict caches
   std::vector<uint8_t> data(10 * 1024 * 1024);  // 10MB > L3
   ```

3. **Empty loop:**

   ```cpp
   // Verify your lambda is non-empty
   auto result = perf.throughputLoop([&] {
     // Make sure there's actual work here!
   }, "oops");
   ```

### Build Errors

**Missing headers:**

```cpp
#include "Perf.hpp"  // Correct path
```

**Linker errors:**

```cmake
# Ensure CMakeLists.txt links benchmarking library
target_link_libraries(MyComponent_PTEST
  PRIVATE
    bench
    GTest::gtest
)
```

### Runtime Errors

**Segmentation fault:**

- Check array bounds in benchmark code
- Verify pointers are valid
- Use AddressSanitizer: `cmake -DCMAKE_CXX_FLAGS="-fsanitize=address"`

**CSV not created:**

```bash
# Ensure path is writable
./MyComponent_PTEST --csv /path/to/results.csv

# Check file was created
ls -la results.csv
```

**Profiler not starting:**

```bash
# perf requires kernel support
sudo sysctl -w kernel.perf_event_paranoid=-1

# RAPL requires root access
sudo ./MyComponent_PTEST --profile rapl
```

---

## See Also

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking guide
- **[Demos](../demo/docs/)** - Interactive CPU demos (01-09) with step-by-step walkthroughs
- **CLI Tools:** `tools/README.md` - CLI tools reference
- **[CI/CD Integration](CI_CD_INTEGRATION.md)** - Automated regression detection
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Main README](../../../README.md)** - Framework overview
