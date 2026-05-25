# Benchmarking Demos

**Location:** `src/bench/demo/`
**Platform:** Linux x86_64 (CPU), CUDA (GPU)

Self-contained demonstrations of the Vernier benchmarking framework. Each demo
is a runnable executable paired with a step-by-step walkthrough. Start with Demo
01 and progress in order -- each demo builds on concepts from the previous one.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [CPU Demos](#2-cpu-demos)
3. [GPU Demos](#3-gpu-demos)
4. [Running Demos](#4-running-demos)
5. [Shared Workloads](#5-shared-workloads)
6. [Learning Path](#6-learning-path)
7. [See Also](#7-see-also)

---

## 1. Getting Started

Build the framework, run a demo, and analyze results in under two minutes:

```bash
# Build framework and demos
make compose-debug

# Build the bench analysis tool
make tools-rust
```

Run Demo 01 (basic throughput measurement) and analyze the results:

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_01_BasicWorkflow --quick --csv /tmp/demo01.csv
  source .env
  bench summary /tmp/demo01.csv
'
```

You will see output like this:

```
[BasicWorkflow.SimpleThroughput]  325 us/call  CV=0.8%  ~3.1K calls/s
  Memory bandwidth: 2457 MB/s (0.8 MB read, 0.0 MB written per call)
  Estimated efficiency: 20.5% of theoretical peak (~12000 MB/s)

[BasicWorkflow.AccumulateVsManualLoop]  325 us/call  CV=0.7%  ~3.1K calls/s
[BasicWorkflow.AccumulateVsManualLoop]  164 us/call  CV=0.5%  ~6.1K calls/s

[BasicWorkflow.QuickModeIteration]  32 us/call  CV=0.2%  ~31.0K calls/s
```

The manual pointer loop is 2x faster than `std::accumulate`. The framework
measured it, exported CSV, and `bench summary` formatted the results. Every demo
follows this same pattern: measure something slow, measure something fast, compare.

Open [docs/01_BASIC_WORKFLOW.md](docs/01_BASIC_WORKFLOW.md) for the full
walkthrough of what the code does and why.

---

## 2. CPU Demos

| #   | Demo                  | Concept                         | Slow Path               | Fast Path                  | Walkthrough                                                 |
| --- | --------------------- | ------------------------------- | ----------------------- | -------------------------- | ----------------------------------------------------------- |
| 01  | Basic Workflow        | Measure-export-analyze cycle    | std::accumulate         | Manual pointer loop        | [01_BASIC_WORKFLOW.md](docs/01_BASIC_WORKFLOW.md)           |
| 02  | perf Profiler         | Hardware counter profiling      | Stride-512 array walk   | Sequential array walk      | [02_PERF_PROFILER.md](docs/02_PERF_PROFILER.md)             |
| 03  | gperftools Profiler   | Function-level flamegraphs      | Bubble sort O(n^2)      | std::sort O(n log n)       | [03_GPERF_PROFILER.md](docs/03_GPERF_PROFILER.md)           |
| 04  | Cache-Friendly Layout | AoS vs SoA data transformation  | 128B struct (81% waste) | Separate arrays (100% use) | [04_CACHE_FRIENDLY.md](docs/04_CACHE_FRIENDLY.md)           |
| 05  | Branch Optimization   | Branch prediction and avoidance | Branchy + random data   | Branchless + multiply      | [05_BRANCH_OPTIMIZATION.md](docs/05_BRANCH_OPTIMIZATION.md) |
| 06  | Thread Scaling        | Lock contention analysis        | Mutex-protected counter | Atomic relaxed counter     | [06_THREAD_SCALING.md](docs/06_THREAD_SCALING.md)           |
| 07  | Callgrind Profiler    | Deterministic instruction count | Linear search O(n)      | Binary search O(log n)     | [07_CALLGRIND_PROFILER.md](docs/07_CALLGRIND_PROFILER.md)   |
| 08  | RAPL Profiler         | Energy/power measurement        | Naive dot product       | Vectorized inner product   | [08_RAPL_PROFILER.md](docs/08_RAPL_PROFILER.md)             |
| 09  | bpftrace Profiler     | Syscall overhead tracing        | One write() per byte    | Single batched write()     | [09_BPFTRACE_PROFILER.md](docs/09_BPFTRACE_PROFILER.md)     |
| 10  | NVTX Annotation       | Timeline labeling for Nsight    | Single opaque region    | Per-phase named ranges     | [13_NVTX_ANNOTATION.md](docs/13_NVTX_ANNOTATION.md)         |
| 11  | Massif Profiler       | Heap usage timeline             | new[] each iteration    | Pooled buffer reuse        | [14_MASSIF_PROFILER.md](docs/14_MASSIF_PROFILER.md)         |
| 12  | Memcheck Profiler     | Leak / UAF detection            | Raw new[] (leaky)       | unique_ptr (clean)         | [15_MEMCHECK_PROFILER.md](docs/15_MEMCHECK_PROFILER.md)     |
| 13  | Off-CPU Profiler      | Where threads go to sleep       | std::mutex contention   | std::atomic counter        | [16_OFFCPU_PROFILER.md](docs/16_OFFCPU_PROFILER.md)         |

The `#` column matches the binary suffix (`BenchDemo_NN_*`); walkthrough
filenames carry their own sequential number across CPU + GPU.

---

## 3. GPU Demos

Requires NVIDIA GPU with CUDA support.

| #   | Demo               | Concept                    | Slow Path              | Fast Path               | Walkthrough                                               |
| --- | ------------------ | -------------------------- | ---------------------- | ----------------------- | --------------------------------------------------------- |
| 01  | GPU Basic Workflow | CPU vs GPU comparison      | CPU loop               | CUDA kernel             | [10_GPU_BASIC_WORKFLOW.md](docs/10_GPU_BASIC_WORKFLOW.md) |
| 02  | Nsight Profiler    | Memory coalescing analysis | Strided global reads   | Sequential global reads | [11_NSIGHT_PROFILER.md](docs/11_NSIGHT_PROFILER.md)       |
| 03  | Shared Memory Opt  | Bank conflicts and padding | Naive global transpose | Padded shared transpose | [12_SHARED_MEMORY_OPT.md](docs/12_SHARED_MEMORY_OPT.md)   |
| 04  | Compute Sanitizer  | GPU memcheck for kernels   | Deliberate OOB write   | Bounds-checked scale    | [17_COMPUTE_SANITIZER.md](docs/17_COMPUTE_SANITIZER.md)   |

Binary names: `BenchDemo_Gpu_NN_*`.

Two GPU profilers ship without a dedicated demo binary because they fire on
*every* GPU run -- consult the walkthroughs to see what to look for:

| Profiler          | Wraps                  | When to use                              | Walkthrough                                            |
| ----------------- | ---------------------- | ---------------------------------------- | ------------------------------------------------------ |
| rocprof (AMD)     | AMD GPU + HIP kernels  | AMD MI / Radeon Instinct profiling       | [18_ROCPROF_PROFILER.md](docs/18_ROCPROF_PROFILER.md)  |
| CUPTI (in-proc)   | Any GPU benchmark      | Per-launch register / smem / count       | [19_CUPTI_KERNEL_METRICS.md](docs/19_CUPTI_KERNEL_METRICS.md) |

---

## 4. Running Demos

All demos are built as performance test executables (ptests):

```bash
# Run a demo
docker compose run --rm -T dev-cuda bash -c '
  ./build/native-linux-debug/bin/ptests/BenchDemo_01_BasicWorkflow --quick
'

# Export results to CSV and analyze
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_04_CacheFriendly --csv results.csv
  source .env
  bench summary results.csv
'

# Run a specific test within a demo
docker compose run --rm -T dev-cuda bash -c '
  ./build/native-linux-debug/bin/ptests/BenchDemo_05_BranchOptimization \
    --gtest_filter="*BranchlessRandomData*"
'
```

### CLI Flags

| Flag                       | Purpose                                          |
| -------------------------- | ------------------------------------------------ |
| `--quick`                  | Fast iteration (fewer cycles/repeats)            |
| `--csv FILE`               | Export results to CSV                            |
| `--repeats N`              | Number of measurement repeats                    |
| `--cycles N`               | Iterations per repeat                            |
| `--threads N`              | Thread count for contention tests                |
| `--profile <backend>`      | Attach any registered backend (see `bench doctor` for the list: `perf`, `gperf`, `callgrind`, `rapl`, `bpftrace`, `massif`, `memcheck`, `offcpu`, `heaptrack`, `jemalloc`, `nsight`, `compute-sanitizer`, `rocprof`) |
| `--profile-output-dir DIR` | Route profile artifacts to a custom root         |
| `--profile-test-timeout N` | SIGALRM watchdog seconds under `--profile`       |
| `--gtest_filter=PATTERN`   | Run specific tests only                          |

---

## 5. Shared Workloads

[helpers/DemoWorkloads.hpp](helpers/DemoWorkloads.hpp) provides reusable
slow/fast workload pairs used across demos:

| Category    | Slow                | Fast                   | Used In |
| ----------- | ------------------- | ---------------------- | ------- |
| Cache       | Stride-512 walk     | Sequential walk        | Demo 02 |
| Cache       | AoS position sum    | SoA position sum       | Demo 04 |
| Branch      | Branchy conditional | Branchless multiply    | Demo 05 |
| Sort        | Bubble sort O(n^2)  | std::sort O(n log n)   | Demo 03 |
| Search      | Linear search O(n)  | Binary search O(log n) | Demo 07 |
| Contention  | Mutex increment     | Atomic increment       | Demo 06 |
| Dot product | Naive (dependency)  | std::inner_product     | Demo 08 |
| I/O         | Per-byte write()    | Batched write()        | Demo 09 |

All workloads are deterministic (fixed seed), compiler-resistant (volatile sinks
and dependency chains), and designed to show measurable differences.

---

## 6. Learning Path

**Getting started (30 min):**

1. Demo 01 -- Learn the basic measure-export-analyze workflow
2. Demo 04 -- See the most common high-leverage optimization (AoS to SoA)

**Profiling tools (1 hour):**

3. Demo 02 -- Hardware counters with perf (cache misses, branch misses)
4. Demo 03 -- Function-level hotspot identification with gperftools

**Intermediate (1 hour):**

5. Demo 05 -- Branch prediction and branchless coding
6. Demo 06 -- Multi-threaded contention analysis

**Advanced profiling:**

7. Demo 07 -- Deterministic instruction counting with Callgrind
8. Demo 08 -- Energy measurement with Intel RAPL
9. Demo 09 -- Syscall tracing with bpftrace

**Memory and contention deep dives:**

10. Demo 11 -- Heap profile timeline with Valgrind Massif
11. Demo 12 -- Leak / UAF detection with Valgrind Memcheck
12. Demo 13 -- Off-CPU profiling: where threads block

**GPU (requires NVIDIA GPU):**

13. Demo 10 -- CPU vs GPU comparison, transfer overhead analysis
14. Demo 11 -- Memory coalescing with Nsight
15. Demo 12 -- Shared memory optimization and bank conflicts
16. Demo 04 (GPU) -- Kernel correctness with Compute Sanitizer

**GPU instrumentation:**

17. Demo 10 (CPU) -- NVTX timeline annotation for any Nsight tool
18. Demo 19 (doc) -- CUPTI in-process kernel metrics (auto-captured)

---

## 7. See Also

- [docs/CPU_GUIDE.md](../docs/CPU_GUIDE.md) -- CPU benchmarking reference
- [docs/GPU_GUIDE.md](../docs/GPU_GUIDE.md) -- GPU benchmarking reference
- [tools/README.md](../../../tools/README.md) -- CLI tools reference
