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
7. [Reference Rigs and the Demo Contract](#7-reference-rigs-and-the-demo-contract)
8. [See Also](#8-see-also)

---

## 1. Getting Started

Build Release first: the
[Raspberry Pi 4 rig document](../docs/rigs/RIG_PI4.md) has the build command
the output below came from, and [docs/rigs/README.md](../docs/rigs/README.md)
covers other machines.

Demo 01 joins 1,000 words two ways. Measure it, then read the rows it wrote:

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_01_BasicWorkflow \
  --target-time 50ms --repeats 10 --csv run1.csv
bench summary run1.csv
```

Captured on the Raspberry Pi 4 rig, 2026-09-20, Release build. The three
result lines, without GoogleTest's framing, the harness's calibration lines
and the end-of-run table:

```
[BasicWorkflow.JoinV0]  927.611 us/call  CV=0.1%  ~1.1K calls/s  (p10=927.056 p90=928.317 sd=0.544)
[BasicWorkflow.JoinV1]  21.044 us/call  CV=0.7%  ~47.5K calls/s  (p10=20.752 p90=21.123 sd=0.149)
[BasicWorkflow.JoinSpeedup]  V0 927.564 us/call  V1 20.511 us/call  45.2x
```

Then the CSV, read back:

```
Test                   Median (us)       P10       P90        CV       Calls/sec  Stable
--------------------  ------------  --------  --------  --------  --------------  ------
BasicWorkflow.JoinV0     927.61100  927.05600  928.31700      0.1%            1078  yes
BasicWorkflow.JoinV1      21.04410  20.75180  21.12340      0.7%           47519  yes

  2 tests, sorted by name
```

Reserving the result once instead of copying it per part is 44 times faster
in this capture. The third test fails if the speedup falls to three times or
less. The framework measured both versions, wrote the CSV, and `bench summary`
read it back. Every demo follows the same pattern: measure something slow,
measure something fast, compare.

Open [docs/01_BASIC_WORKFLOW.md](docs/01_BASIC_WORKFLOW.md) for the full
walkthrough: how to read those lines, what reproduces on another machine, and
how to compare two runs.

---

## 2. CPU Demos

| #   | Demo                  | Concept                            | Slow Path                   | Fast Path                  | Walkthrough                                                 |
| --- | --------------------- | ---------------------------------- | --------------------------- | -------------------------- | ----------------------------------------------------------- |
| 01  | Basic Workflow        | Measure-export-analyze cycle       | join V0 (copy per part)     | join V1 (reserve, append)  | [01_BASIC_WORKFLOW.md](docs/01_BASIC_WORKFLOW.md)           |
| 02  | perf Profiler         | Hardware counter profiling         | Stride-512 array walk       | Sequential array walk      | [02_PERF_PROFILER.md](docs/02_PERF_PROFILER.md)             |
| 03  | gperftools Profiler   | Function-level flamegraphs         | Bubble sort O(n^2)          | std::sort O(n log n)       | [03_GPERF_PROFILER.md](docs/03_GPERF_PROFILER.md)           |
| 04  | Cache-Friendly Layout | AoS vs SoA data transformation     | 128B struct (81% waste)     | Separate arrays (100% use) | [04_CACHE_FRIENDLY.md](docs/04_CACHE_FRIENDLY.md)           |
| 05  | Branch Optimization   | Branch prediction and avoidance    | Branchy + random data       | Branchless + multiply      | [05_BRANCH_OPTIMIZATION.md](docs/05_BRANCH_OPTIMIZATION.md) |
| 06  | Thread Scaling        | Lock contention analysis           | Mutex-protected counter     | Atomic relaxed counter     | [06_THREAD_SCALING.md](docs/06_THREAD_SCALING.md)           |
| 07  | Callgrind Profiler    | Deterministic instruction count    | Linear search O(n)          | Binary search O(log n)     | [07_CALLGRIND_PROFILER.md](docs/07_CALLGRIND_PROFILER.md)   |
| 08  | RAPL Profiler         | Energy/power measurement           | Naive dot product           | Vectorized inner product   | [08_RAPL_PROFILER.md](docs/08_RAPL_PROFILER.md)             |
| 09  | bpftrace Profiler     | Syscall overhead tracing           | One write() per byte        | Single batched write()     | [09_BPFTRACE_PROFILER.md](docs/09_BPFTRACE_PROFILER.md)     |
| 10  | NVTX Annotation       | Timeline labeling for Nsight       | Single opaque region        | Per-phase named ranges     | [13_NVTX_ANNOTATION.md](docs/13_NVTX_ANNOTATION.md)         |
| 11  | Massif Profiler       | Heap usage timeline                | new[] each iteration        | Pooled buffer reuse        | [14_MASSIF_PROFILER.md](docs/14_MASSIF_PROFILER.md)         |
| 12  | Memcheck Profiler     | Leak / UAF detection               | Raw new[] (leaky)           | unique_ptr (clean)         | [15_MEMCHECK_PROFILER.md](docs/15_MEMCHECK_PROFILER.md)     |
| 13  | Off-CPU Profiler      | Where threads go to sleep          | std::mutex contention       | std::atomic counter        | [16_OFFCPU_PROFILER.md](docs/16_OFFCPU_PROFILER.md)         |
| 14  | Helgrind Profiler     | Data-race / thread-error detection | Unguarded shared counter    | std::atomic counter        | [20_HELGRIND_PROFILER.md](docs/20_HELGRIND_PROFILER.md)     |
| 15  | Heaptrack Profiler    | Who allocates, and how often       | join V0 (2 allocs per part) | join V1 (1 alloc per call) | [21_HEAPTRACK_PROFILER.md](docs/21_HEAPTRACK_PROFILER.md)   |
| 16  | jemalloc Profiler     | Sampled allocation hotspots        | Per-iter string churn       | Reserved + reused string   | [22_JEMALLOC_PROFILER.md](docs/22_JEMALLOC_PROFILER.md)     |

The `#` column matches the binary suffix (`BenchDemo_NN_*`); walkthrough
filenames carry their own sequential number across CPU + GPU.

---

## 3. GPU Demos

Requires NVIDIA GPU with CUDA support.

| #   | Demo               | Concept                             | Slow Path               | Fast Path                                   | Walkthrough                                               |
| --- | ------------------ | ----------------------------------- | ----------------------- | ------------------------------------------- | --------------------------------------------------------- |
| 01  | GPU Basic Workflow | CPU vs GPU, and what transfers cost | CPU loop over 1M floats | Same kernel, with and without its transfers | [10_GPU_BASIC_WORKFLOW.md](docs/10_GPU_BASIC_WORKFLOW.md) |
| 02  | Nsight Profiler    | Memory coalescing analysis          | Strided global reads    | Sequential global reads                     | [11_NSIGHT_PROFILER.md](docs/11_NSIGHT_PROFILER.md)       |
| 03  | Shared Memory Opt  | Bank conflicts and padding          | Naive global transpose  | Padded shared transpose                     | [12_SHARED_MEMORY_OPT.md](docs/12_SHARED_MEMORY_OPT.md)   |
| 04  | Compute Sanitizer  | GPU memcheck for kernels            | Deliberate OOB write    | Bounds-checked scale                        | [17_COMPUTE_SANITIZER.md](docs/17_COMPUTE_SANITIZER.md)   |

Binary names: `BenchDemo_Gpu_NN_*`.

Demo 01 measures the shared SAXPY example (see
[Shared Examples](#shared-examples)); the other three carry their own kernels.

Two GPU topics have a walkthrough but no dedicated demo binary:

| Profiler        | Wraps                                   | When to use                                             | Walkthrough                                                   |
| --------------- | --------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| rocprof (AMD)   | AMD GPU + HIP kernels                   | Wraps an AMD GPU run; not validated on AMD hardware     | [18_ROCPROF_PROFILER.md](docs/18_ROCPROF_PROFILER.md)         |
| CUPTI (in-proc) | Tests timed with the GPU kernel builder | Per-kernel launch count, register and shared-memory use | [19_CUPTI_KERNEL_METRICS.md](docs/19_CUPTI_KERNEL_METRICS.md) |

---

## 4. Running Demos

All demos are built as performance test executables (ptests):

```bash
# Run a demo
docker compose run --rm -T dev bash -c '
  ./build/native-linux-debug/bin/ptests/BenchDemo_01_BasicWorkflow --quick
'

# Export results to CSV and analyze
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_04_CacheFriendly --csv results.csv
  source .env
  bench summary results.csv
'

# Run a specific test within a demo
docker compose run --rm -T dev bash -c '
  ./build/native-linux-debug/bin/ptests/BenchDemo_05_BranchOptimization \
    --gtest_filter="*BranchlessRandomData*"
'
```

### CLI Flags

| Flag                       | Purpose                                                                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--quick`                  | Fast iteration (fewer cycles/repeats)                                                                                                                                                                                                                                          |
| `--csv FILE`               | Export results to CSV                                                                                                                                                                                                                                                          |
| `--repeats N`              | Number of measurement repeats                                                                                                                                                                                                                                                  |
| `--cycles N`               | Iterations per repeat                                                                                                                                                                                                                                                          |
| `--threads N`              | Thread count for contention tests                                                                                                                                                                                                                                              |
| `--target-time DUR`        | `throughputLoop` and `contentionRun` only: size each repeat to about this long instead of a fixed cycle count (`us`, `ms`, `s`). `measured()` and GPU kernel tests keep `--cycles`                                                                                             |
| `--warmup N`               | Untimed warmup calls before timing (default 1; `0` lets the framework choose)                                                                                                                                                                                                  |
| `--msg-bytes N`            | Payload size per call; sets the CV threshold used for the stability flag                                                                                                                                                                                                       |
| `--profile <backend>`      | Attach a registered backend: `perf`, `gperf`, `callgrind`, `massif`, `memcheck`, `helgrind`, `heaptrack`, `jemalloc`, `offcpu`, `bpftrace`, `rapl`, `nsight` (alias `nsys`), `ncu`, `compute-sanitizer`, `rocprof`. `bench doctor <binary>` reports which work on this machine |
| `--profile-args STR`       | Backend-specific arguments                                                                                                                                                                                                                                                     |
| `--profile-output-dir DIR` | Route profile artifacts to a custom root                                                                                                                                                                                                                                       |
| `--profile-test-timeout N` | SIGALRM watchdog seconds under `--profile`                                                                                                                                                                                                                                     |
| `--gtest_filter=PATTERN`   | Run specific tests only                                                                                                                                                                                                                                                        |
| `--gpu-device N`           | GPU tests: device index (default 0)                                                                                                                                                                                                                                            |
| `--gpu-warmup N`           | GPU tests: untimed kernel launches before timing                                                                                                                                                                                                                               |

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

### Shared Examples

Code a walkthrough teaches from lives in its own directory beside these
helpers, as `examples/<name>/{inc,src,utst}`: a small library the demo links,
and unit tests that hold the example's versions to the same answers,
registered under the `demo` label (`ctest --test-dir build -L demo`). The
first is [examples/join](examples/join/inc/Join.hpp), measured by demo 01.

| Example                               | Versions                                                                                                  | Used In      |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------ |
| [join](examples/join/inc/Join.hpp)    | V0, rebuilds the string through temporaries for every part; V1, measures, reserves once, appends in place | Demos 01, 21 |
| [saxpy](examples/saxpy/inc/Saxpy.hpp) | CPU loop; G0, one thread per block with per-call allocation; G1, buffers once at 256 threads              | Demo 10      |

The saxpy example and its tests are built only where the GPU demos are.

---

## 6. Learning Path

Walkthroughs are numbered by their file name in `docs/`.

**Getting started:**

1. [01](docs/01_BASIC_WORKFLOW.md) -- the measure, export, compare workflow
2. [04](docs/04_CACHE_FRIENDLY.md) -- a data-layout optimization (AoS to SoA)

**Profiling tools:**

3. [02](docs/02_PERF_PROFILER.md) -- hardware counters with perf
4. [03](docs/03_GPERF_PROFILER.md) -- function-level hotspots with gperftools
5. [07](docs/07_CALLGRIND_PROFILER.md) -- exact instruction counts with Callgrind

**Branches and threads:**

6. [05](docs/05_BRANCH_OPTIMIZATION.md) -- branch prediction and branchless code
7. [06](docs/06_THREAD_SCALING.md) -- contention between threads
8. [16](docs/16_OFFCPU_PROFILER.md) -- off-CPU profiling: where threads block
9. [20](docs/20_HELGRIND_PROFILER.md) -- data races with Helgrind / DRD

**Memory:**

10. [21](docs/21_HEAPTRACK_PROFILER.md) -- ranked allocation sites with heaptrack
11. [14](docs/14_MASSIF_PROFILER.md) -- heap size over time with Massif
12. [15](docs/15_MEMCHECK_PROFILER.md) -- leaks and invalid access with Memcheck
13. [22](docs/22_JEMALLOC_PROFILER.md) -- sampled allocation profiling with jemalloc

**System and energy:**

14. [09](docs/09_BPFTRACE_PROFILER.md) -- kernel tracing with bpftrace
15. [08](docs/08_RAPL_PROFILER.md) -- energy with Intel RAPL

**GPU (requires an NVIDIA GPU):**

16. [10](docs/10_GPU_BASIC_WORKFLOW.md) -- CPU vs GPU, kernel time vs transfers
17. [11](docs/11_NSIGHT_PROFILER.md) -- Nsight Systems and Nsight Compute
18. [13](docs/13_NVTX_ANNOTATION.md) -- NVTX ranges for Nsight timelines
19. [19](docs/19_CUPTI_KERNEL_METRICS.md) -- per-kernel metrics from CUPTI
20. [17](docs/17_COMPUTE_SANITIZER.md) -- kernel correctness with Compute Sanitizer
21. [12](docs/12_SHARED_MEMORY_OPT.md) -- shared memory and bank conflicts (advanced)

---

## 7. Reference Rigs and the Demo Contract

A walkthrough names the machine its output was captured on, in a
`Reference rig` line at its top, and links the rig document instead of
repeating the setup. A walkthrough without that line has not been captured
on a rig. See [docs/rigs/README.md](../docs/rigs/README.md).

| Rig                                                    | Walkthroughs |
| ------------------------------------------------------ | ------------ |
| [Raspberry Pi 4](../docs/rigs/RIG_PI4.md)              | CPU          |
| [NVIDIA Jetson AGX Thor](../docs/rigs/RIG_THOR_AGX.md) | GPU          |

A walkthrough meets this contract:

- **A named rig and a Release build.** Commands were run as written and
  output blocks are pasted from that run, with the capture date and the
  Vernier version.
- **A test that asserts its own effect.** The demo's performance test
  fails if the slow and fast variants stop differing, so a walkthrough
  cannot drift silently.
- **A statement of what reproduces.** Ratios and the profiler's finding
  should match on the same rig; absolute times differ elsewhere.
- **Something runs it.** A walkthrough is re-run on its rig before every
  release, and on the rig's CI lane where one exists. A failing assertion
  blocks the release. An assertion nothing runs protects nothing.

New walkthroughs start from [docs/TEMPLATE.md](docs/TEMPLATE.md).

---

## 8. See Also

- [docs/CPU_GUIDE.md](../docs/CPU_GUIDE.md) -- CPU benchmarking reference
- [docs/GPU_GUIDE.md](../docs/GPU_GUIDE.md) -- GPU benchmarking reference
- [docs/rigs/README.md](../docs/rigs/README.md) -- reference rigs
- [tools/README.md](../../../tools/README.md) -- CLI tools reference
