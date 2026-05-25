# Vernier

**Namespace:** `vernier::bench`
**Platform:** Linux (full), macOS (core harness)
**C++ Standard:** C++23

Performance benchmarking framework with profiler integrations, GPU support,
and statistical analysis.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Key Features](#2-key-features)
3. [Common Workflows](#3-common-workflows)
4. [CLI Tools and Backends](#4-cli-tools-and-backends)
5. [API Reference](#5-api-reference)
6. [Requirements](#6-requirements)
7. [Platform Support](#7-platform-support)
8. [Testing](#8-testing)
9. [Project Structure](#9-project-structure)
10. [License](#10-license)
11. [See Also](#11-see-also)

---

## 1. Quick Start

```cpp
#include "Perf.hpp"

PERF_TEST(MyLib, Throughput) {
  UB_PERF_GUARD(perf);
  perf.warmup([&]{ work(); });
  auto result = perf.throughputLoop([&]{ work(); }, "label");
  EXPECT_GT(result.callsPerSecond, 10000.0);
}

PERF_MAIN()
```

### Build and Run (Docker)

```bash
make compose-debug
make compose-testp

docker compose run --rm -T dev-cuda bash -c '
  ./build/native-linux-debug/bin/ptests/BenchmarkCPU_PTEST --csv results.csv
'
```

### Build Without Docker

```bash
cmake --preset native-linux-debug
cmake --build --preset native-linux-debug
./build/native-linux-debug/bin/ptests/BenchmarkCPU_PTEST --csv results.csv
```

---

## 2. Key Features

- GoogleTest integration with CSV export and end-of-run summary tables
- 13 self-registering profiler backends covering CPU, heap, off-CPU, energy,
  and both NVIDIA + AMD GPU stacks (see Section 4 for the list)
- Per-backend environment doctor (`--profile-check`) with actionable hints
- SIGALRM per-test watchdog so hung profiler runs fail loudly, not silently
- CUDA GPU benchmarking with multi-GPU and Unified Memory support, plus
  in-process CUPTI kernel metrics (register / smem / launch counts) without
  spawning ncu
- NVTX timeline annotation API auto-injected into Nsight Systems runs
- Companion `vernier::monitor` library for lightweight runtime
  instrumentation in production runs (lock-free queue, env-var-driven
  enablement, console + file sinks)
- Statistical analysis: median, percentiles, CV%, adaptive stability detection
- Memory bandwidth analysis with efficiency calculations
- Multi-threaded contention benchmarking with synchronized start gates
- Semantic test macros (PERF_THROUGHPUT, PERF_LATENCY, PERF_MEMORY, etc.)
- CLI tools for analysis, comparison, regression detection, visualization,
  doctor / profile-all / profile-summarize orchestration, project-level
  defaults via `.bench.yaml`

---

## 3. Common Workflows

### Optimization Workflow

```bash
# 1. Baseline measurement
./bin/ptests/MyComponent_PTEST --repeats 30 --csv baseline.csv

# 2. Profile to find hotspots
./bin/ptests/MyComponent_PTEST --profile perf

# 3. Make changes, rebuild, measure again
./bin/ptests/MyComponent_PTEST --repeats 30 --csv optimized.csv

# 4. Statistical comparison
bench compare baseline.csv optimized.csv --threshold 5
```

### Quick Iteration

```bash
./bin/ptests/BenchmarkCPU_PTEST --quick --gtest_filter="*Throughput*"
```

### Install as Library

```bash
make compose-release
make install
```

Consumers use `find_package(vernier)`:

```cmake
find_package(vernier REQUIRED)
target_link_libraries(my_benchmark PRIVATE vernier::bench)
```

The install tree contains headers, shared libraries, CMake config, and documentation
under `build/native-linux-release/install/`.

---

## 4. CLI Tools and Backends

CLI tools build with `make tools-rust` and `make tools-py`; source `.env`
from the build directory to put them on PATH.

| Tool           | Language | Purpose                                                                                                                                                    |
| -------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bench`        | Rust     | Analysis, comparison, validation, run, doctor, profile-all, profile-summarize, init, config-validate, gpu-env, gpu-lock, gpu-monitor, gpu-topo, flamegraph |
| `bench-plot`   | Python   | Visualization (plots, dashboards, charts)                                                                                                                  |
| `nsight-parse` | Python   | Turn `.nsys-rep` / `.ncu-rep` reports into a tidy CSV                                                                                                      |

### Registered profiler backends

`--profile X` dispatches to whichever backend self-registered under that
name; `bench doctor` lists them all with their environment readiness.

| Backend             | Layer | Wraps                                         |
| ------------------- | ----- | --------------------------------------------- |
| `perf`              | CPU   | Linux perf_events (stat / record / mem / c2c) |
| `gperf`             | CPU   | gperftools                                    |
| `callgrind`         | CPU   | valgrind callgrind                            |
| `bpftrace`          | CPU   | bpftrace scripts                              |
| `rapl`              | CPU   | Intel RAPL MSRs                               |
| `massif`            | CPU   | valgrind massif (heap timeline, ~20x)         |
| `memcheck`          | CPU   | valgrind memcheck (errors / leaks)            |
| `offcpu`            | CPU   | bpftrace finish_task_switch (off-CPU stacks)  |
| `heaptrack`         | CPU   | heaptrack (low-overhead heap, ~1.5x)          |
| `jemalloc`          | CPU   | jemalloc prof sampling (~5-10%, LD_PRELOAD)   |
| `nsight`            | GPU   | NVIDIA Nsight Systems / Compute               |
| `compute-sanitizer` | GPU   | NVIDIA Compute Sanitizer (GPU memcheck)       |
| `rocprof`           | GPU   | AMD ROCm rocprof                              |

CUPTI kernel metrics populate the GPU CSV section automatically on every
GPU benchmark; NVTX annotations are available via `BENCH_NVTX_SCOPE`.

```bash
bench summary results.csv
bench compare baseline.csv candidate.csv --fail-on-regression
bench doctor ./build/native-linux-debug/bin/ptests/MyComponent_PTEST
bench profile-all MyComponent --quick
bench-plot plot results.csv --output charts/
```

See [tools/README.md](tools/README.md) for full CLI documentation.

---

## 5. API Reference

| Document                                                 | Purpose                                      |
| -------------------------------------------------------- | -------------------------------------------- |
| [CPU Guide](src/bench/docs/CPU_GUIDE.md)                 | CPU benchmarking patterns and profiler usage |
| [GPU Guide](src/bench/docs/GPU_GUIDE.md)                 | GPU/CUDA benchmarking patterns               |
| [API Reference](src/bench/docs/API_REFERENCE.md)         | Complete API documentation                   |
| [Advanced Guide](src/bench/docs/ADVANCED_GUIDE.md)       | Memory profiling, parameterized tests        |
| [CI/CD Integration](src/bench/docs/CI_CD_INTEGRATION.md) | Automated regression detection               |
| [Docker Setup](src/bench/docs/DOCKER_SETUP.md)           | Container build and profiling setup          |
| [Troubleshooting](src/bench/docs/TROUBLESHOOTING.md)     | Common issues and solutions                  |
| [Demo Walkthroughs](src/bench/demo/docs/)                | 17 step-by-step tutorials (13 CPU + 4 GPU)   |
| [Monitor Guide](src/monitor/docs/MONITOR_GUIDE.md)       | Runtime instrumentation library              |

---

## 6. Requirements

**Required:**

- C++23 compiler (clang-21 recommended, GCC 13+ also works)
- CMake 3.24+
- GoogleTest (auto-fetched via CMake FetchContent)
- POSIX system (Linux or macOS)

**Optional:**

- CUDA toolkit 12+ (GPU benchmarking, NVTX, CUPTI, Compute Sanitizer)
- gperftools (`gperf` backend)
- valgrind (`callgrind` / `massif` / `memcheck` backends)
- bpftrace (`bpftrace` and `offcpu` backends; needs root + tracefs)
- heaptrack (`heaptrack` backend -- low-overhead heap profiler)
- jemalloc with `prof` enabled (`jemalloc` backend; LD_PRELOAD)
- ROCm + rocprof (AMD GPU profiling via the `rocprof` backend)
- Rust toolchain (for `bench` CLI tool)
- Python 3.10+ with Poetry (for `bench-plot` and `nsight-parse` CLI tools)

---

## 7. Platform Support

| Platform                  | Library | Profilers             | CUDA | Pre-built Artifact              |
| ------------------------- | ------- | --------------------- | ---- | ------------------------------- |
| x86_64 Linux              | Full    | All 13                | Yes  | `vernier-*-x86_64-linux[-cuda]` |
| Jetson (aarch64)          | Full    | All except RAPL       | Yes  | `vernier-*-aarch64-jetson`      |
| Raspberry Pi (aarch64)    | Full    | CPU backends, no RAPL | No   | `vernier-*-aarch64-rpi`         |
| RISC-V 64                 | Full    | CPU backends, no RAPL | No   | `vernier-*-riscv64-linux`       |
| macOS (Apple Silicon/x86) | Full    | No-ops                | No   | Build from source               |

`rapl` is Intel-only (energy via MSRs); `rocprof` is AMD-only; `nsight` /
`compute-sanitizer` / `cupti` / `nvtx` are NVIDIA-only. All backends
degrade gracefully when hardware or tools are unavailable -- the core
timing harness always works.

---

## 8. Testing

```bash
# Build and run all tests (Docker)
make compose-debug
make compose-testp

# Run specific library tests
docker compose run --rm -T dev-cuda ctest --test-dir build/native-linux-debug -L bench

# CLI tool tests
make test-rust
make test-py
```

---

## 9. Project Structure

```
vernier/
  CMakeLists.txt              Root project (version, presets, CUDA detection)
  Makefile                    Build entry point (make help for full list)
  docker-compose.yml          Dev containers (CPU, CUDA, cross-compile)
  cmake/vernier/              CMake infrastructure (targets, testing, coverage)
  docker/                     Dockerfiles (base, dev, builder, toolchain)
  mk/                         Make modules (build, test, docker, coverage)
  src/
    bench/                    Benchmarking library (perf, GPU harness, profilers)
      inc/                    Public headers (Perf.hpp, PerfGpu.hpp, Nvtx.hpp, profilers)
      src/                    Profiler implementations + CUPTI collector
      bpf/                    bpftrace scripts (off-cpu, syslat)
      utst/                   Unit tests
      ptst/                   Performance tests (CPU + GPU)
      demo/                   13 CPU + 4 GPU walkthroughs with step-by-step docs
      docs/                   Library documentation
    monitor/                  Runtime instrumentation library (vernier::monitor)
      inc/                    Public headers (Monitor.hpp, MonitorConfig.hpp)
      src/                    Sink implementations
      utst/                   Unit tests
      examples/               End-to-end usage examples
      docs/                   MONITOR_GUIDE.md
  tools/
    rust/                     bench CLI (Rust) -- analysis, doctor, run, gpu-*
    py/                       bench-plot, nsight-parse CLIs (Python)
```

---

## 10. License

MIT License. See [LICENSE](LICENSE) for details.

---

## 11. See Also

- [tools/README.md](tools/README.md) - CLI tools documentation (bench, bench-plot)
- [src/bench/docs/CPU_GUIDE.md](src/bench/docs/CPU_GUIDE.md) - CPU benchmarking guide
- [src/bench/docs/GPU_GUIDE.md](src/bench/docs/GPU_GUIDE.md) - GPU benchmarking guide
- [src/bench/docs/](src/bench/docs/) - Technical documentation
