# Vernier

**Namespace:** `vernier::bench`
**Platform:** Linux (full), macOS (core harness)
**C++ Standard:** C++20 (C++23 used when available)

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
  PERF_GUARD(perf);
  perf.warmup([&]{ work(); });
  auto result = perf.throughputLoop([&]{ work(); }, "label");
  EXPECT_GT(result.callsPerSecond, 10000.0);
}

PERF_MAIN()
```

### Build and Run (Docker)

```bash
make compose-release

docker compose run --rm -T dev-cuda \
  ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST --csv results.csv
```

`make compose-release` builds with the Release preset in the `dev-cuda`
service, which reserves the machine's NVIDIA GPUs (`docker-compose.yml`). The
CPU `dev` service needs none:

```bash
docker compose run --rm -T dev make release
docker compose run --rm -T dev \
  ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST --csv results.csv
```

### Build Without Docker

```bash
cmake --preset native-linux-release
cmake --build --preset native-linux-release
./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST --csv results.csv
```

The presets compile with `clang-21`; for GCC, add
`-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++` to the configure line. The
build also produces the CLI tools when `cargo` is on `PATH`, and the Python
tools when `poetry` and `pip` are; `source build/native-linux-release/.env`
puts them on `PATH`. The Docker and host routes share
`build/native-linux-release`, and CMake refuses a tree configured on the other
side, so remove the directory when switching between them.

---

## 2. Key Features

- GoogleTest integration with CSV export and end-of-run summary tables
- 15 profiler backends covering CPU, heap, off-CPU, energy, thread-safety,
  and both NVIDIA + AMD GPU stacks (see Section 4 for the list)
- Per-backend environment doctor (`--profile-check`) with actionable hints
- SIGALRM watchdog on the `measured()` and `throughputLoop()` loops under
  `--profile`, so such a loop that hangs while profiled fails loudly (exit
  status 2) instead of hanging the run; `contentionRun()` and GPU measurements
  are not covered
- CUDA GPU benchmarking with multi-GPU support, plus in-process CUPTI kernel
  metrics (register / smem / launch counts) without spawning ncu, and, under
  `--capture-um`, CUPTI's unified-memory migration and page-fault counters on
  GPUs whose CUPTI records them
- NVTX timeline annotation API auto-injected into Nsight Systems runs
- Companion `vernier::monitor` library for lightweight runtime
  instrumentation in production runs (lock-free queue, env-var-driven
  configuration, console + file sinks)
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

`MyComponent_PTEST` stands for a benchmark of your own project, defined at the
top of its CMake build (see [Install as Library](#install-as-library)).

```bash
# 1. Baseline measurement
./build/MyComponent_PTEST --repeats 30 --csv baseline.csv

# 2. Profile to find hotspots (perf record; see the CPU guide)
./build/MyComponent_PTEST --profile perf --profile-args "record -g" --target-time 250ms
perf report -i MyComponent.Throughput.perf/perf.data

# 3. Make changes, rebuild, measure again
./build/MyComponent_PTEST --repeats 30 --csv optimized.csv

# 4. Compare the two runs
bench compare baseline.csv optimized.csv --threshold 5
```

### Quick Iteration

```bash
./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST --quick --gtest_filter="*Throughput*"
```

### Install as Library

```bash
make install            # on the host
make compose-install    # or in the dev-cuda service
```

Both run the Release preset build and install it.

Consumers use `find_package(vernier)`:

```cmake
find_package(vernier REQUIRED)
target_link_libraries(my_benchmark PRIVATE vernier::bench)
target_link_libraries(my_app PRIVATE vernier::monitor)
```

`vernier::bench` brings GoogleTest's headers and library with it: `Perf.hpp`
includes GoogleTest and `PERF_MAIN()` runs the GoogleTest cases.
`find_package(vernier)` looks up GTest even for a consumer that uses only the
monitor, so GoogleTest must be installed where CMake finds it and must provide
the `GTest::gtest` target. CMake 3.20 or newer defines that target through its
FindGTest module, and GoogleTest's own package configuration
(`GTestConfig.cmake`) defines it; without it `find_package(vernier)` stops with
a message naming `GTest::gtest`. To point CMake at an installed GoogleTest, set
`GTest_DIR` to the directory holding its `GTestConfig.cmake`, or add its install
prefix to `CMAKE_PREFIX_PATH`. A benchmark may still link `GTest::gtest` itself;
it needs no `GTest::gtest_main`, because `PERF_MAIN()` supplies `main()`. A GPU
benchmark links `vernier::bench_cuda` together with `vernier::bench`.

The install tree contains headers, shared libraries, CMake config, and documentation
under `build/native-linux-release/install/`.

#### Header layout

Headers keep their source-tree paths under one directory,
`<prefix>/<includedir>/vernier/src/<module>/inc/`, where `<includedir>` is
`CMAKE_INSTALL_INCLUDEDIR` (`include` by default). The exported targets put
that `vernier/` directory and the module's directory on the include path, so
`#include "Perf.hpp"` and `#include "src/bench/inc/Perf.hpp"` both work, as do
`"Monitor.hpp"` and `"src/monitor/inc/Monitor.hpp"`, with no include directory
of the consumer's own. With relative install directories (the default), the
package finds its files relative to its own location, so an installed prefix
can be moved.

#### Library versions

A shared library's SONAME is its ABI identity and is separate from the release
number. `libmonitor` follows the release major (`libmonitor.so.1`,
file `libmonitor.so.<release>`). `libbench` and `libbench_cuda` carry their own
ABI number, currently 2 (`libbench.so.2`, file `libbench.so.2.<minor>.<patch>`):
they exchange `PerfConfig`, `Stats` and `PerfGpuConfig` objects with header code
compiled into the benchmark, and that layout changed after the `.so.1`
libraries, so the two ABIs are not interchangeable.

For a packager this means:

- The first component of a bench library's file name is the ABI number, not the
  release major, so installing a new release never overwrites the file an older
  SONAME symlink points at. `libbench.so.1` and `libbench.so.2` can sit in one
  directory, each resolving to the library built for it.
- There is no `libbench.so.1` compatibility link to the current libraries. A
  benchmark built against the older ABI stops at
  `libbench.so.1: cannot open shared object file` instead of running against an
  incompatible layout. Rebuild it against the headers that ship with these
  libraries.
- Raise the ABI number in `src/bench/CMakeLists.txt` (`BENCH_ABI_SOVERSION`) for
  any change to those shared structs, to an exported signature or to
  `Profiler`'s virtual functions (the libraries build the profilers that
  benchmark code calls), even when no struct's size changes. Both bench
  libraries read that one number; until a release ships it,
  `BENCH_ABI_VERSION` in `src/bench/inc/PerfAbi.hpp` alone moves.

---

## 4. CLI Tools and Backends

The Quick Start's preset build also builds the CLI tools (when `cargo`, and
`poetry` with `pip`, are available); `source build/native-linux-release/.env`
puts them on `PATH`. `make tools-rust` and `make tools-py` rebuild only the
tools, in `build/native-linux-debug` unless `BUILD_DIR` names another build
directory.

| Tool         | Language | Purpose                                                                                                                                                                  |
| ------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bench`      | Rust     | Analysis, comparison, validation, run, doctor, profile-all, profile-summarize, nsight-parse, init, config-validate, gpu-env, gpu-lock, gpu-monitor, gpu-topo, flamegraph |
| `bench-plot` | Python   | Visualization (plots, dashboards, charts)                                                                                                                                |

### Registered profiler backends

`--profile X` dispatches to whichever backend self-registered under that
name; `bench doctor` lists them all with their environment readiness
(`--json` for CI capability records, `--require a,b` to gate a profile
lane on the backends it needs, `--profile X` with its options to check one
request).

| Backend             | Layer | Wraps                                         |
| ------------------- | ----- | --------------------------------------------- |
| `perf`              | CPU   | Linux perf_events (stat / record / mem / c2c) |
| `gperf`             | CPU   | gperftools                                    |
| `callgrind`         | CPU   | valgrind callgrind                            |
| `bpftrace`          | CPU   | bpftrace scripts                              |
| `rapl`              | CPU   | Intel RAPL MSRs                               |
| `massif`            | CPU   | valgrind massif (heap timeline, ~20x)         |
| `memcheck`          | CPU   | valgrind memcheck (errors / leaks)            |
| `helgrind`          | CPU   | valgrind helgrind / DRD (data races, locks)   |
| `offcpu`            | CPU   | bpftrace on the sched tracepoints (off-CPU)   |
| `heaptrack`         | CPU   | heaptrack (low-overhead heap, ~1.5x)          |
| `jemalloc`          | CPU   | jemalloc's sampling heap profiler (preloaded) |
| `nsight`            | GPU   | NVIDIA Nsight Systems (timeline)              |
| `ncu`               | GPU   | NVIDIA Nsight Compute (per-kernel analysis)   |
| `compute-sanitizer` | GPU   | NVIDIA Compute Sanitizer (GPU memcheck)       |
| `rocprof`           | GPU   | AMD ROCm rocprof                              |

CUPTI kernel metrics populate the GPU CSV section automatically on every
GPU benchmark; NVTX annotations are available via `BENCH_NVTX_SCOPE`.

```bash
bench summary results.csv
bench compare baseline.csv candidate.csv --fail-on-regression
bench doctor ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST
bench profile-all MyComponent --quick
bench-plot plot results.csv --output charts/
```

See [tools/README.md](tools/README.md) for full CLI documentation.

---

## 5. API Reference

| Document                                                 | Purpose                                                                     |
| -------------------------------------------------------- | --------------------------------------------------------------------------- |
| [CPU Guide](src/bench/docs/CPU_GUIDE.md)                 | CPU benchmarking patterns and profiler usage                                |
| [GPU Guide](src/bench/docs/GPU_GUIDE.md)                 | GPU/CUDA benchmarking patterns                                              |
| [API Reference](src/bench/docs/API_REFERENCE.md)         | Complete API documentation                                                  |
| [Advanced Guide](src/bench/docs/ADVANCED_GUIDE.md)       | Memory profiling, parameterized tests                                       |
| [CI/CD Integration](src/bench/docs/CI_CD_INTEGRATION.md) | Automated regression detection                                              |
| [Docker Setup](src/bench/docs/DOCKER_SETUP.md)           | Container build and profiling setup                                         |
| [Troubleshooting](src/bench/docs/TROUBLESHOOTING.md)     | Common issues and solutions                                                 |
| [Demo Walkthroughs](src/bench/demo/docs/)                | 22 step-by-step walkthroughs (16 CPU + 4 GPU demos, plus rocprof and CUPTI) |
| [Reference Rigs](src/bench/docs/rigs/README.md)          | The machines and conditions the walkthrough outputs were captured on        |
| [Monitor Guide](src/monitor/docs/MONITOR_GUIDE.md)       | Runtime instrumentation library                                             |

---

## 6. Requirements

**Required:**

- C++20 compiler or newer (Clang 12+ / GCC 10+); C++23 is used automatically
  when the toolchain supports it (Clang 21 / GCC 13+)
- CMake 3.24+
- GoogleTest (auto-fetched via CMake FetchContent)
- POSIX system (Linux or macOS)

**Optional:**

- CUDA toolkit 12+ (GPU benchmarking, NVTX, CUPTI, Compute Sanitizer)
- gperftools (`gperf` backend)
- valgrind (`callgrind` / `massif` / `memcheck` backends)
- bpftrace (`bpftrace` and `offcpu` backends; needs root + tracefs)
- heaptrack (`heaptrack` backend -- low-overhead heap profiler)
- a jemalloc built with profiling (`--enable-prof`), for the `jemalloc` backend,
  which preloads it; `jeprof` reads its dumps
- ROCm + rocprof (AMD GPU profiling via the `rocprof` backend)
- Rust toolchain (for `bench` CLI tool)
- Python 3.10+ with Poetry (for the `bench-plot` CLI tool)

---

## 7. Platform Support

| Platform                  | Library | Profilers             | CUDA | Pre-built Artifact              |
| ------------------------- | ------- | --------------------- | ---- | ------------------------------- |
| x86_64 Linux              | Full    | All 15                | Yes  | `vernier-*-x86_64-linux[-cuda]` |
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

Every test CTest runs is labelled with its class, `unit` or `component`, and
also `rig` when it needs hardware, privileges or a device a hosted runner
lacks:

- **`unit`** -- one unit in isolation (a function, class, decision, parser or
  reader), its collaborators values or stand-ins the test controls, with no
  real tool, device, privilege or network. About 1 s a test at most, a whole
  unit program in seconds. `make test` (serial) and `make testp` (parallel)
  run them, and so do `make verify` and CI on every pull request.
- **`component`** -- a component through its real protocol, or an
  integration: a backend started and stopped, a benchmark run end to end, a
  real tool, a consumer or install build. About 10 s a test.
  `make test-component` runs them, and so does the nightly workflow on main,
  off the pull-request path.
- **`rig`**, beside either class -- needs a GPU, hardware counters, the RAPL
  registers, privileged bpftrace or the like. `make test-rig` runs them, on a
  host that has what they need; no other lane does.

The times are guidance for review, not limits: no test fails for its time,
and `make test`, `make testp`, `make test-component` and `make test-rig` end
by listing their slowest tests. A lane whose selection matches no test fails.

The nightly workflow (`.github/workflows/nightly.yml`) runs every check
`mk/checks.mk` registers, each in the CPU dev image, on a schedule and from the
Actions tab; `make list-checks` names them. A check is added by naming its make
target there, with no change to the workflow.

Beside `make testp`, every pull request runs these lanes on unit tests:
`make test-heap` (the allocator-dependent cases, in a tree with tcmalloc
linked), `make asan` (the CPU unit programs' unit tests under
AddressSanitizer), and `make cuda-check` (every CUDA target compiled and
linked, then the cases that need no device), with `make tools-check`, which
checks that the debug build made every CLI tool. Their component cases run
nightly: `make test-heap-component` and `make asan-component`, beside
`make test-component`.

```bash
# Build and run the unit tests (Docker)
make compose-debug
make compose-testp

# The component tests, and the rig tests on a host that has the hardware
make test-component
make test-rig

# Run specific library tests. bash stays the container's first process (the
# trailing exit keeps it from replacing itself with ctest) and reaps the helper
# processes some tests leave behind, which would otherwise keep ctest waiting.
docker compose run --rm -T dev-cuda bash -c 'ctest --test-dir build/native-linux-debug -L bench; exit $?'

# CLI tool tests
make test-rust
make test-py

# Sanitizers, CPU only: the unit selection in build/asan, build/tsan or
# build/ubsan (make asan-component runs the component one under ASan); a
# failing test or any sanitizer report fails the target, whatever
# ASAN_OPTIONS, LSAN_OPTIONS, TSAN_OPTIONS or UBSAN_OPTIONS say
make compose-asan

# The whole suite under a sanitizer, in the tree that target configured, with
# bash as the container's first process as above
docker compose run --rm -T dev bash -c 'cmake --build build/asan -j8 && ctest --test-dir build/asan -j8; exit $?'
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
      bpf/                    bpftrace scripts (write / fsync / wakeup latency, CPU migrations)
      utst/                   Unit tests
      dtst/                   Component tests (backend and probe lifecycles)
      ptst/                   Performance tests (CPU + GPU)
      demo/                   16 CPU + 4 GPU walkthroughs with step-by-step docs
      docs/                   Library documentation
    monitor/                  Runtime instrumentation library (vernier::monitor)
      inc/                    Public headers (Monitor.hpp, MonitorConfig.hpp)
      src/                    Sink implementations
      utst/                   Unit tests
      examples/               End-to-end usage examples
      docs/                   MONITOR_GUIDE.md
  tools/
    rust/                     bench CLI (Rust) -- analysis, doctor, run, gpu-*
    py/                       bench-plot CLI (Python)
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
