# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## v1.0.2 - Unreleased

### Added

- **Self-registering profiler backends** -- profiler dispatch refactored
  from a hardcoded if-chain to a registry. New backends slot in via a
  single `VERNIER_REGISTER_PROFILER_BACKEND` line at file scope.
- **Eight new profiler backends** (registry now lists 14):
  - `massif`            -- valgrind heap profiler (full timeline, ~20x)
  - `memcheck`          -- valgrind memory error / leak detector
  - `helgrind`          -- valgrind thread-error detector: data races, lock
    order, pthread misuse (`--profile-args drd` selects DRD). CPU analog of
    compute-sanitizer's racecheck
  - `offcpu`            -- bpftrace finish_task_switch (where threads sleep)
  - `heaptrack`         -- low-overhead heap profiler (~1.5x)
  - `jemalloc`          -- jemalloc prof sampling (~5-10%, LD_PRELOAD)
  - `compute-sanitizer` -- NVIDIA GPU memcheck (race / sync / init)
  - `rocprof`           -- AMD ROCm GPU profiler
  - `perf` `mem`/`c2c`  -- memory + cache-line-contention submodes
- **CUPTI in-process kernel metrics** -- per-launch register count, static /
  dynamic shared memory, kernel name surface in the GPU CSV section without
  spawning `ncu` as an external process.
- **NVTX annotation API** (`BENCH_NVTX_SCOPE` / `BENCH_NVTX_MARK`) +
  auto-injection by `NsightProfiler` so nsys timelines label by test name
  for free.
- **GPU power + thermal CSV columns** -- `powerDrawW`, `powerLimitW`,
  `temperatureC`, `temperatureDeltaC` populated from NVML on every GPU
  benchmark.
- **Backend environment doctor** -- `--profile-check` now also iterates
  every registered backend and reports `[OK] / [WARN] / [FAIL]` with the
  exact remediation hint (e.g. `sudo sysctl -w kernel.perf_event_paranoid=1`).
- **Per-test watchdog under --profile** -- SIGALRM-based timeout (300s
  default, override with `--profile-test-timeout`) aborts hung tests with
  a diagnostic naming the test and the profile tool.
- **Auto-extract nsys stats** -- after `--profile nsight`, the backend
  drops `cuda_gpu_kern_sum.txt`, `cuda_api_sum.txt`,
  `cuda_gpu_mem_size_sum.txt`, `cuda_gpu_mem_time_sum.txt` next to the
  `.nsys-rep` so the user can grep without a separate CLI round-trip.
- **Docker-aware attach for callgrind + nsight** -- detects PID namespaces
  and prints the precise wrap-externally invocation instead of failing
  silently.
- **New Rust CLI subcommands**:
  - `bench doctor` -- runs `--profile-check` against a binary
  - `bench profile-all` -- iterate every profiler with one invocation
  - `bench profile-summarize` -- tabulate artifacts by tool
  - `bench gpu-topo` -- GPU/CPU NUMA affinity recommendations
  - `bench init` -- scaffold a `.bench.yaml`
  - `bench config-validate` -- sanity-check a `.bench.yaml`
- **Project-level `.bench.yaml`** -- defaults for cycles, repeats,
  profile_output_dir, gtest_filter, bin_roots, bin_subdirs. Read by
  `bench run` and `bench profile-all` when CLI flags omit values;
  unknown keys preserved as extras so projects can layer custom fields.
- **`nsight-parse` Python tool** -- consolidates `.nsys-rep` + `.ncu-rep`
  reports into a tidy CSV the rest of the toolchain can consume.
- **Monitor integration polish** -- `VERNIER_MONITOR_GAUGE` and
  `VERNIER_MONITOR_INCREMENT` macros to match the existing
  `VERNIER_MONITOR_SCOPE`; `configFromEnv()` reads `VERNIER_MONITOR`,
  `VERNIER_MONITOR_FILE`, `VERNIER_MONITOR_CONSOLE`,
  `VERNIER_MONITOR_DISABLE`, `VERNIER_MONITOR_QUEUE` so the same binary
  stays silent in production and emits a full report once an operator
  sets the env var. New `MONITOR_GUIDE.md` plus an end-to-end example
  binary.
- **`UB_PERF_GPU_GUARD` auto-attaches** profiler hooks the same way the
  CPU `UB_PERF_GUARD` does; GPU tests no longer need a manual
  `attachGpuProfilerHooks` call.
- **`--profile-output-dir <path>`** (alias for `--artifact-root`) routes
  every backend's artifacts to a user-chosen root.
- **`--profile-check` filter validation** -- if `--gtest_filter` matched
  zero tests under `--profile`, emit a loud warning naming the profiler
  and pointing at `--gtest_list_tests`.
- **Container-aware `--csv` warning** -- writing the CSV outside the
  workspace mount inside a container now warns at flag-parse time.
- **Seven new CPU demo binaries** with Slow/Fast story arcs:
  `BenchDemo_10_NvtxAnnotation`, `_11_MassifProfiler`,
  `_12_MemcheckProfiler`, `_13_OffCpuProfiler`, `_14_HelgrindProfiler`,
  `_15_HeaptrackProfiler`, `_16_JemallocProfiler`. One new GPU demo:
  `BenchDemo_Gpu_04_ComputeSanitizerProfiler`.
- **Ten new walkthrough docs** (13-22): NVTX, Massif, Memcheck, Off-CPU,
  Compute Sanitizer, rocprof, CUPTI, Helgrind, Heaptrack, jemalloc.
- **TROUBLESHOOTING.md extensions** -- drain-loop / blocking-recv hang
  patterns, the `timeout`+env-var trap, container CSV / artifact path
  routing.
- **`vernier_nvtx_enable()` CMake helper** -- mirrors
  `vernier_nvml_enable()` / `vernier_cupti_enable()`. Propagates the
  CUDA `nvtx3` interface target to a consumer so `Nvtx.hpp`'s
  `BENCH_NVTX_*` macros emit real ranges when the toolkit is present
  and compile to no-ops otherwise.
- **`bench run --profile <X>` auto-wraps wrap-externally backends.**
  When `<X>` is `callgrind`, `massif`, `memcheck`, `helgrind`,
  `heaptrack`, or `compute-sanitizer`, `bench run` invokes the correct wrap command
  (`valgrind --tool=...`, `heaptrack -o ...`, etc.) instead of just
  running the binary directly and leaving the user to copy the
  printed instruction. Artifacts land in
  `<--profile-output-dir>/<binary-stem>.<tool>/`. In-process
  backends (perf, gperf, rapl, bpftrace, offcpu) are unchanged.

### Changed

- **Build `.env` paths are absolute.** `tools/CMakeLists.txt` now
  embeds `${CMAKE_BINARY_DIR}` directly in the generated `.env` file
  instead of writing `$PWD`-relative entries. `. build/<preset>/.env`
  works from anywhere -- the previous "cd into the build dir first"
  step is no longer required. Generation moved to a `configure_file`
  template (`tools/env-helper.sh.in`).

- **Profiler dispatch refactored** to the registry; no behavior change for
  the six pre-existing backends.
- **`VERNIER_BENCH_BIN_ROOTS` / `VERNIER_BENCH_BIN_SUBDIRS`** env-var
  overrides let `bench run <short-name>` work for projects whose build
  layout isn't CMake's default (`build/*/bin/{ptests,tests,examples}`).
- **gtest dependency contained** to `PERF_MAIN` macro expansion -- library
  TUs no longer pull `<gtest/gtest.h>` transitively, which avoids
  collisions when vernier is FetchContent'd alongside a parent project's
  own gtest setup.

### Fixed

- **`HOST_UID` / `HOST_GID` rename end-to-end** -- builder Dockerfiles
  already used these args but compose / base / mk passed bare `UID` /
  `GID` which silently failed in interactive shells (UID is a bash
  readonly builtin). All Dockerfiles, compose, the makefile, and CI
  now use `HOST_UID` / `HOST_GID` uniformly.
- **CSV listener GPU-mode detection** -- previously scanned test names
  for "Gpu" / "CUDA" substrings, which missed `PERF_GPU_BANDWIDTH(Foo,
  Bar)`-style tests. Now reads an explicit flag set by `PERF_GPU_MAIN`.
- **Valgrind under-detection** -- massif / memcheck reported "not running
  under valgrind" (and printed the manual-wrap hint) even when the auto-wrap
  had the binary under valgrind, because they checked `getenv(
  "RUNNING_ON_VALGRIND")` -- a valgrind *client request*, never an env var.
  They now scan `/proc/self/maps` for the `vgpreload` library. compute-sanitizer
  detection gained the same `/proc/self/maps` fallback (its injection env var
  drifted across CUDA releases).
- **perf doctor false `[OK]`** -- `bench doctor` reported perf ready off the
  `command -v perf` + `perf_event_paranoid` checks alone, even when the perf
  wrapper had no kernel-matched build. It now also verifies `perf --version`
  succeeds and warns with the `linux-tools-$(uname -r)` remediation otherwise.
- **`.env` not `set -u`-safe** -- the generated tools `.env` expanded
  `$PYTHONPATH` unguarded, so `set -u` scripts aborted on `source`. Now uses
  `${PYTHONPATH:-}`.
- **`VERNIER_DISABLE_CUPTI` honored in-process** -- now also skips in-process
  CUPTI callback registration in the collector constructor, so an external
  Nsight (nsys / ncu) session can claim the single CUPTI client slot.

### Changed (cont.)

- **`bench profile-all`** accepts `--profile-output-dir` as a visible alias for
  `--out`, matching `bench run`'s flag name.

---

## v1.0.1

### Added

- **GPU CLI tools** -- Three new `bench` subcommands for GPU benchmarking:
  - `bench gpu-env` -- GPU environment validation (driver, toolkit, clocks,
    thermals, profiler availability, P2P topology)
  - `bench gpu-lock` -- Clock management with RAII drop guard for reproducible
    benchmarks (lock, reset, and command-wrapper modes)
  - `bench gpu-monitor` -- Structured GPU state snapshots with field-level diff
    and configurable severity thresholds
- **Runtime performance monitor** (`vernier::monitor`) -- Lightweight
  instrumentation library for real application runs:
  - Lock-free MPMC queue (Vyukov algorithm) for sub-microsecond enqueue
  - Async I/O drain thread with batched writes
  - RAII scope instrumentation (`VERNIER_MONITOR_SCOPE` macro)
  - Counter, gauge, and threshold alerting primitives
  - Console and file sinks with end-of-run summary reporting
- **CMake-driven tool builds** -- Rust, Python, C++, and shell tools now build
  via CMake custom targets with `VERNIER_BUILD_TOOLS` option

### Fixed

- FetchContent compatibility for downstream consumers (use
  `CMAKE_CURRENT_SOURCE_DIR` instead of `CMAKE_SOURCE_DIR` for module path)
- `.env` variable escaping in tools environment helper
- Tool build stamp files moved out of output directories to prevent false
  rebuild triggers
- `cpp_tools` aggregate target scoped with `PROJECT_NAME` to avoid collisions
  in multi-project builds

### Infrastructure

- Build infrastructure hardened from cross-project findings (ccache detection,
  linker selection, split DWARF, warning flags)

---

## v1.0.0

Initial release. CPU and GPU benchmarking framework with profiler integrations
(perf, gperftools, callgrind, bpftrace, Nsight), statistical analysis, CSV
export, regression detection, and CLI tools for visualization and comparison.
