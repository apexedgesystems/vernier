# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## v1.0.3 - 2026-06-28

### Changed

- **Tiered base image** -- `docker/base.Dockerfile` is now two stages:
  `build-base` (lean compile/link/test tier) and `dev-base` (build-base plus
  scanners, formatters, profilers, and docs). `vernier.base` builds the dev-base
  target, so every existing dev shell is unaffected.
- **Hermetic dependency baking** -- the build image pre-fetches the rust tools'
  cargo crates, the FetchContent sources (googletest), and the python tools'
  wheels at their pinned versions, so a release build resolves no external
  registry (crates.io, PyPI, GitHub) at build time. A registry outage, throttle,
  or yank can no longer fail a build; clean builds reuse the caches instead of
  re-fetching. The offline switches are unset in dev-base so interactive dev
  still resolves online.
- **Rust tools build** -- `tools/CMakeLists.txt` no longer overrides
  `CARGO_HOME` to a per-build directory; it inherits the shared registry cache
  (enabling cache reuse and offline builds) and the compiled artifacts still
  isolate to a per-build target directory.
- **Lean release builders** -- every release builder (cpu, cuda, jetson, rpi,
  riscv64) now derives from a build tier (the platform leaf built on build-base)
  instead of the full dev shell, and every build tier carries the offline
  switches for the baked caches. The cpu and cuda tiers drop the dev-only
  scanners, profilers, and formatters outright; the cross tiers (jetson, rpi,
  riscv64) still receive dev tooling through the toolchain overlay (the
  toolchain images build on the dev tier) -- slimming those is a follow-up. The
  platform leaf Dockerfiles are parameterized (a BASE / OVERLAY arg) to produce
  both the dev shell and the build variant from one definition.
- **CI image graph** -- `docker-images.yml` builds and publishes the build tiers
  alongside the dev images, and rebuilds the base image when
  `tools/rust/Cargo.lock`, `tools/py/poetry.lock`, or `ExternalDependencies.cmake`
  change so the baked caches never go stale. The release workflow pulls the build
  tiers to reuse their layers.
- **Resilient image downloads** -- the image's external fetches (the LLVM apt
  key, CMake, UPX, hadolint, shfmt) retry with backoff on connection and HTTP
  5xx/429 errors, so a transient registry blip no longer fails an image rebuild.
- **Registry layer cache for CI image builds** -- CI builds go through a cache
  overlay (`docker-compose.ci-cache.yml`): pushed images embed inline layer
  cache and rebuilds pull unchanged layers from the registry, so an image-input
  change rebuilds only the layers it invalidated instead of the whole image on
  a fresh runner. Toolchain images are now pushed so cross-image and release
  rebuilds cache too.
- **Scoped CI gate** -- a detect job classifies the diff: tool-only changes skip
  the C++ build, the image rebuild runs only when an image input changed (jobs
  otherwise run on the pulled registry images -- previously every PR rebuilt the
  dev image from scratch), and an image-graph check builds the platform leaves
  when they or a gate image change (previously a broken cuda/cross leaf merged
  green). The rust/python tests run on the lean build tier -- the tier the
  release builds with -- and `make test-rust` reuses the shared cargo cache
  instead of re-downloading crates into a per-build throwaway.
- **Release rehearsal mode** -- `release.yml` accepts `workflow_dispatch`: the
  identical artifact build (version from `CMakeLists.txt`, no tag required)
  with only the publish step skipped, so the release pipeline is provable
  before a version tag is cut.

### Added

- **docker/scripts/bake-external-deps.sh** -- clones the FetchContent
  dependencies at their pinned tags into the image for offline configure.
- **build-base + build.\* compose services** -- the lean build/test tier and the
  per-platform lean builder tiers.
- **Dependency-pin guard** -- `scripts/check-pinned-deps.sh` (run as a pre-commit
  hook) fails if any FetchContent `GIT_TAG` is not an immutable ref (a version
  tag or a full commit SHA). A moving ref would let a dependency drift without
  `ExternalDependencies.cmake` changing, silently bypassing the image's
  auto-rebuild and leaving the baked source stale.

## v1.0.2 - 2026-06-04

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
- **`bench validate`** reports host readiness for every backend (the valgrind
  tools, heaptrack, jemalloc, rapl, compute-sanitizer, rocprof), not just
  perf / gperftools / nsight / bpftrace.
- **`bench gpu-env`** compares the CUDA toolkit against the driver's CUDA and
  warns when the toolkit is ahead -- the common, otherwise-silent cause of an
  empty nsys trace (ncu and in-process CUPTI are unaffected).
- **Build floor is C++20** (Clang 12+ / GCC 10+); C++23 is selected
  automatically when the toolchain supports it.

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
