# Vernier CLI Tools

**Location:** `tools/py/`, `tools/rust/`
**Platform:** Linux x86_64
**Tools:** `bench` (Rust), `bench-plot` (Python)

Analysis, comparison, validation, execution, visualization, and flamegraph generation
for Vernier benchmark results.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [bench (Rust)](#2-bench-rust)
3. [bench-plot (Python)](#3-bench-plot-python)
4. [nsight-parse (Python)](#3b-nsight-parse-python)
5. [Common Workflows](#4-common-workflows)
6. [CSV Schema](#5-csv-schema)
7. [Building](#6-building)
8. [Testing](#7-testing)
9. [See Also](#8-see-also)

---

## 1. Quick Start

```bash
# Build tools
make tools-rust        # bench (analysis, comparison, execution)
make tools-py          # bench-plot (visualization, optional)

# Source from anywhere; .env embeds absolute paths.
source build/native-linux-debug/.env

# Verify
bench --help
bench-plot --help      # Only if tools-py built
```

The `.env` file uses absolute paths to the tools in the build tree, so
sourcing it from any cwd puts `bench` / `bench-plot` / `nsight-parse`
on PATH. The build directory itself stays self-contained -- copy it
anywhere, source the `.env` inside it, and the tools work.

---

## 2. bench (Rust)

Single binary with 14 subcommands for benchmarking analysis, profiling
orchestration, GPU environment management, and project setup.

### summary - Display Results

Pretty-print a benchmark CSV with sorting and filtering.

```bash
bench summary results.csv
bench summary results.csv --sort median
bench summary results.csv --sort cv
bench summary results.csv --json
```

**Options:**

| Flag            | Description                           | Default |
| --------------- | ------------------------------------- | ------- |
| `--sort COLUMN` | Sort by: name, median, cv, throughput | name    |
| `--json`        | Machine-readable JSON output          | --      |

### compare - Regression Detection

Statistical comparison of two benchmark CSVs.

```bash
bench compare baseline.csv candidate.csv
bench compare baseline.csv candidate.csv --threshold 3
bench compare baseline.csv candidate.csv --fail-on-regression
bench compare baseline.csv candidate.csv --markdown
```

**Options:**

| Flag                   | Description                                   | Default |
| ---------------------- | --------------------------------------------- | ------- |
| `--threshold PCT`      | Regression threshold in %                     | 5       |
| `--fail-on-regression` | Exit code 1 if regressions detected (CI mode) | --      |
| `--json`               | Machine-readable JSON output                  | --      |
| `--markdown`           | Markdown table output (for PR comments)       | --      |

### validate - Environment Checks

Verify system readiness for benchmarking.

```bash
bench validate
bench validate --json
```

### run - Execute Benchmark Binary

Run a benchmark binary with optional CPU pinning and profiling. The binary
argument can be a full path OR a short name -- the latter auto-resolves
under `build/*/bin/{ptests,tests,examples}` (override with
`VERNIER_BENCH_BIN_ROOTS` / `VERNIER_BENCH_BIN_SUBDIRS` for non-CMake
layouts).

```bash
bench run BasicWorkflow                                   # short name auto-resolve
bench run ./bin/ptests/MyComponent_PTEST                  # full path
bench run MyComponent --csv results.csv --quick
bench run MyComponent --taskset 2-9 --profile perf
bench run MyComponent --csv results.csv --analyze
bench run MyComponent --profile massif                    # auto-wraps with valgrind
bench run MyComponent --profile heaptrack                 # auto-wraps with heaptrack
```

**Options:**

| Flag                       | Description                                          | Default      |
| -------------------------- | ---------------------------------------------------- | ------------ |
| `--csv FILE`               | Export results to CSV                                | --           |
| `--quick`                  | Fewer cycles/repeats for fast iteration              | --           |
| `--taskset CPUS`           | Pin to specific CPU cores                            | --           |
| `--profile MODE`           | Enable profiling (any registered backend; see below) | --           |
| `--profile-output-dir DIR` | Wrap-externally backends' artifact root              | `bench-out/` |
| `--analyze`                | Run summary after execution                          | --           |

When `--profile` names a wrap-externally backend (`callgrind`, `massif`,
`memcheck`, `heaptrack`, `compute-sanitizer`), `bench run` transparently
invokes the correct wrap (`valgrind --tool=...`, `heaptrack -o ...`,
etc.) and writes the artifacts to
`<--profile-output-dir>/<binary-stem>.<tool>/`. In-process backends
(`perf`, `gperf`, `rapl`, `bpftrace`, `offcpu`) run the binary directly
and the C++ harness manages its own per-test artifact subdirs.

Unset `--cycles` / `--repeats` are filled in from `.bench.yaml` (see `init`).

### doctor - Backend Environment Check

Runs `--profile-check` against a ptest binary, printing both the binary
readiness section (frame pointers, DWARF, ASLR, gperftools linkage) and the
per-backend doctor (whether each registered profiler can actually run here).

```bash
bench doctor ./build/native-linux-debug/bin/ptests/MyComponent_PTEST
```

### profile-all - Iterate Every Profiler

Run a benchmark under each profiler in sequence, dropping artifacts under
per-tool subdirectories.

```bash
bench profile-all MyComponent                                       # gperf + perf + callgrind
bench profile-all MyComponent --profilers gperf,callgrind --out out/
bench profile-all MyComponent --quick --filter '*Hot*'
```

### profile-summarize - Tabulate Artifacts

Walks an artifact root and reports per-tool file counts + total bytes.

```bash
bench profile-summarize bench-out/
```

### init / config-validate - Project Defaults

`bench init` scaffolds a `.bench.yaml` at the project root (cycles, repeats,
profile_output_dir, gtest_filter, bin_roots, bin_subdirs). Read by `run` and
`profile-all` when the corresponding CLI flag is omitted.

```bash
bench init                              # writes .bench.yaml in CWD
bench init --path config.yaml --force   # custom path; overwrite existing

bench config-validate                   # walks up from CWD for .bench.yaml
bench config-validate path/to/file.yaml
```

### gpu-topo - GPU/CPU Affinity

Shows the GPU/GPU peer matrix and the NUMA-affine CPU range for each device.

```bash
bench gpu-topo
bench gpu-topo --json
```

### Registered Profiler Backends

`--profile X` dispatches to whichever backend self-registered under name `X`.
The `doctor` command lists all of them with their environment readiness.

| Backend             | Layer | Wraps                                             |
| ------------------- | ----- | ------------------------------------------------- |
| `perf`              | CPU   | `perf stat` / `record` / `mem` / `c2c`            |
| `gperf`             | CPU   | gperftools sampling profiler                      |
| `callgrind`         | CPU   | valgrind callgrind                                |
| `bpftrace`          | CPU   | bpftrace scripts                                  |
| `rapl`              | CPU   | Intel RAPL MSRs                                   |
| `massif`            | CPU   | valgrind massif (heap timeline, ~20x)             |
| `memcheck`          | CPU   | valgrind memcheck (errors / leaks)                |
| `helgrind`          | CPU   | valgrind helgrind / DRD (data races, lock order)  |
| `offcpu`            | CPU   | bpftrace finish_task_switch (off-CPU stacks)      |
| `heaptrack`         | CPU   | heaptrack heap profiler (~1.5x)                   |
| `jemalloc`          | CPU   | jemalloc prof sampling (~5-10%, LD_PRELOAD)       |
| `nsight`            | GPU   | Nsight Systems / Compute (auto-extracts stats)    |
| `compute-sanitizer` | GPU   | NVIDIA Compute Sanitizer (GPU memcheck/race/init) |
| `rocprof`           | GPU   | AMD ROCm rocprof                                  |

CUPTI activity counters (per-launch register count, shared memory, kernel
count) populate the GPU section of the CSV automatically on every GPU run --
no `--profile` flag needed.

### flamegraph - Generate SVG Flamegraphs

Generate flamegraphs from perf profiling data.

```bash
bench flamegraph test.perf/perf.data
bench flamegraph test.perf/perf.data --output hotspots.svg
bench flamegraph candidate.perf/perf.data --baseline baseline.perf/perf.data
```

**Options:**

| Flag              | Description                              | Default        |
| ----------------- | ---------------------------------------- | -------------- |
| `--output FILE`   | Output SVG path                          | flamegraph.svg |
| `--baseline FILE` | Differential flamegraph against baseline | --             |

### gpu-env - GPU Environment Validation

Check GPU readiness for benchmarking: driver, toolkit, devices, clocks, thermals,
profiler availability, and P2P topology.

```bash
bench gpu-env
bench gpu-env --json
```

**Checks performed:**

| Check            | Severity | Description                                      |
| ---------------- | -------- | ------------------------------------------------ |
| nvidia-smi       | FAIL     | Binary exists and runs                           |
| NVIDIA driver    | FAIL     | Driver version query                             |
| CUDA toolkit     | WARN     | nvcc version, falls back to driver-reported CUDA |
| GPU devices      | FAIL     | Device enumeration with name, memory, SM version |
| Persistence mode | WARN     | Cold-start overhead if disabled                  |
| GPU clocks       | WARN     | Current vs max, lock recommendation              |
| ECC memory       | WARN     | Bandwidth impact of ECC                          |
| Power state      | WARN     | Current draw vs limit headroom                   |
| Thermal state    | WARN     | Temperature vs throttle point                    |
| Nsight Systems   | WARN     | nsys version for timeline profiling              |
| Nsight Compute   | WARN     | ncu version for kernel-level profiling           |
| P2P topology     | INFO     | NVLink / PCIe topology (multi-GPU only)          |

**Options:**

| Flag     | Description                  | Default |
| -------- | ---------------------------- | ------- |
| `--json` | Machine-readable JSON output | --      |

### gpu-lock - Clock Management

Lock GPU clocks to a fixed frequency for reproducible benchmarks. Eliminates
clock boost/throttle variance that inflates CV%.

```bash
# Lock clocks (default: max frequency)
bench gpu-lock lock
bench gpu-lock lock --freq 1500

# Lock clocks, run a benchmark, then auto-reset on exit
bench gpu-lock lock -- ./bin/ptests/BenchmarkGPU_PTEST --quick --csv results.csv

# Reset clocks to driver-managed default
bench gpu-lock reset
```

**Subcommands:**

| Subcommand | Description                                       |
| ---------- | ------------------------------------------------- |
| `lock`     | Lock clocks (reset on exit if wrapping a command) |
| `reset`    | Reset clocks to driver-managed default            |

**Lock options:**

| Flag         | Description                            | Default   |
| ------------ | -------------------------------------- | --------- |
| `--device N` | GPU device index                       | 0         |
| `--freq MHz` | Target frequency in MHz                | max clock |
| `-- CMD...`  | Command to run while clocks are locked | --        |

The wrapper mode (`lock -- <command>`) uses a drop guard to guarantee clock
reset even if the wrapped command fails or is interrupted with Ctrl-C.
Persistence mode is auto-enabled if needed.

### gpu-monitor - GPU State Snapshots

Capture GPU state before and after a benchmark run, then diff to detect
environmental drift (thermal throttling, clock changes, memory pressure).

```bash
# Capture current state
bench gpu-monitor snapshot
bench gpu-monitor snapshot -o before.json

# After benchmark run
bench gpu-monitor snapshot -o after.json

# Compare
bench gpu-monitor diff before.json after.json
bench gpu-monitor diff before.json after.json --json
```

**Snapshot fields per device:** temperature, power draw/limit, graphics/memory
clocks, memory usage, GPU/memory utilization, throttle reasons, P-state.

**Diff severity thresholds:**

| Field               | Warning threshold | Meaning                       |
| ------------------- | ----------------- | ----------------------------- |
| temperature_c       | 5 C               | GPU heated up significantly   |
| power_draw_w        | 10 W              | Power budget shifted          |
| clock_graphics_mhz  | 50 MHz            | Clock speed changed           |
| clock_mem_mhz       | 50 MHz            | Memory clock changed          |
| memory_used_mib     | 100 MiB           | Other process grabbed GPU mem |
| gpu_utilization_pct | 20%               | Background GPU load           |
| pstate              | any change        | Performance state shifted     |
| throttle_reasons    | any change        | Throttling started or stopped |

---

## 3. bench-plot (Python)

Visualization tool for generating charts, dashboards, and reports from benchmark CSVs.
Requires `make tools-py`.

### plot - Standard Charts

```bash
bench-plot plot results.csv
bench-plot plot results.csv --output charts/
```

### dashboard - Interactive HTML Dashboard

```bash
bench-plot dashboard results.csv
bench-plot dashboard results.csv --output perf_dashboard.html
```

### report - Analysis Report

```bash
bench-plot report results.csv
bench-plot report results.csv --output analysis/
```

### scaling - Payload Size Analysis

```bash
bench-plot scaling 1kb.csv 64kb.csv 1mb.csv
bench-plot scaling 1kb.csv 64kb.csv 1mb.csv --output scaling.html
```

---

## 3b. nsight-parse (Python)

Turn raw Nsight reports into a tidy CSV the rest of the toolchain can consume.

```bash
# Single .nsys-rep -> CSV with the four canonical nsys reports
nsight-parse parse run.nsys-rep --csv kernels.csv

# Single .ncu-rep -> CSV with ncu's per-kernel summary
nsight-parse parse run.ncu-rep --csv compute.csv

# Directory walk: handle every .nsys-rep / .ncu-rep under the path
nsight-parse parse bench-out/nsight/ --csv combined.csv
```

Output columns: `source` (nsys / ncu), `report`, `kernel`, `instances`,
`time_total_ns`, `time_avg_ns`, `time_pct`, plus per-tool metric columns
(metric names normalized to snake-case).

---

## 4. Common Workflows

### Development Iteration

Quick test with immediate summary:

```bash
source build/native-linux-debug/.env
bench run MyComponent_PTEST --quick --csv results.csv --analyze
```

### Optimization Workflow

```bash
source build/native-linux-debug/.env

# 1. Validate environment
bench validate

# 2. Baseline measurement
bench run MyComponent_PTEST -- --repeats 30 --csv baseline.csv

# 3. Profile to find hotspots
bench run MyComponent_PTEST -- --profile perf --cycles 100000
bench flamegraph MyComponent.Throughput.perf/perf.data --output before.svg

# 4. Make changes, rebuild

# 5. Measure again
bench run MyComponent_PTEST -- --repeats 30 --csv optimized.csv

# 6. Statistical comparison
bench compare baseline.csv optimized.csv --threshold 5

# 7. Visualize (optional)
bench-plot plot optimized.csv --output analysis/
```

### GPU Benchmark Workflow

Full GPU benchmarking pipeline with environment validation, clock locking,
and state monitoring:

```bash
source build/native-linux-debug/.env

# 1. Validate GPU environment
bench gpu-env

# 2. Lock clocks for reproducibility
bench gpu-lock lock

# 3. Snapshot before
bench gpu-monitor snapshot -o before.json

# 4. Run benchmark
./bin/ptests/BenchmarkGPU_PTEST --repeats 30 --csv gpu_results.csv

# 5. Snapshot after
bench gpu-monitor snapshot -o after.json

# 6. Check for environmental drift
bench gpu-monitor diff before.json after.json

# 7. Analyze results
bench summary gpu_results.csv

# 8. Reset clocks
bench gpu-lock reset
```

Or use the wrapper mode to combine steps 2, 4, and 8:

```bash
bench gpu-lock lock -- ./bin/ptests/BenchmarkGPU_PTEST --repeats 30 --csv gpu_results.csv
```

### CI Regression Detection

```bash
bench compare baseline.csv candidate.csv \
  --threshold 5 \
  --fail-on-regression \
  --markdown > pr_comment.md
```

Exit code 1 on regression. `--markdown` produces a table suitable for PR comments.

---

## 5. CSV Schema

The benchmarking framework outputs CSV files with the following columns.

**Base columns:** test, cycles, repeats, warmup, threads, msgBytes, wallMedian,
wallP10, wallP90, wallMin, wallMax, wallMean, wallStddev, wallCV, callsPerSecond,
stable, cvThreshold

**GPU columns (when present):** gpuModel, computeCapability, kernelTimeUs,
transferTimeUs, h2dBytes, d2hBytes, speedupVsCpu, memBandwidthGBs, occupancy,
smClockMHz, throttling, powerDrawW, powerLimitW, temperatureC,
temperatureDeltaC, cuptiKernelLaunches, cuptiRegistersMedian,
cuptiRegistersMax, cuptiStaticSmemBytes, cuptiDynamicSmemBytes, deviceId,
deviceCount, multiGpuEfficiency, p2pBandwidthGBs, umPageFaults,
umH2DMigrations, umD2HMigrations, umMigrationTimeUs, umThrashing

**Metadata columns:** timestamp, gitHash, hostname, platform

The `stable` and `cvThreshold` columns are optional. All tools accept CSVs with or
without these columns.

---

## 6. Building

### Rust Tools (bench)

```bash
make tools-rust
```

Produces a single `bench` binary in `build/native-linux-debug/bin/tools/rust/`.
CUDA-related features are enabled automatically when `nvcc` is on PATH.

**Requirements:** Rust toolchain (rustup)

### Python Tools (bench-plot)

```bash
make tools-py
```

Installs `bench-plot` and all dependencies into the build directory.

**Requirements:** Python >=3.10, Poetry

### Adding New Tools

**Rust:** Add a `src/bin/mytool.rs` file and a `[[bin]]` entry in `Cargo.toml`.
Rebuild with `make tools-rust`.

**Python:** Add a module in `src/vernier_tools/` and a `[tool.poetry.scripts]`
entry in `pyproject.toml`. Rebuild with `make tools-py`.

---

## 7. Testing

```bash
# Rust tool tests
make test-rust

# Python tool tests
make test-py

# Or directly
cd tools/rust && cargo test
cd tools/py && poetry run pytest -v
```

---

## 8. See Also

- `src/bench/docs/CPU_GUIDE.md` - CPU benchmarking patterns
- `src/bench/docs/GPU_GUIDE.md` - GPU benchmarking patterns
- `src/monitor/inc/Monitor.hpp` - Runtime performance monitor API
- `src/bench/docs/TROUBLESHOOTING.md` - Common issues and solutions
