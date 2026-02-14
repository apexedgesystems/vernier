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
4. [Common Workflows](#4-common-workflows)
5. [CSV Schema](#5-csv-schema)
6. [Building](#6-building)
7. [Testing](#7-testing)
8. [See Also](#8-see-also)

---

## 1. Quick Start

```bash
# Build tools
make tools-rust        # bench (analysis, comparison, execution)
make tools-py          # bench-plot (visualization, optional)

# Setup environment (from build directory)
cd build/native-linux-debug
source .env

# Verify
bench --help
bench-plot --help      # Only if tools-py built
```

The build directory is self-contained and relocatable. Copy it anywhere, source `.env`,
and the tools work.

---

## 2. bench (Rust)

Single binary with 5 subcommands for all non-plotting benchmarking tasks.

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

Run a benchmark binary with optional CPU pinning and profiling.

```bash
bench run ./bin/ptests/MyComponent_PTEST
bench run ./bin/ptests/MyComponent_PTEST --csv results.csv --quick
bench run ./bin/ptests/MyComponent_PTEST --taskset 2-9 --profile perf
bench run ./bin/ptests/MyComponent_PTEST --csv results.csv --analyze
```

**Options:**

| Flag             | Description                             | Default |
| ---------------- | --------------------------------------- | ------- |
| `--csv FILE`     | Export results to CSV                   | --      |
| `--quick`        | Fewer cycles/repeats for fast iteration | --      |
| `--taskset CPUS` | Pin to specific CPU cores               | --      |
| `--profile MODE` | Enable profiling (perf, gperf)          | --      |
| `--analyze`      | Run summary after execution             | --      |

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

## 4. Common Workflows

### Development Iteration

Quick test with immediate summary:

```bash
cd build/native-linux-debug
source .env
bench run ./bin/ptests/MyComponent_PTEST --quick --csv results.csv --analyze
```

### Optimization Workflow

```bash
cd build/native-linux-debug
source .env

# 1. Validate environment
bench validate

# 2. Baseline measurement
./bin/ptests/MyComponent_PTEST --repeats 30 --csv baseline.csv

# 3. Profile to find hotspots
./bin/ptests/MyComponent_PTEST --profile perf --cycles 100000
bench flamegraph MyComponent.Throughput.perf/perf.data --output before.svg

# 4. Make changes, rebuild

# 5. Measure again
./bin/ptests/MyComponent_PTEST --repeats 30 --csv optimized.csv

# 6. Statistical comparison
bench compare baseline.csv optimized.csv --threshold 5

# 7. Visualize (optional)
bench-plot plot optimized.csv --output analysis/
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
smClockMHz, throttling, deviceId, deviceCount, multiGpuEfficiency, p2pBandwidthGBs,
umPageFaults, umH2DMigrations, umD2HMigrations, umMigrationTimeUs, umThrashing

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
- `src/bench/docs/TROUBLESHOOTING.md` - Common issues and solutions
