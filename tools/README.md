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
# Build: the Release preset build (README Quick Start) includes the tools
cmake --preset native-linux-release
cmake --build --preset native-linux-release

# Put them on PATH; .env holds absolute paths, so any working directory works
source build/native-linux-release/.env

# Verify
bench --help
bench-plot --help      # Only if the Python tools were built (poetry and pip)
```

The `.env` file holds absolute paths into the build tree, so sourcing it from
any working directory puts `bench`, `bench-plot` and `nsight-parse` on `PATH`.
It works only where the tree was configured: a moved or copied tree's `.env`
still names the original location, and a tree configured in a container names
the container's paths. Configure a build where you need the tools instead. The
`bench` executable itself needs only the C runtime and can be copied on its
own; the Python tools need the tree's `lib/python`, which the `.env` puts on
`PYTHONPATH`.

`make tools-rust` and `make tools-py` rebuild only the tools, in
`build/native-linux-debug` (configured with the Debug preset if it is not yet)
unless `BUILD_DIR` names another build directory:
`make tools-rust BUILD_DIR=build/native-linux-release`.

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

**Input it refuses.** Every row must have a test name that is not empty or
only whitespace, and a finite number in `wallMedian`, `wallCV` and
`callsPerSecond`. `wallP10`, `wallP90`, `stable`, `cvThreshold`, `cycles` and
`repeats` may be left out, by the CSV's layout, by a row that stops early or
by an empty field, and then show their defaults, but where a row gives one it
must be a finite number of its kind. Anything else exits 1 with nothing on
stdout, as text and as JSON, and a message naming the file and line and what
it found there: the test, column and value, or the blank name. Zeros are
values.

### compare - Median Change Between Two Runs

Compare two benchmark CSVs test by test, and label each test by how far its
reported median moved.

```bash
bench compare baseline.csv candidate.csv
bench compare baseline.csv candidate.csv --threshold 3
bench compare baseline.csv candidate.csv --fail-on-regression
bench compare baseline.csv candidate.csv --markdown
```

**Options:**

| Flag                   | Description                                            | Default |
| ---------------------- | ------------------------------------------------------ | ------- |
| `--threshold PCT`      | Median change, in %, beyond which a test is labelled   | 5       |
| `--fail-on-regression` | Exit code 1 on a regression or a missing baseline test | --      |
| `--json`               | Machine-readable JSON output                           | --      |
| `--markdown`           | Markdown table output (for PR comments)                | --      |

**What the labels mean.** A test is `REGRESSION` when the candidate's median
is more than `--threshold` percent above the baseline's, `IMPROVEMENT` when
it is more than that below it, and `neutral` otherwise, including at exactly
the threshold as the CSVs' decimals give it: from 1 to 1.05 is 5%, neutral at
`--threshold 5`. The labels describe the difference between two runs. They are
not a test for statistical significance: the CSVs carry summary statistics,
not the observations such a test needs. Each row shows both runs' CV as
context for how spread out each run was on its own; neither CV measures the
spread between the two runs, so a small median change on noisy tests is worth
re-running before acting on it. Accounting for run-to-run noise is a known
limit of this comparison rather than something it does.

**Tests that are not in both runs.** The comparison covers the tests both CSVs
report. Tests only the baseline reports are listed as missing from the
candidate, tests only the candidate reports as new, and a renamed test shows
up as both. Names are matched exactly as written, case and any spaces around
them included. Two CSVs with no test name in common are an error: the command
exits 1 rather than reporting that nothing regressed.

**Input it refuses.** The command exits 1 with nothing on stdout, and a
message naming the file or run, the test, the column and the value involved,
when given a `--threshold` that is not a finite percentage of zero or more, a
row whose test name is empty or only whitespace, a test name a CSV reports
twice, a `wallMedian` or `wallCV` that is missing from its row, empty, not a
number or not finite, a `wallMedian` of zero or less, a negative `wallCV`, or
a candidate median so many times its baseline that the percentage change
overflows.

**Advisory or gate.** Plain `bench compare` reports and exits 0 whatever the
labels say. `--fail-on-regression` is the gate: it exits 1 when a test is
labelled `REGRESSION`, or when the candidate does not run a test the baseline
did, since nothing measured that test. A test only the candidate runs is
reported as new and does not fail on its own; adopt it into the baseline
deliberately.

**JSON shape.** `--json` prints one document: `threshold_pct`, a `results`
array of per-test objects, and the `baseline_only` and `candidate_only` name
lists. Each result's `p_value` is `null`, because none was computed. Read the
label from `classification`: `delta_pct` is the change as binary floating
point computes it, so a change exactly at the threshold in the CSVs' decimals
can read a few units in its last digit past it (`5.000000000000004`).

### validate - Profiling Tools on This Host

An advisory inventory: it exits 0 whatever it finds. Without a binary it
reports facts, not readiness: the profiling tools PATH finds, each with its
path and the version its `--version` prints (`ncu` has a row of its own, and
the `gperftools` row names the analyzer `--profile-analyze` runs), the msr
device `rapl` reads, ASLR, the FlameGraph scripts, and
`kernel.perf_event_paranoid` with what the kernel allows at its value. A tool
PATH does not find, or finds without an execute bit, is `[WARN]`. Whether a
profiler can run here, and in which modes, is what `bench doctor <binary>`
checks.

```bash
bench validate
bench validate --json
bench validate ./build/native-linux-release/bin/ptests/MyComponent_PTEST
```

With a binary, the tool rows give way to the binary's own rows for each
profiler's default mode (its doctor's JSON document), after the ASLR,
FlameGraph and `perf_event_paranoid` rows: `ok` stays `[OK]`, `warn` stays
`[WARN]`, and a profiler the binary cannot use here is `[WARN]` with
`not usable here: <message>`, the doctor's remedy on the line after it. No
row fails the command; to fail a lane on the profilers it needs, use
`bench doctor <binary> --require <backends>`. A binary that is missing, does
not start, or prints no usable doctor document is an error (exit 1, the cause
on stderr, nothing on stdout).

`--json` prints one array of rows: `label`, `status` (`ok` or `warn`),
`detail`, and `hint` where the binary's row has a remedy.

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
| `--profile-args ARGS`      | The profile's mode (see below); may start with `-`   | --           |
| `--profile-output-dir DIR` | Wrap-externally backends' artifact root              | `bench-out/` |
| `--profile-analyze`        | The profile's analysis (callgrind: see below)        | --           |
| `--analyze`                | Run summary after execution                          | --           |

When `--profile` names a wrap-externally backend (`callgrind`, `massif`,
`memcheck`, `helgrind`, `heaptrack`, `compute-sanitizer`, `nsight` or its
other name `nsys`, `ncu`), `bench run` starts the binary under that tool
(`valgrind --tool=...`, `heaptrack -o ...`, `nsys profile ...`, etc.) and
writes the artifacts to `<--profile-output-dir>/<binary-stem>.<tool>/`, where
`<tool>` is the canonical name (`nsys` writes to `.nsight`). The words of
`--profile-args` select the tool's mode, and any other word is refused before
anything starts:

| `--profile`         | `--profile-args` words                                   | Selects                                                  |
| ------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `massif`            | `pages` or `stacks`                                      | `--pages-as-heap=yes` or `--stacks=yes`                  |
| `memcheck`          | `leak-full`, `track-origins`                             | the full leak check (the default), `--track-origins=yes` |
| `helgrind`          | `drd`                                                    | valgrind's DRD instead of Helgrind                       |
| `compute-sanitizer` | one of `memcheck`, `racecheck`, `synccheck`, `initcheck` | that `--tool` (default `memcheck`)                       |
| `nsight`            | `compute` (or `ncu`)                                     | Nsight Compute (`ncu`) instead of Nsight Systems         |

`callgrind`, `heaptrack` and `ncu` take no mode. A kernel replay
(`--profile-args replay`) is refused: its metrics are the benchmark's own, so
run the benchmark directly, and it prints the `ncu` command that replays its
kernels. In-process backends
(`perf`, `gperf`, `rapl`, `bpftrace`, `offcpu`) run the binary directly, which
reads `--profile-args` itself, and the C++ harness manages its own per-test
artifact subdirs.

**A wrapped run's output.** Before it starts the benchmark, `bench run`
creates the wrap's folder (a folder it cannot create stops the run before
anything starts) and removes from that folder the files a previous run of
the same wrap left: `callgrind.out`, `massif.out`, `memcheck.log`,
`helgrind.log`, `run.zst` and `run.gz` (heaptrack), `sanitizer.log`,
`profile.nsys-rep` with `profile.sqlite` and the four summaries (nsight),
`kernel_profile.ncu-rep` (ncu, and nsight's compute mode) and
`jeprof.*.heap` (jemalloc), each removal printed as `[bench] removed <file>
from a previous run`. Nothing else in the folder, and nothing outside it, is
removed. valgrind, compute-sanitizer, nsys and ncu read a `%` in an output
option as the start of a macro (`%p`, `%q{VAR}`), so `bench run` writes the
folder there with each `%` doubled, and the output lands in the folder as
named. heaptrack replaces `%h` and `%p` in its `-o` with the host name and
its process id and has no escape for them, so `bench run` refuses a heaptrack
folder holding either before anything starts; any other `%` reaches
heaptrack as it is. After the benchmark exits 0, the file must exist and hold
something: the run prints `[bench] <tool> wrote <file> (<n> bytes)`, and
otherwise fails with `completion: <file> was not written` (or `is empty`).
For nsight, a `nsys stats` summary that fails fails the run as `analysis:`,
and the report is kept. With `--profile-analyze` (given to `bench run`, or
forwarded after `--`), a callgrind run's profile is annotated after valgrind
has written it: `callgrind_annotate --auto=yes <profile>`, its first 40 lines
printed; a missing or failing `callgrind_annotate`, or one still running after
120 s, fails the run as `analysis:`, and the profile is kept. The annotator
runs in a process group of its own: a process of that group still running
when the annotator exits is ended, and `bench run` says so; SIGINT, SIGTERM or
SIGHUP sent to `bench run` meanwhile ends the group, then `bench run` itself.

**compute-sanitizer's verdict.** The route passes `--error-exitcode 5`: the
tool ends with status 5 when it reports errors, whatever the benchmark itself
returned, and otherwise with the benchmark's own status. `bench run` reads
this run's `sanitizer.log` to tell them apart. Errors counted by its summary
fail the run, and the count printed is the summary's: the `ERROR SUMMARY`
line of memcheck, synccheck and initcheck, or racecheck's `RACECHECK SUMMARY:
H hazards displayed (E errors, W warnings)`, whose errors count and whose
warnings do not. With no errors counted, the status is the benchmark's own,
5 included. A report that
holds only the tool's own `Error:` line fails the run as `collection:`, a
missing report or one without its summary as `completion:`, and nothing is
counted.

Unset `--cycles` / `--repeats` / `--target-time` are filled in from `.bench.yaml` (see `init`).

**How a run ends.** `bench run` exits 0 when the benchmark does and 1
otherwise, and its last line names how the benchmark ended:

```text
Error: the benchmark exited with status 1
Error: the benchmark was ended by signal 9
Error: the requested profile failed (the benchmark's report above says why); the benchmark exited with status 4
Error: compute-sanitizer reported 3 errors in the benchmark; the report is bench-out/probe.compute-sanitizer/sanitizer.log (the tool exited with status 5)
```

Status 4 means the tests passed and the `--profile` request failed: the
benchmark's own `[profile]` report, printed just above, lists each failure. A
benchmark run directly exits with that status itself (see the exit status 4
entry in [Troubleshooting](../src/bench/docs/TROUBLESHOOTING.md)).

### doctor - Backend Environment Check

Runs `--profile-check` against a ptest binary, printing both the binary
readiness section (frame pointers, DWARF, ASLR, gperftools linkage) and the
per-backend doctor (whether each registered profiler can actually run here).

```bash
bench doctor ./build/native-linux-release/bin/ptests/MyComponent_PTEST
bench doctor ./build/native-linux-release/bin/ptests/MyComponent_PTEST --json > doctor.json
bench doctor ./build/native-linux-release/bin/ptests/MyComponent_PTEST --require offcpu,heaptrack
bench doctor ./build/native-linux-release/bin/ptests/MyComponent_PTEST \
  --profile massif --profile-args pages --require massif
```

With `--json`, stdout is the binary's JSON document and nothing else. With
`--require`, `bench doctor` exits 1 unless every named backend reports OK,
and prints one `[require]` line per backend: on stderr when `--json` is also
given, so that stdout stays one document, and on stdout otherwise. A binary
whose doctor prints no valid document is an error (exit 1).

`--profile`, `--profile-args`, `--profile-analyze` and the arguments after
`--` are passed to the binary as `bench run` passes them, and the doctor adds
a row for that request. `--require` judges the requested backend by that row
and every other backend by its default mode's row, so a requirement on
`massif` with `--profile-args pages` is met only when that mode is ready. A
binary built before the requested row existed cannot answer such a
requirement: rebuild it against this vernier, or drop `--profile`.

`bench validate <binary>` shows the same default-mode rows as an advisory
report that never fails on them.

**What the doctor runs, and what it costs.** To say whether each profiler can
run here, the doctor starts most of the tools once, one after another: each
valgrind tool and heaptrack on `/bin/true`, a short `perf stat` of its own
process, the bpftrace and off-CPU probe scripts, and `--version` of nsys, ncu
and compute-sanitizer. Those starts take most of its time, which depends on
the machine and on the tools installed; no time is promised. One run of this
release's `bench doctor build/bin/ptests/BenchmarkCPU_PTEST` on each rig,
timed on core 3 after one untimed run: on the
[Raspberry Pi 4 rig](../src/bench/docs/rigs/RIG_PI4.md), with the governor at
performance, 3.96 s, and 8.60 s with `BENCH_SUDO=1`, under which the
bpftrace and off-CPU probes attach and run; on the
[Jetson AGX Thor rig](../src/bench/docs/rigs/RIG_THOR_AGX.md), with its clocks
as found, 1.35 s, and 3.26 s with `BENCH_SUDO=1`. Other runs and machines take
their own time. A benchmark run checks only its own request.

### profile-all - Iterate Every Profiler

Run a benchmark under each profiler in sequence, dropping artifacts under
per-tool subdirectories.

```bash
bench profile-all MyComponent                                       # gperf + perf + callgrind
bench profile-all MyComponent --profilers gperf,callgrind --out out/
bench profile-all MyComponent --quick --filter '*Hot*'
```

Each profiler runs as `bench run --profile <name>` does, into
`<out>/<name>/`, whatever the others did. The run ends with one line per
profiler, `completed` or `failed` with the reason, and exits 1 when any of them
failed:

```
=== bench profile-all: summary ===
  gperf      completed  bench-out/gperf
  perf       failed     bench-out/perf -- the requested profile failed (the benchmark's report above says why); the benchmark exited with status 4
  callgrind  completed  bench-out/callgrind
Error: 1 of 3 profile runs failed: perf
```

Every profiler in the list is required, the default three included: on a
machine that lacks one, name the others with `--profilers`.

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
The `doctor` command lists all of them with whether each can run here, and
checks a given `--profile` request by its own mode. The backends whose tool
records the whole process (`callgrind`, `massif`, `memcheck`, `helgrind`,
`heaptrack`, `nsight`, `ncu`, `compute-sanitizer`, `rocprof`) collect only in a
process that tool started: `bench run` starts it for all but `rocprof`, and a
run started without it fails with the command that would start it.

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
| `offcpu`            | CPU   | bpftrace on the sched tracepoints (off-CPU)       |
| `heaptrack`         | CPU   | heaptrack heap profiler (~1.5x)                   |
| `jemalloc`          | CPU   | jemalloc prof sampling (~5-10%, LD_PRELOAD)       |
| `nsight`            | GPU   | Nsight Systems / Compute (auto-extracts stats)    |
| `ncu`               | GPU   | NVIDIA Nsight Compute (per-kernel analysis)       |
| `compute-sanitizer` | GPU   | NVIDIA Compute Sanitizer (GPU memcheck/race/init) |
| `rocprof`           | GPU   | AMD ROCm rocprof                                  |

CUPTI activity counters (per-launch register count, shared memory, kernel
count) populate the GPU section of the CSV automatically on every GPU run --
no `--profile` flag needed.

### flamegraph - Generate SVG Flamegraphs

Generate flamegraphs from the `perf.data` of a record-mode perf run
(`--profile perf --profile-args "record -g"`; a plain `--profile perf` writes
only `stat.txt`). Needs `perf` and the FlameGraph scripts, looked up in
`$FLAMEGRAPH_DIR`, then `~/FlameGraph`, `/usr/local/FlameGraph` and
`/opt/FlameGraph`, then as `flamegraph.pl` on `PATH`.

```bash
bench flamegraph MyComponent.Test.perf/perf.data
bench flamegraph MyComponent.Test.perf/perf.data --output hotspots.svg
bench flamegraph optimized/MyComponent.Test.perf/perf.data --baseline baseline/MyComponent.Test.perf/perf.data
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

Reads Nsight reports and writes what the tools print as one CSV of its own. It
is not a benchmark CSV: `bench summary`, `bench compare` and `bench-plot` need
`test`, `wallMedian`, `wallCV` and `callsPerSecond` columns and refuse it
(`missing required column 'test'`, `Missing required columns`). Read it with a
CSV tool.

```bash
nsight-parse parse run.nsys-rep --csv summaries.csv   # one report
nsight-parse parse bench-out/ --csv combined.csv      # every .nsys-rep and .ncu-rep under a directory
```

**How it reads a report.**

- `.nsys-rep`: `nsys export --type sqlite` once, to a private temporary file,
  then `nsys stats --report <summary> --format csv` on that export for
  `cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_size_sum` and
  `cuda_gpu_mem_time_sum`. An export that `bench run --profile nsight` left
  beside the report is neither used nor changed; a plain `nsys stats` on such a
  report can refuse it ("Existing SQLite export found ... older than input
  file").
- `.ncu-rep`: `ncu --import <report> --csv --print-summary per-kernel`.

**What it writes.** For an `.nsys-rep`, one row per row of the four summaries:
one per kernel name, one per CUDA call name, one per kind of copy, not one per
launch. The columns are `source` (`nsys`), `report` (the summary), `kernel` (its
`Name`; empty for the copy summaries, which name the row in `Operation`),
`instances` (`Instances` or `Num Calls`; empty for the copy summaries, which
count in `Count`), `time_total_ns`, `time_avg_ns`, `time_pct`, then every other
column the summaries print, under their own names (`Med (ns)`, `Min (ns)`,
`Max (ns)`, `StdDev (ns)`, `Count`, `Total (MB)` and so on), in alphabetical
order. From walkthrough 11's Nsight Systems step, 13 rows:

```
source,report,kernel,instances,time_total_ns,time_avg_ns,time_pct,Avg (MB),Count,Max (MB),Max (ns),Med (MB),Med (ns),Min (MB),Min (ns),Operation,StdDev (MB),StdDev (ns),Total (MB)
nsys,cuda_gpu_kern_sum,"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *, unsigned long)",61,178481856,2925932.1,100.0,,,,3774688,,2890848.0,,2852032,,,149253.5,
nsys,cuda_api_sum,cudaMemcpy,183,199253613,1088817.6,58.5,,,,3841722,,131481.0,,46630,,,1397635.5,
...
```

For an `.ncu-rep`, one row per launch shape, section and metric (88 for
walkthrough 11's two kernel shapes): `source` (`ncu`), `report` (`per_kernel`),
`kernel`, then ncu's own columns in snake case (`block_size`, `grid_size`,
`invocations`, `section_name`, `metric_name`, `metric_unit`, `minimum`,
`maximum`, `average` and the process columns).

**Exit status.** 0 when every requested report was read; 1 when any was not:
a tool failed or is missing, an input is not a report, or a directory holds
none. Each failure is an error line on stderr, and the rows of the reports that
were read are written all the same. A summary with no data, such as the kernel
summary of a report with no kernel, is a warning. A directory with one good
report and a truncated copy of an ncu report, on the Jetson AGX Thor rig (nsys
2025.3.2, ncu 2025.3.1), exits 1 after:

```
[nsight-parse] error: ncu --import failed for mixed/damaged.ncu-rep: exit status 1: ==ERROR== An unexpected incompatibility with this Nsight Compute version occurred. Try opening the file with the same tool version it was created with.
[nsight-parse] wrote 13 rows to mixed.csv
```

---

## 4. Common Workflows

### Development Iteration

Quick test with immediate summary:

```bash
source build/native-linux-release/.env
bench run MyComponent_PTEST --quick --csv results.csv --analyze
```

### Optimization Workflow

```bash
source build/native-linux-release/.env

# 1. Validate environment
bench validate

# 2. Baseline measurement
bench run MyComponent_PTEST -- --repeats 30 --csv baseline.csv

# 3. Profile to find hotspots (record mode writes perf.data)
bench run MyComponent_PTEST -- --profile perf --profile-args "record -g" --target-time 250ms
bench flamegraph MyComponent.Throughput.perf/perf.data --output before.svg

# 4. Make changes, rebuild

# 5. Measure again
bench run MyComponent_PTEST -- --repeats 30 --csv optimized.csv

# 6. Compare the medians
bench compare baseline.csv optimized.csv --threshold 5

# 7. Visualize (optional)
bench-plot plot optimized.csv --output analysis/
```

### GPU Benchmark Workflow

Full GPU benchmarking pipeline with environment validation, clock locking,
and state monitoring:

```bash
source build/native-linux-release/.env

# 1. Validate GPU environment
bench gpu-env

# 2. Lock clocks for reproducibility
bench gpu-lock lock

# 3. Snapshot before
bench gpu-monitor snapshot -o before.json

# 4. Run benchmark
./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST --repeats 30 --csv gpu_results.csv

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
bench gpu-lock lock -- ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST --repeats 30 --csv gpu_results.csv
```

### CI Regression Detection

```bash
bench compare baseline.csv candidate.csv \
  --threshold 5 \
  --fail-on-regression \
  --markdown > pr_comment.md
```

`--fail-on-regression` is what makes this a gate: exit code 1 on a median
more than 5% above its baseline, and on a baseline test the candidate did not
run. Without the flag the command reports and exits 0. Unusable input --
including two CSVs with no test in common -- exits 1 either way, so a job
that compares the wrong pair of files fails instead of passing silently.
`--markdown` produces a table suitable for PR comments.

---

## 5. CSV Schema

The benchmarking framework outputs CSV files with the following columns.

**Base columns:** test, cycles, repeats, warmup, threads, msgBytes, console,
nonBlocking, minLevel, wallMedian, wallP10, wallP90, wallMin, wallMax, wallMean,
wallStddev, wallCV, callsPerSecond, stable, cvThreshold

**Profile columns (when profiling):** profileTool, profileDir

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

Produces a single `bench` binary in `<build dir>/bin/tools/rust/`
(`build/native-linux-debug` unless `BUILD_DIR` says otherwise; see
[Quick Start](#1-quick-start)).
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
