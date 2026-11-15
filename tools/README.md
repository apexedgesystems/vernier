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
any working directory puts `bench` and `bench-plot` on `PATH`.
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

Single binary with 15 subcommands for benchmarking analysis, profiling
orchestration, profiler report extraction, GPU environment management, and
project setup.

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
PATH does not find, or finds without an execute bit, is `[WARN]`. Each
`--version` runs for at most 5 s in a process group of its own, which
`bench validate` ends before it moves on, so nothing a tool starts outlives
it. Whether a profiler can run here, and in which modes, is what
`bench doctor <binary>` checks.

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
under `build/*/bin/{ptests,tests,examples}` (for non-CMake layouts, set
`.bench.yaml`'s `bin_roots` / `bin_subdirs`, or `VERNIER_BENCH_BIN_ROOTS` /
`VERNIER_BENCH_BIN_SUBDIRS`, which take precedence; see `init`). A name that
is a file in the working directory is that file, as
`bench doctor` reads it: it starts as `./<name>`, never as a program of that
name on PATH, and must be executable.

```bash
bench run BasicWorkflow                                   # short name auto-resolve
bench run ./bin/ptests/MyComponent_PTEST                  # full path
bench run MyComponent --csv results.csv --quick
bench run MyComponent --taskset 2-9 --profile perf
bench run MyComponent --csv results.csv --analyze
bench run MyComponent --profile massif                    # auto-wraps with valgrind
bench run MyComponent --profile heaptrack                 # auto-wraps with heaptrack
bench run MyComponent --profile jemalloc                  # preloads jemalloc, checks its dump
```

**Options:**

| Flag                       | Description                                          | Default      |
| -------------------------- | ---------------------------------------------------- | ------------ |
| `--csv FILE`               | Export results to CSV                                | --           |
| `--quick`                  | Fewer cycles/repeats for fast iteration              | --           |
| `--taskset CPUS`           | Pin to specific CPU cores                            | --           |
| `--profile MODE`           | Enable profiling (any registered backend; see below) | --           |
| `--profile-args ARGS`      | The profile's mode (see below); may start with `-`   | --           |
| `--profile-frequency HZ`   | gperf's CPU sampling rate, 1 to 4000 (see below)     | --           |
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

| `--profile`         | `--profile-args` words                                      | Selects                                                  |
| ------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| `massif`            | `pages` or `stacks`                                         | `--pages-as-heap=yes` or `--stacks=yes`                  |
| `memcheck`          | `leak-full`, `track-origins`                                | the full leak check (the default), `--track-origins=yes` |
| `helgrind`          | `drd`                                                       | valgrind's DRD instead of Helgrind                       |
| `compute-sanitizer` | one of `memcheck`, `racecheck`, `synccheck`, `initcheck`    | that `--tool` (default `memcheck`)                       |
| `nsight`            | `compute` (or `ncu`)                                        | Nsight Compute (`ncu`) instead of Nsight Systems         |
| `ncu`, or `nsight`  | `replay`, `--set NAME`, `--section NAME`, `--metrics NAMES` | what Nsight Compute collects (below)                     |

`callgrind`, `heaptrack` and `jemalloc` take no mode.

**Nsight Compute's options.** `--profile ncu`, and `--profile nsight` when it
selects Nsight Compute, take Vernier's supported subset of ncu's own options:

- `--set NAME`, once: a section set (`ncu --list-sets`: `basic`, `detailed`,
  `full`, ...);
- `--section NAME`, as often as needed: a section by its identifier
  (`ncu --list-sections`: `LaunchStats`, `Occupancy`, ...);
- `--metrics NAME[,NAME...]`, as often as needed: metrics by name, the list
  running on through further words to the next option or mode word;
- `replay`: the replay metric list (throughputs, achieved occupancy, warp
  execution and branch efficiency, registers per thread, the ALU, FMA and
  load/store pipes), written to `kernel_replay.ncu-rep`.

Each option may also be given as `--option=VALUE`. NAME is letters, digits
and `_`, and a metric name may also hold `.`; ncu itself checks that the set,
section or metric exists. Under `nsight`, any of these selects Nsight Compute,
as `compute` does; `replay` takes no other selection. `bench run` starts
`ncu -o <folder>/kernel_profile -f --target-processes all`, then the set,
each section and one `--metrics` with every name, in that order:

```bash
bench run MyKernel_PTEST --profile ncu --profile-args '--set full' --cycles 3 --repeats 1
bench run MyKernel_PTEST --profile ncu --profile-args \
  '--section LaunchStats --section Occupancy --metrics smsp__inst_executed_op_shared_ld.sum'
bench run MyKernel_PTEST --profile nsight --profile-args replay --cycles 3 --repeats 1
```

Any other ncu option (`--kernel-name`, `--launch-count`, `--replay-mode`,
`--nvtx`, `-o`, ...), a value outside its form (a regular expression, a
metric group), an option without its value and a second set are refused
before anything starts, naming what is accepted: the report, its overwrite
and `--target-processes` are `bench run`'s own. For any other option, start
the binary under ncu yourself; `bench doctor` and the binary print the
command.

Under ncu no row is timed: the timing cells stay empty, since ncu's
instrumentation adds to what the benchmark would measure, and the run says
why ([GPU guide](../src/bench/docs/GPU_GUIDE.md#using-nsight-compute)).
`bench summary` and `bench compare` refuse such a row, saying that the run
was a profiling capture, and `bench run` refuses `--analyze` for a run under
ncu before anything starts.

In-process backends (`perf`, `gperf`, `rapl`, `bpftrace`, `offcpu`) run the
binary directly, which reads `--profile-args` itself, and the C++ harness
manages its own per-test artifact subdirs.

With `--profile gperf`, `--profile-frequency HZ` starts the binary with
`CPUPROFILE_FREQUENCY=HZ`, the variable gperftools reads its sampling rate
from only as the process starts, and passes the binary
`--profile-frequency HZ`, which checks that each profile it writes records
that rate's period; the `Running:` line shows the variable. Nothing else is
added: no wrap, no folder. The rate is refused before anything starts
when it is not a whole number from 1 to 4000, or when no profiler, or another
one, is requested. Without the flag `bench run` sets nothing, and a
`CPUPROFILE_FREQUENCY` it inherited reaches the binary as it is.

Arguments after `--` reach the binary as they are, except the request's own
fields: `--profile`, `--profile-args`, `--profile-frequency` and
`--profile-output-dir` (or the binary's `--artifact-root`) given there are the
request `bench run` routes, as if given to it, and reach the binary once. A
field given twice with different values, to `bench run` and after `--`, is
refused before anything starts, and so is `--profile-frequency=HZ` after
`--`, which the binary does not read.

**A wrapped run's output.** Before it starts the benchmark, `bench run`
creates the wrap's folder (a folder it cannot create stops the run before
anything starts) and removes from that folder the files a previous run of
the same wrap left: `callgrind.out`, `massif.out`, `memcheck.log`,
`helgrind.log`, `run.zst` and `run.gz` (heaptrack), `sanitizer.log`,
`profile.nsys-rep` with `profile.sqlite` and the four summaries (nsight),
`kernel_profile.ncu-rep` (ncu, and nsight's compute mode),
`kernel_replay.ncu-rep` (a replay) and
`jeprof.*.heap` (jemalloc), each removal printed as `[bench] removed <file>
from a previous run`. Nothing else in the folder, and nothing outside it, is
removed. valgrind, compute-sanitizer, nsys and ncu read a `%` in an output
option as the start of a macro (`%p`, `%q{VAR}`), so `bench run` writes the
folder there with each `%` doubled, and the output lands in the folder as
named. compute-sanitizer joins a relative log name to its working directory
before it reads the macros, so `bench run` gives it the folder named whole,
from the working directory, as its `Running:` line shows. heaptrack replaces `%h` and `%p` in its `-o` with the host name and
its process id and has no escape for them, so `bench run` refuses a heaptrack
folder holding either before anything starts; any other `%` reaches
heaptrack as it is. After the benchmark exits 0, the file must exist and hold
something: the run prints `[bench] <tool> wrote <file> (<n> bytes)`, and
otherwise fails with `completion: <file> was not written` (or `is empty`).
For nsight, a `nsys stats` summary that fails, or still runs after 300 s,
fails the run as `analysis:`, and the report is kept. With
`--profile-analyze` (given to `bench run`, or forwarded after `--`), a
callgrind run's profile is annotated after valgrind has written it:
`callgrind_annotate --auto=yes <profile>`, its first 40 lines printed; a
missing or failing `callgrind_annotate`, or one still running after 120 s,
fails the run as `analysis:`, and the profile is kept. Each `nsys stats` and
the annotator run in a process group of their own: a process of that group
still running when the program exits is ended, and `bench run` says so, while
the program's own exit status decides the analysis; SIGINT, SIGTERM or SIGHUP
sent to `bench run` meanwhile ends the group, then `bench run` itself (a
signal `bench run` was started with ignored stays ignored).

**jemalloc's wrap.** jemalloc has no program to start the benchmark under:
`bench run` starts the benchmark itself with `LD_PRELOAD=libjemalloc.so.2`
and `MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:<folder>/jeprof`,
so jemalloc samples allocations with their stacks and writes, when the
process exits, a dump holding the bytes allocated over the whole run. The
loader resolves `libjemalloc.so.2` through `LD_LIBRARY_PATH` first, which is
how a jemalloc built with profiling outside the system's folders is selected.
Before anything starts, `bench run` refuses a library the loader cannot
preload (quoting the loader) and a folder holding `,`, which `MALLOC_CONF`
cannot carry; whether the library profiles is the benchmark's to decide, and
a run whose jemalloc cannot exits with status 4. After the benchmark exits 0,
only the dump named after its own process, `jeprof.<pid>.<n>.f.heap`, counts:
it must be a heap profile that maps the benchmark, with its totals line
(`t*:`). The run prints
`[bench] jemalloc wrote <dump> (<n> bytes) when the benchmark exited` (a run
that sampled nothing is a valid run) and the `jeprof --text --alloc_space`
command that reads it, and names any other process's dump in the folder as
not this run's. It prints no number from the totals line: jemalloc writes it
in an encoding of its own for `jeprof`, not as counts of the samples taken,
and `jeprof` estimates the bytes from each stack's own record. Otherwise it
fails with
`completion: jemalloc wrote no exit dump of the benchmark ...`, or says what
is wrong with the dump. Without `jeprof` on `PATH` it says that only reading
the dump needs it.

**compute-sanitizer's verdict.** The route passes `--error-exitcode 5`: the
tool ends with status 5 when it reports errors, whatever the benchmark itself
returned, and otherwise with the benchmark's own status. `bench run` reads
this run's `sanitizer.log` to tell them apart. Errors counted by its summary
fail the run, and the count printed is the summary's: the
`ERROR SUMMARY: N errors` line of memcheck, synccheck and initcheck (not the
`ERROR SUMMARY: N errors were not printed` line the print limit adds after
it), or racecheck's `RACECHECK SUMMARY: H hazards displayed (E errors, W
warnings)`, whose errors count and whose warnings do not. With no errors
counted, the status is the benchmark's own, 5 included. A report that holds
only the tool's own `Error:` line fails the run as `collection:`, a missing
report or one without its summary as `completion:`, and nothing is counted.

Unset `--cycles` / `--repeats` / `--target-time` are filled in from `.bench.yaml` (see `init`).
Its `gtest_filter` is passed to the binary as `--gtest_filter=<filter>`,
before the arguments after `--`, unless one of those starts with
`--gtest_filter=`: a filter given on the command line is the only one the
binary gets.

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

`--profile`, `--profile-args`, `--profile-frequency`, `--profile-analyze` and
the arguments after `--` make the request `bench run` would make of them, and
are passed to the binary as `bench run` passes them, with the
`CPUPROFILE_FREQUENCY` it would set; the doctor adds a row for that request.
`--require` judges the requested backend by that row and every other backend
by its default mode's row, so a requirement on `massif` with
`--profile-args pages` is met only when that mode is ready. The row must
answer the request, by its backend and its mode; a row for another request,
like no row at all, leaves the requirement unmet. A binary built before the
requested row existed cannot answer such a requirement: rebuild it against
this vernier, or drop `--profile`.

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
`<out>/<name>/`, whatever the others did. Each is given `--filter` as
`--gtest_filter`, or without it `.bench.yaml`'s `gtest_filter`, as `bench run`
gives it. The run ends with one line per
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

### nsight-parse - Nsight Reports as CSV

Reads saved Nsight Systems (`.nsys-rep`) and Nsight Compute (`.ncu-rep`)
reports into one CSV of its own. It needs `nsys` on `PATH` for Systems reports
and `ncu` for Compute reports, and nothing else.

```bash
bench nsight-parse run.nsys-rep --csv summaries.csv   # one report
bench nsight-parse bench-out/ --csv combined.csv      # every report under a directory
```

**Options:**

| Flag                | Description                                                                                               | Default  |
| ------------------- | --------------------------------------------------------------------------------------------------------- | -------- |
| `--csv FILE`        | The CSV to write; written even when no report could be read                                               | required |
| `--timeout SECONDS` | How long one `nsys` or `ncu` command may run before it and every process in its process group are stopped | 600      |

**How it reads a report.** A directory stands for every `.nsys-rep` under it,
then every `.ncu-rep`, each sorted by path.

- `.nsys-rep`: `nsys export --type sqlite` once, into a private temporary
  directory removed afterwards, then
  `nsys stats --report <summary> --format csv` on that export for
  `cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_size_sum` and
  `cuda_gpu_mem_time_sum`. Nothing is written beside the report: the export
  `bench run --profile nsight` leaves there is neither used nor changed (a
  plain `nsys stats` on such a report can refuse it: "Existing SQLite export
  found ... older than input file").
- `.ncu-rep`: `ncu --import <report> --csv --print-summary per-kernel`.

`bench run --profile nsight` writes the same four summaries as text beside the
report it made, for reading; this command reads saved reports into one CSV,
for other tools.

**What it writes.** For an `.nsys-rep`, one row per row of the four summaries:
one per kernel name, one per CUDA call name, one per kind of copy, not one per
launch. The columns are `source` (`nsys`), `report` (the summary), `kernel` (its
`Name`; empty for the copy summaries, which name the row in `Operation`),
`instances` (`Instances` or `Num Calls`; empty for the copy summaries, which
count in `Count`), `time_total_ns`, `time_avg_ns`, `time_pct`, then every other
column the summaries print, under their own names (`Med (ns)`, `Min (ns)`,
`Max (ns)`, `StdDev (ns)`, `Count`, `Total (MB)` and so on), in alphabetical
order. For a report of walkthrough 11's `G0` test, 13 rows:

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
`maximum`, `average` and the process columns). Lines end in CRLF. A CSV built
from several reports has no column naming the report a row came from.

It is not a benchmark CSV: `bench summary`, `bench compare` and `bench-plot`
need `test`, `wallMedian`, `wallCV` and `callsPerSecond` columns and refuse it
(`missing required column 'test'`, `Missing required columns`). Read it with a
CSV tool.

**Exit status.** 0 when every requested input was read; 1 when any was not: a
tool failed, is missing, ran past `--timeout` or left a process holding its
output 2 s after it ended, an input is not a report or is empty, or a directory
holds none. Each failure is an error line on stderr
naming the input and its cause, and the rows of the reports that were read are
written all the same. A summary with no data, such as the kernel summary of a
report with no kernel, is a warning. A `--csv` that is one of the reports, by
its own path, a symbolic link or a hard link, is refused before anything is
read. Each `nsys` or `ncu` command runs in a process
group of its own, and what is left of that group is killed when the command
ends, however it ends: a process the tool started outlives it only by leaving
the group. SIGHUP, SIGINT or SIGTERM stops the extraction: the tool's group is
killed, the private export removed and nothing written, and `bench` ends by
that signal; a signal ignored when it starts, as under `nohup`, stays ignored.
On the Jetson AGX Thor rig (nsys 2025.3.2, ncu 2025.3.1), a directory with one
good report and a truncated copy of an Nsight Compute report:

```bash
bench nsight-parse mixed/ --csv mixed.csv   # exits 1
```

```
[nsight-parse] error: ncu --import failed for mixed/damaged.ncu-rep: exit status 1: ==ERROR== An unexpected incompatibility with this Nsight Compute version occurred. Try opening the file with the same tool version it was created with.
[nsight-parse] wrote 13 rows to mixed.csv
```

### init / config-validate - Project Defaults

`bench init` scaffolds a `.bench.yaml` at the project root (cycles, repeats,
target_time, profile_output_dir, gtest_filter, bin_roots, bin_subdirs). Read
by `run` and `profile-all` when the corresponding CLI flag is omitted;
`target_time` (`--target-time`) by `run` alone, passed as written for the
binary to read. A `--quick` on the command line takes precedence over the
file's cycles, repeats and target_time: the binary sizes the run as
`--quick` says, and a `--cycles` or `--repeats` given beside it is passed
too. `gtest_filter` is
the default test filter: `run` passes it unless a `--gtest_filter=` follows
`--`, and `profile-all` unless `--filter` is given. A filter passed this way,
or typed, takes precedence over `GTEST_FILTER` in the environment, which
GoogleTest reads only when no `--gtest_filter` is given: to filter with
`GTEST_FILTER`, remove the key. In a filter, everything after the first `-`
excludes: `-*Large*:*PayloadScaling*` runs every test but those two kinds.

A relative path in the file, `profile_output_dir` or a `bin_roots` entry, is
read from the folder of the `.bench.yaml` that names it, wherever the command
runs from below it; `bin_subdirs` are below each root. A path given on the
command line (`--profile-output-dir`, `--out`) is read from the working
directory.

`bin_roots` and `bin_subdirs` say where `run` and `profile-all` look for a
binary given by a short name: under each root (an entry ending in `/*` stands
for every folder in it), in each subfolder. The roots are tried in the order
listed, and the first with a match decides; among several matches under one
root (an entry ending in `/*` is one root) the first in sorted order is taken,
and `bench` says which. `VERNIER_BENCH_BIN_ROOTS` and
`VERNIER_BENCH_BIN_SUBDIRS` (colon-separated; a relative root in
`VERNIER_BENCH_BIN_ROOTS` is relative to the working directory) take
precedence over the keys, each over its own; without either, the lookup is
`build/*`, from the working directory, and `bin/ptests`, `bin/tests`,
`bin/examples`. A name found nowhere is an error naming the roots and
subfolders searched and where each came from.

`run` and `profile-all` read the whole file before they start anything, and
stop on one they cannot read or use, with an error naming the file and, for
each problem, its line, the key and what the key takes. A problem is a line
that is not a top-level `key: value`, a `- item` below a `key:` line or a
comment; a quote left open; a key given twice; or a value of the wrong kind:
`cycles` and `repeats` take a whole number above 0, `target_time` a duration
the binary reads (at least `1us`), `profile_output_dir` a path, `gtest_filter`
one value, `bin_roots` and `bin_subdirs` a list. `config-validate` checks the
file the same way: it lists each problem as `[FAIL]` and exits 1 with the
error `run` would print. Its warnings do not stop a run: a key it does not
know is the project's own (kept, read by nothing in bench; a misspelt key
shows here), and fewer than 5 repeats or an absolute `profile_output_dir` are
allowed.

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
| `jemalloc`          | CPU   | jemalloc's sampling heap profiler (preloaded)     |
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

# Lock clocks, run a benchmark, then reset them once it has ended
bench gpu-lock lock -- ./bin/ptests/BenchmarkGPU_PTEST --quick --csv results.csv

# Reset clocks to driver-managed default
bench gpu-lock reset
```

**Subcommands:**

| Subcommand | Description                                     |
| ---------- | ----------------------------------------------- |
| `lock`     | Lock clocks (reset when a wrapped command ends) |
| `reset`    | Reset clocks to driver-managed default          |

**Lock options:**

| Flag         | Description                            | Default   |
| ------------ | -------------------------------------- | --------- |
| `--device N` | GPU device index                       | 0         |
| `--freq MHz` | Target frequency in MHz                | max clock |
| `-- CMD...`  | Command to run while clocks are locked | --        |

**The wrapper mode** (`lock -- <command>`) locks the clocks, runs the
command, and resets the clocks after the command has ended, however it ends:

| The run ends with                                                      | The clocks                                                       | `bench`                                                                                                           |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| the command exiting 0                                                  | reset                                                            | exits 0                                                                                                           |
| the command exiting with another status N                              | reset                                                            | exits N, saying so                                                                                                |
| the command ended by a signal N of its own                             | reset                                                            | exits 128 + N, saying so (137 for SIGKILL), as a shell reports it                                                 |
| a lock that `nvidia-smi` refuses, or that does not finish within 30 s  | not locked (one past 30 s is reset, as it may have taken effect) | exits 125, naming why, and starts no command                                                                      |
| a command that cannot start                                            | reset                                                            | exits 125, naming the command                                                                                     |
| SIGINT (Ctrl-C), SIGTERM or SIGHUP sent to `bench`                     | reset after the command has ended                                | ends by the first of them it received (130, 143 or 129 in a shell)                                                |
| a reset that `nvidia-smi` refuses, or that does not finish within 30 s | not reset                                                        | names the commands that reset them and the command's status if not 0, and exits 125 unless a signal above ends it |

Status 125 is `bench`'s own: a failure of the lock, the command's start or
the reset, which an `Error:` line on stderr names. Every other status is the
command's, passed through, so a script reads the command's own result; a
command that itself exits 125 is passed through as 125, without that line.

The clocks are reset only after the command has ended: whatever `bench`
receives, it waits for the command, so the clocks stay locked exactly as long
as the command runs. A command that does not end on the signal keeps
`bench` waiting. End the command yourself with a stronger signal,
`kill -KILL <pid>` with the command's process id, which `bench` prints when
it receives a signal; the reset follows at once, and `bench` then ends by
the first signal it received.

The command runs in `bench`'s own process group, with its terminal, as it
would without the wrapper: Ctrl-C, Ctrl-Z and a hangup reach the command and
`bench` together. `bench` passes a SIGTERM on to the command, since a
terminal never sends one; SIGINT and SIGHUP reach the command from the
terminal. Two consequences: a SIGTERM sent to the whole process group (as
`kill %1` sends it) reaches the command twice, from the sender and from
`bench`; and a SIGINT sent to `bench` alone (`kill -INT <pid>`) is not passed
on, so `bench` waits for the command to end by itself. A signal `bench`
receives while the command is being started, before the command exists,
ends the command as it starts. A signal ignored when
`bench` started (under `nohup`, or in a background job of a shell without job
control) stays ignored, for `bench` and the command. Each `nvidia-smi` call
runs in a process group of its own, which no signal sent to `bench` reaches,
so a lock or a reset is never stopped halfway; a signal received meanwhile
is acted on once the call has finished. The lines `bench` prints about the
lock, the reset and each signal, and the error it reports before it ends by
a signal, hold up neither the reset nor that end: each waits for stderr only
while stderr has room, a second at the most, so a stderr that takes nothing
more (a full pipe nobody reads, a terminal that hung up) loses them instead.
Two writes are ordinary ones that a full pipe nobody reads can hold up: the
error line of a run that fails with no signal received, which comes after
the reset, and Rust's own message if `bench` panics.

The SIGTERM `bench` passes on reaches the command it started, the process
named after `--`, and no process that command starts. A shell script that
runs its benchmark as a child is ended by the signal (a shell without a trap
for it ends at once) while the benchmark goes on: `bench` then resets the
clocks under the running benchmark and ends by the signal. So a wrapper
script should `exec` its benchmark (`exec ./MyBench_PTEST "$@"` as its last
line), which makes the benchmark the command `bench` started, or forward the
signal to it: start it in the background, `trap` SIGTERM to `kill` it, and
`wait` for it. To end a whole job, `bench` and every process the command
started, send the signal to the job's process group (`kill -TERM -- -<pgid>`,
as a shell's `kill %1` does): each process in it receives the signal, and
`bench` still waits for the command to end before it resets the clocks.

Not covered: SIGKILL sent to `bench`, SIGQUIT (Ctrl-\, which also dumps
core), a crash of `bench` and a power loss end it without the reset. Reset
the clocks by hand:

```bash
bench gpu-lock reset --device N      # or, as root: nvidia-smi -i N -rgc
```

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
