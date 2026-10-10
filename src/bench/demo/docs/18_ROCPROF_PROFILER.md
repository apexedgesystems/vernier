# Demo 18: rocprof for AMD GPU Profiling

**Reference rig:** none. rocprof needs an AMD GPU, and neither reference rig
has one ([Reference Rigs](../../docs/rigs/README.md)).
**Build:** Release
**Example:** none. The demo tree has no HIP workload; see
[The Example](#the-example).
**Captured:** 2026-10-03 (UTC), in the project's development container on an
x86-64 laptop with no AMD GPU, no ROCm and no rocprof installed.
**Versions:** written for the Vernier 1.0.4 release. The captures come from
the development tree ahead of it, whose CMake project version is 1.0.3 and
whose CLI reported `bench 1.0.3` at capture time.

**This walkthrough is not validated on AMD hardware.** The backend ships and
is selectable, and nobody on this project has run it against an AMD GPU. The
output blocks below come from the machine named above, which cannot profile
anything with rocprof: they show what the backend does when its tool is
absent, which is the state most readers are in. Every step that needs an AMD
GPU is written as an instruction and marked _Not run_, with no invented
output. [Validate This Walkthrough on an AMD GPU](#validate-this-walkthrough-on-an-amd-gpu)
says what is missing and how to close it. It does not meet the walkthrough
contract in [the demo README](../README.md#7-reference-rigs-and-the-demo-contract):
no rig, no example, no test asserting the effect. Each section below says so
where it applies.

## Overview

`rocprof` is the command-line profiler of the first ROCProfiler generation on
AMD GPUs. Vernier's backend targets that command by name, and AMD deprecates
it in favour of a newer tool, so this page is an unvalidated integration with
a legacy tool rather than a route to current ROCm profiling. It describes
what the backend does and does not do, shows what you see when `rocprof` is
missing, and gives the first reader with an AMD GPU a short recipe to
validate the backend and report what it prints.

## What is rocprof?

`rocprof` is the CLI of the first version of ROCProfiler. AMD's
[tool status page](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
lists ROCProfiler, ROCTracer, `rocprof` and `rocprofv2` as deprecated and
strongly recommends upgrading to the ROCprofiler-SDK library and its
[`rocprofv3` tool](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html),
whose own page calls it backward compatible with `rocprof`. **Vernier's
backend provides no `rocprofv3` integration:** it looks for the exact name
`rocprof`, and an installation that ships only the newer command does not
satisfy it.

Per AMD's documentation the tool records kernel timings and HIP and HSA
activity, in formats that documentation defines. Nothing on this page was
produced by running it, so its behaviour is not described further here. What
Vernier assumes of it is narrower and is visible in the backend's code: a
tool invoked from outside the process, as
`rocprof [mode flag] -o <file> <binary> [args]`.

- **Best for:** what AMD's documentation points it at -- kernel timing and API
  tracing of HIP workloads on AMD GPUs.
- **How it works, for Vernier's purposes:** the measured binary runs under
  rocprof. Nothing is linked into the binary and nothing is called
  in-process; the backend only reacts to the wrap.
- **Overhead:** not measured by this project.
- **Skip it for:** NVIDIA GPUs ([Demo 11](11_NSIGHT_PROFILER.md)), CPU work
  ([Demo 02](02_PERF_PROFILER.md)), and hosts without a working ROCm install.

**In Vernier:** `--profile rocprof` selects the backend. What it does:

- Without a program named `rocprof` on `PATH`, the request fails: the
  measurement runs, and the run exits with status 4 when its tests pass
  (Step 1).
- It never starts, stops or configures rocprof. rocprof wraps the process;
  Vernier only reacts to being wrapped.
- When rocprof is on `PATH` and the process does not run under it, the
  request fails the same way, and the run prints the command that starts it
  under rocprof, `rocprof -o ./results.csv <this-binary> --profile rocprof [...]`.
  No artifact directory is created for a request that fails.
- It decides that rocprof runs the process from `ROCP_TOOL_LIB`,
  `ROCPROFILER_LIBRARY`, or `rocprof` inside `LD_PRELOAD`, expecting rocprof
  to set one of them. Whether current rocprof releases do is not verified
  here. With one present the request proceeds, reported unverified
  (`rocprof's injection is present (...); AMD collection is not validated (legacy rocprof)`),
  and the backend creates a per-test directory `<TestName>.rocprof/` (under
  `--profile-output-dir <dir>` when that flag is given) and writes nothing
  into it: rocprof writes where its `-o` says.
- `--profile-args` picks the mode: `stats`, `hsa-trace` and `hip-trace` map
  to rocprof's `--stats`, `--hsa-trace` and `--hip-trace`, which the printed
  command then carries; any other word is refused.
- `bench run --profile rocprof` does **not** wrap the binary in rocprof, the
  way it wraps the Valgrind tools, heaptrack, Compute Sanitizer, `nsys` and
  `ncu`. It runs the binary directly, so the request fails unless you start
  the run under rocprof yourself.

**Needs:** three separate things, none of which Vernier ships or tests here.
Working AMD GPU hardware with its kernel driver. A ROCm installation that
exposes the legacy `rocprof` command under that name; no ROCm version is
claimed to have been tested with this backend. And a HIP workload to profile.

## The Example

There is no example for this walkthrough. Vernier's GPU demos
(`BenchDemo_Gpu_*`) are CUDA, and the shared workloads in
[helpers/DemoWorkloads.hpp](../helpers/DemoWorkloads.hpp) are CPU code.
Profiling an AMD GPU through Vernier means pointing rocprof at a performance
test of your own, built for HIP.

The two steps below use the CPU demo `BenchDemo_01_BasicWorkflow`, because
what they exercise -- backend selection and the readiness check -- happens
before any workload runs and does not depend on which one it is.

## Step 1: Ask for the Backend Without ROCm

```bash
taskset -c 13 ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow \
    --profile rocprof --quick --gtest_filter='BasicWorkflow.*'
```

`taskset` keeps the measurement on one core; core 13 is this capture host's
choice, and any core, or no `taskset` at all, reaches the same backend path.

Captured output, trimmed to the first test's request line and the end of the
run (exit status 4):

```
...
[ RUN      ] BasicWorkflow.JoinV0

[FAIL] Profiler 'rocprof': missing: rocprof not found on PATH
   Install ROCm's rocprofiler (apt install rocprofiler on Debian and Ubuntu).
   Nothing is collected for this request; the run will fail (exit status 4 if the tests pass).

...
[  PASSED  ] 3 tests.

[profile] --profile rocprof failed; the run exits with status 4:
[profile]   rocprof: missing: rocprof not found on PATH
...
```

What to read: the measurement still runs and the tests pass, the request's
line says once what is missing, and the run ends with a report of the request
and exit status 4. No artifact directory is created.

The install line is generic. What the backend needs is a program named exactly
`rocprof`; which package provides it, and whether the ROCm release you install
still carries the legacy command, depends on your distribution and ROCm
version.

Note what is absent: the command that starts the run under rocprof. The
request prints it only when rocprof is on `PATH`; without it, the remedy is to
install it.

Routing the same request through the CLI changes nothing:

```bash
bench run ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow \
    --profile rocprof --quick --taskset 13
```

Captured output, first and last lines:

```
Running: taskset -c 13 ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow --quick --profile rocprof
...
Error: the requested profile failed (the benchmark's report above says why); the benchmark exited with status 4
```

The runner passes `--profile rocprof` to the binary and runs it directly. It
builds no rocprof command, so the same failure follows, and `bench run`
exits 1.

## Step 2: Check the Machine with the Doctor

```bash
bench doctor ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow
```

Captured output, rocprof's row and the summary (the rows for the other
backends and the readiness section above them are cut):

```
...
=== Profiler Backend Doctor (default mode of each backend) ===

  ...
  [FAIL] rocprof    missing: rocprof not found on PATH
             Install ROCm's rocprofiler (apt install rocprofiler on Debian and Ubuntu).

  15 backend(s), 8 fail.
...
```

This row is a lookup, not a capability test. The backend looks for a program
named `rocprof` on `PATH`; it enumerates no GPUs, tests no device
permissions, runs no HIP code and never starts the profiler. Three states
follow from that lookup. The capture above is the first of them; the messages
of the other two are quoted from the backend's check in the source, not from
a run on a machine in that state:

| Tag      | Message                                                                                                                       | What was found                                                                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[FAIL]` | `missing: rocprof not found on PATH`                                                                                          | no `rocprof` on `PATH`. This is the row captured above                                                                                                       |
| `[FAIL]` | `unusable: <path> is not an executable file`                                                                                  | a file named `rocprof` on `PATH` without an execute bit                                                                                                      |
| `[WARN]` | `unverified: AMD collection is not validated (legacy rocprof): rocprof is <path>; no AMD device access or capture is checked` | `rocprof` found. The most this row reports: a precondition for Step 3, not proof of GPU access, of a compatible stack, or that a capture will produce output |

The row is never `[OK]`. For a script or a CI lane, ask for the verdict
instead of the table:

```bash
bench doctor ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow \
    --require rocprof
```

Captured output (exit status 1):

```
[require] rocprof: NOT READY (missing: rocprof not found on PATH)
          Install ROCm's rocprofiler (apt install rocprofiler on Debian and Ubuntu).
[require] 1 requirement(s) unmet
```

`--require` accepts only `[OK]`, so on a machine where rocprof is found the
requirement is still unmet, on the `unverified` row: until the backend is
validated on AMD hardware, no lane can require it.

## Step 3: Profile on an AMD Host

_Not run: needs an AMD GPU. No output is shown for this step because nobody
has run it._

This step needs three things the doctor does not establish: working AMD GPU
hardware, a ROCm installation that provides the legacy `rocprof` command, and
a HIP performance test of your own. With those, wrap the test in rocprof and
let Vernier measure inside it:

```bash
# Not run: needs an AMD GPU.
rocprof -o ./results.csv \
    ./build/bin/ptests/<YourHipPtest> --profile rocprof \
    --gtest_filter='<Suite>.<Test>'
```

That block is a template, not a runnable example: every `<...>` is yours to
fill, and no binary in this repository matches it.

Run the binary once without the wrap first. The request fails and prints the
rocprof command it expects, `rocprof -o ./results.csv <this-binary> --profile
rocprof [...]`, with placeholders for the binary and its arguments, so the
line above comes from your own build rather than from this page.
`--profile-args stats`, `hsa-trace` and `hip-trace` change that printed
command: the backend adds rocprof's `--stats`, `--hsa-trace` or `--hip-trace`
to it.

## Step 4: Read the Report

_Not run: needs an AMD GPU._

The report is expected at the path the `-o` argument names, `./results.csv`
in the printed command; Vernier writes nothing there itself. When it detects
the wrap, the run proceeds with one readiness line,
`[WARN] Profiler 'rocprof': unverified: rocprof's injection is present (<markers>); AMD collection is not validated (legacy rocprof)`,
in place of the failure, and the backend creates the per-test directory
`<Suite>.<Test>.rocprof/`, which stays empty.

This page does not name the columns of `results.csv` and does not describe
the trace file, because nobody on this project has seen either. A walkthrough
states what a run produced.

## Validate This Walkthrough on an AMD GPU

If you have an AMD GPU and ten minutes, this is the missing piece. Four
outputs are enough to replace the stub with a walkthrough written from a real
run. The recipe applies to an installation that still exposes the legacy
`rocprof` command; on a ROCm release that ships only `rocprofv3`, the backend
has nothing to find, and that result is worth reporting too.

1. Build Release and check the lookup:

   ```bash
   # Not run here: needs an AMD GPU.
   bench doctor ./build/bin/ptests/<YourHipPtest>
   ```

   Expect rocprof's row to be `[WARN]` and `unverified`, naming the command it
   found. Whether a capture works is what the next two steps test.

2. Run your HIP test once with `--profile rocprof` and no wrap, and keep the
   `[FAIL] Profiler 'rocprof'` lines it prints: they hold the command to run.
3. Run that command with the placeholders filled in, and keep the readiness
   line the run prints and whatever rocprof writes.
4. Report, through the project's issue tracker: your `rocprof --version`, the
   doctor row, those lines, and a listing of the working directory with the
   first lines of the CSV.

That is enough to turn this page into a walkthrough written from a real run,
and to tell whether the backend needs work for current ROCm releases.

## What Should Reproduce

| Reading                                   | On the capture host                         | Elsewhere                                                            |
| ----------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------- |
| doctor's rocprof row                      | fails: `missing: rocprof not found on PATH` | depends on the machine; three states, listed in Step 2, never `[OK]` |
| `--profile rocprof` run without rocprof   | the tests run, exit status 4                | the same on any host where rocprof does not run the process          |
| artifact directory after an unwrapped run | none created                                | none created                                                         |
| kernel timings, `results.csv`             | none                                        | unknown: needs an AMD GPU, see the validation recipe                 |

**Expected numbers:** none. This walkthrough has no measured reading of its
own, on this host or any other, so there is nothing to compare against. The
timing line in Step 1 belongs to a CPU demo and is incidental.

## If It Does Not Match

- **The doctor fails with `missing: rocprof not found on PATH`.** No program
  named `rocprof` on this `PATH`. That is not a statement about what is
  installed. Check whether an installation is present but outside `PATH`, and
  whether it provides the legacy command at all, rather than only the current
  one: a ROCm installation whose CLI is `rocprofv3` reaches this state, since
  the backend looks for the older name and invokes nothing else.
- **The doctor fails with `unusable: <path> is not an executable file`.** A
  file named `rocprof` is on `PATH` without an execute bit.
- **The doctor warns `unverified: AMD collection is not validated (legacy
rocprof)`.** rocprof was found. That is the most this row reports, and a
  `--require rocprof` stays unmet on it.
- **You wrapped the run in rocprof and it still failed, with `missing: rocprof
collects only when rocprof runs the process, and rocprof does not run this
one`.** The backend infers the wrap from `ROCP_TOOL_LIB`,
  `ROCPROFILER_LIBRARY` or `LD_PRELOAD`; a rocprof release that sets none of
  them leaves the request failing whatever the wrap is doing. Judge by the
  files rocprof leaves, and please report the case.
- **`bench run --profile rocprof` fails.** Expected: the runner does not wrap
  rocprof, so the benchmark runs without it and the request fails. Invoke
  rocprof yourself, as in Step 3.

## Check Against the Reference

There is no reference CSV for this walkthrough, and there cannot be one until
a run on an AMD GPU produces the first set of numbers. Other walkthroughs
compare against a capture from their reference rig; this one has neither rig
nor capture.

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- the NVIDIA tools, and
  a walkthrough with a rig behind it
- [Demo 17 (Compute Sanitizer)](17_COMPUTE_SANITIZER.md) -- another backend
  that wraps the process from outside
- [Reference Rigs](../../docs/rigs/README.md) -- why a walkthrough names a
  machine, and which walkthroughs fall outside the two rigs
- [GPU Guide](../../docs/GPU_GUIDE.md) -- the backend's reference section
