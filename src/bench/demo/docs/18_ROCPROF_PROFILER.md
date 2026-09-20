# Demo 18: rocprof for AMD GPU Profiling

**Reference rig:** none. rocprof needs an AMD GPU, and neither reference rig
has one ([Reference Rigs](../../docs/rigs/README.md)).
**Build:** Release
**Example:** none. The demo tree has no HIP workload; see
[The Example](#the-example).
**Captured:** 2026-09-19, on an x86-64 Linux host with no AMD GPU, no ROCm
and no rocprof installed.
**Versions:** written for the Vernier 1.0.4 release. The captures come from
the development tree ahead of it, whose CMake project version is 1.0.3 and
whose CLI reported `bench 1.0.2` at capture time.

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

- It is created only when a program named `rocprof` is on `PATH`. Without
  one, the profiler registry reports the backend as unavailable and the
  measurement runs unprofiled (Step 1).
- It never starts, stops or configures rocprof. rocprof wraps the process;
  Vernier only reacts to being wrapped.
- When rocprof is on `PATH` and the process is not wrapped, the backend
  prints the rocprof command line to use, naming a per-test artifact
  directory `<TestName>.rocprof/` (placed under `--profile-output-dir <dir>`
  when that flag is given). Vernier creates that directory but writes nothing
  into it: the files inside are rocprof's, and only a wrapped run produces
  them.
- It decides "am I wrapped" by looking for `ROCP_TOOL_LIB`,
  `ROCPROFILER_LIBRARY`, or `rocprof` inside `LD_PRELOAD`, expecting rocprof
  to set one of them. Whether current rocprof releases do is not verified
  here. When one is present it prints a line naming the mode and the artifact
  directory instead of the wrap instruction.
- `--profile-args` picks the mode: `stats`, `hsa-trace` and `hip-trace` map
  to rocprof's `--stats`, `--hsa-trace` and `--hip-trace`; anything else is
  the default mode with no extra flag.
- `bench run --profile rocprof` does **not** wrap the binary in rocprof, the
  way it wraps the Valgrind tools, heaptrack, Compute Sanitizer, `nsys` and
  `ncu`. It runs the binary directly, so the run is unprofiled unless you
  invoke rocprof yourself.

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
taskset -c 8 ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow \
    --profile rocprof --quick --gtest_filter='BasicWorkflow.*'
```

`taskset` keeps the measurement on one core; core 8 is this capture host's
choice, and any core, or no `taskset` at all, reaches the same backend path.

Captured output, trimmed to the last of the three tests:

```
...
[ RUN      ] BasicWorkflow.QuickModeIteration

[WARN] Profiler 'rocprof' requested but unavailable on this platform.
   Install ROCm + rocprof (apt install rocprofiler on Debian/Ubuntu).
   Falling back to no-op (measurements will proceed without profiling).

[BasicWorkflow.QuickModeIteration]  4.577 us/call  CV=0.2%  ~218.5K calls/s  (p10=4.568 p90=4.587 sd=0.008)
[       OK ] BasicWorkflow.QuickModeIteration (115 ms)
...
```

What to read: the measurement still runs and the binary still exits 0. The
only sign that nothing was profiled is that warning, once per test. No
artifact directory is created, because the backend was never constructed.

The install line in that warning is generic. What the backend needs is a
program named exactly `rocprof`; which package provides it, and whether the
ROCm release you install still carries the legacy command, depends on your
distribution and ROCm version.

Note what is absent: the backend's wrap instruction. That hint comes from the
backend, and the backend exists only when rocprof is on `PATH`.

Routing the same request through the CLI changes nothing:

```bash
bench run ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow \
    --profile rocprof --quick --taskset 8
```

Captured output, first line:

```
Running: taskset -c 8 ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow --quick --profile rocprof
...
```

The runner passes `--profile rocprof` to the binary and runs it directly. It
builds no rocprof command, so the same warning follows.

## Step 2: Check the Machine with the Doctor

```bash
bench doctor ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow
```

Captured output, rocprof's row and the summary (the rows for the other
backends and the readiness section above them are cut):

```
...
=== Profiler Backend Doctor ===

  ...
  [FAIL] rocprof    ROCm not detected (no rocprof on PATH, no /opt/rocm)
             Install ROCm + roctracer (https://rocm.docs.amd.com).

  12 backend(s), 5 fail.
```

This row is a marker check, not a capability test. The backend looks for two
things: a program named `rocprof` on `PATH`, and either
`/opt/rocm/lib/libhsa-runtime64.so` or `/sys/class/kfd/kfd/topology/nodes`
openable as a file. It enumerates no GPUs, tests no device permissions, runs
no HIP code and never starts the profiler. The four states below are the four
combinations of those two lookups. The capture above is the first of them;
the messages of the other three are quoted from the backend's environment
check in the source, not from a run on a machine in that state:

| Tag      | Message                                                            | What was found                                                                                                                                   |
| -------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[FAIL]` | `ROCm not detected (no rocprof on PATH, no /opt/rocm)`             | no `rocprof` on `PATH` and neither marker path openable. This is the row captured above                                                          |
| `[FAIL]` | `ROCm runtime present but rocprof binary missing`                  | a marker path opened, no `rocprof` on `PATH`. The command may be installed under another name, or outside this `PATH`                            |
| `[WARN]` | `rocprof present but no ROCm runtime / GPU kernel driver detected` | `rocprof` found, neither marker path openable. A GPU and a driver may still be present at paths this check does not look at                      |
| `[OK]`   | `rocprof + ROCm runtime available`                                 | both lookups succeeded. That is a precondition for Step 3, not proof of GPU access, of a compatible stack, or that a capture will produce output |

For a script or a CI lane, ask for the verdict instead of the table:

```bash
bench doctor ./build/native-linux-release/bin/ptests/BenchDemo_01_BasicWorkflow \
    --require rocprof
```

Captured output (exit status 1):

```
[require] rocprof: NOT READY (ROCm not detected (no rocprof on PATH, no /opt/rocm))
          Install ROCm + roctracer (https://rocm.docs.amd.com).
[require] 1 requirement(s) unmet
```

## Step 3: Profile on an AMD Host

_Not run: needs an AMD GPU. No output is shown for this step because nobody
has run it._

This step needs three things the doctor does not establish: working AMD GPU
hardware, a ROCm installation that provides the legacy `rocprof` command, and
a HIP performance test of your own. With those, wrap the test in rocprof and
let Vernier measure inside it:

```bash
# Not run: needs an AMD GPU.
rocprof -o <Suite>.<Test>.rocprof/results.csv \
    ./build/bin/ptests/<YourHipPtest> --profile rocprof \
    --gtest_filter='<Suite>.<Test>'
```

That block is a template, not a runnable example: every `<...>` is yours to
fill, and no binary in this repository matches it. The artifact directory is
named after the test the backend is attached to, which is why the path
carries the full GoogleTest name.

Run the binary once without the wrap first. The backend prints the rocprof
command it expects, naming the artifact directory it will use and leaving
placeholders for the binary and its arguments, so the line above comes from
your own build rather than from this page. `--profile-args stats`,
`hsa-trace` and `hip-trace` change that printed command: the backend adds
rocprof's `--stats`, `--hsa-trace` or `--hip-trace` to it.

## Step 4: Read the Report

_Not run: needs an AMD GPU._

The report is expected at the path the `-o` argument names: the backend
builds its printed command as `-o <artifact directory>/results.csv`, and it
writes nothing there itself. When it detects the wrap, Vernier's own output
gains one line naming the mode and that directory, in place of the wrap
instruction.

This page does not name the columns of `results.csv` and does not describe
the trace file, because nobody on this project has seen either. A walkthrough
states what a run produced.

## Validate This Walkthrough on an AMD GPU

If you have an AMD GPU and ten minutes, this is the missing piece. Four
outputs are enough to replace the stub with a walkthrough written from a real
run. The recipe applies to an installation that still exposes the legacy
`rocprof` command; on a ROCm release that ships only `rocprofv3`, the backend
has nothing to find, and that result is worth reporting too.

1. Build Release and confirm both lookups succeed:

   ```bash
   # Not run here: needs an AMD GPU.
   bench doctor ./build/bin/ptests/<YourHipPtest> --require rocprof
   ```

   Expect the check to report rocprof ready, and exit status 0. That means
   the command and a marker path were found; whether a capture works is what
   the next two steps test.

2. Run your HIP test once with `--profile rocprof` and no wrap, and keep the
   `[rocprof]` block it prints.
3. Run the command from that block with the placeholders filled in, and keep
   whatever rocprof writes.
4. Report, through the project's issue tracker: your `rocprof --version`, the
   doctor row, the `[rocprof]` block, and a listing of the artifact directory
   with the first lines of the CSV.

That is enough to turn this page into a walkthrough written from a real run,
and to tell whether the backend needs work for current ROCm releases.

## What Should Reproduce

| Reading                                     | On the capture host      | Elsewhere                                             |
| ------------------------------------------- | ------------------------ | ----------------------------------------------------- |
| doctor's rocprof row                        | fails: ROCm not detected | depends on the machine; four states, listed in Step 2 |
| `--profile rocprof` warning, run unprofiled | yes                      | the same on any host without rocprof on `PATH`        |
| artifact directory after an unwrapped run   | none created             | created, and empty, when rocprof is on `PATH`         |
| kernel timings, `results.csv`               | none                     | unknown: needs an AMD GPU, see the validation recipe  |

**Expected numbers:** none. This walkthrough has no measured reading of its
own, on this host or any other, so there is nothing to compare against. The
timing line in Step 1 belongs to a CPU demo and is incidental.

## If It Does Not Match

- **The doctor fails with `ROCm not detected`.** Neither lookup succeeded: no
  program named `rocprof` on this `PATH`, and neither marker path openable.
  That is not a statement about what is installed. Check whether an
  installation is present but outside `PATH`, and whether it provides the
  legacy command at all, rather than only the current one.
- **The doctor fails because the `rocprof` binary is missing.** A marker path
  opened and the command did not resolve. A ROCm installation whose CLI is
  `rocprofv3` reaches this state: the backend looks for the older name and
  invokes nothing else.
- **The doctor warns that rocprof has no runtime or kernel driver.** The
  command resolved and neither of the two marker paths could be opened. A GPU
  and its driver may still be present; the marker paths may be outside the
  container or installed elsewhere. Missing driver devices alone does not
  determine this row.
- **You wrapped the run in rocprof and still got the "NOT running under
  rocprof" hint.** The backend infers the wrap from `ROCP_TOOL_LIB`,
  `ROCPROFILER_LIBRARY` or `LD_PRELOAD`; a rocprof release that sets none of
  them leaves the hint printed whatever the wrap is doing. Judge by the files
  rocprof leaves, not by the hint, and please report the case.
- **`bench run --profile rocprof` produced no profile.** Expected: the runner
  does not wrap rocprof. Invoke rocprof yourself, as in Step 3.

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
