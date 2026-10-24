# Demo 08: Intel RAPL -- the Energy a Call Costs

**Reference rig:** none. RAPL needs an Intel CPU, and neither reference rig
has one ([Reference Rigs](../../docs/rigs/README.md)). This page is captured
on an Intel laptop without dedicated reference-rig qualification.
**Captured on:** an x86-64 laptop with an Intel Core Ultra 7 165H, Ubuntu
22.04.5, Linux 6.8.0, `intel_pstate` with governor `powersave`, energy
preference `balance_performance` and power profile `balanced`, on AC power;
timing pinned to CPU 10, one of its performance cores. It is a development
machine shared with other work, not a reference: the load average was about
25 while the steps below ran.
**Build:** Release, on the laptop itself ([Prerequisites](#prerequisites)).
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-10-17 (UTC), steps 1 and 2. Written for the Vernier 1.0.4
release from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`.

**The energy steps of this page have not run.** Reading the energy counter
needs access this laptop's ordinary user does not have, and no run with that
access has been made for this release yet. Steps 1 and 2 are captured as an
ordinary user: the time per call, and what a run without access does. Steps
3 and 4, which need the access, are written as instructions and marked _Not
run_, with no invented output. [Validate the Energy Steps](#validate-the-energy-steps)
says what is missing. The page meets the walkthrough contract in
[the demo README](../README.md#7-reference-rigs-and-the-demo-contract) only
in part, and says so where it applies.

## Overview

A timer says how long a call takes. RAPL says how much energy the processor
package used while the calls ran, and from that, how much energy a call
costs. This walkthrough measures the two versions of the `join` example, which
build the same string 30 times apart in time on this laptop, under Vernier's
`rapl` backend. Step 1 measures them; step 2 shows what an ordinary user
gets: a run that refuses, says why, and fails; steps 3 and 4 measure the
energy with access, and read the report.

## What is RAPL?

RAPL (Running Average Power Limit) is a set of model-specific registers on
Intel CPUs. One of them, `MSR_PKG_ENERGY_STATUS` (0x611), counts the energy
package 0 has used, in units that another register, `MSR_RAPL_POWER_UNIT`
(0x606), gives, and the backend decodes. On this laptop the kernel's powercap
interface, which any user can read, puts the counter's range at 262,143,328,850
uJ over 2^32 counts: 61.035 uJ a count, 2^-14 J. Linux reads model-specific
registers through the `msr` driver's device, `/dev/cpu/0/msr` for CPU 0,
which is in package 0.

- **Scope:** the whole package, every core, cache and the integrated graphics,
  whatever runs on them, not the test's thread alone; nothing outside the
  package (no DRAM counter is read, and no other device). A quiet machine
  makes the reading cleaner. It does not make a package counter an
  attribution to one process.
- **Resolution:** the counter is 32 bits wide; at this laptop's unit it wraps
  after 262,144 J, which takes more than half an hour even at the package's
  short-term limit (115 W here). A window is taken to wrap it at most once.
- **Overhead:** two register reads per measured window, before the first
  repeat and after the last, outside the timed calls.
- **Not for:** where the energy goes inside a function (use a profiler for
  time: [02](02_PERF_PROFILER.md), [03](03_GPERF_PROFILER.md)), CPUs that
  are not Intel's (ARM boards, AMD), or the whole machine's power.

**In Vernier:** `--profile rapl` reads the package counter before and after
each measured window and writes `<Test>.rapl/energy.txt` under the output
root (`--profile-output-dir`, or the working directory), one numbered section
per window, then prints a summary that names the file. The CSV gets no
energy columns: a row's `profileDir` names its case's folder. Before the first
case, the backend opens the device and reads both registers; if it cannot, the
run says why (`denied`, `missing`, `unsupported`), creates no folder, and
fails with exit status 4 once its tests pass.

**Needs:** an Intel CPU, the `msr` driver loaded (`sudo modprobe msr`), and a
process that may read `/dev/cpu/0/msr`: root, or a process with
CAP_SYS_RAWIO, which the driver checks, that also has read permission on the
device, which the capability does not grant (the device is
`crw------- root root` by default). Vernier does not raise its privileges for
RAPL. In a container the device must be passed in and the process inside must
meet the same two requirements
([Docker setup](../../docs/DOCKER_SETUP.md#rapl-energy-profiling)).

## Prerequisites

Build Release on the machine that will measure, with its own toolchain (the
build used here, run in the repository's top folder):

```bash
cmake --preset native-linux-release -B build -DVERNIER_BUILD_GPU=OFF
cmake --build build
source build/.env     # puts the bench CLI on PATH
```

## The Example

The two versions of the shared [`join`](../examples/join/inc/Join.hpp)
example, which [walkthrough 01](01_BASIC_WORKFLOW.md#the-example) explains:
`joinV0` rebuilds the string through temporaries for every part, `joinV1`
measures the answer, reserves once and appends in place. Both return the same
string; the example's unit tests hold them to that.

The demo, [`cpu/08_RaplProfiler_Demo.cpp`](../cpu/08_RaplProfiler_Demo.cpp),
measures each in its own test over 1,000 words, the profiler guard creating
the profiler `--profile rapl` asks for:

```cpp
PERF_THROUGHPUT(RaplProfiler, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}
```

`RaplProfiler.JoinV1` is the same with `joinV1`. A window of about two
seconds a case (`--target-time 100ms --repeats 20`) holds some two thousand
of the counter's updates, which Intel documents as happening about once a
millisecond; this page has not measured that interval.

## Step 1: Measure

```bash
taskset -c 10 ./build/bin/ptests/BenchDemo_08_RaplProfiler \
  --target-time 100ms --repeats 20 --csv rapl_timing.csv
```

Captured output (the harness's progress line, which it overwrites, left out):

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from RaplProfiler
[ RUN      ] RaplProfiler.JoinV0
[target-time] 100.000 ms -> cycles=903 (calibrated 110.6875 us/call, batch of 16)
[RaplProfiler.JoinV0]  122.659 us/call  CV=3.5%  ~8.2K calls/s  (p10=120.165 p90=128.841 sd=4.337)
[       OK ] RaplProfiler.JoinV0 (2249 ms)
[ RUN      ] RaplProfiler.JoinV1
[target-time] 100.000 ms -> cycles=24450 (calibrated 4.0898 us/call, batch of 256)
[RaplProfiler.JoinV1]  4.023 us/call  CV=2.8%  ~248.5K calls/s  (p10=3.904 p90=4.202 sd=0.111)
[       OK ] RaplProfiler.JoinV1 (1970 ms)
[----------] 2 tests from RaplProfiler (4219 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (4219 ms total)
[  PASSED  ] 2 tests.

=====================================================================
Test                  Median (us)     CV%     Calls/s  Status
---------------------------------------------------------------------
RaplProfiler.JoinV0       122.659    3.5%        8.2K  OK
RaplProfiler.JoinV1         4.023    2.8%      248.5K  OK
---------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

JoinV0 takes 122.7 us a call and JoinV1 4.0 us, 30 times less, each with a
spread of a few percent on this busy machine. Each window is about two
seconds (903 and 24,450 calls a repeat, 20 repeats). That ratio is what the
energy should follow, unless the package draws very different power for the
two.

## Step 2: Check Access

As the same ordinary user, ask the doctor about the request:

```bash
bench doctor ./build/bin/ptests/BenchDemo_08_RaplProfiler --profile rapl
```

The `rapl` row, and the row for the selected request, captured (the other
backends' rows left out):

```
  [FAIL] rapl       denied: cannot open /dev/cpu/0/msr: Permission denied
             RAPL needs access set up beforehand, and Vernier does not raise its privileges: run as root, or give the process CAP_SYS_RAWIO (the msr driver checks it) and read access to /dev/cpu/0/msr (the capability does not grant it).
...
  Selected request: --profile rapl
  [FAIL] rapl       denied: cannot open /dev/cpu/0/msr: Permission denied
             RAPL needs access set up beforehand, and Vernier does not raise its privileges: run as root, or give the process CAP_SYS_RAWIO (the msr driver checks it) and read access to /dev/cpu/0/msr (the capability does not grant it).
```

Then run it anyway:

```bash
bench run ./build/bin/ptests/BenchDemo_08_RaplProfiler --taskset 10 \
  --profile rapl --target-time 100ms --repeats 20
```

Captured output:

```
Running: taskset -c 10 ./build/bin/ptests/BenchDemo_08_RaplProfiler --target-time 100ms --repeats 20 --profile rapl
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from RaplProfiler
[ RUN      ] RaplProfiler.JoinV0

[FAIL] Profiler 'rapl': denied: cannot open /dev/cpu/0/msr: Permission denied
   RAPL needs access set up beforehand, and Vernier does not raise its privileges: run as root, or give the process CAP_SYS_RAWIO (the msr driver checks it) and read access to /dev/cpu/0/msr (the capability does not grant it).
   Nothing is collected for this request; the run will fail (exit status 4 if the tests pass).

[target-time] 100.000 ms -> cycles=636 (calibrated 157.0000 us/call, batch of 8)
[RaplProfiler.JoinV0]  127.331 us/call  CV=8.0%  ~7.9K calls/s  (p10=123.780 p90=137.766 sd=10.513)
[       OK ] RaplProfiler.JoinV0 (1672 ms)
[ RUN      ] RaplProfiler.JoinV1
[target-time] 100.000 ms -> cycles=22614 (calibrated 4.4219 us/call, batch of 256)
[RaplProfiler.JoinV1]  4.038 us/call  CV=2.4%  ~247.7K calls/s  (p10=3.965 p90=4.226 sd=0.096)
[       OK ] RaplProfiler.JoinV1 (1842 ms)
[----------] 2 tests from RaplProfiler (3515 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (3515 ms total)
[  PASSED  ] 2 tests.

[profile] --profile rapl failed; the run exits with status 4:
[profile]   rapl: denied: cannot open /dev/cpu/0/msr: Permission denied
...
Error: the requested profile failed (the benchmark's report above says why); the benchmark exited with status 4
```

The doctor and the run give the same answer, because they make the same
decision: the backend opened `/dev/cpu/0/msr` and the device's mode refused
this user. The tests still ran and passed, timed as in step 1, but nothing
was collected: the run wrote no `RaplProfiler.*.rapl` folder, and the
benchmark exited with status 4 (`bench run` itself exits 1, reporting it). A
run that measures without energy is not reported as one that measured it.

## Step 3: Measure the Energy

_Not run._ With access, as root here:

```bash
sudo "$(command -v bench)" run ./build/bin/ptests/BenchDemo_08_RaplProfiler \
  --taskset 10 --profile rapl --target-time 100ms --repeats 20 --csv rapl.csv
```

For each case the backend reads the counter before the first repeat and after
the last, prints a summary (`=== Energy Measurement (RAPL, package 0) ===`:
the window's number, its energy in joules, average power in watts, energy per
call in millijoules, its duration, and the report's path), and writes
`RaplProfiler.JoinV0.rapl/energy.txt` and `RaplProfiler.JoinV1.rapl/energy.txt`
in the working directory. Run with `sudo`, the folders and the CSV belong to
root. The window spans the measured repeats and the harness's summary and
row for the case, which end before the second reading.

## Step 4: Read the Report

_Not run._ The report of a case holds a header, then one numbered section per
measured window, in the order of the case's CSV rows:

| Field                                           | What it is                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| `Scope`                                         | package 0, read through `/dev/cpu/0/msr` (MSR 0x611); nothing off it     |
| `Counter`                                       | 32 bits, the energy at which it wraps at this unit                       |
| `Energy per operation`                          | (in the header) its definition: average package power x median call time |
| `Energy consumed`                               | the window's counter delta in units, in joules                           |
| `Duration`                                      | the window's wall-clock duration, in seconds                             |
| `Average power`                                 | energy over duration, in watts                                           |
| `Energy per operation`                          | average power x `Median time per call`, in millijoules a call            |
| `Energy unit`                                   | one count, in microjoules (from MSR 0x606)                               |
| `Start counter`, `End counter`, `Delta counter` | the counter's low 32 bits at each end, and their difference              |

A window over which the counter did not advance says so in its section. Read
energy per call against time per call: JoinV1's should be a fraction of
JoinV0's, by about the time ratio unless the package's power differs much
between the two, and each version's average power says whether it does.

## What Should Reproduce

| Reading                               | On this laptop                              | Elsewhere                                                      |
| ------------------------------------- | ------------------------------------------- | -------------------------------------------------------------- |
| JoinV0 / JoinV1 time per call         | 30x (122.7 / 4.0 us), step 1                | same direction; size depends on CPU and allocator              |
| the refusal without access            | `denied`, exit status 4, no folder (step 2) | the cause of that machine: denied, missing, unsupported        |
| JoinV1's energy per call against V0's | _Not run_                                   | _Not run_                                                      |
| joules and watts                      | _Not run_                                   | depend on the machine, its state and other load on the package |

## If It Does Not Match

- **`denied`**, as in step 2: the process may not read the device. Run as
  root, or give it CAP_SYS_RAWIO and read permission on the device; see
  [troubleshooting](../../docs/TROUBLESHOOTING.md#rapl-energy-measurement-fails).
- **`missing`**: the `msr` driver is not loaded (`sudo modprobe msr`), or a
  container was started without the device (`--device /dev/cpu/0/msr`). The
  notice reads
  `missing: cannot open /dev/cpu/0/msr: No such file or directory; the msr driver provides it`.
- **`unsupported`**: an ARM board, such as the reference rigs, where the
  backend prints
  `unsupported: rapl reads Intel's RAPL energy registers, and /proc/cpuinfo names no CPU vendor here`
  (its unit test's ARM case; not yet captured on a rig); an AMD CPU names its
  vendor instead. A CPU that does not implement the register is
  `unsupported` too, naming the read that failed.
- **A counter that did not advance** over a window: the report says so;
  that window measured no energy.
- **Numbers far from another run's:** other work on the package is in the
  reading. Measure on a quiet machine, and compare runs of one session.

## Validate the Energy Steps

What this page lacks is one session with access on this laptop: step 3's
command run five times on a quiet machine, the report of each read as step 4
says, and the check below run with that access. Until then the steps that
need it stay _Not run_.

## What Keeps This Page True

- `TestDemoRapl`, a test program beside the demo
  ([`08_RaplProfiler_uTest.cpp`](../cpu/utst/08_RaplProfiler_uTest.cpp)):
  `RaplDemoTest.MeasuresBothVersions` runs the demo with no profiler and
  fails unless its two tests are the run's only ones, pass, and publish a row
  each. `Rapl.EnergyPerCall` runs it as step 3 does, with windows of a third
  of a second: where RAPL cannot measure it fails unless the run refuses as
  step 2 shows (exit status 4, the notice once, no folder), then skips
  quoting the notice; where RAPL can, it fails unless each case's report
  holds one window over which the counter advanced, its numbers agree with
  one another, and JoinV1's energy per call is at most a third of JoinV0's.
  `RaplReportTest` holds its reading of notices and reports to text from
  real runs and from the backend.
- Vernier's own tests hold the backend: `RaplCheckTest` and
  `RaplAccessRefusalTest` its decision, cause by cause, on files that stand
  in for `/proc/cpuinfo` and the device; `RaplRunTest` the refusal through
  the registry, the constructor and the factory; `RaplCaptureTest` the
  report, against a device file rewritten between the readings.

All of them are registered with `ctest`, the demo's check under the `demo`
label and its profiled test also under `rapl`, alone:

```bash
ctest --test-dir build -L demo
ctest --test-dir build -L rapl
```

On this laptop, as its ordinary user, `Rapl.EnergyPerCall` skips on the
`denied` refusal: the effect is checked only where the energy counter can be
read, which no automatic run here can.

## See Also

- [CPU guide: energy measurement with RAPL](../../docs/CPU_GUIDE.md#energy-measurement-with-rapl)
- [Advanced guide: using RAPL](../../docs/ADVANCED_GUIDE.md#using-rapl)
- [Troubleshooting: RAPL energy measurement fails](../../docs/TROUBLESHOOTING.md#rapl-energy-measurement-fails)
- [Walkthrough 01](01_BASIC_WORKFLOW.md): the `join` example and how to read the timing
