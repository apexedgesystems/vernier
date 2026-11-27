# Demo 08: Intel RAPL -- the Energy a Call Costs

**Reference rig:** none. RAPL needs an Intel CPU, and neither reference rig
has one ([Reference Rigs](../../docs/rigs/README.md)).
**Machine:** an x86-64 laptop with an Intel Core Ultra 7 165H, Ubuntu
22.04.5, Linux 6.8.0, `intel_pstate` with governor `powersave`, energy
preference `balance_performance` and power profile `balanced`, on AC power;
timing pinned to CPU 10, one of its performance cores. It is a laptop shared
with other work, not a reference: steps 1 and 2 with other work running, at a
load average of about 25; steps 3 and 4 with no build or test running, at
6.05 to 7.51 ([Validate the Energy Steps](#validate-the-energy-steps)).
**Build:** Release, on the laptop itself ([Prerequisites](#prerequisites)).
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))

**The energy steps run as root on this laptop's Intel Core Ultra 7 165H,
which is not a reference rig.** Over five runs of step 3's command, JoinV0's
estimated energy per call was 27.2 to 33.5 times JoinV1's, 32.6 times in the
run step 3 shows. JoinV1 does less work
([The Example](#the-example)), so its calls are shorter, and the ratio of the
estimates is the two versions' time ratio multiplied by the ratio of the
power the package drew over their windows. Five runs on one laptop show those
ratios, not what another machine gives: measure both versions' time per
call and power on yours. The joules and watts belong to this package, its
settings and the other work running on it. Steps 1 and 2 are captured as an
ordinary user: the time per call, and what a run without access does.
[Validate the Energy Steps](#validate-the-energy-steps) gives the measurements
that check steps 3 and 4. The page meets the walkthrough contract in
[the demo README](../README.md#7-reference-rigs-and-the-demo-contract) only
in part, and says so where it applies.

## Overview

A timer says how long a call takes. RAPL says how much energy the processor
package used while the calls ran, and from that and the median time per call
Vernier estimates the energy a call costs. This walkthrough measures the two
versions of the `join` example, which build the same string 30 times apart in
time on this laptop, under Vernier's `rapl` backend. Step 1 measures them;
step 2 shows what an ordinary user gets: a run that refuses, says why, and
fails; steps 3 and 4 measure the energy with access, and read the report.

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
  short-term limit (115 W here). A window's delta is its energy only when the
  window uses less than those 262,144 J (2^32 counts).
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
seconds a case (`--target-time 100ms --repeats 20`) holds some 1,700 of the
counter's updates: on this laptop the counter changed 429 times in
half a second, four times that in two seconds, a median 1.143 ms apart, close
to the once a millisecond Intel documents
([Validate the Energy Steps](#validate-the-energy-steps)).

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
spread of a few percent with other work running (a load average of about 25).
Each window is about two
seconds (903 and 24,450 calls a repeat, 20 repeats). That ratio is what the
estimates of energy per call in step 3 should follow, unless the package
draws very different power for the two.

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

With access, as root here:

```bash
sudo "$(command -v bench)" run ./build/bin/ptests/BenchDemo_08_RaplProfiler \
  --taskset 10 --profile rapl --target-time 100ms --repeats 20 --csv rapl.csv
```

Captured output, one of five runs of this command, listed first under
[Validate the Energy Steps](#validate-the-energy-steps):

```
Running: taskset -c 10 ./build/bin/ptests/BenchDemo_08_RaplProfiler --csv rapl.csv --target-time 100ms --repeats 20 --profile rapl
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from RaplProfiler
[ RUN      ] RaplProfiler.JoinV0
[target-time] 100.000 ms -> cycles=892 (calibrated 112.0000 us/call, batch of 16)
[RaplProfiler.JoinV0]  110.459 us/call  CV=0.3%  ~9.1K calls/s  (p10=110.235 p90=110.625 sd=0.301)

=== Energy Measurement (RAPL, package 0) ===
Window:              1
Energy consumed:     76.222 Joules
Average power:       38.67 Watts
Energy per op:       4.272 mJ/call (average power x median call time)
Duration:            1.971 seconds
Scope:               package 0, every core included; nothing outside it
Report:              ./RaplProfiler.JoinV0.rapl/energy.txt

[       OK ] RaplProfiler.JoinV0 (1985 ms)
[ RUN      ] RaplProfiler.JoinV1
[target-time] 100.000 ms -> cycles=26975 (calibrated 3.7070 us/call, batch of 512)
[RaplProfiler.JoinV1]  3.558 us/call  CV=0.3%  ~281.1K calls/s  (p10=3.546 p90=3.571 sd=0.011)

=== Energy Measurement (RAPL, package 0) ===
Window:              1
Energy consumed:     70.528 Joules
Average power:       36.73 Watts
Energy per op:       0.131 mJ/call (average power x median call time)
Duration:            1.920 seconds
Scope:               package 0, every core included; nothing outside it
Report:              ./RaplProfiler.JoinV1.rapl/energy.txt

[       OK ] RaplProfiler.JoinV1 (1924 ms)
[----------] 2 tests from RaplProfiler (3909 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (3909 ms total)
[  PASSED  ] 2 tests.

=====================================================================
Test                  Median (us)     CV%     Calls/s  Status
---------------------------------------------------------------------
RaplProfiler.JoinV0       110.459    0.3%        9.1K  OK
RaplProfiler.JoinV1         3.558    0.3%      281.1K  OK
---------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

For each case the backend read the counter before the first repeat and after
the last, printed its summary, and wrote `RaplProfiler.JoinV0.rapl/energy.txt`
and `RaplProfiler.JoinV1.rapl/energy.txt` in the working directory. Run with
`sudo`, the folders and the CSV belong to root. The window spans the measured
repeats and the harness's summary and row for the case, which end before the
second reading: JoinV0's window is 1.971 seconds of the test's 1985 ms.

The report's energy per call is an estimate for a call of median duration at
the window's average package power: that power times the median time per
call. JoinV0's is 4.272 mJ and JoinV1's 0.131 mJ, 32.6 times less
(4.272 / 0.131). The median time per call fell 31.0 times
(110.459 / 3.558); the rest is the package's power, 38.67 W over JoinV0's
window against 36.73 W over JoinV1's, 1.053 times as much
(31.0 x 1.053 = 32.6). Both windows drew more than the 32.176 and 29.315 W
the package drew with no test running, before and after the five runs
([Validate the Energy Steps](#validate-the-energy-steps)): the estimates
include that background draw, and neither is what the calls alone added. Nor
is an estimate an equal share of the window's energy: the calls' mean time
can differ from their median, and the window also holds work that is not a
measured call, so the window's energy divided by its calls can give another
figure, as it does in the fourth run listed under
[Validate the Energy Steps](#validate-the-energy-steps).

## Step 4: Read the Report

JoinV0's report from that run, `RaplProfiler.JoinV0.rapl/energy.txt`:

```
=== RAPL Energy Measurement ===
Test: RaplProfiler.JoinV0
CPU: Intel(R) Core(TM) Ultra 7 165H
Family: 6, Model: 170
Scope: package 0, read through /dev/cpu/0/msr (MSR 0x611): every core, cache and the integrated graphics of the package, whatever runs on them; nothing outside the package
Counter: 32 bits, wrapping after 262144.000 J at this unit; a window's delta is its energy only when the window uses less than that
Energy per operation: the window's average package power x its median time per call

--- Window 1 ---
Energy consumed: 76.222107 J
Duration: 1.970944 s
Average power: 38.672893 W
Energy per operation: 4.271751 mJ/call
Median time per call: 110.459 us
Energy unit: 61.0352 uJ
Start counter: 1358076129
End counter: 1359324952
Delta counter: 1248823
```

It holds a header, then one numbered section per measured window, in the
order of the case's CSV rows:

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

The section's numbers follow from one another: 1,248,823 counts of
61.0352 uJ are 76.222 J; over 1.970944 s that is 38.673 W; and 38.672893 W
for a median call of 110.459 us is 4.2718 mJ, the report's 4.271751 rounded.
A window over which the counter did not advance says so in its section. Read
the estimated energy per call against time per call: JoinV1's is a fraction
of JoinV0's, by about the time ratio unless the package's power differs much
between the two, and each version's average power says whether it does, as
step 3 shows.

## What Should Reproduce

| Reading                                   | On this laptop                                                                                                                      | Elsewhere                                                                                                              |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| JoinV0 / JoinV1 time per call             | 30x (122.7 / 4.0 us), step 1                                                                                                        | same direction; size depends on CPU and allocator                                                                      |
| the refusal without access                | `denied`, exit status 4, no folder (step 2)                                                                                         | the cause of that machine: denied, missing, unsupported                                                                |
| JoinV0 / JoinV1 estimated energy per call | 32.6x (4.272 / 0.131 mJ), step 3; 27.2x to 33.5x over five runs                                                                     | measure both versions' time and power there: the estimates' ratio is their time ratio times their windows' power ratio |
| joules and watts                          | JoinV0's estimate 4.272 to 4.979 mJ a call, JoinV1's 0.131 to 0.164, at 36.73 to 45.28 W; with no test running, 32.176 and 29.315 W | depend on the machine, its state and other load on the package                                                         |

The ranges are five runs of step 3's command, one after
another ([Validate the Energy Steps](#validate-the-energy-steps)). They
describe those runs, not a bound another run has to meet.

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
  (the line its unit test holds for an ARM CPU); an AMD CPU names its
  vendor instead. A CPU that does not implement the register is
  `unsupported` too, naming the read that failed.
- **A counter that did not advance** over a window: the report says so;
  that window measured no energy.
- **Numbers far from another run's:** other work on the package is in the
  reading. Measure on a quiet machine, and compare runs taken back to back
  with the same settings.

## Validate the Energy Steps

Steps 3 and 4 run with access, as root, on this laptop with no build or test
running; the one-minute load average read 7.51 before the runs and 6.05 after
them:

```
load average: 7.51 6.42 8.89 5/3902 570853
...
load average: 6.05 6.15 8.65 4/3916 596581
```

Read outside Vernier, over ten seconds with no test running, before and after
the runs: the energy register and the kernel's powercap counter side by side;
then the register's unit, and the times between the counter's updates:

```
window: 10.009 s
register 0x611: 1346110702 -> 1351387315: 5276613 counts, 322.058899 J, 32.176 W
powercap package-0 energy_uj: 82159866696 -> 82481924771: 322.058075 J, 32.176 W
register / powercap: 1.000003
...
register 0x606: energy-status unit 14
energy unit from the register: 61.035156250 uJ
powercap max_energy_range_uj: 262143328850
powercap range / 2^32: 61.034999986 uJ
register unit / powercap unit: 1.000002560
...
reads: 168365 in 0.500 s; the counter changed 429 times
ms between changes: min 0.696, median 1.143, max 1.851
...
window: 10.010 s
register 0x611: 1374668466 -> 1379476199: 4807733 counts, 293.440735 J, 29.315 W
powercap package-0 energy_uj: 83902889822 -> 84196329805: 293.439983 J, 29.315 W
register / powercap: 1.000003
```

The register and powercap agree on the energy and on the unit to within 3
parts in a million (1.000003 and 1.000002560). With no test running, the
package drew 32.176 W before the runs and 29.315 W after them; that draw is
in every window below.

Step 3's command ran five times in a row. Each run's two cases, in order (the
first run is step 3's; the harness's progress line, which it overwrites, left
out):

```
[RaplProfiler.JoinV0]  110.459 us/call  CV=0.3%  ~9.1K calls/s  (p10=110.235 p90=110.625 sd=0.301)
Average power:       38.67 Watts
Energy per op:       4.272 mJ/call (average power x median call time)
[RaplProfiler.JoinV1]  3.558 us/call  CV=0.3%  ~281.1K calls/s  (p10=3.546 p90=3.571 sd=0.011)
Average power:       36.73 Watts
Energy per op:       0.131 mJ/call (average power x median call time)

[RaplProfiler.JoinV0]  110.540 us/call  CV=0.2%  ~9.0K calls/s  (p10=110.350 p90=110.852 sd=0.169)
Average power:       38.88 Watts
Energy per op:       4.298 mJ/call (average power x median call time)
[RaplProfiler.JoinV1]  3.565 us/call  CV=6.7%  ~280.5K calls/s  (p10=3.542 p90=3.607 sd=0.242)
Average power:       40.57 Watts
Energy per op:       0.145 mJ/call (average power x median call time)

[RaplProfiler.JoinV0]  111.254 us/call  CV=0.3%  ~9.0K calls/s  (p10=111.041 p90=111.548 sd=0.313)
Average power:       40.02 Watts
Energy per op:       4.453 mJ/call (average power x median call time)
[RaplProfiler.JoinV1]  3.614 us/call  CV=2.5%  ~276.7K calls/s  (p10=3.547 p90=3.724 sd=0.091)
Average power:       45.28 Watts
Energy per op:       0.164 mJ/call (average power x median call time)

[RaplProfiler.JoinV0]  110.552 us/call  CV=15.2%  ~9.0K calls/s  (p10=110.025 p90=118.082 sd=17.640) [UNSTABLE]
Average power:       45.04 Watts
Energy per op:       4.979 mJ/call (average power x median call time)
[RaplProfiler.JoinV1]  3.614 us/call  CV=0.6%  ~276.7K calls/s  (p10=3.593 p90=3.641 sd=0.021)
Average power:       41.72 Watts
Energy per op:       0.151 mJ/call (average power x median call time)

[RaplProfiler.JoinV0]  110.372 us/call  CV=3.2%  ~9.1K calls/s  (p10=109.979 p90=111.691 sd=3.602)
Average power:       42.20 Watts
Energy per op:       4.658 mJ/call (average power x median call time)
[RaplProfiler.JoinV1]  3.581 us/call  CV=0.4%  ~279.3K calls/s  (p10=3.562 p90=3.598 sd=0.015)
Average power:       38.94 Watts
Energy per op:       0.139 mJ/call (average power x median call time)
```

The time per call barely moved: 110.372 to 111.254 us for JoinV0, 3.558 to
3.614 us for JoinV1. The package's power moved more, from 36.73 to 45.28 W,
and the estimates of energy per call with it, each the power times a median
that barely moved: 4.272 to 4.979 mJ for JoinV0, 0.131 to 0.164 mJ for
JoinV1. JoinV0's estimate over JoinV1's, run by run, is 32.6, 29.6, 27.2,
33.0 and 33.5 times (4.272 / 0.131, 4.298 / 0.145, 4.453 / 0.164,
4.979 / 0.151 and 4.658 / 0.139). Where the ratio fell furthest, in the third
run, JoinV1's window drew 45.28 W against JoinV0's 40.02 W while their times
kept a ratio of 30.8 (111.254 / 3.614): the package's power moved it, not the
version's work. The fourth run's JoinV0 window was unstable (CV 15.2%) and
drew the most, 45.04 W, at the same median time as the others. From its
report and CSV, which this page does not show, the window's energy divided by
its calls is 94.276 J over 18,000 calls (900 a repeat, 20 repeats), 5.238 mJ
a call: 5% more than the estimate's 4.979 mJ, as its mean call, 116.292 us,
was 5% longer than its median.

The same case over two window lengths, JoinV0 alone with 50 ms and 200 ms a
repeat (the progress line left out):

```
[target-time] 50.000 ms -> cycles=446 (calibrated 111.9375 us/call, batch of 16)
[RaplProfiler.JoinV0]  110.822 us/call  CV=0.3%  ~9.0K calls/s  (p10=110.689 p90=111.100 sd=0.302)
Energy consumed:     18.182 Joules
Average power:       36.76 Watts
Energy per op:       4.074 mJ/call (average power x median call time)
Duration:            0.495 seconds
...
[target-time] 200.000 ms -> cycles=1807 (calibrated 110.6250 us/call, batch of 16)
[RaplProfiler.JoinV0]  110.881 us/call  CV=5.0%  ~9.0K calls/s  (p10=110.769 p90=124.134 sd=5.740)
Energy consumed:     83.631 Joules
Average power:       40.41 Watts
Energy per op:       4.481 mJ/call (average power x median call time)
Duration:            2.069 seconds
```

The energy follows the window: 83.631 J against 18.182 J is 4.6 times as
much, for 2.069 s against 0.495 s, 4.18 times as long, the rest the power
(40.41 W against 36.76 W, 1.099 times as much). The estimated energy per
call, 4.481 and 4.074 mJ, differs by the same 10% as the power.

`TestDemoRapl`, run as root, passed its eight tests; its effect test put
JoinV0's estimate at 31.4 times JoinV1's:

```
[Rapl.EnergyPerCall]  JoinV0 4.279301 mJ/call, JoinV1 0.136410 mJ/call (31.4x)
...
[  PASSED  ] 8 tests.
```

With the demo's JoinV1 calling `joinV0`, the effect test fails its assertion
that JoinV1's estimate is at most a third of JoinV0's; with the demo as
shipped, it passes:

```
[Rapl.EnergyPerCall]  JoinV0 5.048026 mJ/call, JoinV1 5.517446 mJ/call (0.9x)
...
Expected: (3.0 * V1) <= (V0), actual: 16.552337999999999 vs 5.0480260000000001
JoinV1's energy per call is not at most a third of JoinV0's
...
[  FAILED  ] 1 test, listed below:
...
[Rapl.EnergyPerCall]  JoinV0 4.265530 mJ/call, JoinV1 0.141699 mJ/call (30.1x)
...
[  PASSED  ] 1 test.
```

The performance sample `Profilers.RAPLIntegration` measured with access, and,
run as the ordinary user, was refused and exited with status 4:

```
[Profilers.RAPLIntegration]  0.455 us/call  CV=24.8%  ~2.2M calls/s  (p10=0.452 p90=0.761 sd=0.138) [UNSTABLE]
...
Energy consumed:     1.865 Joules
Average power:       33.58 Watts
Energy per op:       0.015 mJ/call (average power x median call time)
...
[  PASSED  ] 1 test.
...
[FAIL] Profiler 'rapl': denied: cannot open /dev/cpu/0/msr: Permission denied
[Profilers.RAPLIntegration]  0.482 us/call  CV=0.2%  ~2.1M calls/s  (p10=0.481 p90=0.483 sd=0.001)
[  PASSED  ] 1 test.
[profile] --profile rapl failed; the run exits with status 4:
```

A copy of the demo binary with the file capability `cap_sys_rawio=ep`, run as
the ordinary user with the device at its default mode, loads its libraries and
lists its tests, and its read of the counter is refused:

```
./build/bin/ptests/BenchDemo_08_RaplProfiler.setcap-check cap_sys_rawio=ep
...
RaplProfiler.
  JoinV0
  JoinV1
...
[FAIL] Profiler 'rapl': denied: cannot open /dev/cpu/0/msr: Permission denied
```

So a file capability alone does not let the user open the device, whose
mode still refuses it, as the access requirement in
[What is RAPL?](#what-is-rapl) says. The other half, that the driver checks
the capability, rests on the driver's code: no run here gives an ordinary
user read permission without the capability.

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
  one another, and JoinV1's estimated energy per call is at most a third of
  JoinV0's. `RaplReportTest` holds its reading of notices and reports to text
  from real runs and from the backend.
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
read: as root, or with the access [What is RAPL?](#what-is-rapl) describes.

## See Also

- [CPU guide: energy measurement with RAPL](../../docs/CPU_GUIDE.md#energy-measurement-with-rapl)
- [Advanced guide: using RAPL](../../docs/ADVANCED_GUIDE.md#using-rapl)
- [Troubleshooting: RAPL energy measurement fails](../../docs/TROUBLESHOOTING.md#rapl-energy-measurement-fails)
- [Walkthrough 01](01_BASIC_WORKFLOW.md): the `join` example and how to read the timing
