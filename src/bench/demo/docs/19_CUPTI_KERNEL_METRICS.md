# Demo 19: GPU Kernel Metrics

**Reference rig:** [NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
**Build:** Release
**Example:** `saxpy`, through demo 02's two kernel tests (see [Shared Examples](../README.md#shared-examples))
**Captured:** 2026-10-03 (UTC). Written for the Vernier 1.0.4 release;
captured from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`. The run in Check Against the Reference was made later
the same day, by this page's commands as written. Step 4's console output and
kernel summaries come from a session of 2026-10-10 (UTC) on a later
development tree at the same version, with Step 1's command run just before
them and the clocks locked the same way.

## Overview

A GPU kernel test writes one CSV row, and its GPU columns come from four
places: CUDA events around the launches, CUPTI's record of each launch, an
estimate the harness makes from the launch shape, and NVML's readings of the
device. They cover different things, and each can be missing for its own
reason. This walkthrough reads the rows of the SAXPY kernel at one and at 256
threads per block, column by column: where each value comes from, what it
covers, and what an empty cell means. On this rig NVML answers none of the
readings the harness asks for, so its cells are empty, and the run says so.

## What a GPU Kernel Test Records

A kernel test hands the harness a launch, and measures it with
`perf.cudaKernel(launch).withLaunchConfig(grid, block).measure()`:

- **CUDA events.** In each of `--repeats` repeats the harness records an
  event on the test's stream, makes `--cycles` launches, and records a second
  event. The time between the two, divided by the cycle count, is one sample
  of the time per launch.
- **CUPTI**, the profiling interface NVIDIA's tools use, records every kernel
  the GPU runs while the measured repeats run, with the registers and shared
  memory the launch was given.
- **NVML**, the driver's management library, is read once just before the
  measured repeats and once just after them: clocks, power and temperature.
- **The occupancy estimate**, computed by the harness from the launch
  configuration that `withLaunchConfig` declares and the device's limits.

The test's warmup runs before all of this.

**In Vernier:** no flag. Every kernel test of a GPU build does this, and
`--csv` writes the row. A cell its source cannot fill (a build without
CUPTI, CUPTI standing down for an Nsight session, NVML that does not answer)
is empty, never 0, and the run says so on stderr: once per process for what
holds for the whole run, and naming the test when it concerns one
measurement.

**Needs:** a CUDA device, the toolkit on `PATH`, and a GPU build: see the
rig's [setup](../../docs/rigs/RIG_THOR_AGX.md#2-one-time-setup) and
[build](../../docs/rigs/RIG_THOR_AGX.md#3-build). CUPTI and NVML are optional;
`-DVERNIER_USE_CUPTI=OFF` and `-DVERNIER_USE_NVML=OFF` build without them.

## The Example

Demo 02 ([`02_NsightProfiler_Demo.cu`](../gpu/02_NsightProfiler_Demo.cu)),
walkthrough 11's demo, times the bare SAXPY kernel in the launch shapes of the
example's two GPU versions: `NsightProfiler.KernelOneThreadPerBlock`, 1,048,576
blocks of one thread, and `NsightProfiler.Kernel256ThreadsPerBlock`, 4,096
blocks of 256, both over 1,048,576 floats already on the device. Both tests go
through one function:

```cpp
ub::PerfGpuResult measureKernelShape(ub::PerfGpuCase& perf, const DeviceVectors& device,
                                     int threadsPerBlock, const char* label) {
  const auto LAUNCH = [&device, threadsPerBlock](cudaStream_t s) {
    ubd::launchSaxpy(A, device.x(), device.y(), N, threadsPerBlock, s);
  };
  perf.cudaWarmup(LAUNCH);
  return perf.cudaKernel(LAUNCH, label)
      .withLaunchConfig(gridFor(threadsPerBlock), dim3(threadsPerBlock))
      .measure();
}
```

and the kernel is the example's, one element per thread
([`SaxpyKernel.cu`](../examples/saxpy/src/SaxpyKernel.cu)):

```cpp
__global__ void saxpyKernel(float a, const float* x, float* y, std::size_t n) {
  const std::size_t I = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (I < n) {
    y[I] = a * x[I] + y[I];
  }
}
```

The tests declare no transfer, so each row is the kernel alone. Walkthrough 11
profiles these two shapes; this one reads what their CSV rows say.

## Step 1: Measure

Run the two kernel tests inside the rig document's
[measurement procedure](../../docs/rigs/RIG_THOR_AGX.md#4-running-a-measurement):
the clocks locked with `jetson_clocks` and restored afterwards, the host thread
on core 13. From the source tree, after the rig's build:

```bash
taskset -c 13 ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --gtest_filter='NsightProfiler.Kernel*' --cycles 20 --repeats 10 --csv kernel_metrics.csv
```

The CSV lands in the working directory, under the name given; the demo writes
nothing else. Captured output (the reference run):

```
Note: Google Test filter = NsightProfiler.Kernel*
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from NsightProfiler
[ RUN      ] NsightProfiler.KernelOneThreadPerBlock
[gpu] NVML reported no SM clock, maximum SM clock, power draw, power limit or GPU temperature (Not Supported): smClockMHz, throttling, powerDrawW, powerLimitW, temperatureC and temperatureDeltaC stay empty.
[NsightProfiler.KernelOneThreadPerBlock]  2998.613 us/call  CV=0.0%  ~333 calls/s  (p10=2998.119 p90=2999.197 sd=0.405)
[       OK ] NsightProfiler.KernelOneThreadPerBlock (813 ms)
[ RUN      ] NsightProfiler.Kernel256ThreadsPerBlock
[NsightProfiler.Kernel256ThreadsPerBlock]  20.109 us/call  CV=0.5%  ~49.7K calls/s  (p10=20.073 p90=20.359 sd=0.110)
[       OK ] NsightProfiler.Kernel256ThreadsPerBlock (25 ms)
[----------] 2 tests from NsightProfiler (839 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (839 ms total)
[  PASSED  ] 2 tests.

=========================================================================================
Test                                      Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------------------------------
NsightProfiler.KernelOneThreadPerBlock       2998.613    0.0%         333  OK
NsightProfiler.Kernel256ThreadsPerBlock        20.109    0.5%       49.7K  OK
-----------------------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

What to read: the `[gpu]` line first. It holds for the whole run, so it is
printed once, at the first measurement, and it names the six NVML cells that
stay empty in both rows and why: NVML on this rig answers `Not Supported` to
each reading. Then the medians: 2,998.6 us per launch at one thread per block,
20.11 us at 256, 149.1 times as long.

## Step 2: Read the GPU Columns

The GPU columns of both rows, by name, one column per line (`-` is an empty
cell):

```bash
awk -F, -v cols=test,wallMedian,kernelTimeUs,transferTimeUs,h2dBytes,d2hBytes,occupancy,cuptiKernelLaunches,cuptiRegistersMedian,cuptiRegistersMax,cuptiStaticSmemBytes,cuptiDynamicSmemBytes,smClockMHz,throttling,powerDrawW,powerLimitW,temperatureC,temperatureDeltaC,speedupVsCpu,memBandwidthGBs '
  NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
  { for (i = 1; i <= NF; i++) cell[NR, i] = ($i == "" ? "-" : $i) }
  END { n = split(cols, name, ",")
        for (k = 1; k <= n; k++) { line = name[k]; for (r = 2; r <= NR; r++) line = line " " cell[r, col[name[k]]]; print line } }' \
  kernel_metrics.csv | column -t
```

```
test                   NsightProfiler.KernelOneThreadPerBlock  NsightProfiler.Kernel256ThreadsPerBlock
wallMedian             2998.61                                 20.1088
kernelTimeUs           2998.612785                             20.108800
transferTimeUs         0.000000                                0.000000
h2dBytes               0                                       0
d2hBytes               0                                       0
occupancy              0.500000                                1.000000
cuptiKernelLaunches    200                                     200
cuptiRegistersMedian   16                                      16
cuptiRegistersMax      16                                      16
cuptiStaticSmemBytes   0                                       0
cuptiDynamicSmemBytes  0                                       0
smClockMHz             -                                       -
throttling             -                                       -
powerDrawW             -                                       -
powerLimitW            -                                       -
temperatureC           -                                       -
temperatureDeltaC      -                                       -
speedupVsCpu           -                                       -
memBandwidthGBs        -                                       -
```

The header row names the rest: the device and its compute capability, and
the multi-GPU and unified-memory columns, which a single-GPU test without
unified memory leaves empty except `deviceCount`, 1.

## Where Each Column Comes From

| Column                                          | Source                                                                                                                                            | What it covers                                                                                             | Here                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `kernelTimeUs`                                  | CUDA events before the first and after the last of `--cycles` launches, divided by the cycle count; the median of the `--repeats` samples         | the stream from the first launch to the end of the last, per launch: the kernels and any time between them | 2,998.6 and 20.11 us                                       |
| `wallMedian`                                    | the same samples, plus each repeat's copy legs                                                                                                    | one call: the copies in, one launch, the copies out                                                        | the kernel time, to the printed digits: no copies declared |
| `transferTimeUs`, `h2dBytes`, `d2hBytes`        | event pairs around the copies the test declares, and their bytes                                                                                  | declared copies only                                                                                       | 0: none declared, so none timed; a measured zero           |
| `occupancy`                                     | the harness's estimate (below)                                                                                                                    | the launch configuration and the device's limits, not the run                                              | 0.5 and 1.0                                                |
| `cuptiKernelLaunches`                           | CUPTI's records of the kernels the GPU ran while the measured repeats ran                                                                         | the measured launches; not the warmup                                                                      | 200: 20 cycles times 10 repeats                            |
| `cuptiRegistersMedian`, `cuptiRegistersMax`     | the records' registers per thread                                                                                                                 | what each thread was allocated at launch                                                                   | 16; the compiler's count is 10 (Step 3)                    |
| `cuptiStaticSmemBytes`, `cuptiDynamicSmemBytes` | the records' static and dynamic shared memory per block                                                                                           | the kernel's `__shared__` arrays, and what the launch asked for                                            | 0: the kernel declares none and the launch asks for none   |
| `smClockMHz`, `throttling`                      | NVML: the SM clock at the end of the measured repeats; whether it fell more than 10% from start to end (only when the maximum clock was read too) | two instants, not the window between them                                                                  | empty: `Not Supported`                                     |
| `powerDrawW`, `powerLimitW`                     | NVML: the mean of the power draw read at the start and at the end; the power limit                                                                | two instantaneous readings, not the energy the measurement used                                            | empty: `Not Supported`                                     |
| `temperatureC`, `temperatureDeltaC`             | NVML: the GPU temperature at the end, and end minus start                                                                                         | two instants                                                                                               | empty: `Not Supported`                                     |
| `speedupVsCpu`                                  | a CPU baseline the test or its suite measured, over `wallMedian` ([walkthrough 10](10_GPU_BASIC_WORKFLOW.md))                                     | tests with a baseline                                                                                      | empty: these tests have none                               |
| `memBandwidthGBs`                               | the declared copies' bytes over their time                                                                                                        | declared copies only                                                                                       | empty: none declared                                       |

**An empty cell is a statement, not a zero.** A source that has nothing for a
cell leaves it empty. Where something stops a source (a build without it, a
device that does not answer, a test without a launch configuration), the run
names the cells in a `[gpu] ... stay empty.` line: once per process for what
holds for the whole run, as Step 1's NVML line does, and with the test's name
for one measurement. Cells a test has nothing to measure for (a speedup
without a baseline, a bandwidth without a transfer) are empty without a line.
A 0, as in `transferTimeUs`, is a measurement.

**`kernelTimeUs` is measured.** The events time the test's stream, so the
cell includes whatever the stream does between one kernel and the next; for
a 3 ms kernel that is nothing to speak of, and Step 4 shows it for a 22 us
one. It is the time per launch when launches run back to back, which is how
the test runs them.

**`occupancy` is not measured.** The harness divides the warps that blocks of
this shape can keep resident on an SM by the warps an SM can hold. An SM of
this GPU holds 1,536 threads (48 warps) and at most 24 blocks. One thread per
block is one warp per block, and 24 blocks are 24 warps: 0.5. Blocks of 256
threads are 8 warps each, and 6 of them fill the 1,536 threads: 48 warps, 1.0.
The estimate counts the thread and block limits and the dynamic shared memory
the launch declares, not registers or the kernel's static shared memory, and
it knows nothing of how the kernel then runs. Nsight Compute measures the
occupancy a kernel achieves; walkthrough 11 reads it for these two shapes
([Step 4 there](11_NSIGHT_PROFILER.md#step-4-what-limits-the-kernel-nsight-compute)),
below the estimate in both. In code the estimate is
`OccupancyMetrics::achievedOccupancy`; despite the name, it is this estimate.

**The `cupti*` cells are records of the measured launches.** CUPTI records
every kernel the GPU ran while the measured repeats ran: 200 launches here,
the warmup excluded. The registers are what each thread was given at launch:
16 per thread, where the kernel compiles to 10 (Step 3). Turning the
collector off changes the times by little here: with it off (the override and
the build without CUPTI in [If It Does Not Match](#if-it-does-not-match)), the
one-thread kernel read 2,994.4 and 2,993.6 us, against 2,996.6 to 3,000.0 us
with it on over six locked runs: the reference session's five and the run in
[Check Against the Reference](#check-against-the-reference).

**The NVML cells are two instants.** NVML is read once at each end of the
measured repeats: the power cell is the mean of two instantaneous readings, not
an energy trace, and the clock and temperature are the end's readings. On this
rig NVML finds the device, which it looks up by the CUDA device's UUID, and
answers `Not Supported` to every reading, so all six cells are empty. On a GPU
whose NVML answers, they are filled. One run of Step 1's command on an NVIDIA
RTX 5000 Ada laptop GPU (driver 580.178.04, in the project's CUDA development
container; none of its times are used on this page) read `smClockMHz` 1680,
`powerDrawW` 28.2 and 29.9 W, `temperatureC` 60, `temperatureDeltaC` 0 and
`throttling` 0, and named `powerLimitW` empty: that GPU's NVML does not report
its power limit.

## Step 3: The Registers as Compiled

```bash
cuobjdump --dump-resource-usage ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler
```

Captured output, trimmed to the kernel:

```
...
Resource usage:
 Common:
  GLOBAL:0
 Function _ZN7vernier5bench4demo47_GLOBAL__N__d3424f2f_14_SaxpyKernel_cu_2e849e8111saxpyKernelEfPKfPfm:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:928 TEXTURE:0 SURFACE:0 SAMPLER:0
```

The compiler gave the kernel 10 registers per thread and no shared memory
(`REG:10`, `SHARED:0`); the CSV says 16. `cuptiRegistersMedian` is what each
thread was allocated at launch, and a launch is given registers in larger
units than one: on this GPU and on the RTX 5000 Ada, the 10 compiled registers
were allocated as 16, the count rounded up to a multiple of 8. Nsight Compute,
in walkthrough 11's capture on this rig, reports `Registers Per Thread` 16.00
for both shapes. Read the column as the kernel's footprint on the register
file, and the compiler's report for the count the code needs.

## Step 4: Against Nsight Systems' Kernel Durations

The events time the stream; Nsight Systems records each kernel's own start
and end. With the clocks still locked, run each shape under it:

```bash
source build/.env
bench run ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile nsight -- \
  --gtest_filter=NsightProfiler.KernelOneThreadPerBlock --cycles 20 --repeats 3
```

```
Running: nsys profile -o bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile -t cuda,nvtx --force-overwrite true ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile nsight --gtest_filter=NsightProfiler.KernelOneThreadPerBlock --cycles 20 --repeats 3
...
[ RUN      ] NsightProfiler.KernelOneThreadPerBlock

[WARN] Profiler 'nsight': unverified: nsys started this process and writes its report when the process exits; whether it captured the benchmark's GPU work is not checked from inside the process

[nsight] this process runs under nsys, which writes the report when the process exits.
[gpu] in-process CUPTI collection disabled for this run (external Nsight session or VERNIER_DISABLE_CUPTI); CUPTI CSV columns will be empty.
[gpu] NVML reported no SM clock, maximum SM clock, power draw, power limit or GPU temperature (Not Supported): smClockMHz, throttling, powerDrawW, powerLimitW, temperatureC and temperatureDeltaC stay empty.
[NsightProfiler.KernelOneThreadPerBlock]  2888.810 us/call  CV=0.0%  ~346 calls/s  (p10=2888.418 p90=2890.900 sd=1.362)
[       OK ] NsightProfiler.KernelOneThreadPerBlock (370 ms)
...
[bench] nsight wrote bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.nsys-rep (138941 bytes)
[nsight] wrote the nsys stats summaries into bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight
```

Under `nsys` the in-process CUPTI collector stands down, and says so:
registered beside `nsys`, it would keep `nsys` from recording the kernels. The
kernels as the GPU ran them,
`bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/cuda_gpu_kern_sum.txt`:

```
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                             Name
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ------------------------------------------------------------------------------------------
    100.0      205,428,256         71  2,893,355.7  2,888,320.0  2,851,552  3,271,008     46,870.7  vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *, unsigned long)
```

The same command with `--gtest_filter=NsightProfiler.Kernel256ThreadsPerBlock`
overwrites the report and its summaries with the other shape's:

```
[NsightProfiler.Kernel256ThreadsPerBlock]  24.944 us/call  CV=0.3%  ~40.1K calls/s  (p10=24.847 p90=24.998 sd=0.078)
```

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                             Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ------------------------------------------------------------------------------------------
    100.0        1,619,328         71  22,807.4  22,400.0    22,336    40,256      2,291.7  vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *, unsigned long)
```

Each test ran 71 kernels: 11 warmup launches and the 60 measured (20 cycles
times 3 repeats). Within each run:

| Shape                 | events, per launch | Nsight Systems, median kernel | difference |
| --------------------- | ------------------ | ----------------------------- | ---------- |
| one thread per block  | 2,888.8 us         | 2,888.3 us                    | 0.02%      |
| 256 threads per block | 24.94 us           | 22.40 us                      | 2.5 us     |

For the 3 ms kernel the two agree. For the 22 us kernel the events also hold
the time on the stream between one kernel and the next, which a kernel's own
duration leaves out: 2.5 us a launch here.

Both runs also differ from Step 1's command, run just before them in the same
session without a profiler: 24.94 against 20.18 us at 256 threads, 2,889
against 2,993 us at one. A run under a profiler is a run of its own, so set
`kernelTimeUs` against the kernel durations of the same run.

## What Should Reproduce

| Reading                                    | On this rig                                                                                  | Elsewhere                                                                                    |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| one thread vs 256 per block, per launch    | 149.1x; 146.9x to 149.4x over five runs, clocks locked; 147.3x to 149.3x over three as found | one thread per block stays far slower; the size depends on the GPU                           |
| `cuptiKernelLaunches`                      | 200 in both rows (`--cycles` times `--repeats`)                                              | the same wherever CUPTI collects                                                             |
| `cuptiRegistersMedian` / compiled          | 16 / 10                                                                                      | the compiled count depends on the compiler and architecture; 16 / 10 on the RTX 5000 Ada too |
| `occupancy`                                | 0.5 and 1.0                                                                                  | depends on the SM's limits; 0.5 and 1.0 on the RTX 5000 Ada too                              |
| NVML cells                                 | all six empty, `Not Supported`                                                               | filled where NVML answers; all but `powerLimitW` on the RTX 5000 Ada                         |
| `kernelTimeUs` vs Nsight Systems, same run | 0.02 to 0.1% apart at one thread per block; 2.5 to 2.6 us a launch apart at 256 (three runs) | close for long kernels; the gap between launches shows on short ones                         |
| absolute times                             | 2,998.6 / 20.11 us                                                                           | will differ                                                                                  |

The non-timing cells were the same in all eight runs. The ranges describe
those runs; another run can land outside them. The 256-thread kernel is the
noisiest reading on the page: its within-run CV ran from 0.2% to 6.2% over the
locked runs and up to 7.2% as found, against 0.0% for the one-thread kernel.

## If It Does Not Match

- **The `cupti*` cells are empty.** The common causes each say so. Inside an
  Nsight session (Step 4), or with `VERNIER_DISABLE_CUPTI=1`, every
  measurement prints:

  ```
  [gpu] in-process CUPTI collection disabled for this run (external Nsight session or VERNIER_DISABLE_CUPTI); CUPTI CSV columns will be empty.
  ```

  A build without CUPTI (`-DVERNIER_USE_CUPTI=OFF`, or a toolkit without the
  library) prints, once:

  ```
  [gpu] this build has no CUPTI: cuptiKernelLaunches, cuptiRegistersMedian, cuptiRegistersMax, cuptiStaticSmemBytes and cuptiDynamicSmemBytes stay empty.
  ```

  The timing cells are measured either way: the override run read 2,994.4 and
  20.64 us, the build without CUPTI 2,993.6 and 20.59 us.

- **`occupancy` is empty.** A kernel measured without `withLaunchConfig` has
  no launch shape to estimate from; the run names it, as for this test of the
  harness's own:

  ```
  [gpu] GpuBandwidth45.Kernel declares no launch configuration (.withLaunchConfig(grid, block)), so its occupancy stays empty.
  ```

- **No CSV was written.** A `--csv` the run cannot write stops it before any
  test runs, with exit status 2:

  ```
  [csv] cannot write --csv 'results/kernel_metrics.csv': No such file or directory
  ```

  Create the directory, or give a path in an existing one.

## Check Against the Reference

```bash
bench compare src/bench/demo/reference/thor/19_cupti_kernel_metrics.csv kernel_metrics.csv
```

A later run on the same rig, clocks locked:

```
Test                                         Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
---------------------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
NsightProfiler.Kernel256ThreadsPerBlock      20.10880      20.20720    +0.09840     +0.5%      0.5%      5.7%  neutral
NsightProfiler.KernelOneThreadPerBlock     2998.61000    2996.55000    -2.06000     -0.1%      0.0%      0.0%  neutral

  2 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference CSV is one of five runs captured with the clocks locked: the one
whose two kernel times sit closest to the median of each test across the five.
It was captured from a copy of the development tree without its git metadata,
so its `gitHash` column reads `unknown`, and its `hostname` is what the board
reported, `kalex`.

## What Keeps This Page True

- `KernelColumns.MatchTheirSources`, a test program of its own beside the demo
  ([`02_NsightProfiler_KernelColumns_uTest.cpp`](../gpu/utst/02_NsightProfiler_KernelColumns_uTest.cpp),
  built as `TestDemoKernelColumns`), runs demo 02's two kernel tests with
  `--csv` and holds each row to this page: `kernelTimeUs` equal to
  `wallMedian`; the copy columns 0; `occupancy` equal to the harness's estimate
  for the row's block; one CUPTI record per measured launch, registers equal to
  the compiled count rounded up to a multiple of 8, the kernel's static shared
  memory and no dynamic, or, in a build without CUPTI, the five cells stated
  empty; every cell the run states empty empty, every NVML cell it does not
  state a reading, and `speedupVsCpu`, `memBandwidthGBs` and the multi-GPU and
  unified-memory cells empty. It compares no timing. It skips only without a
  CUDA device or inside an Nsight session, saying which, and a failing run
  keeps its files and names the folder.
- Demo 02's two kernel tests fail if the occupancy estimate stops matching
  their shapes, and the SAXPY example's unit tests, in `TestDemoExamples`,
  hold its versions to the same answers.

An ordinary test run includes the check, and every test it runs should pass:

```bash
ctest --test-dir build -L demo
```

The demo's timing tests are not registered: what they measure belongs to the
machine they run on. This repository has no continuous-integration lane on
the reference board, so before a release the page's commands are run on the
rig by hand, and the page and its reference CSV are re-captured when what
they show changes.

## See Also

- [Demo 10: GPU Basic Workflow](10_GPU_BASIC_WORKFLOW.md) -- the GPU harness,
  its timing columns and the CPU baseline
- [Demo 11: Nsight Systems and Nsight Compute](11_NSIGHT_PROFILER.md) -- the
  same two shapes under both tools, and the occupancy Nsight Compute measures
- [GPU Guide: understanding occupancy](../../docs/GPU_GUIDE.md#understanding-occupancy)
- [Reference Rig: NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
