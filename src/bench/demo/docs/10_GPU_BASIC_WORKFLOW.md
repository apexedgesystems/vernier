# Demo 10: GPU Basic Workflow

**Reference rig:** [NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
**Build:** Release
**Example:** `saxpy` (see [Shared Examples](../README.md#shared-examples))
**Captured:** 2026-09-23 (UTC). Written for the Vernier 1.0.4 release;
captured from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`.

## Overview

The first GPU measurement: a GPU function timed the way a CPU function is,
the same kernel timed on its own, and both compared with a CPU loop. On this
rig the kernel alone is about nine times faster than the loop; with its copies
in and out, the round trip is barely faster than the loop with the clocks
locked, and slower than it with the clocks left to the governor. That gap is
what this walkthrough teaches you to see.

## What the GPU Harness Measures

A GPU benchmark is a GoogleTest test that builds a `PerfGpuCase` with
`PERF_GPU_GUARD(perf)` and asks it for a CPU baseline, a kernel measurement,
or both:

- `perf.cpuBaseline(fn)` times `fn` like any CPU benchmark and records the
  median as the suite's baseline.
- `perf.cudaKernel(launch)` times the launches, and each
  `.withHostToDevice(...)` / `.withDeviceToHost(...)` adds a copy to time
  around them. Every repeat records one CUDA event pair around the
  host-to-device copies, one around `--cycles` launches, and one around the
  device-to-host copies. A leg with no copies declared is not timed.

**In Vernier:** the console line and the CSV row of each test. For a GPU test
the columns mean:

| Column                                         | Unit                                                      |
| ---------------------------------------------- | --------------------------------------------------------- |
| `kernelTimeUs`                                 | one launch (the launch events divided by the cycle count) |
| `transferTimeUs`                               | both copy legs of one repeat, host-to-device plus back    |
| `wallMedian`, the console `us/call`, `calls/s` | one round trip: one copy in, one launch, one copy out     |
| `speedupVsCpu`                                 | the suite's CPU baseline median divided by `wallMedian`   |

The CSV carries more GPU columns than these (the device, bytes moved,
bandwidth, occupancy, clocks, power and temperature, the in-process CUPTI
launch metrics, multi-GPU and unified-memory fields), and its header row names
them all. On this rig the clock, power and temperature cells are 0 or empty.

So the wall time of a test with transfers is what a call would cost if every
call copied its data in and out, and the wall time of a kernel-only test is its
kernel time. A GPU test is compared against the baseline its own test measured,
or against its suite's baseline while exactly one test of the suite has
recorded one; otherwise `speedupVsCpu` is an empty cell.

**Needs:** a CUDA device, the toolkit on `PATH`, and a GPU build: see the
rig's [setup](../../docs/rigs/RIG_THOR_AGX.md#2-one-time-setup) and
[build](../../docs/rigs/RIG_THOR_AGX.md#3-build).

## The Example

SAXPY, `y = a*x + y` over 1,048,576 floats: one multiply and one add per
element, every element independent, so nothing about the arithmetic can be
slow. The CPU reference
([`SaxpyCpu.cpp`](../examples/saxpy/src/SaxpyCpu.cpp)):

```cpp
void saxpyCpu(float a, const std::vector<float>& x, std::vector<float>& y) {
  const std::size_t N = x.size();
  const float* __restrict__ px = x.data();
  float* __restrict__ py = y.data();
  for (std::size_t i = 0; i < N; ++i) {
    py[i] = a * px[i] + py[i];
  }
}
```

and the kernel, one element per thread
([`SaxpyGpu.cu`](../examples/saxpy/src/SaxpyGpu.cu)):

```cpp
__global__ void saxpyKernel(float a, const float* x, float* y, std::size_t n) {
  const std::size_t I = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (I < n) {
    y[I] = a * x[I] + y[I];
  }
}
```

The example also carries two ways of driving that kernel end to end, `G0`
(allocate and copy on every call, one thread per block) and `G1` (buffers once,
256 threads per block); its unit test, `TestDemoExamples`, holds both to the CPU
loop within one part in a million. This walkthrough does not time them: the
demo gives the harness the bare launch and lets it time the copies.

The demo ([`01_GpuBasicWorkflow_Demo.cu`](../gpu/01_GpuBasicWorkflow_Demo.cu))
has three tests. `CpuBaseline` times the loop. `GpuWithTransfers` times the
launch with both vectors copied in and `y` copied out:

```cpp
  const ub::PerfGpuResult RESULT = perf.cudaKernel(LAUNCH, "saxpy_gpu_with_transfers")
                                       .withHostToDevice(X.data(), device.x(), BYTES)
                                       .withHostToDevice(y.data(), device.y(), BYTES)
                                       .withDeviceToHost(device.y(), y.data(), BYTES)
                                       .withLaunchConfig(GRID, BLOCK)
                                       .measure();
```

`GpuKernelOnly` copies the data to the device once, outside the measurement,
and times the launch with no copies declared.

Each test asserts the effect it demonstrates, with a bound well inside what
this rig measures: the copies of a round trip cost more than 2.5 times the
kernel (6.60 to 7.00 over sixteen runs with the clocks locked), and the kernel
alone beats the CPU loop by more than 3x (8.91x to 9.60x over the same runs).
Both bounds were also checked on a discrete GPU, where the copies cost 234 to
255 times the kernel and the kernel alone beat the loop roughly 19x to 22x
(Step 3). The speedup of the round trip with its copies is not asserted,
because its direction depends on the clocks and on the memory system (below).

## Step 1: Measure

Run the demo inside the rig document's
[measurement procedure](../../docs/rigs/RIG_THOR_AGX.md#4-running-a-measurement):
the clocks locked with `jetson_clocks` and restored afterwards, the host thread
on core 13. From the source tree, after the rig's build:

```bash
taskset -c 13 ./build/bin/ptests/BenchDemo_Gpu_01_GpuBasicWorkflow --repeats 10 --csv gpu_basic_workflow.csv
```

The CSV lands in the working directory, under the name given; the demo writes
nothing else. Captured output (the reference run; the baseline's progress
lines are trimmed):

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from GpuBasicWorkflow
[ RUN      ] GpuBasicWorkflow.CpuBaseline
...
[GpuBasicWorkflow.CpuBaseline]  186.248 us/call  CV=0.1%  ~5.4K calls/s  (p10=186.061 p90=186.417 sd=0.159)
[       OK ] GpuBasicWorkflow.CpuBaseline (20658 ms)
[ RUN      ] GpuBasicWorkflow.GpuWithTransfers
[GpuBasicWorkflow.GpuWithTransfers]  155.930 us/call  CV=6.3%  ~6.4K calls/s  (p10=152.809 p90=164.007 sd=9.950)
[       OK ] GpuBasicWorkflow.GpuWithTransfers (2038 ms)
[ RUN      ] GpuBasicWorkflow.GpuKernelOnly
[GpuBasicWorkflow.GpuKernelOnly]  20.277 us/call  CV=0.1%  ~49.3K calls/s  (p10=20.266 p90=20.304 sd=0.017)
[       OK ] GpuBasicWorkflow.GpuKernelOnly (2050 ms)
[----------] 3 tests from GpuBasicWorkflow (24747 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (24747 ms total)
[  PASSED  ] 3 tests.

===================================================================================
Test                                Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------------------------
GpuBasicWorkflow.CpuBaseline            186.248    0.1%        5.4K  OK
GpuBasicWorkflow.GpuWithTransfers       155.930    6.3%        6.4K  OK
GpuBasicWorkflow.GpuKernelOnly           20.277    0.1%       49.3K  OK
-----------------------------------------------------------------------------------
3 tests | 3 stable | 0 unstable
```

What to read: the median of each line, then its CV. The round trip with its
copies reads 155.9 us against 186.2 us for the loop; the kernel alone reads
20.3 us.

## Step 2: Read the GPU Columns

`bench summary` prints the timing columns of the CSV:

```bash
source build/.env
bench summary gpu_basic_workflow.csv
```

```
Test                                Median (us)       P10       P90        CV       Calls/sec  Stable
---------------------------------  ------------  --------  --------  --------  --------------  ------
GpuBasicWorkflow.CpuBaseline          186.24800  186.06100  186.41700      0.1%            5369  yes
GpuBasicWorkflow.GpuKernelOnly         20.27680  20.26620  20.30430      0.1%           49317  yes
GpuBasicWorkflow.GpuWithTransfers     155.93000  152.80900  164.00700      6.3%            6413  yes

  3 tests, sorted by name
```

It does not print the GPU columns. They are in the CSV, and can be read by
name:

```bash
awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; print "test kernelTimeUs transferTimeUs h2dBytes d2hBytes speedupVsCpu"; next }
         { print $col["test"], $col["kernelTimeUs"], $col["transferTimeUs"], $col["h2dBytes"], $col["d2hBytes"], $col["speedupVsCpu"] }' \
  gpu_basic_workflow.csv | column -t
```

```
test                               kernelTimeUs  transferTimeUs  h2dBytes  d2hBytes  speedupVsCpu
GpuBasicWorkflow.CpuBaseline
GpuBasicWorkflow.GpuWithTransfers  20.138642     136.160001      8388608   4194304   1.194436
GpuBasicWorkflow.GpuKernelOnly     20.276845     0.000000        0         0         9.185268
```

The baseline row has no GPU values; its GPU cells are empty, not zero.

## Step 3: Where the Time Goes

The round trip of `GpuWithTransfers` is its three legs: 136.16 us of copies
(8 MiB in, `x` and `y`; 4 MiB out, `y`) and 20.14 us of kernel, 156.30 us in
all, against a wall median of 155.93 us. The two agree without being equal,
because each is the median of its own ten samples. The copies are 87% of the
round trip: the kernel is cheap, and moving its data is not.

On this rig the CPU and the GPU share DRAM, so a "transfer" is a copy from one
place in the same memory to another, at about 92 GB/s here
(`memBandwidthGBs` in the CSV). **These transfer costs are specific to this
rig.** A discrete GPU moves the same bytes over PCIe. One pass of this demo
on an NVIDIA RTX 5000 Ada laptop GPU (compute capability 8.9), with the clocks
as found and the laptop busy with other work, so that only its GPU-side figures
are exact, read: the kernel about 8 us, the copies 1.87 to 2.06 ms, 234 to 255
times the kernel. The round trip took roughly 11 to 13 times as long as the CPU
loop there, while the kernel alone was roughly 19x to 22x faster than it.

## Step 4: The Kernel on Its Own

`GpuKernelOnly` declares no copies, so no transfer is timed: its
`transferTimeUs` is 0.000000 and its wall time is its kernel time, 20.28 us.
With the clocks locked, the reference run reads:

| Test               | `speedupVsCpu`, clocks locked |
| ------------------ | ----------------------------- |
| `GpuKernelOnly`    | 9.19x                         |
| `GpuWithTransfers` | 1.19x                         |

The kernel is nine times faster than the loop; a call that copies its data in
and out is not. With the clocks left to the governor the kernel alone still
reads 8.61x to 8.75x (six runs), but the round trip with its copies reads
0.82x to 0.98x: slower than the CPU loop. Every speedup on this page is stated
with its clock procedure for that reason. The round trip is also the noisiest
of the three measurements: its CV reached 6.3% in the locked runs on record,
the reference run's own included, so a single run can land a few percent
outside any range stated here.

## What Should Reproduce

| Reading                            | On this rig                                                                                              | Elsewhere                                                                                           |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| kernel alone vs the CPU loop       | 9.19x, clocks locked (8.91x to 9.60x over sixteen runs)                                                  | the kernel wins by a wide margin: roughly 19x to 22x on the discrete GPU in Step 3                  |
| round trip with copies vs the loop | 1.19x, clocks locked (1.13x to 1.23x over sixteen runs, wall 153.5 to 163.0 us); 0.82x to 0.98x as found | may go either way: on the discrete GPU in Step 3 it took roughly 11 to 13 times as long as the loop |
| copies / kernel in one round trip  | 6.76 (6.60 to 7.00, clocks locked)                                                                       | the copies dominate: 234 to 255 over PCIe on the discrete GPU in Step 3                             |
| absolute times                     | 186.2 / 155.9 / 20.3 us                                                                                  | will differ                                                                                         |

The demo fails if the copies stop costing 2.5 times the kernel or the kernel
stops beating the loop by 3x.

## If It Does Not Match

- **The clocks were not locked.** The GPU readings move and the round trip
  with copies can drop below the CPU loop. A run with the clocks as found,
  compared against the reference:

  ```
  Test                                   Baseline     Candidate       Delta         %   p-value        Result
  ---------------------------------  ------------  ------------  ----------  --------  --------  ------------
  GpuBasicWorkflow.CpuBaseline          186.24800     183.67100    -2.57700     -1.4%    0.0002  neutral
  GpuBasicWorkflow.GpuKernelOnly         20.27680      21.32940    +1.05260     +5.2%    0.0002  REGRESSION
  GpuBasicWorkflow.GpuWithTransfers     155.93000     188.04500   +32.11500    +20.6%    0.0002  REGRESSION

    2 regression(s)  1 neutral
  ```

  Lock them as the rig document says and run again. With the clocks locked, a
  round trip a few percent away from the reference is within what this rig
  produces (its CV reached 6.3% in the locked runs on record).

- **The speedup check was skipped.** `GpuKernelOnly` takes its speedup from
  `CpuBaseline`, which has to run first in the same process. A run filtered to
  one GPU test still measures it and writes its CSV row, then skips the
  comparison and says why. The run passes no test, lists this one as skipped,
  and exits 0:

  ```
  no CPU baseline in this run: run the whole suite for the speedup
  ...
  [  PASSED  ] 0 tests.
  [  SKIPPED ] 1 test, listed below:
  [  SKIPPED ] GpuBasicWorkflow.GpuKernelOnly
  ```

- **No device is visible.** With `CUDA_VISIBLE_DEVICES` empty, every test of
  the demo, the baseline included, fails before it measures anything, because
  the GPU harness needs a device to set up any of its tests:

  ```
  C++ exception with description "no CUDA-capable device is detected" thrown in the test body.
  ```

  `TestDemoExamples` instead runs its CPU-loop check and skips the GPU cases.

## Check Against the Reference

```bash
bench compare src/bench/demo/reference/thor/10_gpu_basic_workflow.csv gpu_basic_workflow.csv
```

A later run on the same rig, clocks locked:

```
Test                                   Baseline     Candidate       Delta         %   p-value        Result
---------------------------------  ------------  ------------  ----------  --------  --------  ------------
GpuBasicWorkflow.CpuBaseline          186.24800     186.07800    -0.17000     -0.1%    0.0757  neutral
GpuBasicWorkflow.GpuKernelOnly         20.27680      20.27640    -0.00040     -0.0%    0.5205  neutral
GpuBasicWorkflow.GpuWithTransfers     155.93000     159.18900    +3.25900     +2.1%    0.7337  neutral

  3 neutral
```

`bench compare` labels a test `REGRESSION` or `IMPROVEMENT` when its median
moved by more than the threshold (5% unless `--threshold` says otherwise) and
the p-value it prints is below 0.05. That p-value is computed from samples the
CLI reconstructs from each row's percentiles, not from the measured samples, so
read the label as a threshold label and nothing more.

The reference CSV is one of five runs captured with the clocks locked: the one
whose three medians sit closest to the median of each test across the five. It
was captured from a copy of the development tree without its git metadata, so
its `gitHash` column reads `unknown`.

## See Also

- [GPU Guide](../../docs/GPU_GUIDE.md) -- the GPU harness in full
- [Reference Rig: NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
- [Shared Examples](../README.md#shared-examples) -- the SAXPY example and its tests
