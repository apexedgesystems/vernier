# Demo 17: Compute Sanitizer

**Reference rig:** [NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
**Build:** Release
**Example:** `saxpy` (see [Shared Examples](../README.md#shared-examples)), plus a copy of its kernel without the bounds guard, private to the demo
**Captured:** 2026-09-30 (UTC), written for the Vernier 1.0.4 release;
captured from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`; compute-sanitizer 2025.3.1 (CUDA 13.0). The doctor's
row, the readiness report under the wrap, the output of the `bench run` steps
(2 and 4) and the plain run with `--profile compute-sanitizer` in
[If It Does Not Match](#if-it-does-not-match) are from a later session on
2026-10-10 (UTC), from a later development tree at the same version with the
same tool; the report in Step 3, the job commands and every other figure are
from the first.

## Overview

A kernel that runs one thread past the end of its buffers computes the right
answer on this rig, never faults, and passes every test of its result. This
walkthrough takes the SAXPY kernel from [walkthrough 10](10_GPU_BASIC_WORKFLOW.md),
adds a copy of it without its `if (I < n)`, and runs both under Compute
Sanitizer's memcheck. The tool names the read, the line it is on, the thread
and block that made it, and the allocation it ran past; the same run on the
guarded kernel reports nothing.

## What is Compute Sanitizer?

Compute Sanitizer ships with the CUDA toolkit. Its memcheck tool starts the
program, instruments its device code as each module loads, and checks every
global memory access a kernel makes against the allocations the tool knows
about. An access that misses is reported with the kernel and, when the code
carries line information, the source line; the thread and block that made
it; the address; and the nearest allocation with its size. By default the
tool then ends the CUDA context (`--destroy-on-device-error context`), so
the kernel stops at its first invalid access and every later CUDA call in
the process fails. At exit it prints an error summary. The tool has three
more tools, `racecheck` (shared-memory hazards), `synccheck` (barrier misuse)
and `initcheck` (reads of uninitialized device memory); this example gives
them nothing to find, so this page runs memcheck only.

- **Best for:** device code that reads or writes outside its buffers, a
  bug the hardware often lets through and a test of the result cannot see.
- **Overhead:** large. On this rig the kernel that takes about 20 us alone
  took 1149.5 us under memcheck, about 56 times as long
  (step 4). Run one case at a time under it, with few cycles, and never read
  its timings as measurements.
- **Not for:** time (Nsight, [walkthrough 11](11_NSIGHT_PROFILER.md)), host
  memory (memcheck, [walkthrough 15](15_MEMCHECK_PROFILER.md)), or a kernel
  that never touches memory it does not own.

**In Vernier:** `--profile compute-sanitizer` selects the compute-sanitizer
backend, which does not start the tool itself. `bench run <binary> --profile
compute-sanitizer` builds the wrap and prints it on its first line:
`compute-sanitizer --tool=memcheck --error-exitcode 5 --log-file <working directory>/<dir>/<binary>.compute-sanitizer/sanitizer.log <binary> --profile compute-sanitizer ...`,
where `<dir>` is `--profile-output-dir` (`bench-out` when you give none). The
log is named whole because the tool joins a relative name to its working
directory and reads a `%` there as a macro. The tool writes one report for
the whole process, so profile one case per run with `--gtest_filter`; no
per-test folder is created. The wrap runs the tool `--profile-args` names
(`memcheck`, `racecheck`, `synccheck` or `initcheck`; memcheck by default),
and when the report counts errors `bench run` fails the run, naming the
report (step 2). Under the wrap a case that measures prints, once per run,
the readiness report of its profiler request:

```
[WARN] Profiler 'compute-sanitizer': unverified: compute-sanitizer started this process and reports when it exits; which tool it runs, and what it finds, is not seen from inside the process
```

which says that the tool started the process and that its report, not the
benchmark, holds what it found; the backend then prints that the wrap was
detected and where the report goes. Run by hand instead, with
`--profile compute-sanitizer` on the binary's own command line and nothing
around it, the request fails: the tests run, the run ends with status 4, and
its report names the command that wraps the binary in the tool (see
[If It Does Not Match](#if-it-does-not-match)). No folder is created for it.

**Needs:** the CUDA toolkit on `PATH`, with `compute-sanitizer`, and a GPU
build: see the rig's [setup](../../docs/rigs/RIG_THOR_AGX.md#2-one-time-setup)
and [build](../../docs/rigs/RIG_THOR_AGX.md#3-build). No privileges: every
run on this page was made as the user, on a rig whose GPU performance
counters are restricted to administrators (the rig document's note on
Nsight Compute). `bench doctor` reports it as:

```
  [WARN] compute-sanitizer unverified: /usr/local/cuda/bin/compute-sanitizer runs here (Version 2025.3.1.0 (build 36400806) (public-release)); whether its memcheck checks the benchmark's kernels is not checked before the run
```

A run is unverified for the same reason: whether the tool checked the
kernels is in its report.

## The Example

The shared kernel, one element per thread
([`SaxpyKernel.cu`](../examples/saxpy/src/SaxpyKernel.cu)):

```cpp
__global__ void saxpyKernel(float a, const float* x, float* y, std::size_t n) {
  const std::size_t I = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (I < n) {
    y[I] = a * x[I] + y[I];
  }
}
```

A grid is whole blocks, so unless `n` is a multiple of the block size the
last block has threads with no element; the guard is what stops them. The
copy without it, private to the demo
([`04_ComputeSanitizerProfiler_Unguarded.cu`](../gpu/04_ComputeSanitizerProfiler_Unguarded.cu)):

```cpp
__global__ void saxpyUnguarded(float a, const float* x, float* y) {
  const std::size_t I = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  y[I] = a * x[I] + y[I]; // the shared kernel's `if (I < n)` is missing
}
```

It is not a version of the shared example. The example's versions are held
to the same answers by its unit tests and are safe to run anywhere; this one
is not, so it lives with the demo, compiled with device line information
whatever the build type (`-lineinfo`), so the tool's report can name its
line.

The demo ([`04_ComputeSanitizerProfiler_Demo.cu`](../gpu/04_ComputeSanitizerProfiler_Demo.cu))
runs both at 1,048,575 elements, one less than 4,096 blocks of 256 cover,
so the last block has exactly one thread past the end. `ComputeSanitizer.SaxpyKernel`
times the shared kernel on its own, as walkthrough 10's kernel-only case
does, and after the measurement holds the first and the last element to the
scalar applied once per launch, so a kernel that missed either one fails.
Those two values say nothing of memory past the end, where a stray access
can leave both right: whether the kernel stays inside its buffers is
memcheck's to check, as steps 2 to 4 show. It is the demo's one CSV row. The
unguarded copy has a case of its own that measures nothing:

```cpp
PERF_GPU_TEST(ComputeSanitizer, SaxpyUnguarded) {
  DEMO_SKIP_UNLESS_UNDER_COMPUTE_SANITIZER();

  DeviceVectors device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  ASSERT_TRUE(device.fill()) << "copy to the device failed";

  wrong::launchSaxpyUnguarded(A, device.x(), device.y(), N, BLOCK_SIZE, nullptr);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "the launch was refused";
  const cudaError_t END = cudaDeviceSynchronize();
  std::printf("[ComputeSanitizer.SaxpyUnguarded]  one launch over %zu elements, %u blocks of %d; "
              "the device reported: %s\n",
              N, static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE,
              cudaGetErrorString(END));
}
```

It launches the copy once, asserts the launch was accepted, and prints what
the device reported when the launch was waited for; what happens to the
kernel is the tool's doing, and what the tool reports is checked apart from
the demo ([What Keeps This Page True](#what-keeps-this-page-true)). Its
first line is the demo helper
[`SkipUnlessUnderComputeSanitizer.hpp`](../helpers/SkipUnlessUnderComputeSanitizer.hpp):
a case that runs past its buffers on purpose must not run in an ordinary
test run, so unless the process is under compute-sanitizer the case skips
itself before any CUDA call and says how to run it. The demo therefore runs
anywhere: with a GPU, without the tool, and, the skip apart, without a
device.

## Step 1: Measure

Run the demo inside the rig document's
[measurement procedure](../../docs/rigs/RIG_THOR_AGX.md#4-running-a-measurement):
the clocks locked with `jetson_clocks` and restored afterwards, the host
thread on core 13. From the source tree, after the rig's build:

```bash
taskset -c 13 ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --repeats 10 --csv compute_sanitizer.csv
```

The CSV lands in the working directory, under the name given; the demo
writes nothing else. Captured output (the reference run), with the
checkout's path in the skip line shortened to `...`:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from ComputeSanitizer
[ RUN      ] ComputeSanitizer.SaxpyKernel
[ComputeSanitizer.SaxpyKernel]  20.473 us/call  CV=0.1%  ~48.8K calls/s  (p10=20.438 p90=20.507 sd=0.030)
[       OK ] ComputeSanitizer.SaxpyKernel (2228 ms)
[ RUN      ] ComputeSanitizer.SaxpyUnguarded
.../src/bench/demo/gpu/04_ComputeSanitizerProfiler_Demo.cu:165: Skipped
this case launches a kernel past the end of its buffers for Compute Sanitizer to find, so it runs only under it: run this binary under `compute-sanitizer --tool=memcheck`, or through `bench run --profile compute-sanitizer`

[  SKIPPED ] ComputeSanitizer.SaxpyUnguarded (0 ms)
[----------] 2 tests from ComputeSanitizer (2229 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (2229 ms total)
[  PASSED  ] 1 test.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] ComputeSanitizer.SaxpyUnguarded
```

What to read: the kernel's median, 20.473 us, and its CV.
`ComputeSanitizer.SaxpyUnguarded` reports `SKIPPED` with the helper's
message, because the run is not under the tool. The CSV holds the one
timing row.

## Step 2: Run the Unguarded Kernel Under the Tool

```bash
bench run ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --profile compute-sanitizer \
  --profile-output-dir sanitizer-unguarded -- --gtest_filter=ComputeSanitizer.SaxpyUnguarded
```

Captured output, with the working directory shortened to `...`:

```
Running: compute-sanitizer --tool=memcheck --error-exitcode 5 --log-file .../sanitizer-unguarded/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --profile compute-sanitizer --profile-output-dir sanitizer-unguarded --gtest_filter=ComputeSanitizer.SaxpyUnguarded
Note: Google Test filter = ComputeSanitizer.SaxpyUnguarded
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ComputeSanitizer
[ RUN      ] ComputeSanitizer.SaxpyUnguarded
[ComputeSanitizer.SaxpyUnguarded]  one launch over 1048575 elements, 4096 blocks of 256; the device reported: unspecified launch failure
[       OK ] ComputeSanitizer.SaxpyUnguarded (428 ms)
[----------] 1 test from ComputeSanitizer (428 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (428 ms total)
[  PASSED  ] 1 test.

[profile] --profile compute-sanitizer: no case that ran was built with the profiler guard; the compute-sanitizer wrap still recorded the whole process.

Error: compute-sanitizer reported 4 errors in the benchmark; the report is sanitizer-unguarded/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log (the tool exited with status 5)
```

The `Running:` line is the wrap `bench run` built: the tool asked for, an
exit status for its findings, and the log named whole. No readiness report is
printed here: the case builds no measurement, so no profiler is created for
it, as the `[profile]` line says, and the tool around the process checks it
all the same. The case ran, because under the tool the helper lets it, and it
passed: the launch was accepted, and what the device reported when the case
waited for it is `unspecified launch failure`, which is the tool ending the
context at the kernel's first invalid access. The tool then ended with the
status `bench run` gave it for findings, and `bench run` read the report's
count and failed the run, naming the report: it exits 1. The one file the run
wrote is the report, in a folder named for the binary and the tool:

```bash
ls sanitizer-unguarded/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer
```

```
sanitizer.log
```

## Step 3: Read the Report

```bash
cat sanitizer-unguarded/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log
```

Captured output. GoogleTest's frames are cut from each host backtrace and
marked `...`:

```
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at vernier::bench::demo::sanitizer_demo::<unnamed>::saxpyUnguarded(float, const float *, float *)+0x100 in 04_ComputeSanitizerProfiler_Unguarded.cu:31
=========     by thread (255,0,0) in block (4095,0,0)
=========     Access to 0xd67bffffc is out of bounds
=========     and is 1 bytes after the nearest allocation at 0xd67800000 of size 4,194,300 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: ComputeSanitizer_SaxpyUnguarded_Test::TestBody() [0x8ccb] in BenchDemo_Gpu_04_ComputeSanitizerProfiler
...
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: ComputeSanitizer_SaxpyUnguarded_Test::TestBody() [0x8d3f] in BenchDemo_Gpu_04_ComputeSanitizerProfiler
...
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaFree.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: ComputeSanitizer_SaxpyUnguarded_Test::TestBody() [0x8d6b] in BenchDemo_Gpu_04_ComputeSanitizerProfiler
...
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaFree.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: ComputeSanitizer_SaxpyUnguarded_Test::TestBody() [0x8d73] in BenchDemo_Gpu_04_ComputeSanitizerProfiler
...
=========
========= ERROR SUMMARY: 4 errors
```

Every line of the report begins with the tool's prefix. One invalid access
is reported, with the lines that place it:

| Line                                                                                                       | What it says                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `Invalid __global__ read of size 4 bytes`                                                                  | a four-byte load from global memory the kernel may not touch: one `float`                                                                 |
| `at ...saxpyUnguarded(float, const float *, float *)+0x100 in 04_ComputeSanitizerProfiler_Unguarded.cu:31` | the kernel and, from its line information, the source line: `y[I] = a * x[I] + y[I];`                                                     |
| `by thread (255,0,0) in block (4095,0,0)`                                                                  | the last thread of the last block: the one thread past the end                                                                            |
| `Access to 0x... is out of bounds`                                                                         | the address is in no allocation the tool knows                                                                                            |
| `and is 1 bytes after the nearest allocation at 0x... of size 4,194,300 bytes`                             | it lies right after one of the two vectors, 1,048,575 floats of four bytes                                                                |
| `Saved host backtrace up to driver entry point at kernel launch time`                                      | the host stack of the launch: the demo's case, then GoogleTest's frames                                                                   |
| `Program hit cudaErrorLaunchFailure (error 719) ... on CUDA API call to cudaDeviceSynchronize`             | the tool ended the context at the invalid access, so the wait that followed failed, and so did the two `cudaFree` of the buffers after it |
| `ERROR SUMMARY: 4 errors`                                                                                  | the read and the three failed calls                                                                                                       |

The line to act on is the one with a file and a line number: the read at
line 31 is the kernel loading one of its two vectors at index `n`, one past
the end. The other load and the store would have followed; the kernel never
got there, because the tool stopped it at the first invalid access.

What not to conclude:

- **That the hardware would have caught this.** It did not: a plain launch
  of this kernel on this rig computes every element and returns `no error`
  from the wait that follows, so nothing but the tool sees the read.
- **That four things went wrong.** The summary counts the consequences of
  the tool's own context termination too. One access was invalid.
- **That the addresses or the offset travel.** `0x...` and `+0x100`
  belong to this build and this run; the kernel's name, the line, the thread,
  the block and the allocation's size do.

## Step 4: Confirm the Fix

The same command on the shared kernel, with one cycle and one repeat (the
harness's warmup launches and one measured launch):

```bash
bench run ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --profile compute-sanitizer \
  --cycles 1 --repeats 1 --profile-output-dir sanitizer-kernel -- --gtest_filter=ComputeSanitizer.SaxpyKernel
cat sanitizer-kernel/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log
```

Captured output of the run, with the working directory shortened to `...`:

```
Running: compute-sanitizer --tool=memcheck --error-exitcode 5 --log-file .../sanitizer-kernel/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --cycles 1 --repeats 1 --profile compute-sanitizer --profile-output-dir sanitizer-kernel --gtest_filter=ComputeSanitizer.SaxpyKernel
Note: Google Test filter = ComputeSanitizer.SaxpyKernel
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ComputeSanitizer
[ RUN      ] ComputeSanitizer.SaxpyKernel

[WARN] Profiler 'compute-sanitizer': unverified: compute-sanitizer started this process and reports when it exits; which tool it runs, and what it finds, is not seen from inside the process

[compute-sanitizer] tool=memcheck -- wrapping detected; compute-sanitizer reports at process exit, in its --log-file or on its stdout. Artifact directory: sanitizer-kernel/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer
[gpu] CUPTI refused the collector's activity callbacks (CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED): cuptiKernelLaunches, cuptiRegistersMedian, cuptiRegistersMax, cuptiStaticSmemBytes and cuptiDynamicSmemBytes stay empty.
[gpu] NVML reported no SM clock, maximum SM clock, power draw, power limit or GPU temperature (Not Supported): smClockMHz, throttling, powerDrawW, powerLimitW, temperatureC and temperatureDeltaC stay empty.
[ComputeSanitizer.SaxpyKernel]  1149.472 us/call  CV=0.0%  ~870 calls/s  (p10=1149.472 p90=1149.472 sd=0.000)
[       OK ] ComputeSanitizer.SaxpyKernel (243 ms)
[----------] 1 test from ComputeSanitizer (243 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (243 ms total)
[  PASSED  ] 1 test.
[bench] compute-sanitizer wrote sanitizer-kernel/BenchDemo_Gpu_04_ComputeSanitizerProfiler.compute-sanitizer/sanitizer.log (62 bytes)
```

and of the report:

```
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

The `[WARN]` line is the readiness report explained above, printed because
this case measures; the line after it is the backend's notice that the wrap
was detected and where the report goes. The two `[gpu]` lines name the GPU
columns this run leaves empty: CUPTI takes one subscriber, and under the tool
the harness's collector is refused; this rig's NVML reads no clocks or
power. The last line is `bench run`'s check that the run left its report.
`ERROR SUMMARY: 0 errors`: the tool watched every launch of the guarded
kernel and found nothing to report, and the run exits 0. The result line is
one launch under the tool, 1149.5 us against step 1's 20.473 us; it says what
the tool costs, not how fast the kernel is. Measured with more launches
(`--cycles 100 --repeats 3`, three runs each way), the kernel read
20.10 to 20.93 us plainly and 1135.3 to 1136.4 us under
the tool: 54 to 57 times as long.

## Failing a Job on a Memory Error

The wrap `bench run` builds gives the tool an exit status for its findings
and fails the run when the report counts errors, as step 2 shows, so a job
that runs it stops on an invalid access. A job that runs the tool itself
gives it an exit code of its own:

```bash
compute-sanitizer --tool=memcheck --error-exitcode 1 --log-file unguarded.log \
  ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --gtest_filter=ComputeSanitizer.SaxpyUnguarded
echo "exit $?"
```

```
Note: Google Test filter = ComputeSanitizer.SaxpyUnguarded
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ComputeSanitizer
[ RUN      ] ComputeSanitizer.SaxpyUnguarded
[ComputeSanitizer.SaxpyUnguarded]  one launch over 1048575 elements, 4096 blocks of 256; the device reported: unspecified launch failure
[       OK ] ComputeSanitizer.SaxpyUnguarded (456 ms)
[----------] 1 test from ComputeSanitizer (456 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (456 ms total)
[  PASSED  ] 1 test.
exit 1
```

The test passed and the tool exited 1: the report is in `unguarded.log`, and
the job fails. The same on the shared kernel:

```bash
compute-sanitizer --tool=memcheck --error-exitcode 1 --log-file kernel.log \
  ./build/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler --cycles 1 --repeats 1 --gtest_filter=ComputeSanitizer.SaxpyKernel
echo "exit $?"
```

```
Note: Google Test filter = ComputeSanitizer.SaxpyKernel
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ComputeSanitizer
[ RUN      ] ComputeSanitizer.SaxpyKernel
[ComputeSanitizer.SaxpyKernel]  1147.040 us/call  CV=0.0%  ~872 calls/s  (p10=1147.040 p90=1147.040 sd=0.000)
[       OK ] ComputeSanitizer.SaxpyKernel (236 ms)
[----------] 1 test from ComputeSanitizer (236 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (236 ms total)
[  PASSED  ] 1 test.
exit 0
```

## What Should Reproduce

| Reading                        | On this rig                                                                                                                                                                    | Elsewhere                                                                                                                                                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the read                       | `Invalid __global__ read of size 4 bytes`, once, by thread (255,0,0) in block (4095,0,0), `1 bytes after` an allocation of 4,194,300 bytes, at line 31 of the unguarded source | should match: it depends on the code, not the machine; the same read, thread, block and offset on an RTX 5000 Ada laptop GPU with compute-sanitizer 2025.4, which prints the size without thousands separators |
| errors counted                 | 4: the read and three failed calls after the context ended                                                                                                                     | 4 with the tool's default context policy; more calls after the read count more                                                                                                                                 |
| what the shared kernel reports | `0 errors`, in every run for this page                                                                                                                                         | should match                                                                                                                                                                                                   |
| the plain run                  | `SaxpyKernel` OK, `SaxpyUnguarded` skipped, exit 0                                                                                                                             | the same anywhere: the case skips before any CUDA call                                                                                                                                                         |
| the kernel's time              | 20.473 us; 20.406 to 20.497 us over the session's five runs                                                                                                                    | will differ; about 8.7 us on the RTX 5000 Ada                                                                                                                                                                  |
| the tool's cost                | 54 to 57 times the kernel's time at 100 launches per repeat                                                                                                                    | tens to hundreds of times; about 116 times on the RTX 5000 Ada at 100 launches per repeat (8.746 us plainly, 1017.1 us under the tool), one run each way with the clocks as found and other work on the laptop |

The five runs are step 1's command run five times in one session on this
rig, the reference capture among them. That range describes those runs; it
is not a bound a run has to meet, and another run of the same command can
land outside it. The kernel's time is the noisiest reading on this page:
0.4% across the five runs, and 4% across the three plain runs at 100
launches per repeat in step 4 (20.10 to 20.93 us), which is what moves the
tool's cost between 54 and 57 times. The tool's lines are the readings to
carry elsewhere.

## If It Does Not Match

- **`ComputeSanitizer.SaxpyUnguarded` reports `SKIPPED` in step 2.** The
  run was not under the tool. The binary run on its own prints:

  ```
  Note: Google Test filter = ComputeSanitizer.SaxpyUnguarded
  [==========] Running 1 test from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 1 test from ComputeSanitizer
  [ RUN      ] ComputeSanitizer.SaxpyUnguarded
  .../src/bench/demo/gpu/04_ComputeSanitizerProfiler_Demo.cu:165: Skipped
  this case launches a kernel past the end of its buffers for Compute Sanitizer to find, so it runs only under it: run this binary under `compute-sanitizer --tool=memcheck`, or through `bench run --profile compute-sanitizer`

  [  SKIPPED ] ComputeSanitizer.SaxpyUnguarded (0 ms)
  [----------] 1 test from ComputeSanitizer (0 ms total)

  [----------] Global test environment tear-down
  [==========] 1 test from 1 test suite ran. (0 ms total)
  [  PASSED  ] 0 tests.
  [  SKIPPED ] 1 test, listed below:
  [  SKIPPED ] ComputeSanitizer.SaxpyUnguarded
  ```

  Run it through `bench run --profile compute-sanitizer`, as step 2 does, or
  under `compute-sanitizer --tool=memcheck` yourself.

- **`bench run` stops before the test starts:**

  ```
  Error: tool not found: 'compute-sanitizer' is not on PATH; --profile compute-sanitizer runs the benchmark under it. Install compute-sanitizer, or run `bench doctor` to see which profilers this machine can use
  ```

  The toolkit's `bin` is not on `PATH`; the rig document's setup exports it.

- **The binary run by hand with `--profile compute-sanitizer` fails with
  status 4 and names the wrap.** Nothing started it under the tool, so the
  request cannot check anything: the tests run and pass, and the report names
  the command that wraps the binary in the tool, and `bench run`, which does.
  The run as this page's later session printed it, with the working
  directory shortened to `...`:

  ```
  [ RUN      ] ComputeSanitizer.SaxpyKernel

  [FAIL] Profiler 'compute-sanitizer': missing: compute-sanitizer checks a process only when it starts it, and it did not start this one
     Wrap it: compute-sanitizer --tool=memcheck --error-exitcode 5 --log-file=.../sanitizer.log <this-binary> --profile compute-sanitizer [...]; or run it with bench run --profile compute-sanitizer, which wraps it and reads the report.
     Nothing is collected for this request; the run will fail (exit status 4 if the tests pass).

  [gpu] NVML reported no SM clock, maximum SM clock, power draw, power limit or GPU temperature (Not Supported): smClockMHz, throttling, powerDrawW, powerLimitW, temperatureC and temperatureDeltaC stay empty.
  [ComputeSanitizer.SaxpyKernel]  21.856 us/call  CV=0.0%  ~45.8K calls/s  (p10=21.856 p90=21.856 sd=0.000)
  [       OK ] ComputeSanitizer.SaxpyKernel (180 ms)
  ...
  [profile] --profile compute-sanitizer failed; the run exits with status 4:
  [profile]   compute-sanitizer: missing: compute-sanitizer checks a process only when it starts it, and it did not start this one
  ```

  The wrap names the log whole, in the directory the run started from, with
  each `%` in it doubled for the tool and the name quoted for the shell where
  it needs it: the tool joins a relative log name to its working directory
  and reads a `%` there as a macro. Filled in with the binary and the run's
  arguments, it checks the case and writes the log where it says. No folder
  is created for the refused request. With `--profile-args racecheck`,
  `synccheck` or `initcheck`, both commands carry that tool.

- **The report names no line, only `saxpyUnguarded(...)+0x...`.** The
  build has no device line information for that source; this tree compiles
  the unguarded copy with `-lineinfo` in every build type.

- **`Error: Target application terminated before first instrumented API
call`, exit 255.** The program the tool started never called CUDA: a
  filter that selects no GPU test, or a CPU binary. `--require-cuda-init no`
  lets such a run end with the program's own status.

- **Hundreds of reads and a truncated listing.** A size that leaves more
  threads past the end reports each one, and the tool stops printing after
  100 (`--print-limit`); at 1,048,577 elements, 255 threads of a 4,097th
  block, this rig reported 259 errors.

- **`--profile-args` is refused before anything runs.** A word that is not
  one of the tool's four (`memcheck`, `racecheck`, `synccheck`,
  `initcheck`), or two of them at once, is refused by `bench run`, naming
  the four; the binary run by hand reads the same words and fails the
  request the same way.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/thor/17_compute_sanitizer.csv compute_sanitizer.csv
```

A later run on the same rig, clocks locked:

```
Test                              Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
----------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
ComputeSanitizer.SaxpyKernel      20.47330      20.24260    -0.23070     -1.1%      0.1%      0.2%  neutral

  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

`bench compare` labels a test `REGRESSION` when its median is more than the
threshold (5% unless `--threshold` says otherwise) above the reference's,
`IMPROVEMENT` when it is more than that below it, and `neutral` otherwise.
The labels describe the difference between two runs and are not a
significance test; the two CV columns are each run's own spread, not the
spread between the runs. The CSV holds a time only; what the tool reports is
checked by the demo's check, below.

The reference CSV is the one of the session's five runs whose median sits
closest to the median of the five. It was captured from a copy of the
development tree without its git metadata, so its `gitHash` column reads
`unknown`, and its `hostname` column records the name this rig reports.

## What Keeps This Page True

Three things check what this page shows, and all fail loudly:

- `TestDemoComputeSanitizer`, a test program of its own beside the demo
  ([`04_ComputeSanitizerProfiler_uTest.cpp`](../gpu/utst/04_ComputeSanitizerProfiler_uTest.cpp)),
  runs the demo binary under compute-sanitizer as `bench run` wraps it, with
  its own exit code for errors. `ComputeSanitizer.FindsTheUnguardedRead` fails
  unless the unguarded case runs to its end, the report holds exactly one
  invalid access, a read of four bytes in `saxpyUnguarded` at the line of
  its statement (found in the source), by thread (255,0,0) in block
  (4095,0,0), one byte after an allocation of the vectors' size, and the
  tool exits with the code it was given; a copy given its guard fails it.
  `ComputeSanitizer.KernelReportsNothing` fails on any access or error for
  the shared kernel. `ComputeSanitizer.UnguardedSkipsOutsideTheTool` holds
  the plain run to its skip; `ComputeSanitizer.PlainRunIsNotWrapped` the
  binary run by hand with `--profile compute-sanitizer` to its refusal
  (status 4 after its test passed, no wrap reported where there is none, the
  wrap named); `ComputeSanitizer.PlainRunHintShape` the wrap its report names
  to its shape (the tool asked for, `--error-exitcode 5`, the log named whole
  in the run's working directory, and `bench run` with the same request, for
  memcheck and for racecheck); `ComputeSanitizer.HintRunsOnTheFirstRun` that
  wrap, filled in and run as printed from another directory, to writing its
  log where it says, under a directory whose name holds a `%`, also where the
  run's own directory's name holds spaces
  (`ComputeSanitizer.HintRunsWithSpacesInItsPath`) or a quote
  (`ComputeSanitizer.HintRunsWithAQuoteInItsPath`); and
  `ComputeSanitizer.NamedToolHintRunsThatTool` the report for racecheck,
  synccheck and initcheck to offering `bench run` with that tool, and a wrap
  which, run as printed against a stand-in that records its arguments, starts
  compute-sanitizer with the tool named. These are registered with `ctest`
  under the `demo` and `compute-sanitizer` labels wherever the GPU demos are
  built, and skip only where the tool is not on `PATH`, where the CUDA
  runtime sees no device, or in a build with the address or the thread
  sanitizer, in which the demo does not run, saying which. Six more tests,
  in `TestDemoComputeSanitizerReport`
  ([`04_ComputeSanitizerProfiler_Report_uTest.cpp`](../gpu/utst/04_ComputeSanitizerProfiler_Report_uTest.cpp)),
  hold the report reading to real report text from both tool versions; they
  need neither the tool nor CUDA, so every build runs them, under the same
  labels:

  ```bash
  ctest --test-dir build -L compute-sanitizer
  ```

- The helper's own tests, in `TestDemoHelpers`, run their binary on a probe
  case that uses the helper, plainly and under the tool, and fail if the
  probe runs in the plain run or skips under the tool. The probe itself is
  reported as skipped by `ctest` in an ordinary run, which is the helper
  working.
- The example's unit tests hold every version of SAXPY to the CPU loop's
  answers, under the same `demo` label.

The demo's timing case is not registered: what it measures belongs to the
machine it runs on. This repository has no continuous-integration lane on
the reference board, so before a release the page's commands are run on the
rig by hand, and the page and its reference CSV are re-captured when what
they show changes.

## See Also

- [Demo 10: GPU basic workflow](10_GPU_BASIC_WORKFLOW.md) -- the same
  kernel, timed with and without its transfers
- [Demo 11: Nsight Systems and Nsight Compute](11_NSIGHT_PROFILER.md) -- the
  performance side of GPU profiling
- [Demo 15: memcheck](15_MEMCHECK_PROFILER.md) -- the same kind of check for
  host memory
- [Demo 19: CUPTI kernel metrics](19_CUPTI_KERNEL_METRICS.md) -- per-kernel
  metrics from inside the process
- [GPU Guide](../../docs/GPU_GUIDE.md) -- the GPU harness in full
- [Reference Rig: NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
