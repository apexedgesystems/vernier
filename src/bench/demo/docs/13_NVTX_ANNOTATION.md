# Demo 13: NVTX Annotation

**Reference rig:** [NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
**Build:** Release
**Example:** `saxpy` (see [Shared Examples](../README.md#shared-examples))
**Captured:** 2026-10-03 (UTC). Written for the Vernier 1.0.4 release;
captured from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`, with Nsight Systems 2025.3.2. Steps 1 to 4 come from
one session. A second session the same day, from the same tree with this page
and its reference CSV added, ran every command on this page again; the
`bench compare` output is from it, and the ranges this page states include
its runs.

## Overview

Nsight Systems records a run's CUDA calls, copies and kernels with their
times, but not which part of the program each belongs to, and of the host's
own work between them it keeps at most periodic CPU samples. An NVTX range is
a name the program puts around a stretch of its own code; Nsight Systems
records it with the work of the same moment. On the SAXPY example's `G1`, the
plain call's capture shows the GPU busy about 70 us of a call of about 880 us.
The same call in three named ranges, `copy_in`, `kernel` and `copy_out`, shows
where the rest goes: the host copying x and y into its pinned buffers, a wait
for the GPU to start the first copy, and the host copying y back out.

## What is NVTX?

- **NVTX** (the NVIDIA Tools Extension) is a C API that ships with the CUDA
  toolkit as headers (`nvtx3`). `nvtxRangePushA("name")` opens a range on the
  calling thread and `nvtxRangePop()` closes the innermost one, so ranges nest.
  NVTX records nothing itself: a tool that starts the program, such as `nsys`
  or `ncu`, receives the calls, and without one they do nothing.
- **A range belongs to the host thread.** It opens when the thread pushes and
  closes when it pops. A CUDA copy or kernel launch returns once it is queued,
  so a range around those calls alone holds the queueing and none of the GPU's
  work. To hold the work, the code waits for it before the range closes.
- **It labels; it does not measure.** Nsight Systems records the ranges and
  `nsys stats` reports them. The timings on this page still come from the
  benchmark.

**In Vernier:** `BENCH_NVTX_SCOPE("name")` ([`Nvtx.hpp`](../../inc/Nvtx.hpp))
opens a range and closes it when the enclosing scope ends; `BENCH_NVTX_MARK`
records an instant. Both compile to nothing in a build without the CUDA
toolkit's `nvtx3` headers (CMake's `CUDA::nvtx3`), so the same source builds
everywhere. With `--profile nsight` (or `--profile ncu`), the harness adds one
range per test, named after the test (`NvtxAnnotation.G1`): opened when the
measured calls start, after the warmup, and closed once the measurement is over.
Without that flag there is no range named after the test.
`bench run <binary> --profile nsight` starts the binary under
`nsys profile -t cuda,nvtx` and writes
`bench-out/<binary>.nsight/profile.nsys-rep`, the SQLite export next to it and
four summaries ([walkthrough 11](11_NSIGHT_PROFILER.md)); the ranges are read
from the report with `nsys stats`.

**Needs:** the CUDA toolkit on `PATH`, with `nsys`, and a GPU build: see the
rig's [setup](../../docs/rigs/RIG_THOR_AGX.md#2-one-time-setup) and
[build](../../docs/rigs/RIG_THOR_AGX.md#3-build).

## The Example

SAXPY, `y = a*x + y` over 1,048,576 floats
([`SaxpyGpu.cpp`](../examples/saxpy/src/SaxpyGpu.cpp)). `G1` (`SaxpyG1`)
allocates two device buffers, two pinned host buffers and a stream once, in
its constructor; a call copies x and y into the pinned buffers, queues the two
copies to the device, the launch at 256 threads per block and the copy back,
and waits once:

```cpp
void SaxpyG1::apply(float a, const std::vector<float>& x, std::vector<float>& y) {
  Impl& s = *impl_;
  const std::size_t BYTES = s.n * sizeof(float);
  std::memcpy(s.hX.get(), x.data(), BYTES);
  std::memcpy(s.hY.get(), y.data(), BYTES);
  check(cudaMemcpyAsync(s.dX.get(), s.hX.get(), BYTES, cudaMemcpyHostToDevice, s.stream.get()),
        "HtoD x");
  check(cudaMemcpyAsync(s.dY.get(), s.hY.get(), BYTES, cudaMemcpyHostToDevice, s.stream.get()),
        "HtoD y");
  launchSaxpy(a, s.dX.get(), s.dY.get(), s.n, /*threadsPerBlock=*/256, s.stream.get());
  check(cudaGetLastError(), "launch");
  check(cudaMemcpyAsync(s.hY.get(), s.dY.get(), BYTES, cudaMemcpyDeviceToHost, s.stream.get()),
        "DtoH y");
  check(cudaStreamSynchronize(s.stream.get()), "sync"); // one wait, at the end
  std::memcpy(y.data(), s.hY.get(), BYTES);
}
```

The demo ([`05_NvtxAnnotation_Demo.cu`](../gpu/05_NvtxAnnotation_Demo.cu))
times that call two ways, one test and one CSV row each:

- `NvtxAnnotation.G1` calls `SaxpyG1::apply` as it is, and opens no range of
  its own.
- `NvtxAnnotation.G1Phases` does the same work with buffers set up the same
  way, once, and the example's launch, one phase at a time, each in a named
  range (`SaxpyG1InPhases::apply`):

```cpp
void apply(float a, const std::vector<float>& x, std::vector<float>& y) {
  {
    BENCH_NVTX_SCOPE(phase::COPY_IN);
    std::memcpy(hX_, x.data(), BYTES);
    std::memcpy(hY_, y.data(), BYTES);
    check(cudaMemcpyAsync(dX_, hX_, BYTES, cudaMemcpyHostToDevice, stream_), "HtoD x");
    check(cudaMemcpyAsync(dY_, hY_, BYTES, cudaMemcpyHostToDevice, stream_), "HtoD y");
    check(cudaStreamSynchronize(stream_), "copy in");
  }
  {
    BENCH_NVTX_SCOPE(phase::KERNEL);
    ubd::launchSaxpy(a, dX_, dY_, N, THREADS_PER_BLOCK, stream_);
    check(cudaGetLastError(), "launch");
    check(cudaStreamSynchronize(stream_), "kernel");
  }
  {
    BENCH_NVTX_SCOPE(phase::COPY_OUT);
    check(cudaMemcpyAsync(hY_, dY_, BYTES, cudaMemcpyDeviceToHost, stream_), "DtoH y");
    check(cudaStreamSynchronize(stream_), "copy out");
    std::memcpy(y.data(), hY_, BYTES);
  }
}
```

Each phase waits for its stream before its range closes, so each range holds
its phase's GPU work: that is what lets the capture show the phases. It costs
`G1Phases` two more waits a call than `G1`, and its kernel is queued only after
the copies have finished, its copy back only after the kernel. The three names
(`copy_in`, `kernel`, `copy_out`) live in
[`05_NvtxAnnotation_Phases.hpp`](../gpu/05_NvtxAnnotation_Phases.hpp), which
the demo and its check both read. Both tests check `y` after all their calls,
and the example's unit tests hold `G1` to the CPU loop's answers.

## Step 1: Measure

Inside the rig document's
[measurement procedure](../../docs/rigs/RIG_THOR_AGX.md#4-running-a-measurement)
(clocks locked with `jetson_clocks` and restored afterwards, the host thread on
core 13), from the source tree, after the rig's build:

```bash
taskset -c 13 ./build/bin/ptests/BenchDemo_Gpu_05_NvtxAnnotation --cycles 20 --repeats 10 --csv nvtx_annotation.csv
```

A call takes under a millisecond, so give the run a cycle count; the default
of 10,000 cycles per repeat would keep each test busy for over a minute.
Captured output (the reference run):

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from NvtxAnnotation
[ RUN      ] NvtxAnnotation.G1
[NvtxAnnotation.G1]  874.800 us/call  CV=0.8%  ~1.1K calls/s  (p10=869.870 p90=880.725 sd=6.874)
[       OK ] NvtxAnnotation.G1 (329 ms)
[ RUN      ] NvtxAnnotation.G1Phases
[NvtxAnnotation.G1Phases]  908.925 us/call  CV=0.2%  ~1.1K calls/s  (p10=906.880 p90=909.950 sd=2.028)
[       OK ] NvtxAnnotation.G1Phases (189 ms)
[----------] 2 tests from NvtxAnnotation (519 ms total)
...
[  PASSED  ] 2 tests.
```

What to read: `G1` takes 874.8 us a call and `G1Phases` 908.9 us, 34.1 us
more: what the phased call's two extra waits cost. Over fifteen runs of this
command in two sessions the difference read 26.6 to 42.6 us in eleven; the
other four, and the two bands both tests move between on this rig, are in
[What Should Reproduce](#what-should-reproduce).

## Step 2: The Call Without Ranges of Its Own

With the clocks still locked, run `G1` under Nsight Systems:

```bash
source build/.env
bench run ./build/bin/ptests/BenchDemo_Gpu_05_NvtxAnnotation --profile nsight -- \
  --gtest_filter=NvtxAnnotation.G1 --cycles 20 --repeats 3
```

```
Running: nsys profile -o bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile -t cuda,nvtx --force-overwrite true ./build/bin/ptests/BenchDemo_Gpu_05_NvtxAnnotation --profile nsight --gtest_filter=NvtxAnnotation.G1 --cycles 20 --repeats 3
...
[ RUN      ] NvtxAnnotation.G1

[WARN] Profiler 'nsight': unverified: collection is owned by the nsight wrap; completion is checked at exit

[nsight] this process runs under nsys, which writes the report when the process exits.
[NvtxAnnotation.G1]  882.550 us/call  CV=0.6%  ~1.1K calls/s  (p10=881.590 p90=891.750 sd=5.725)
[       OK ] NvtxAnnotation.G1 (209 ms)
...
Generated:
    .../bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile.nsys-rep
[nsight] auto-extracted nsys stats reports into bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight
```

The `[WARN]` line is the harness's readiness report for the profiler request,
as in walkthrough 11: under `bench run`'s wrap the capture belongs to `nsys`.
The run made 61 calls, one warmup and 60 measured. Read the report's ranges
before the next `bench run`, which writes the same file:

```bash
nsys stats --force-export=true --report nvtx_pushpop_sum bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile.nsys-rep
```

```
...
 ** NVTX Push/Pop Range Summary (nvtx_pushpop_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)        Range
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ------------------
    100.0       61,369,464          1  61,369,464.0  61,369,464.0  61,369,464  61,369,464          0.0  :NvtxAnnotation.G1
```

One range, the test's, 61.37 ms long (`nsys` writes each name as
`domain:name`, and the demo's ranges are in the default domain, which has no
name). The summaries `bench run` wrote beside the report count what happened
inside it: three `cudaMemcpyAsync`, one launch and one `cudaStreamSynchronize`
a call (183, 61 and 61 in `cuda_api_sum.txt`, warmup included); the kernel took
22.5 us on the GPU by its median (`cuda_gpu_kern_sum.txt`) and each 4.19 MB
copy about 15.5 us (`cuda_gpu_mem_time_sum.txt`). Matched call by call in the
report, as in Step 4, the GPU's four operations add up to about 70 us of each
882 us call. Each call's wait, `cudaStreamSynchronize`, lasts 281 us by its
median; after it returns, about 572 us pass, with no CUDA call and nothing on
the GPU, before the next call queues its first copy, and the GPU starts that
copy about 216 us after it was queued.

The CUDA trace does not say what the host does in those 572 us. Nsight Systems
also samples the CPU by default on this rig, and all 36 samples it took of the
test's thread inside the range landed in `__memcpy_sve`, the C library's copy
(33 of 37 and 35 of 37 in two more captures): the thread is copying, but a
sample does not say which copy, in which call, or for how long. A range does.

The range itself holds more than its calls: 60 calls of about 882 us take
53 ms, and the range lasts 61.37 ms. It opens 0.68 ms before the first copy
runs on the GPU (the first call's host work, then the wait for the GPU to
start the copy), and it closes 8.4 ms after the last copy back. That tail
belongs to the first measured test of a process, not to `G1`: before the
harness closes the range, it builds the test's CSV row, and the first row a
process builds runs `git describe` once, as a subprocess, for the `gitHash`
column. In a capture of both tests in one process, the first test's range
ended 8.1 ms after its last copy back and the second's 0.24 ms after. In a
capture of an earlier session that also traced the OS runtime, the range's
thread called `popen` within a millisecond of the last copy back, and after it
the trace shows nothing on that thread until the range closed.

## Step 3: The Same Call in Three Ranges

```bash
bench run ./build/bin/ptests/BenchDemo_Gpu_05_NvtxAnnotation --profile nsight -- \
  --gtest_filter=NvtxAnnotation.G1Phases --cycles 20 --repeats 3
```

```
Running: nsys profile -o bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile -t cuda,nvtx --force-overwrite true ./build/bin/ptests/BenchDemo_Gpu_05_NvtxAnnotation --profile nsight --gtest_filter=NvtxAnnotation.G1Phases --cycles 20 --repeats 3
...
[NvtxAnnotation.G1Phases]  953.700 us/call  CV=0.4%  ~1.0K calls/s  (p10=949.820 p90=957.460 sd=3.899)
[       OK ] NvtxAnnotation.G1Phases (212 ms)
...
```

```bash
nsys stats --force-export=true --report nvtx_pushpop_sum bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile.nsys-rep
```

```
...
 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)           Range
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ------------------------
     52.7       64,930,807          1  64,930,807.0  64,930,807.0  64,930,807  64,930,807          0.0  :NvtxAnnotation.G1Phases
     33.3       40,984,001         61     671,868.9     669,703.0     645,741     753,602     17,993.0  :copy_in
     11.5       14,225,526         61     233,205.3     233,111.0     219,509     257,741      8,370.7  :copy_out
      2.5        3,112,543         61      51,025.3      46,843.0      44,639     232,667     24,387.5  :kernel
```

The test's range, and 61 of each phase, one per call. Which phases are whose,
and how they nest, is in the trace:

```bash
nsys stats --force-export=true --report nvtx_pushpop_trace bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile.nsys-rep
```

```
...
 Start (ns)    End (ns)    Duration (ns)  DurChild (ns)  DurNonChild (ns)            Name             PID     TID    Lvl  NumChild  RangeId  ParentId  RangeStack          NameTree
 -----------  -----------  -------------  -------------  ----------------  ------------------------  ------  ------  ---  --------  -------  --------  ----------  ------------------------
 178,263,740  179,017,342        753,602              0           753,602  :copy_in                  35,518  35,518    0         0        1            :1          :copy_in
 179,018,944  179,251,611        232,667              0           232,667  :kernel                   35,518  35,518    0         0        2            :2          :kernel
 179,252,278  179,496,981        244,703              0           244,703  :copy_out                 35,518  35,518    0         0        3            :3          :copy_out
 179,522,935  244,453,742     64,930,807     57,091,098         7,839,709  :NvtxAnnotation.G1Phases  35,518  35,518    0       180        4            :4          :NvtxAnnotation.G1Phases
 179,533,426  180,223,592        690,166              0           690,166  :copy_in                  35,518  35,518    1         0        5         4  :4:5        --:copy_in
 180,224,305  180,274,991         50,686              0            50,686  :kernel                   35,518  35,518    1         0        6         4  :4:6        --:kernel
 180,275,472  180,519,685        244,213              0           244,213  :copy_out                 35,518  35,518    1         0        7         4  :4:7        --:copy_out
...
```

- **Level 0, before the test's range:** the warmup call's three phases. The
  warmup runs before the harness opens the test's range, so its phases have no
  parent. Its `kernel` range is the longest, 232.7 us: it holds the process's
  first launch, which loads the kernel's code (`cuLibraryLoadData`, 44.6 us),
  and the kernel's first run (40.2 us).
- **The test's range** (`RangeId` 4) has 180 children, three for each of the 60
  measured calls. They take 57.09 ms of it (`DurChild`); the other 7.84 ms
  (`DurNonChild`) are almost all after the last call: Step 2's tail.
- **Level 1, inside it:** each measured call's `copy_in`, `kernel` and
  `copy_out`, in that order, each closed before the next opens, with `ParentId`
  4 and a `--:` in `NameTree` for their depth.

`cuda_api_sum.txt` counts the extra waits: 183 `cudaStreamSynchronize`, three
a call.

## Step 4: Read the Ranges

By the summary's medians, a `G1Phases` call is 670 us in `copy_in`, 47 us in
`kernel` and 233 us in `copy_out`: 950 us of the 954 us a call took under
`nsys`. The start and end of every range, CUDA call and GPU operation are in
the report's traces, which `nsys stats` writes as CSV files:

```bash
nsys stats --force-export=true --report nvtx_pushpop_trace --report cuda_api_trace \
  --report cuda_gpu_trace --format csv --output phases \
  bench-out/BenchDemo_Gpu_05_NvtxAnnotation.nsight/profile.nsys-rep
```

That writes `phases_nvtx_pushpop_trace.csv`, `phases_cuda_api_trace.csv` and
`phases_cuda_gpu_trace.csv` in the current directory. Matching each range with
the CUDA calls and GPU operations that start and end inside it splits every
phase in four (medians in microseconds over the 60 measured calls of Step 3's
capture; each column is a median of its own, so a row's parts need not add up
to its median):

| Range      | Median | Before its first CUDA call | From that call to the GPU's first operation | GPU operations | From the last one to the range's end |
| ---------- | ------ | -------------------------- | ------------------------------------------- | -------------- | ------------------------------------ |
| `copy_in`  | 669.4  | 398.2                      | 215.2                                       | 35.4           | 17.3                                 |
| `kernel`   | 46.8   | 1.9                        | 7.6                                         | 22.6           | 14.3                                 |
| `copy_out` | 233.1  | 0.3                        | 7.1                                         | 15.3           | 210.3                                |

What the ranges say about the call:

- **The GPU works about 73 us of it:** the two copies in (35.4 us from the
  first one's start to the second one's end), the kernel (22.6 us) and the copy
  back (15.3 us).
- **The host copies take most of it.** `copy_in` spends 398 us before its first
  CUDA call, which is the host copying x and y, 4.19 MB each, into the pinned
  buffers; `copy_out` spends 210 us after its copy, which is the wait returning
  and the host copying y back out. In Step 2's capture these copies were the
  time with no CUDA call between one call's wait and the next call's first copy;
  the ranges name them.
- **The first copy starts late.** From queueing the first copy to the GPU
  starting it, 215 us pass, as in Step 2's `G1` (216 us): this rig's slow band
  (see [What Should Reproduce](#what-should-reproduce)). The ranges place the
  delay: it is the first copy's, which follows about 600 us in which the GPU
  had nothing to do. The kernel and the copy back, each queued right after a
  wait, start within 8 us.
- **A range is not its work.** The `kernel` range, 46.8 us, holds the launch
  call, the kernel's 22.6 us and the wait returning; read a kernel's own time
  from the GPU's summary (`cuda_gpu_kern_sum.txt`), as walkthrough 11 does.

What not to conclude: that `G1` spends its time the same way to the
microsecond. The ranges describe `G1Phases`, which waits three times a call and
queues each phase only after the last one finished; it took 34.1 us a call more
than `G1` in Step 1, and under `nsys` it took 62.2 to 71.2 us more than `G1`
in four pairs of captures. The GPU's work and the host's copies are the same in both
calls; what the waits change is when each piece starts.

## What Should Reproduce

| Reading                                    | On this rig                                                                                                              | Elsewhere                                                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Step 2's ranges                            | one, `NvtxAnnotation.G1`, with no range inside                                                                           | the same: they are the code's                                                                                             |
| Step 3's ranges                            | the test's, with 60 calls' `copy_in`, `kernel`, `copy_out` at level 1 inside it; the warmup's three at level 0 before it | the same; the check below holds them                                                                                      |
| CUDA calls a call                          | `G1`: three copies, one launch, one wait; `G1Phases`: the same with three waits                                          | the same                                                                                                                  |
| phase medians (`nvtx_pushpop_sum`)         | `copy_in` 655.6 to 676.4 us, `kernel` 46.6 to 47.7 us, `copy_out` 227.4 to 250.0 us, over five captures in two sessions  | the order and what each holds should match; their lengths depend on the host's copies, the GPU and how they are connected |
| the GPU's operations in a `G1` call        | 69.6 to 70.9 us, over five captures                                                                                      | depends on the GPU; on a discrete GPU the copies cross PCIe and take longer ([walkthrough 10](10_GPU_BASIC_WORKFLOW.md))  |
| the first test's range after its last call | 7.9 to 8.5 ms over nine captures; the second test's 0.24 ms (below)                                                      | the first test of a process carries it; its size depends on how long starting a process takes                             |
| `G1Phases` minus `G1` (Step 1)             | 26.6 to 42.6 us in eleven of fifteen runs (below)                                                                        | what a wait costs depends on the GPU and its driver                                                                       |
| absolute times                             | 874.8 / 908.9 us (the reference run)                                                                                     | will differ                                                                                                               |

The demo's two tests fail if their answer is wrong; the ranges are held by the
check described in [What Keeps This Page True](#what-keeps-this-page-true).

**The first test's range runs on after its last call.** With the clocks locked
it ended 7.9 to 8.5 ms after the last copy back for the first test of each of
nine captures in two sessions, and 0.24 ms after it for the second test of the
capture of both. In an earlier session, with the clocks as found, five captures
traced as here read 12.4 to 13.9 ms for the first test, three beside a parallel
build 15.2 to 35.1 ms, and the second test 0.23 to 0.36 ms in eight.

**Both tests move between two bands on this rig.** Over fifteen runs of Step
1's command in two sessions, with the clocks locked, `G1` read 629.5 to
643.5 us in three runs and 872.1 to 906.2 us in twelve; `G1Phases` read 666.3
to 676.3 us in five and 903.1 to 945.8 us in ten. In two runs the two
tests landed in different bands (`G1` 872.1 and 875.5 us, `G1Phases` 666.3 and
676.3 us), and a test that changed band part-way through a run shows a CV above
5% (6.6% to 14.2%, in five runs). Where both landed in the same band,
`G1Phases` read 26.6 to 42.6 us more than `G1` in eleven runs, 0.9 us more in
one and 71.3 us more in one. All nine captures (Steps 2 and 3, and one of both
tests) were in the slow band, with the GPU starting a call's first copy 212 to
219 us after the host queued it; the rig document describes the bands
([rig-specific behavior](../../docs/rigs/RIG_THOR_AGX.md#5-rig-specific-behavior)).
Another run of unchanged code can land outside every range on this page.

## If It Does Not Match

- **No range named after the test.** It comes from `--profile nsight` (or
  `--profile ncu`). Started under `nsys` without that flag, the demo records
  only its own ranges: a capture of `G1Phases` typed by hand without it held 61
  of each phase and no `NvtxAnnotation.G1Phases`. `bench run --profile nsight`
  passes the flag to the binary, as its `Running:` line shows.
- **No ranges at all.** In a build without the CUDA toolkit's `nvtx3` headers,
  `BENCH_NVTX_SCOPE` and the test's range compile to nothing, and the demo still
  runs and passes. The check below skips in such a build and says why.
- **A range that holds none of its GPU work.** A range closes when the host
  pops it; without the wait at the end of a phase, the range closes before the
  GPU runs the phase's work. With the `kernel` phase's wait removed, the check
  reports every call, the first as:

  ```
  call 0, range kernel holds 0 kernel(s), 0 copy(ies) to the device, 0 copy(ies) back, 0 other
  ```

- **The run was not captured.** `--profile nsight` without `nsys` around the
  process captures nothing and prints the command that would; see
  [walkthrough 11](11_NSIGHT_PROFILER.md#if-it-does-not-match).
- **`nsys stats` refuses the report.** After `bench run` has summarized a
  report, `nsys stats` without `--force-export=true` can stop at the SQLite
  export left beside it; every read on this page passes the flag (walkthrough
  11 shows the message).
- **`G1` or `G1Phases` moved by a quarter or more.** They move between two
  bands on this rig (above): down 25% to 31% from the slow band, up 33% to 44%
  from the fast one. The reference run has both in the slow band.

## Check Against the Reference

```bash
bench compare src/bench/demo/reference/thor/13_nvtx_annotation.csv nvtx_annotation.csv
```

```
Test                         Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
-----------------------  ------------  ------------  ----------  --------  --------  --------  ------------
NvtxAnnotation.G1           874.80000     629.47500  -245.32500    -28.0%      0.8%      3.4%  IMPROVEMENT
NvtxAnnotation.G1Phases     908.92500     666.77500  -242.15000    -26.6%      0.2%      0.4%  IMPROVEMENT

  2 improvement(s)  0 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The replay's run landed in the fast band, 28.0% and 26.6% below the reference,
which is in the slow band: both tests read `IMPROVEMENT` with nothing changed.
`bench compare` labels a test `REGRESSION` when its median is more than the
threshold (5% unless `--threshold` says otherwise) above the reference's,
`IMPROVEMENT` when it is more than that below it, and `neutral` otherwise. The
labels describe the difference between two runs and are not a significance
test; the two CV columns are each run's own spread, not the spread between the
runs. Compare a run with the reference only when both landed in the same band
(above).

The reference CSV is one of the ten runs of Step 1 in the first session: the
one whose two medians sit closest to the median of each test across the ten
(874.7 and 908.0 us); across all fifteen runs of both sessions it is the same
run. It was captured from a copy of the development tree without its git
metadata, so its `gitHash` column reads `unknown`.

## What Keeps This Page True

- `NvtxRanges.RecordedByNsightSystems`, a test program of its own beside the
  demo
  ([`05_NvtxAnnotation_Ranges_uTest.cpp`](../gpu/utst/05_NvtxAnnotation_Ranges_uTest.cpp),
  built as `TestDemoNvtxRanges`), runs the demo's two tests under
  `nsys profile -t cuda,nvtx` with `--profile nsight`, eight measured calls and
  one warmup each, and reads the report's push/pop trace and GPU trace with
  `nsys stats`. It fails unless each test has one range of its own; `G1`'s
  holds no range and its eight kernels; `G1Phases`'s holds `copy_in`, `kernel`
  and `copy_out` for each call, in that order, directly inside it, each around
  its own copies or kernel; and the warmup's three phases lie outside both. It
  counts and orders; it compares no durations. It skips, saying why, where
  `nsys` is not on `PATH`, where there is no CUDA device and in a build without
  NVTX headers, all decided before anything runs, and where `nsys` stops before
  the demo's tests start because it cannot create its temporary files, quoting
  `nsys`. Otherwise it fails unless both tests pass under `nsys`, `nsys` exits
  0 and the report is written, and a failure says how the run ended and where
  its files are kept. Beside it, `NsysCsvTest`, `NsysReportTest` (two real
  reports of the demo, from Nsight Systems 2025.3.2 and 2026.3.1),
  `NvtxTraceTest` and `NsysCaptureTest` (stand-ins for `nsys` that exit early,
  are killed or stop on their temporary files) hold its reading and its
  judgement to their cases.
- `TestBenchNsightRange`
  ([`ProfilerNsightRange_uTest.cu`](../../utst/ProfilerNsightRange_uTest.cu))
  holds the range named after the test to its measurement without `nsys`: a
  small library stands in for the NVTX tool and records every push and pop,
  and the test fails unless the range opens after the warmup and closes before
  the measurement returns, for a kernel, a CPU baseline, a multi-GPU and a CPU
  measurement, and for two measurements in one test.
- The example's unit tests hold `G1` to the CPU loop's answers.

All of them are registered with `ctest`, the range tests under the `nsight`
label (the check also under `demo`), so a GPU build's test run includes them:

```bash
ctest --test-dir build -L nsight
```

The demo's two timing tests are not registered: what they measure belongs to
the machine they run on. This repository has no continuous-integration lane on
the reference board, so before a release the page's commands are run on the
rig by hand, and the page and its reference CSV are re-captured when what they
show changes.

## See Also

- [Demo 11: Nsight Systems and Nsight Compute](11_NSIGHT_PROFILER.md) -- the
  same example's CUDA calls and kernels, and `bench run --profile nsight`
- [Demo 10: GPU Basic Workflow](10_GPU_BASIC_WORKFLOW.md) -- the GPU harness and
  the SAXPY example
- [GPU Guide: NVTX annotation API](../../docs/GPU_GUIDE.md#nvtx-annotation-api)
- [Reference Rig: NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
- [Shared Examples](../README.md#shared-examples) -- the SAXPY example and its
  tests
