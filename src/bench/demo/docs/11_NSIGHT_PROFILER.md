# Demo 11: Nsight Systems and Nsight Compute

**Reference rig:** [NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
**Build:** Release
**Example:** `saxpy` (see [Shared Examples](../README.md#shared-examples))
**Captured:** 2026-09-24 (UTC). Written for the Vernier 1.0.4 release;
captured from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`. The two `bench compare` outputs were produced from the
saved CSVs of the runs they show, by the CLI of a later development tree, which
also reports `bench 1.0.3`.

## Overview

Two NVIDIA tools, two questions. Nsight Systems records what happened and
when: every CUDA call, copy and kernel of a run. Nsight Compute takes one
kernel and measures what limits it. On the SAXPY example's two GPU versions,
Nsight Systems shows `G0` allocating and freeing its buffers on every call and
spending most of each call in its kernel; Nsight Compute shows why that kernel
is slow: it runs one thread per block, so 31 of the 32 lanes of every warp sit
idle and at most half of the warps an SM can hold are resident.

## What are Nsight Systems and Nsight Compute?

- **Nsight Systems (`nsys`)** starts the program, records its CUDA calls, the
  copies and kernels on the GPU, and NVTX ranges with their times, and writes a
  report (`.nsys-rep`) when the program exits. Reach for it first:
  it says where a run's time goes. It does not say why a kernel is slow.
- **Nsight Compute (`ncu`)** stops at each kernel launch and replays the
  kernel several times (ten passes per launch here) to read the GPU's
  performance counters: launch shape, occupancy, throughput, and rules that
  name what limits the kernel. The replays make a profiled launch take about a
  second on this rig, so give it a handful of launches, and take the timings
  from a run without it.

**In Vernier:** `--profile nsight` (or its alias `nsys`) and `--profile ncu`.
Neither tool can attach to a process that is already running, so the process
has to start under the tool. `bench run` does that:

- `bench run <binary> --profile nsight` runs the binary under
  `nsys profile -t cuda,nvtx` and writes
  `bench-out/<binary>.nsight/profile.nsys-rep`, the SQLite export next to it,
  and four summaries read from it (`cuda_gpu_kern_sum.txt`, `cuda_api_sum.txt`,
  `cuda_gpu_mem_time_sum.txt`, `cuda_gpu_mem_size_sum.txt`). Each measured test
  gets an NVTX range named after the test, opened when its measurement starts
  and closed when it ends.
- `bench run <binary> --profile ncu` runs it under
  `ncu --target-processes all` and writes
  `bench-out/<binary>.ncu/kernel_profile.ncu-rep`.

A binary started with `--profile nsight` but without a tool around it prints
the command that would capture it and runs uncaptured (see
[If It Does Not Match](#if-it-does-not-match)). A wrap typed by hand, such as
`nsys profile ... <binary> --profile nsight`, works too, but writes only the
report: run `nsys stats` on it yourself.

The GPU harness also has its own CUPTI collector, which fills the CSV's
`cupti*` columns. With it registered, Nsight Systems 2025.3.2 recorded no
kernels on this rig, and under Nsight Compute 2025.3.1 the collector recorded
nothing, so the collector stands down when the process runs inside an Nsight
session: `bench run`'s wrap, or a wrap typed by hand, which it recognizes by
the `NSYS_PROFILING_SESSION_ID` or `NV_NSIGHT_INJECTION_PORT_BASE` variable
those versions put in the program's environment. It says so on stderr, and the
`cupti*` cells of that run are empty:

```
[gpu] in-process CUPTI collection disabled for this run (external Nsight session or VERNIER_DISABLE_CUPTI); CUPTI CSV columns will be empty.
```

A tool version that does not set those variables is not recognized: set
`VERNIER_DISABLE_CUPTI=1` for that run. A `--profile` value alone does not
turn the collector off, and `VERNIER_DISABLE_CUPTI=0` or `false` does not turn
it back on inside a session.

**Needs:** the CUDA toolkit on `PATH`, with `nsys` and `ncu`, and a GPU build:
see the rig's [setup](../../docs/rigs/RIG_THOR_AGX.md#2-one-time-setup) and
[build](../../docs/rigs/RIG_THOR_AGX.md#3-build). On this rig the GPU's
performance counters are for administrators only, so `ncu` runs under `sudo`.

## The Example

SAXPY, `y = a*x + y` over 1,048,576 floats, driven two ways
([`SaxpyGpu.cpp`](../examples/saxpy/src/SaxpyGpu.cpp)). `G0` does everything
on every call:

```cpp
void saxpyG0(float a, const std::vector<float>& x, std::vector<float>& y) {
  const std::size_t N = x.size();
  const std::size_t BYTES = N * sizeof(float);

  const DeviceBuffer D_X(BYTES, "cudaMalloc x"); // every call
  const DeviceBuffer D_Y(BYTES, "cudaMalloc y");
  // Pageable host memory: the copy cannot overlap anything and blocks here.
  check(cudaMemcpy(D_X.get(), x.data(), BYTES, cudaMemcpyHostToDevice), "HtoD x");
  check(cudaMemcpy(D_Y.get(), y.data(), BYTES, cudaMemcpyHostToDevice), "HtoD y");
  launchSaxpy(a, D_X.get(), D_Y.get(), N, /*threadsPerBlock=*/1, nullptr); // one thread per block
  check(cudaGetLastError(), "launch");
  check(cudaMemcpy(y.data(), D_Y.get(), BYTES, cudaMemcpyDeviceToHost), "DtoH y");
} // both buffers are freed here, every call, however the function leaves
```

`G1` (`SaxpyG1`) allocates two device buffers, two pinned host buffers and a
stream once, in its constructor; a call copies the vectors into the pinned
buffers, queues two copies, the launch at 256 threads per block and the copy
back, and waits once:

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

Both launch the same kernel, one element per thread
([`SaxpyKernel.cu`](../examples/saxpy/src/SaxpyKernel.cu)). The example's unit
tests hold both versions to the CPU loop's answers, and `TestDemoSaxpyDriving`
checks, without a GPU, the CUDA calls each version makes: two allocations and
two frees per `G0` call, five acquisitions once for `G1`, three copies per
call for both, and one thread per block against 256.

The demo ([`02_NsightProfiler_Demo.cu`](../gpu/02_NsightProfiler_Demo.cu)) has
five tests. `G0` and `G1` time one call of each version end to end.
`KernelOneThreadPerBlock` and `Kernel256ThreadsPerBlock` time the bare kernel
in `G0`'s and in `G1`'s launch shape with the GPU harness, and each fails when
the harness's occupancy estimate stops matching its shape: at most 0.5 for one
thread per block, at least 0.75 for 256. `LaunchShapeSpeedup` times both
shapes itself with CUDA events, writes no CSV row, and fails unless one thread
per block is more than 30 times slower. It compares timings, so it stays out of
a run under Nsight Compute. `G0` against `G1` end to end is not asserted: on a
discrete GPU, where the copies dominate, the two take about as long.

## Step 1: Measure

Inside the rig document's
[measurement procedure](../../docs/rigs/RIG_THOR_AGX.md#4-running-a-measurement)
(clocks locked with `jetson_clocks` and restored afterwards, the host thread on
core 13), from the source tree, after the rig's build:

```bash
taskset -c 13 ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --cycles 20 --repeats 10 --csv nsight_profiler.csv
```

`G0` and the one-thread kernel take milliseconds per call, so give the run a
cycle count; the default of 10,000 cycles per repeat would keep each of them
busy for minutes. Captured output (the reference run):

```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from NsightProfiler
[ RUN      ] NsightProfiler.G0
[NsightProfiler.G0]  3640.375 us/call  CV=2.7%  ~275 calls/s  (p10=3430.315 p90=3696.925 sd=98.536)
[       OK ] NsightProfiler.G0 (872 ms)
[ RUN      ] NsightProfiler.G1
[NsightProfiler.G1]  653.250 us/call  CV=2.1%  ~1.5K calls/s  (p10=650.635 p90=665.635 sd=13.597)
[       OK ] NsightProfiler.G1 (141 ms)
[ RUN      ] NsightProfiler.KernelOneThreadPerBlock
[NsightProfiler.KernelOneThreadPerBlock]  2993.558 us/call  CV=0.0%  ~334 calls/s  (p10=2993.350 p90=2993.813 sd=0.188)
[       OK ] NsightProfiler.KernelOneThreadPerBlock (659 ms)
[ RUN      ] NsightProfiler.Kernel256ThreadsPerBlock
[NsightProfiler.Kernel256ThreadsPerBlock]  20.174 us/call  CV=0.7%  ~49.6K calls/s  (p10=20.126 p90=20.296 sd=0.151)
[       OK ] NsightProfiler.Kernel256ThreadsPerBlock (24 ms)
[ RUN      ] NsightProfiler.LaunchShapeSpeedup
[NsightProfiler.LaunchShapeSpeedup]  1 thread/block 2993.203 us/launch  256 threads/block 20.922 us/launch  143.1x
[       OK ] NsightProfiler.LaunchShapeSpeedup (81 ms)
[----------] 5 tests from NsightProfiler (1779 ms total)
...
[  PASSED  ] 5 tests.
```

What to read: `G0` takes 3.64 ms a call and `G1` 0.65 ms, 5.6 times less. The
kernel alone takes 2,994 us in `G0`'s launch shape and 20.2 us in `G1`'s, and
`LaunchShapeSpeedup`, timing both shapes the same way, reads 143.1x. So most of
`G0`'s call is its kernel. The CSV has the harness's view of the two kernel
rows:

```bash
awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; print "test wallMedian kernelTimeUs occupancy cuptiKernelLaunches"; next }
         { print $col["test"], $col["wallMedian"], $col["kernelTimeUs"], $col["occupancy"], $col["cuptiKernelLaunches"] }' \
  nsight_profiler.csv | column -t
```

```
test                                     wallMedian  kernelTimeUs  occupancy  cuptiKernelLaunches
NsightProfiler.G0                        3640.38
NsightProfiler.G1                        653.25
NsightProfiler.KernelOneThreadPerBlock   2993.56     2993.557644   0.500000   200
NsightProfiler.Kernel256ThreadsPerBlock  20.1736     20.173600     1.000000   200
```

`occupancy` is the harness's estimate from the launch shape and the device's
limits: 0.5 for one thread per block, 1.0 for 256. `G0` and `G1` are timed as
plain calls, so their GPU cells are empty. The CUPTI collector counted the 200
measured launches of each kernel row (20 cycles times 10 repeats).

## Step 2: What Happened, and When (Nsight Systems)

With the clocks still locked, run `G0` under Nsight Systems:

```bash
source build/.env
bench run ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile nsight -- \
  --gtest_filter=NsightProfiler.G0 --cycles 20 --repeats 3
```

```
Running: nsys profile -o bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile -t cuda,nvtx --force-overwrite true ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile nsight --gtest_filter=NsightProfiler.G0 --cycles 20 --repeats 3
...
[ RUN      ] NsightProfiler.G0
[nsight] this process runs under nsys, which writes the report when the process exits.
[NsightProfiler.G0]  3367.300 us/call  CV=0.2%  ~297 calls/s  (p10=3366.940 p90=3376.380 sd=5.460)
[       OK ] NsightProfiler.G0 (352 ms)
...
Generated:
    .../bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.nsys-rep
[nsight] auto-extracted nsys stats reports into bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight
```

The report and the four summaries are in
`bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/`. The capture cost little
here: `G0` read 3,367 us a call under `nsys`, against 3,438 to 3,664 us in
Step 1's runs. The run made 61 calls, one warmup and 60 measured.
`cuda_api_sum.txt`, the CUDA calls:

```
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)  Min (ns)   Max (ns)   StdDev (ns)           Name
 --------  ---------------  ---------  -----------  --------  --------  ----------  -----------  ----------------------
     63.9      184,694,694        183  1,009,260.6  56,824.0    38,324   3,335,648  1,361,768.7  cudaMemcpy
     32.1       92,759,919        122    760,327.2  77,130.0    62,880  83,080,798  7,514,896.6  cudaMalloc
      3.9       11,151,974        122     91,409.6  91,903.0    47,528     854,676    108,588.1  cudaFree
      0.1          370,961         61      6,081.3   4,722.0     4,269      72,648      8,790.7  cudaLaunchKernel
      0.0           38,814          1     38,814.0  38,814.0    38,814      38,814          0.0  cuLibraryLoadData
      0.0            8,775         61        143.9     130.0        92         333         37.2  cuKernelGetName
      0.0              778          1        778.0     778.0       778         778          0.0  cuModuleGetLoadingMode
      0.0              481          1        481.0     481.0       481         481          0.0  cuLibraryGetKernel
```

`cuda_gpu_kern_sum.txt`, the kernels as the GPU ran them:

```
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                                             Name
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ------------------------------------------------------------------------------------------
    100.0      176,629,184         61  2,895,560.4  2,889,504.0  2,867,328  3,269,536     49,815.5  vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *, unsigned long)
```

`cuda_gpu_mem_time_sum.txt`, the copies as the GPU ran them:

```
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     74.6        2,358,240    122  19,329.8  19,792.0    17,248    40,480      3,281.2  [CUDA memcpy Host-to-Device]
     25.4          802,592     61  13,157.2  12,864.0    11,680    16,000      1,417.9  [CUDA memcpy Device-to-Host]
```

The test's NVTX range is in the report too. `nsys stats` reads any report;
give it `--force-export=true` for one that `bench run` has summarized (see
[If It Does Not Match](#if-it-does-not-match)):

```bash
nsys stats --force-export=true --report nvtx_sum bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.nsys-rep
```

```
 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)   Style         Range
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  -------  ------------------
    100.0      210,950,836          1  210,950,836.0  210,950,836.0  210,950,836  210,950,836          0.0  PushPop  :NsightProfiler.G0
```

## Step 3: Read the Report

- **The calls per `G0` call.** 122 `cudaMalloc` and 122 `cudaFree` over 61
  calls: two allocations and two frees on every call, 77 us and 92 us each by
  their medians. The longest `cudaMalloc`, 83 ms, is the process's first CUDA
  runtime call, which pays for setting CUDA up; the medians are the per-call
  cost. 183 `cudaMemcpy`, three per call, each 4.19 MB
  (`cuda_gpu_mem_size_sum.txt`).
- **Where a call's time goes.** The kernel averages 2.90 ms on the GPU, 86% of
  the 3.37 ms call. The two allocations and two frees take about 0.34 ms by
  their medians, the copies about 52 us on the GPU. The synchronous
  `cudaMemcpy` back to the host waits for the kernel, which is why its average
  (1.01 ms) is far above its median (57 us): read the kernel's time from the
  kernel summary, not from the call that waited for it.
- **The range.** The 210.95 ms range `NsightProfiler.G0` is the measured
  window: on the timeline it holds 60 of the 61 kernels, 120 `cudaMalloc` and
  120 `cudaFree`; the warmup call comes before it.

So `G0` spends its time in the kernel, and the allocations are the next cost.
Nsight Systems does not say why the kernel takes 2.9 ms to add two vectors.
That is Nsight Compute's question.

## Step 4: What Limits the Kernel (Nsight Compute)

Profile the two kernel tests, which launch the bare kernel in each shape.
Nsight Compute replays every launch, so keep the launches few with
`--cycles 3 --repeats 1`, and filter to the two tests, which leaves
`LaunchShapeSpeedup` out. As root on this rig, with the clocks locked, then
give the output back to your user:

```bash
sudo env PATH="$PATH" bench run ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile ncu --cycles 3 --repeats 1 -- \
  --gtest_filter='NsightProfiler.Kernel*'
sudo chown -R "$(id -u):$(id -g)" bench-out
```

```
Running: ncu -o bench-out/BenchDemo_Gpu_02_NsightProfiler.ncu/kernel_profile -f --target-processes all ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --cycles 3 --repeats 1 --profile ncu --gtest_filter=NsightProfiler.Kernel*
...
[ RUN      ] NsightProfiler.KernelOneThreadPerBlock
==PROF== Connected to process 86816 (.../build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler)
[nsight] this process runs under ncu, which writes the report when the process exits.
==WARNING== Unable to access the following 8 metrics: mcc__cycles_active.avg, mcc__cycles_active.max, mcc__cycles_active.min, mcc__cycles_active.sum, mcc__cycles_elapsed.avg, mcc__cycles_elapsed.max, mcc__cycles_elapsed.min, mcc__cycles_elapsed.sum.
...
==PROF== Profiling "saxpyKernel" - 0: 0%....50%....100% - 10 passes
...
[gpu] in-process CUPTI collection disabled for this run (external Nsight session or VERNIER_DISABLE_CUPTI); CUPTI CSV columns will be empty.
...
[NsightProfiler.KernelOneThreadPerBlock]  910898.844 us/call  CV=0.0%  ~1 calls/s  (p10=910898.844 p90=910898.844 sd=0.000)
[       OK ] NsightProfiler.KernelOneThreadPerBlock (17345 ms)
...
[       OK ] NsightProfiler.Kernel256ThreadsPerBlock (12257 ms)
...
==PROF== Report: .../bench-out/BenchDemo_Gpu_02_NsightProfiler.ncu/kernel_profile.ncu-rep
```

The run took 30 seconds for 28 profiled launches, 14 of each shape (the
harness's warmup launches and the three measured ones). The times it prints
are `ncu`'s replays, not the kernel's: the timings are Step 1's. The eight
`mcc__` metrics are not readable on this rig; the sections below do not use
them. Read the report with the same `ncu`:

```bash
ncu --import bench-out/BenchDemo_Gpu_02_NsightProfiler.ncu/kernel_profile.ncu-rep --print-summary per-kernel
```

The launch statistics and occupancy of each shape, minimum, maximum and
average over its 14 launches:

```
  unnamed>::saxpyKernel(float, const float *, float *, unsigned long) (4096, 1, 1)x(256, 1, 1), Device 0, CC 11.0, Invocations 14
    ...
    Section: Launch Statistics
    -------------------------------- --------------- ------------ ------------ ------------
    Metric Name                          Metric Unit      Minimum      Maximum      Average
    -------------------------------- --------------- ------------ ------------ ------------
    Block Size                                             256.00       256.00       256.00
    Cluster Size                                             0.00         0.00         0.00
    Grid Size                                            4,096.00     4,096.00     4,096.00
    ...
    Waves Per SM                                            34.13        34.13        34.13
    -------------------------------- --------------- ------------ ------------ ------------

    Section: Occupancy
    ------------------------------- ----------- ------- ------- -------
    Metric Name                     Metric Unit Minimum Maximum Average
    ------------------------------- ----------- ------- ------- -------
    ...
    Block Limit Barriers                  block   24.00   24.00   24.00
    Block Limit SM                        block   24.00   24.00   24.00
    Block Limit Registers                 block   16.00   16.00   16.00
    Block Limit Shared Mem                block   16.00   16.00   16.00
    Block Limit Warps                     block    6.00    6.00    6.00
    Theoretical Active Warps per SM        warp   48.00   48.00   48.00
    Theoretical Occupancy                     %  100.00  100.00  100.00
    Achieved Occupancy                        %   74.45   77.70   76.66
    Achieved Active Warps Per SM           warp   35.74   37.30   36.80
    ------------------------------- ----------- ------- ------- -------

  unnamed>::saxpyKernel(float, const float *, float *, unsigned long) (1048576, 1, 1)x(1, 1, 1), Device 0, CC 11.0, Invocations 14
    ...
    Section: Launch Statistics
    -------------------------------- --------------- ------------ ------------ ------------
    Metric Name                          Metric Unit      Minimum      Maximum      Average
    -------------------------------- --------------- ------------ ------------ ------------
    Block Size                                               1.00         1.00         1.00
    Cluster Size                                             0.00         0.00         0.00
    Grid Size                                        1,048,576.00 1,048,576.00 1,048,576.00
    ...
    Waves Per SM                                         2,184.53     2,184.53     2,184.53
    -------------------------------- --------------- ------------ ------------ ------------

    Section: Occupancy
    ------------------------------- ----------- ------- ------- -------
    Metric Name                     Metric Unit Minimum Maximum Average
    ------------------------------- ----------- ------- ------- -------
    ...
    Block Limit Barriers                  block   24.00   24.00   24.00
    Block Limit SM                        block   24.00   24.00   24.00
    Block Limit Registers                 block  128.00  128.00  128.00
    Block Limit Shared Mem                block   32.00   32.00   32.00
    Block Limit Warps                     block   48.00   48.00   48.00
    Theoretical Active Warps per SM        warp   24.00   24.00   24.00
    Theoretical Occupancy                     %   50.00   50.00   50.00
    Achieved Occupancy                        %   29.02   31.00   29.84
    Achieved Active Warps Per SM           warp   13.93   14.88   14.33
    ------------------------------- ----------- ------- ------- -------
```

`--page details` prints every launch with the rules `ncu` applies to it. For
the first launch in `G0`'s shape:

```bash
ncu --import bench-out/BenchDemo_Gpu_02_NsightProfiler.ncu/kernel_profile.ncu-rep --page details
```

```
    OPT   Est. Speedup: 96.88%
          Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 1
          threads per block. Consequently, some threads in a warp are masked off and those hardware resources are
          unused. Try changing the number of threads per block to be a multiple of 32 threads. Between 128 and 256
          threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one
          large thread block per multiprocessor if latency affects performance.  This is particularly beneficial to
          kernels that frequently call __syncthreads(). See the Hardware Model
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more
          details on launch configurations.
...
    OPT   Est. Local Speedup: 50%
          The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
          hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the number of blocks that
          can fit on the SM.
```

What it says:

- **The launch shape.** `G0` launches 1,048,576 blocks of one thread,
  `(1048576, 1, 1)x(1, 1, 1)`; `G1` launches 4,096 blocks of 256,
  `(4096, 1, 1)x(256, 1, 1)`. The same million threads either way.
- **Idle lanes.** A warp is 32 threads and a block of one thread still takes
  one warp, so 31 of its 32 lanes do nothing. The first rule says so, and
  estimates what a multiple of 32 threads per block would save (96.88%, which
  is 31/32).
- **Occupancy.** An SM here holds at most 24 blocks (`Block Limit SM`) and 48
  warps. One-warp blocks fill 24 of the 48 warp slots: theoretical occupancy
  50%, and 29.02% to 31.00% achieved. Blocks of 256 threads are eight warps,
  six blocks fill all 48 slots: 100% theoretical, 74.45% to 77.70% achieved.
  The harness's `occupancy` column in Step 1 is the theoretical figure (0.5
  and 1.0); the achieved figure is `ncu`'s measurement.

The fix is `G1`'s launch shape: in Step 1 the same kernel took 20.2 us at 256
threads per block against 2,994 us at one.

## Step 5: Confirm the Fix (Nsight Systems on G1)

The same command as Step 2, on `G1`:

```bash
bench run ./build/bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile nsight -- \
  --gtest_filter=NsightProfiler.G1 --cycles 20 --repeats 3
```

```
[NsightProfiler.G1]  656.850 us/call  CV=0.2%  ~1.5K calls/s  (p10=655.570 p90=658.130 sd=1.306)
```

`cuda_api_sum.txt`:

```
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)    Max (ns)   StdDev (ns)            Name
 --------  ---------------  ---------  ------------  ------------  ---------  ----------  ------------  ----------------------
     87.0       87,289,536          2  43,644,768.0  43,644,768.0     79,334  87,210,202  61,610,827.6  cudaMalloc
      4.1        4,164,057         61      68,263.2      68,296.0     54,991      94,833       6,041.0  cudaStreamSynchronize
      3.4        3,366,675          2   1,683,337.5   1,683,337.5  1,617,564   1,749,111      93,017.8  cudaHostAlloc
      2.7        2,663,082          2   1,331,541.0   1,331,541.0  1,222,805   1,440,277     153,775.9  cudaFreeHost
      1.5        1,511,605        183       8,260.1       5,463.0      3,732      28,935       4,690.7  cudaMemcpyAsync
      0.9          858,741         61      14,077.7       9,843.0      6,574     267,222      32,973.8  cudaLaunchKernel
      0.3          277,388          2     138,694.0     138,694.0     65,796     211,592     103,093.3  cudaFree
      0.1          106,722          1     106,722.0     106,722.0    106,722     106,722           0.0  cudaStreamCreate
      0.0           44,139          1      44,139.0      44,139.0     44,139      44,139           0.0  cuLibraryLoadData
      0.0           39,501         61         647.6         667.0        167       1,250         248.5  cuKernelGetName
      0.0           24,851          1      24,851.0      24,851.0     24,851      24,851           0.0  cudaStreamDestroy
      0.0            1,139          1       1,139.0       1,139.0      1,139       1,139           0.0  cuModuleGetLoadingMode
      0.0              426          1         426.0         426.0        426         426           0.0  cuLibraryGetKernel
```

`cuda_gpu_kern_sum.txt`:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                             Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ------------------------------------------------------------------------------------------
    100.0        1,392,096         61  22,821.2  22,464.0    22,368    40,224      2,305.2  vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *, unsigned long)
```

`cuda_gpu_mem_time_sum.txt`:

```
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     68.1        2,008,544    122  16,463.5  15,712.0    15,392    30,048      1,952.8  [CUDA memcpy Host-to-Device]
     31.9          939,616     61  15,403.5  15,360.0    15,232    17,344        279.8  [CUDA memcpy Device-to-Host]
```

- **Allocations moved out of the call.** Two `cudaMalloc`, two
  `cudaHostAlloc` and one stream for the whole run, made by the constructor
  before the measurement; the range `NsightProfiler.G1` (47.71 ms) holds the
  60 measured calls and none of them. Per call: three `cudaMemcpyAsync`, one
  launch, one `cudaStreamSynchronize`.
- **The kernel.** 22.8 us on average, against 2.90 ms in `G0`.
- **What is left.** The GPU works about 71 us of each 657 us call: the kernel
  and three copies of about 16 us each. The CUDA calls add about 110 us on the
  host, most of it the 68 us wait. The remaining time is outside any CUDA call:
  `G1`'s three `memcpy` into and out of its pinned buffers. `-t cuda,nvtx`
  does not trace those; on the timeline they are the gaps between one call's
  wait and the next call's first copy.

**One reading moves between runs on this rig.** In some runs a `G1` call waits
about 280 us in `cudaStreamSynchronize` instead of about 68 us: on the
timeline, the GPU starts each call's first copy about 217 us after the host
queued it, instead of about 16 us, while the copies and the kernel take as long
as ever. `G1` then reads about 0.9 ms a call instead of about 0.65 ms (see
[What Should Reproduce](#what-should-reproduce)).

## The Reports as CSV

Vernier's `nsight-parse` is meant to turn these reports into one CSV. It is
one of Vernier's Python tools, which a build puts in `build/bin/tools/py` when
it finds Poetry and pip; this rig's build has no Poetry, so for this page the
wheel built from the same tree was installed with `pip3 install --target` and
put on `PATH`. In this release it does not read what `bench run` leaves
reliably:

- **Step 2's folder.** It runs `nsys stats` without `--force-export=true`, so
  it writes no rows whenever `nsys` refuses the SQLite export beside the report
  (see [If It Does Not Match](#if-it-does-not-match)). Run after each of eight
  runs of Step 2's command, it wrote no rows four times:

  ```bash
  nsight-parse parse bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/ --csv nsys_summaries.csv
  ```

  ```
  [nsight-parse] nsys stats --report cuda_gpu_kern_sum failed for profile.nsys-rep
  [nsight-parse] nsys stats --report cuda_api_sum failed for profile.nsys-rep
  [nsight-parse] nsys stats --report cuda_gpu_mem_size_sum failed for profile.nsys-rep
  [nsight-parse] nsys stats --report cuda_gpu_mem_time_sum failed for profile.nsys-rep
  [nsight-parse] wrote 0 rows to nsys_summaries.csv
  ```

  and 13 rows the other four, one per row of the four summaries.

- **Step 4's report.** It runs `ncu` without `--import`, so `ncu` takes the
  report for a program to launch ("The target application is not an
  executable binary", when that command is run by hand), and the parse writes
  no rows:

  ```bash
  nsight-parse parse bench-out/BenchDemo_Gpu_02_NsightProfiler.ncu/ --csv ncu_summary.csv
  ```

  ```
  [nsight-parse] ncu --csv failed for kernel_profile.ncu-rep
  [nsight-parse] wrote 0 rows to ncu_summary.csv
  ```

Both tools print CSV themselves, so until `nsight-parse` reads these reports,
ask them. The output below is from a second run of Steps 2 and 4 with the
clocks as found, so its numbers differ a little from those steps:

```bash
nsys stats --force-export=true --format csv --report cuda_gpu_kern_sum bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.nsys-rep
```

```
Generating SQLite file bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.sqlite from bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.nsys-rep
Processing [bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.sqlite] with [/opt/nvidia/nsight-systems/2025.3.2/host-linux-armv8/reports/cuda_gpu_kern_sum.py]...
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
100.0,179106560,61,2936173.1,2889824.0,2864096,3829824,169012.8,"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *, unsigned long)"
```

The other summaries come the same way (`--report cuda_api_sum`, and so on).
Nsight Compute prints one row per launch shape, section and metric, 88 rows
after the header here:

```bash
ncu --import bench-out/BenchDemo_Gpu_02_NsightProfiler.ncu/kernel_profile.ncu-rep --csv --print-summary per-kernel
```

```
"Process ID","Process Name","Host Name","Kernel Name","Block Size","Grid Size","Device","CC","Invocations","Section Name","Metric Name","Metric Unit","Minimum","Maximum","Average"
...
"109199","BenchDemo_Gpu_02_NsightProfiler","127.0.0.1","unnamed>::saxpyKernel(float, const float *, float *, unsigned long)","(256, 1, 1)","(4096, 1, 1)","0","11.0","14","Occupancy","Theoretical Occupancy","%","100.00","100.00","100.00"
"109199","BenchDemo_Gpu_02_NsightProfiler","127.0.0.1","unnamed>::saxpyKernel(float, const float *, float *, unsigned long)","(256, 1, 1)","(4096, 1, 1)","0","11.0","14","Occupancy","Achieved Occupancy","%","73.56","79.38","74.92"
...
"109199","BenchDemo_Gpu_02_NsightProfiler","127.0.0.1","unnamed>::saxpyKernel(float, const float *, float *, unsigned long)","(1, 1, 1)","(1048576, 1, 1)","0","11.0","14","Occupancy","Theoretical Occupancy","%","50.00","50.00","50.00"
"109199","BenchDemo_Gpu_02_NsightProfiler","127.0.0.1","unnamed>::saxpyKernel(float, const float *, float *, unsigned long)","(1, 1, 1)","(1048576, 1, 1)","0","11.0","14","Occupancy","Achieved Occupancy","%","27.46","30.08","28.52"
...
```

`nsight-parse`'s CSV, when it has rows, is not a benchmark CSV: the benchmark
tools need `test`, `wallMedian`, `wallCV` and `callsPerSecond` columns.
`bench summary` and `bench compare` refuse it (`missing required column
'test'`), and `bench-plot` stops with `Missing required columns`.

## What Should Reproduce

| Reading                                       | On this rig                                                                                                  | Elsewhere                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `G0`'s CUDA calls (Nsight Systems)            | two `cudaMalloc`, two `cudaFree`, three `cudaMemcpy` and one launch per call                                 | the same: they are the code's, and `TestDemoSaxpyDriving` checks them without a GPU                              |
| `G1`'s CUDA calls                             | two `cudaMalloc`, two `cudaHostAlloc`, one stream once; three `cudaMemcpyAsync`, one launch, one wait a call | the same                                                                                                         |
| launch shapes (Nsight Compute)                | `(1048576, 1, 1)x(1, 1, 1)` and `(4096, 1, 1)x(256, 1, 1)`                                                   | the same                                                                                                         |
| theoretical occupancy (ncu; the CSV's column) | 50% and 100% (0.5 and 1.0)                                                                                   | depends on the GPU's limits; both occupancy checks also pass on an RTX 5000 Ada laptop GPU                       |
| achieved occupancy (ncu)                      | 29.02% to 31.00% and 74.45% to 77.70% over 14 launches each                                                  | depends on the GPU                                                                                               |
| kernel alone, one thread vs 256 per block     | 143.1x; 141.7x to 143.3x over six runs, clocks locked; 138.7x to 138.9x over four as found                   | one thread per block stays far slower: 94.9x to 96.0x on the RTX 5000 Ada (three runs)                           |
| `G0` vs `G1` end to end                       | 5.6x; 4.1x to 5.7x over six runs, clocks locked (below)                                                      | depends on what the copies cost: 1.1x on the RTX 5000 Ada, where copies over PCIe dominate both (walkthrough 10) |
| absolute times                                | 3,640 / 653 / 2,994 / 20.2 us                                                                                | will differ                                                                                                      |

The demo fails if one thread per block stops being 30 times slower than 256,
or if the occupancy estimate stops being at most 0.5 for one thread per block
and at least 0.75 for 256.

**The end-to-end times sit in two bands on this rig.** Over the six runs of
Step 1's command with the clocks locked, `G0` read about 3.44 ms twice
(3,438 and 3,447 us) and 3.64 to 3.66 ms four times; `G1` read 634 to 653 us
four times and 876 to 888 us twice; the kernel rows did not move. In 40 more
runs of `G0` and `G1`, 3 s apart with the clocks read between runs, `G1`'s
median was 629 to 654 us in 33, 875 to 929 us in 6 and 695 us in one; in six of
the 40 it changed band part-way through, which shows as a CV above 10%. Core
13's clock, the GPU clocks and the memory clock read the same before every one
of those runs. Run on its own, `G1` landed in the slower band more often than
not: in 23 of 30 runs of it alone, 3 s apart, and in 8 of 10 captures of it
alone under Nsight Systems, where the timeline showed the late start described
in Step 5. The two bands are this rig's; a discrete GPU was not checked for
them.

The RTX 5000 Ada figures come from three runs of the demo with Step 1's
settings (`--cycles 20 --repeats 10`) in the project's `dev-cuda` container,
with the clocks as found and the laptop busy with other work, so that only the
GPU-side figures are exact: the kernel took 622 us per launch in one-thread
blocks and 6.5 us in 256-thread ones; `G0` took 2.62 to 2.63 ms a call and
`G1` 2.32 to 2.34 ms.

## If It Does Not Match

- **The run was not captured.** `--profile nsight` without `nsys` around the
  process captures nothing; the test still runs and passes, and the backend
  prints the command that would capture it:

  ```
  [nsight] No nsys session: nsys cannot attach to a running process, so this
  [nsight] run is not captured. Start the binary under nsys:
  [nsight]   nsys profile -o ./NsightProfiler.G0.nsight/profile -t cuda,nvtx --force-overwrite true \
  [nsight]       <this-binary> --profile nsight [...]
  [nsight] or let bench run start it, which also writes the summary reports:
  [nsight]   bench run <this-binary> --profile nsight -- [...]
  ```

  Start it under `bench run` as Steps 2 and 5 do, or under the printed
  command. `--profile ncu` without `ncu` prints the same for `ncu`.

- **`ncu` has no permission for the counters.** Run without `sudo`, the tests
  pass but no kernel is profiled, and `bench run` exits non-zero:

  ```
  ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
  ...
  [  PASSED  ] 1 test.
  ==PROF== Disconnected from process 87873
  Error: parse error: benchmark exited with code 1
  ```

  Run it as Step 4 does, with `sudo env PATH="$PATH"`.

- **`ncu` runs for hours.** It replays every launch: about a second each here.
  Without `--cycles 3 --repeats 1`, the kernel tests' 10,000 cycles per repeat
  would each take hours; keep the filter too, since `LaunchShapeSpeedup`
  compares timings that `ncu`'s replay distorts.

- **`nsys stats` refuses the report.** After `bench run` has summarized a
  report, `nsys stats` without `--force-export=true` can stop at the SQLite
  export left beside it. After six runs of Step 2's command it stopped four
  times:

  ```
  WARNING: Existing SQLite export found: bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.sqlite
           File is older than input file: bench-out/BenchDemo_Gpu_02_NsightProfiler.nsight/profile.nsys-rep
           Use --force-export=true to update export file.
  ...
  ```

- **The report has no kernels.** `nsys stats` says a report "does not contain
  CUDA kernel data" when the harness's CUPTI collector stayed registered inside
  the capture of a kernel test (`G0` and `G1` do not use the collector). The harness recognizes a session only by the variables named in
  [What are Nsight Systems and Nsight Compute?](#what-are-nsight-systems-and-nsight-compute);
  with a tool version that does not set them, run again with
  `VERNIER_DISABLE_CUPTI=1`.

- **`G0` or `G1` moved by 5% or more with the clocks locked.** They sit in two
  bands on this rig (above). A run that lands in the other band from the
  reference's is flagged; the kernel rows are not. With the clocks locked, a
  later run against the reference:

  ```
  Test                                         Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
  ---------------------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
  NsightProfiler.G0                          3640.38000    3446.97000  -193.41000     -5.3%      2.7%      0.9%  IMPROVEMENT
  NsightProfiler.G1                           653.25000     648.10000    -5.15000     -0.8%      2.1%      0.5%  neutral
  NsightProfiler.Kernel256ThreadsPerBlock      20.17360      20.28960    +0.11600     +0.6%      0.7%      5.9%  neutral
  NsightProfiler.KernelOneThreadPerBlock     2993.56000    2990.93000    -2.63000     -0.1%      0.0%      0.1%  neutral

    1 improvement(s)  3 neutral

    Labels compare the median change against the 5.0% threshold.
    They describe the difference between two runs, not a significance test;
    the CV of each run is its own spread, not the spread between the runs.
  ```

- **The clocks were not locked.** Over four runs with the clocks as found,
  `G1` read 782 to 807 us a call with a CV near 10%, and the one-thread kernel
  2,922 to 2,928 us. One of them against the reference:

  ```
  Test                                         Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
  ---------------------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
  NsightProfiler.G0                          3640.38000    3710.35000   +69.97000     +1.9%      2.7%      1.8%  neutral
  NsightProfiler.G1                           653.25000     794.10000  +140.85000    +21.6%      2.1%      9.4%  REGRESSION
  NsightProfiler.Kernel256ThreadsPerBlock      20.17360      21.04720    +0.87360     +4.3%      0.7%      5.1%  neutral
  NsightProfiler.KernelOneThreadPerBlock     2993.56000    2927.93000   -65.63000     -2.2%      0.0%      0.0%  neutral

    1 regression(s)  3 neutral

    Labels compare the median change against the 5.0% threshold.
    They describe the difference between two runs, not a significance test;
    the CV of each run is its own spread, not the spread between the runs.
  ```

  Lock them as the rig document says and run again.

## Check Against the Reference

```bash
bench compare src/bench/demo/reference/thor/11_nsight_profiler.csv nsight_profiler.csv
```

`bench compare` labels a test `REGRESSION` when its median is more than the
threshold (5% unless `--threshold` says otherwise) above the reference's,
`IMPROVEMENT` when it is more than that below it, and `neutral` otherwise. The
labels describe the difference between two runs and are not a significance
test; the two CV columns are each run's own spread, not the spread between the
runs. On this rig the kernel rows are the ones to hold to the reference; `G0`
and `G1` can land in the other band (above).

The reference CSV is one of five runs of Step 1's command captured with the
clocks locked: the one whose four medians sit closest to the median of each
test across the five. It was captured from a copy of the development tree
without its git metadata, so its `gitHash` column reads `unknown`.

## See Also

- [Demo 10: GPU Basic Workflow](10_GPU_BASIC_WORKFLOW.md) -- the GPU harness and the SAXPY example
- [GPU Guide](../../docs/GPU_GUIDE.md) -- the GPU harness in full
- [Reference Rig: NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
- [Shared Examples](../README.md#shared-examples) -- the SAXPY example and its tests
