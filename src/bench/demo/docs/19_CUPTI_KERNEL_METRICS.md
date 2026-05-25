# Demo 19: CUPTI In-Process Kernel Metrics

## Overview

CUPTI (CUDA Profiling Tools Interface) is the C API that Nsight uses
under the hood. The vernier harness uses the CUPTI Activity API to
capture per-kernel metrics directly inside the benchmark process --
register count, static / dynamic shared memory, kernel name, launch
geometry -- without spawning ncu as an external program.

There's no dedicated demo binary. CUPTI fires for **any** GPU benchmark
in the harness. The metrics surface automatically in the CSV columns
when at least one kernel launch is captured during a measured window.

## What is CUPTI?

CUPTI is the C library NVIDIA's own profilers (Nsight Systems, Nsight
Compute, `ncu`, `nsys`) sit on top of. It exposes two API surfaces:

- **Activity API** -- post-event records of kernel launches, memory
  copies, API calls. Cheap, in-process, no replay needed. vernier uses
  this surface.
- **Profiler API** -- per-counter sampling that replays kernels to
  collect deep metrics like achieved occupancy or warp efficiency. This
  is what `ncu` uses; vernier deliberately doesn't (replay breaks the
  single-launch measurement model).

- **Best for:** capturing per-launch register count, shared-memory
  allocation, and launch geometry alongside vernier's own timing
  numbers -- without spawning `ncu` as an external process (fragile in
  Docker PID namespaces).
- **How it works:** the harness registers a CUPTI activity callback
  before the measured window and flushes records after. No replay, no
  extra process.
- **Overhead:** negligible (~1% per launch).
- **Skip it for:** deep per-launch counters like achieved occupancy or
  cache hit rates -- those need `ncu` and kernel replay.

**In vernier:** no flag needed. CUPTI fires on every GPU benchmark
when libcupti was linked at build time; the metrics appear in the GPU
section of the CSV (see columns below).

## CSV Columns

| Column | Source | Meaning |
|---|---|---|
| `cuptiKernelLaunches` | `CUpti_ActivityKernel9.kind` | Number of kernel launches in this measure() window |
| `cuptiRegistersMedian` | `.registersPerThread` | Median registers per thread across launches |
| `cuptiRegistersMax` | `.registersPerThread` | Max registers per thread observed |
| `cuptiStaticSmemBytes` | `.staticSharedMemory` | Median static __shared__ allocation |
| `cuptiDynamicSmemBytes` | `.dynamicSharedMemory` | Median dynamic shared memory passed at launch |

## Usage

Run any existing GPU demo and inspect the CSV:

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_Gpu_02_NsightProfiler \
      --quick --csv /home/$USER/workspace/cupti_demo.csv \
      --gtest_filter="*Coalesced*"
'

# The cupti* columns appear in the GPU section of the CSV
head -1 cupti_demo.csv | tr ',' '\n' | grep cupti
```

Sample values from a typical run (vector-add kernel, NVIDIA GPU):

```
cuptiKernelLaunches:    25000
cuptiRegistersMedian:   16
cuptiRegistersMax:      16
cuptiStaticSmemBytes:   0
cuptiDynamicSmemBytes:  0
```

What that tells you:

- **25,000 launches** -- the harness ran the kernel that many times in
  the measured window. Useful for sanity-checking that the harness
  actually issued the work you expected.
- **16 registers/thread** -- the kernel fits comfortably in a single SM
  register file; you have headroom to add work-per-thread without
  spilling. A jump from 16 to (say) 96 would warn you that a refactor
  is reducing achievable occupancy.
- **0 shared memory** -- this kernel doesn't use `__shared__`. Useful
  contrast against a kernel that's expected to use shared memory: a
  zero there is a bug, a non-zero value is the bytes/block you can
  cross-check against your `__shared__` declarations.

## When CUPTI is a No-Op

The collector compiles to a no-op when:
- libcupti is not linked at build time (the CMake target falls back
  cleanly via `__has_include(<cupti.h>)`).
- The CUDA toolkit is too old to expose `CUpti_ActivityKernel9`.

In either case the CSV columns appear but stay empty.

## Comparison with ncu

For a benchmark binary that runs hot kernels in a measured loop, CUPTI
in-process captures the same per-launch fields that `ncu --csv
--print-summary per-kernel` would report -- without:

- The ~5x replay overhead ncu adds (CUPTI Activity samples the actual
  launch; ncu re-launches the kernel multiple times to gather metrics).
- The container PID-namespace failure mode that affects `ncu -p <pid>`
  attach (CUPTI runs in-process so namespaces don't apply).
- The need to install a second tool on the deployment target.

For metrics that ncu *does* capture and CUPTI Activity doesn't (achieved
occupancy, warp efficiency, cache hit rates), keep using `--profile
nsight --profile-args replay` or wrap externally with ncu.

## Key Takeaways

- CUPTI fires automatically on every GPU benchmark -- no `--profile`
  flag needed.
- Five CSV columns surface per-launch register count, shared-memory,
  and launch count alongside vernier's timing data.
- In-process, near-zero overhead -- no replay, no second process.
- Reach for ncu when you need deep counters that require replay
  (occupancy, cache hit rates, warp efficiency).

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- external Nsight
  Systems / Compute integration
- [Demo 17 (Compute Sanitizer)](17_COMPUTE_SANITIZER.md) -- correctness
  side of the GPU profiling story
