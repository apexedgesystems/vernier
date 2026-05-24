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

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- external Nsight
  Systems / Compute integration
- [Demo 17 (Compute Sanitizer)](17_COMPUTE_SANITIZER.md) -- correctness
  side of the GPU profiling story
