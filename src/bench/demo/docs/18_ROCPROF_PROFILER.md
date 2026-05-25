# Demo 18: rocprof for AMD GPU Profiling

## Overview

rocprof is AMD's profiler for HIP / OpenCL / OpenMP kernels on Radeon
Instinct and MI GPUs. The vernier backend follows the same wrap-externally
pattern as the Nsight + Compute Sanitizer backends: the binary stays
passive unless detected wrapping, and the backend prints the exact
rocprof invocation when it's not.

vernier's CUDA GPU demos (`gpu/01-04`) target NVIDIA hardware. To exercise
the rocprof backend on an AMD GPU you need a HIP-ported kernel. The
workflow below uses one of the CUDA demos as the structural template;
the kernel body would be identical except for the `<<<...>>>` -> `hipLaunchKernelGGL`
substitution.

## What is rocprof?

`rocprof` is AMD's equivalent of NVIDIA's Nsight Systems / Compute. It
captures kernel timings, hardware counters, and API traces on AMD GPUs
running ROCm, and writes the results as CSV + a Chrome-trace JSON.

- **Best for:** AMD GPU benchmarking and optimization -- kernel
  duration, occupancy, memory throughput, and host/device traces on
  Radeon Instinct / MI hardware.
- **How it works:** wrap-externally; rocprof injects its agent into
  HSA/HIP and intercepts kernel dispatches.
- **Overhead:** minimal for kernel timings; per-counter modes add a
  small fixed cost per kernel launch.
- **Skip it for:** NVIDIA GPUs (use Nsight), CPU profiling (perf), or
  hosts without a working ROCm stack.

**In vernier:** `--profile rocprof` is wrap-externally -- run the
binary under `rocprof -o <path>`. Per-test artifacts land in
`<TestName>.rocprof/`; `results.csv` is the per-kernel timing table and
`results.json` opens in `chrome://tracing`.

## Prerequisites

```bash
# AMD GPU host with ROCm installed
sudo apt install rocm-dev rocprofiler
which rocprof          # /opt/rocm/bin/rocprof
```

vernier's `bench doctor` (or `--profile-check` on any test binary)
reports the rocprof + ROCm runtime status so you can confirm both are
available before running a profile.

## Step 1: Run Unwrapped (Hint Path)

```bash
./build/native-linux-debug/bin/ptests/MyHipPtest \
    --profile rocprof --quick --gtest_filter='Gpu.MyKernel'
```

Expected:

```
[rocprof] NOT running under rocprof; this measurement will execute
[rocprof] normally but no profile is collected. To collect:
[rocprof]   rocprof -o MyHipPtest.Gpu.MyKernel.rocprof/results.csv \
[rocprof]       <this-binary> --profile rocprof --profile-args default [...]
```

The backend prints the precise invocation including the per-test
artifact directory. Copy/paste it.

## Step 2: Run Under rocprof

```bash
rocprof -o MyHipPtest.Gpu.MyKernel.rocprof/results.csv \
    ./build/native-linux-debug/bin/ptests/MyHipPtest \
    --profile rocprof --quick --gtest_filter='Gpu.MyKernel'

# results.csv -- per-kernel time table
# results.json -- Chrome trace; open with chrome://tracing
column -t -s ',' MyHipPtest.Gpu.MyKernel.rocprof/results.csv | head -10
```

## Step 3: Trace Modes

```bash
# Kernel statistics (--stats): same data plus per-kernel min/avg/max
./MyHipPtest --profile rocprof --profile-args stats [...]
# -> rocprof --stats -o ...

# HSA-level tracing (cross-references with kernel launches)
./MyHipPtest --profile rocprof --profile-args hsa-trace [...]
# -> rocprof --hsa-trace -o ...

# HIP API tracing (cudaMalloc / cudaMemcpy equivalents)
./MyHipPtest --profile rocprof --profile-args hip-trace [...]
# -> rocprof --hip-trace -o ...
```

## When to Use

- AMD MI / Radeon Instinct workloads where vendor-neutral profiling
  isn't available.
- Cross-vendor benchmarking exercises (port the same kernel to CUDA +
  HIP, profile with nsight + rocprof, compare timings).
- Verifying kernel timing parity between drivers (host nsys timeline
  vs ROCm rocprof JSON).

## Key Takeaways

- rocprof is to AMD GPUs what Nsight is to NVIDIA -- the primary perf
  tool for HIP / OpenCL kernels on ROCm.
- vernier uses the same wrap-externally pattern as nsight and
  compute-sanitizer; the backend prints the precise rocprof invocation
  when not wrapped.
- `results.csv` (per-kernel timings) and `results.json` (Chrome trace)
  are the two artifacts to look at.
- Trace modes (`--stats`, `--hsa-trace`, `--hip-trace`) trade detail
  for overhead; pick by what question you're answering.

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- NVIDIA
  equivalent
- [Demo 17 (Compute Sanitizer)](17_COMPUTE_SANITIZER.md) -- shares the
  wrap-externally pattern
