# Demo 17: Compute Sanitizer -- GPU Memcheck for Kernel Correctness

## Overview

Compute Sanitizer (the successor to cuda-memcheck) is NVIDIA's GPU
equivalent of valgrind memcheck. It catches device-side bugs that often
don't crash but silently corrupt results. The same binary can be wrapped
with any of four sub-tools depending on what bug class you're hunting:

| Sub-tool    | Catches                                              |
| ----------- | ---------------------------------------------------- |
| `memcheck`  | OOB reads/writes, leaks, misaligned access (default) |
| `racecheck` | Shared-memory data races                             |
| `synccheck` | Missed or invalid `__syncthreads` barriers           |
| `initcheck` | Reads from uninitialized device memory               |

The story: a kernel with an off-by-one bug returns plausible-looking
results most of the time and ships. Compute Sanitizer flags it with the
exact source line, thread, and block.

Two variants in this demo:

- **Safe:** `ComputeSanitizer.SafeKernel` -- bounds-checked scale; 0 errors
- **Buggy:** `ComputeSanitizer.WithDeliberateOob` -- writes one past end

## What is Compute Sanitizer?

Compute Sanitizer is NVIDIA's correctness tool for CUDA kernels. It
wraps your binary externally, intercepts every device-side memory
access (and depending on the sub-tool, every sync barrier / shared-mem
access / load), and produces a host-side report pinning each violation
to a source line, thread, and block.

- **Best for:** correctness bugs that pass tests on your machine but
  fail elsewhere -- OOB by a few bytes, racy shared-memory writes,
  uninitialized reads. Particularly useful before promoting a kernel
  from prototype to production.
- **How it works:** binary patching of device code at load time, with
  the host driver intercepting each kind of event the chosen sub-tool
  watches for.
- **Overhead:** ~5-10x kernel time. Use `--cycles 5` or lower; the goal
  is detection, not benchmarking.
- **Skip it for:** performance work (use Nsight), CPU code (compute
  sanitizer is GPU-only), or fully validated kernels you trust.

**In vernier:** `--profile compute-sanitizer` is wrap-externally -- run
the binary under `compute-sanitizer --tool=<sub-tool>`. Logs land in
`<TestName>.compute-sanitizer/`.

## Prerequisites

```bash
make compose-debug
# compute-sanitizer ships with the CUDA toolkit; available in dev-cuda.
```

## Step 1: Run the Safe Kernel Under the Sanitizer

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  compute-sanitizer --tool=memcheck \
    --log-file=/tmp/safe.sanitizer.log \
    ./bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler \
    --profile compute-sanitizer --quick \
    --gtest_filter="ComputeSanitizer.SafeKernel"
  tail -3 /tmp/safe.sanitizer.log
'
```

Expected:

```
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

## Step 2: Run the Buggy Kernel

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  compute-sanitizer --tool=memcheck \
    --log-file=/tmp/oob.sanitizer.log \
    ./bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler \
    --profile compute-sanitizer --cycles 5 --repeats 1 \
    --gtest_filter="ComputeSanitizer.WithDeliberateOob"
  grep -B1 -A8 "Invalid __global__ write" /tmp/oob.sanitizer.log | head -20
  grep "ERROR SUMMARY" /tmp/oob.sanitizer.log
'
```

Expected: an `Invalid __global__ write` error with a host backtrace
pinning the bug to `out[n] = 99.0f;` in
`04_ComputeSanitizerProfiler_Demo.cu`.

```
========= Invalid __global__ write of size 4 bytes
=========     at scaleKernelWithOob(const float *, float *, int)+0x6b0 in
              04_ComputeSanitizerProfiler_Demo.cu:73
=========     by thread (0,0,0) in block (4095,0,0)
=========     Access to 0x... is out of bounds
========= ERROR SUMMARY: 5 errors
```

The gtest case may report `FAILED` depending on driver / architecture:
some CUDA runtimes catch the page-boundary overrun and surface it as
"an illegal memory access", others silently let the OOB write through.
The sanitizer log above is what reliably pinpoints the bug -- exact
source line, thread, and block -- regardless of how the runtime reacts.

## Step 3: Run the Other Sanitizer Tools

The same demo binary works with each tool:

```bash
# Shared-memory race detection
compute-sanitizer --tool=racecheck \
  ./bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler \
  --profile compute-sanitizer --profile-args racecheck [...]

# Missed __syncthreads
compute-sanitizer --tool=synccheck [...]

# Uninitialized memory reads
compute-sanitizer --tool=initcheck [...]
```

## When to Use

- After any change to a CUDA kernel's indexing math.
- After changing block / grid configuration -- bounds that used to hold
  on a different launch shape may now slip.
- Before promoting a kernel from prototype to production.

## Overhead

~5-10x kernel time. Use `--cycles 5` or lower; the goal is detection,
not benchmarking.

## Key Takeaways

- Compute Sanitizer is a _correctness_ tool for CUDA kernels -- catches
  bugs that don't crash but corrupt results.
- Four sub-tools cover four bug classes: `memcheck`, `racecheck`,
  `synccheck`, `initcheck`. Same binary, different `--tool=`.
- The value is the source-line attribution: thread + block + line.
- Run it whenever kernel indexing math or launch geometry changes.

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- the _performance_
  side of GPU profiling
- [Demo 15 (Memcheck)](15_MEMCHECK_PROFILER.md) -- the CPU equivalent
  (Valgrind memcheck)
