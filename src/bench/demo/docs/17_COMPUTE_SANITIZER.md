# Demo 17: Compute Sanitizer -- GPU Memcheck for Kernel Correctness

## Overview

Compute Sanitizer (formerly cuda-memcheck) is NVIDIA's GPU equivalent of
valgrind memcheck. It catches device-side bugs that often don't crash but
silently corrupt results:

- Out-of-bounds device reads / writes (`--tool=memcheck`, default)
- Shared-memory data races (`--tool=racecheck`)
- Missed `__syncthreads` barriers (`--tool=synccheck`)
- Reads from uninitialized device memory (`--tool=initcheck`)

The story: a kernel with an off-by-one bug returns plausible-looking
results most of the time and ships. Compute Sanitizer flags it with the
exact source line.

Two variants in this demo:

- **Safe:** `ComputeSanitizer.SafeKernel` -- bounds-checked scale; 0 errors
- **Buggy:** `ComputeSanitizer.WithDeliberateOob` -- writes one past end

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
    --profile compute-sanitizer --quick \
    --gtest_filter="ComputeSanitizer.WithDeliberateOob"
  grep -B1 -A8 "Invalid __global__ write" /tmp/oob.sanitizer.log | head -20
'
```

Expected: a `Invalid __global__ write` error with a host backtrace pinning
the bug to `out[n] = 99.0f;` in `04_ComputeSanitizerProfiler_Demo.cu`.

```
========= Invalid __global__ write of size 4 bytes
=========     at scaleKernelWithOob(const float *, float *, int)+0xa0
=========     by thread (0,0,0) in block (4095,0,0)
=========     Address 0x... is out of bounds
========= ERROR SUMMARY: 5 errors
```

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

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- GPU performance
- [Demo 15 (Memcheck)](15_MEMCHECK_PROFILER.md) -- CPU equivalent
