# Demo 11: NVIDIA Nsight Profiler for Memory Coalescing

## Overview

Demonstrates using NVIDIA Nsight to identify and fix uncoalesced global memory
access patterns. Memory coalescing is the GPU equivalent of CPU cache-friendly
access -- the single most important optimization for memory-bound CUDA kernels.

When threads in a warp access consecutive addresses, the memory controller
combines 32 requests into 1-2 transactions. When threads access scattered
addresses, each request becomes a separate transaction (up to 32x slower).

## What is Nsight?

NVIDIA's Nsight is actually two complementary tools shipped together:

- **Nsight Systems (`nsys`)** -- whole-application timeline. Shows
  CPU/GPU activity, kernel launches, memory copies, NVTX ranges, and
  CUDA API calls on a unified trace. Reach for it first when the
  question is "where does the GPU time go?"
- **Nsight Compute (`ncu`)** -- per-kernel deep dive. Replays each
  kernel multiple times to collect hardware metrics: achieved occupancy,
  memory throughput, cache hit rates, warp-execution efficiency. Reach
  for it when the question is "why is *this kernel* slow?"

- **Best for:** GPU kernel optimization. Nsight Systems pinpoints which
  kernel matters; Nsight Compute explains why it's slow.
- **How it works:** both use NVIDIA's CUPTI under the hood. `nsys` reads
  the Activity API (in-process, cheap); `ncu` uses the Profiler API with
  kernel replay (~5x slower per kernel).
- **Overhead:** `nsys` is near-native; `ncu` adds ~5x per profiled
  kernel because of replay.
- **Skip it for:** CPU-only work (perf / gperf), AMD GPUs (rocprof), or
  microsecond-scale single-launch measurement where replay would
  distort results.

**In vernier:** `--profile nsight` is wrap-externally -- run the binary
under `nsys profile -o <path>` or `ncu -o <path>`. Per-test artifacts
land in `<TestName>.nsight/`; the backend auto-extracts
`cuda_gpu_kern_sum.txt`, `cuda_api_sum.txt`, and the memory summaries
next to the `.nsys-rep` so you can grep without a separate CLI round-trip.

## Prerequisites

```bash
make compose-debug
make tools-rust
# Requires NVIDIA GPU with CUDA support
# Nsight Systems: nsys (for timeline profiling)
# Nsight Compute: ncu (for kernel analysis)
```

## Step 1: Baseline Measurement

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  timeout 120 ./bin/ptests/BenchDemo_Gpu_02_NsightProfiler --quick \
    --csv /tmp/nsight_demo.csv
  ./bin/tools/rust/bench summary /tmp/nsight_demo.csv
'
```

Expected output:

```
NsightProfiler.UncoalescedAccess    ~0.08 ms/call   ~12.5K calls/s
  Kernel: 75 us | Bandwidth: 1.2 GB/s

NsightProfiler.CoalescedAccess      ~0.01 ms/call  ~100.0K calls/s
  Kernel: 10 us | Bandwidth: 9.5 GB/s
```

## Step 2: Profile with Nsight

### Nsight Systems (timeline)

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_Gpu_02_NsightProfiler --profile nsight \
    --gtest_filter="*UncoalescedAccess*" --cycles 100
'
```

This produces an `nsys` report showing kernel launch timelines and
memory transfer patterns.

### Nsight Compute (kernel analysis)

For detailed memory efficiency metrics, use `ncu` directly:

```bash
ncu --target-processes all \
  ./bin/ptests/BenchDemo_Gpu_02_NsightProfiler --quick \
  --gtest_filter="*UncoalescedAccess*"
```

## Step 3: Diagnose

### Uncoalesced kernel analysis

```
Kernel: uncoalescedReadKernel
Global Load Efficiency:    3.1%    <-- Only 1/32 of loaded bytes are used
L2 Cache Hit Rate:        12.4%    <-- Cache thrashing from scattered access
Memory Throughput:         1.2 GB/s <-- Far below theoretical maximum
```

Each thread reads `input[idx * 32]` -- addresses 128 bytes apart within a
warp. The memory controller must issue 32 separate transactions per warp
because no two threads access the same cache line.

```
Thread 0 reads: input[0]       -> cache line 0
Thread 1 reads: input[32]      -> cache line 1
Thread 2 reads: input[64]      -> cache line 2
...
Thread 31 reads: input[992]    -> cache line 31

Result: 32 cache line loads for 32 floats (128 bytes useful / 4096 loaded = 3.1%)
```

### Coalesced kernel analysis

```
Kernel: coalescedReadKernel
Global Load Efficiency:  100.0%    <-- All loaded bytes are consumed
L2 Cache Hit Rate:        95.2%    <-- Sequential access is cache-friendly
Memory Throughput:         9.5 GB/s <-- Near theoretical memory bandwidth
```

Each thread reads `input[idx]` -- consecutive addresses within a warp.
The memory controller combines all 32 requests into a single transaction.

```
Thread 0 reads: input[0]       -> cache line 0
Thread 1 reads: input[1]       -> cache line 0
Thread 2 reads: input[2]       -> cache line 0
...
Thread 31 reads: input[31]     -> cache line 0

Result: 1 cache line load for 32 floats (128 bytes useful / 128 loaded = 100%)
```

## Step 4: Compare

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_Gpu_02_NsightProfiler --quick \
    --gtest_filter="*UncoalescedAccess*" --csv /tmp/nsight_slow.csv
  ./bin/ptests/BenchDemo_Gpu_02_NsightProfiler --quick \
    --gtest_filter="*CoalescedAccess*" --csv /tmp/nsight_fast.csv
  ./bin/tools/rust/bench compare /tmp/nsight_slow.csv /tmp/nsight_fast.csv
'
```

| Metric          | Uncoalesced | Coalesced | Improvement |
| --------------- | ----------- | --------- | ----------- |
| Kernel time     | ~75 us      | ~10 us    | 7.5x faster |
| Load efficiency | 3.1%        | 100%      | 32x better  |
| Bandwidth       | 1.2 GB/s    | 9.5 GB/s  | 7.9x higher |

## Common Coalescing Patterns

### Bad: Column-major access in row-major layout

```cpp
// Threads in warp access different rows (strided)
output[threadIdx.x * width + col] = ...;
```

### Good: Row-major access in row-major layout

```cpp
// Threads in warp access same row (consecutive)
output[row * width + threadIdx.x] = ...;
```

### Bad: Struct-of-arrays scattered access

```cpp
// Each field in a different memory region
float x = particles.x[idx];
float y = particles.y[idx];  // Separate load transaction
float z = particles.z[idx];  // Separate load transaction
```

### Good: Array-of-structs with aligned access

```cpp
// All fields in one contiguous struct
Particle p = particles[idx]; // One transaction loads all fields
```

Note: The optimal layout depends on whether you access all fields or only
some. SoA is better when you only read x (see Demo 04 for the CPU version
of this trade-off).

## Key Takeaways

- Memory coalescing is the most impactful GPU optimization
- Coalesced: threads in a warp access consecutive addresses
- Uncoalesced: threads access scattered addresses (up to 32x penalty)
- Nsight Compute shows "Global Load Efficiency" as a percentage
- 100% efficiency = all loaded bytes consumed, 3% = 97% wasted bandwidth
- The fix is usually a data layout change, not an algorithm change
- Profile with `--profile nsight` or use `ncu` directly for kernel analysis

## See Also

- `docs/GPU_GUIDE.md` -- Memory coalescing patterns
- Demo 10 (GPU Basic) -- Framework fundamentals and transfer overhead
- Demo 12 (Shared Memory) -- Use shared memory to fix coalescing issues
