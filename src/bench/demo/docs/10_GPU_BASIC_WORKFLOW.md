# Demo 10: GPU Basic Benchmarking Workflow

## Overview

Teaches the fundamental GPU measurement workflow using the CUDA benchmarking
API. Demonstrates how to measure CPU baselines, GPU kernels with and without
transfer overhead, and how the framework automatically calculates speedup.

Workload: Vector addition (a[i] + b[i] = c[i]) on 1M elements.

## Prerequisites

```bash
make compose-debug
make tools-rust
# Requires NVIDIA GPU with CUDA support
```

## Step 1: Run All GPU Workflow Tests

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  timeout 120 ./bin/ptests/BenchDemo_Gpu_01_GpuBasicWorkflow --quick \
    --csv /tmp/gpu_workflow.csv
  ./bin/tools/rust/bench summary /tmp/gpu_workflow.csv
'
```

Expected output (your GPU model and numbers will differ):

```
GPU: NVIDIA RTX 5000 Ada Generation Laptop GPU (Compute 8.9)
SM Clock: 1905 MHz | Device 0 of 1

GpuBasicWorkflow.CpuBaseline       ~2.5 ms/call      ~400 calls/s
  Transfer: N/A (CPU only)

GpuBasicWorkflow.GpuWithTransfers  ~0.4 ms/call    ~2,500 calls/s
  Kernel: 17 us | Transfer: 380 us | H2D: 8 MB | D2H: 4 MB
  Speedup vs CPU: 6.3x
  Bandwidth: 6.5 GB/s

GpuBasicWorkflow.GpuKernelOnly     ~0.02 ms/call  ~50,000 calls/s
  Kernel: 17 us | Transfer: 0 us
  Speedup vs CPU: 125x
  Bandwidth: N/A (no transfers)
```

## Step 2: Understanding the Results

### Three-way comparison

| Test             | What it measures              | Time     |
| ---------------- | ----------------------------- | -------- |
| CpuBaseline      | CPU loop over 1M floats       | ~2.5 ms  |
| GpuWithTransfers | H2D + kernel + D2H            | ~0.4 ms  |
| GpuKernelOnly    | Kernel alone (data on device) | ~0.02 ms |

### Where time goes in GpuWithTransfers

```
H2D transfer:  8 MB (two float arrays)     ~180 us  (47%)
Kernel:        1M float additions            ~17 us   (4%)
D2H transfer:  4 MB (result array)          ~180 us  (47%)
Overhead:      CUDA API calls                 ~3 us   (2%)
                                           --------
Total:                                      ~380 us
```

Transfer overhead dominates for this simple workload. The kernel itself is
only 4% of total time. This is typical for memory-bound kernels and is
the most common GPU performance trap.

### Speedup analysis

- **With transfers**: 6.3x faster than CPU -- modest, limited by PCIe bandwidth
- **Kernel only**: 125x faster than CPU -- the true compute advantage

The difference tells you: this workload would benefit greatly from keeping
data on the GPU across multiple kernel launches (persistent device data).

## Step 3: Understanding the API

### cpuBaseline()

```cpp
auto result = perf.cpuBaseline(
    [&] { vectorAddCPU(h_a.data(), h_b.data(), h_c.data(), N); },
    "cpu_vector_add");
```

Measures a CPU function to establish a reference. The framework uses this
to calculate `speedupVsCpu` in the CSV output.

### cudaKernel() with transfers

```cpp
auto result = perf.cudaKernel(
        [&](cudaStream_t s) {
          vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
        },
        "gpu_vector_add")
    .withHostToDevice(h_a.data(), d_a, SIZE)
    .withHostToDevice(h_b.data(), d_b, SIZE)
    .withDeviceToHost(d_c, h_c.data(), SIZE)
    .withLaunchConfig(grid, block)
    .measure();
```

The builder pattern separates concerns:

- `.withHostToDevice()` -- Registers H2D transfers (measured separately)
- `.withDeviceToHost()` -- Registers D2H transfers (measured separately)
- `.withLaunchConfig()` -- Records grid/block dims for CSV metadata
- `.measure()` -- Runs warmup + measurement cycles

### cudaKernel() without transfers

```cpp
auto result = perf.cudaKernel(
        [&](cudaStream_t s) {
          vectorAddKernel<<<grid, block, 0, s>>>(d_a, d_b, d_c, N);
        },
        "gpu_kernel_only")
    .withLaunchConfig(grid, block)
    .measure();
```

Omit `.withHostToDevice()` / `.withDeviceToHost()` to measure kernel time
alone. Pre-transfer data before measurement with regular `cudaMemcpy()`.

## Step 4: Analyze the CSV

The GPU CSV extends the CPU format with additional columns:

```
gpuModel, computeCapability, kernelTimeUs, transferTimeUs,
h2dBytes, d2hBytes, speedupVsCpu, memBandwidthGBs,
occupancy, smClockMHz, throttling
```

Use `bench summary` to display these:

```bash
./bin/tools/rust/bench summary /tmp/gpu_workflow.csv
```

## When GPU Wins vs CPU

| Factor               | Favors GPU              | Favors CPU              |
| -------------------- | ----------------------- | ----------------------- |
| Data size            | Large (10K+ elements)   | Small (<1K)             |
| Arithmetic intensity | High (many ops/byte)    | Low (1 op/byte)         |
| Transfer frequency   | Infrequent (data stays) | Every call              |
| Parallelism          | Embarrassingly parallel | Sequential dependencies |

Vector addition is low arithmetic intensity (1 add per 12 bytes loaded).
GPU wins only at large scale or when data is already resident.

## Key Takeaways

- Always measure CPU baseline to get meaningful speedup numbers
- Transfer overhead often dominates for simple kernels
- "Kernel only" vs "with transfers" reveals how much PCIe costs you
- Keep data on the GPU as long as possible to amortize transfer costs
- The framework separates kernel time, transfer time, and total time
- Use `PERF_GPU_COMPARISON` macro for CPU-vs-GPU benchmarks

## Further Reading

- `docs/GPU_GUIDE.md` -- Complete GPU benchmarking reference
- Demo 11 (Nsight) -- Profile GPU memory access patterns
- Demo 12 (Shared Memory) -- Optimize GPU memory hierarchy
