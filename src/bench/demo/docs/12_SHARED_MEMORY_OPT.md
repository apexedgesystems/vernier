# Demo 12: Shared Memory Optimization and Bank Conflicts

## Overview

Demonstrates the progression from global memory to shared memory for GPU
matrix transpose, showing how bank conflicts degrade performance and how
padding eliminates them. Three-way comparison:

1. Global memory only (baseline -- uncoalesced writes)
2. Shared memory with bank conflicts (partial win)
3. Shared memory without bank conflicts (full win)

Matrix transpose is the classic shared memory use case because naive
implementations always have either uncoalesced reads or uncoalesced writes.

## Prerequisites

```bash
make compose-debug
make tools-rust
# Requires NVIDIA GPU with CUDA support
```

## Step 1: Run All Three Variants

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  timeout 120 ./bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt --quick \
    --csv /tmp/shared_mem.csv
  ./bin/tools/rust/bench summary /tmp/shared_mem.csv
'
```

Expected output (1024x1024 float matrix):

```
SharedMemoryOpt.NaiveGlobalMemory         ~0.10 ms/call   ~10.0K calls/s
  Kernel: 100 us | Bandwidth: 1.8 GB/s

SharedMemoryOpt.SharedWithBankConflicts   ~0.04 ms/call   ~25.0K calls/s
  Kernel: 40 us | Bandwidth: 4.5 GB/s

SharedMemoryOpt.SharedConflictFree        ~0.02 ms/call   ~50.0K calls/s
  Kernel: 20 us | Bandwidth: 9.0 GB/s
```

## Step 2: Understanding the Three Approaches

### Approach 1: Naive Global Memory

```cpp
// Read: coalesced (row-major)
// Write: UNCOALESCED (column-major, stride = MATRIX_DIM)
output[x * dim + y] = input[y * dim + x];
```

Reads are coalesced because threads in a warp access consecutive elements
in the same row. But writes are uncoalesced because each thread writes to
a different column -- addresses `dim` elements apart.

```
Warp reads:  input[0], input[1], ..., input[31]     -> 1 transaction
Warp writes: output[0], output[1024], ..., output[31*1024] -> 32 transactions
```

### Approach 2: Shared Memory with Bank Conflicts

```cpp
__shared__ float tile[32][32];

// Step 1: Global -> Shared (coalesced read)
tile[threadIdx.y][threadIdx.x] = input[y * dim + x];
__syncthreads();

// Step 2: Shared -> Global (coalesced write, BUT bank conflicts on read)
output[outY * dim + outX] = tile[threadIdx.x][threadIdx.y];
```

Both global reads and writes are now coalesced. The shared memory staging
tile handles the transpose. But reading `tile[threadIdx.x][threadIdx.y]`
means all 32 threads in a warp read from the same column -- the same bank.

GPU shared memory has 32 banks, each 4 bytes wide. Bank index for
`tile[row][col]` = `(row * 32 + col) % 32 = col` (since row\*32 is a
multiple of 32). When all threads read column `threadIdx.y` (same value),
they all hit the same bank.

```
Thread 0 reads:  tile[0][0]   -> bank 0
Thread 1 reads:  tile[1][0]   -> bank 0  (conflict!)
Thread 2 reads:  tile[2][0]   -> bank 0  (conflict!)
...
Thread 31 reads: tile[31][0]  -> bank 0  (conflict!)

Result: 32-way bank conflict = serialized to 32 sequential accesses
```

### Approach 3: Shared Memory with Padding (No Bank Conflicts)

```cpp
__shared__ float tile[32][32 + 1]; // +1 padding per row
```

Adding one extra element per row shifts the bank mapping:

```
tile[row][col] is at index (row * 33 + col)
Bank = (row * 33 + col) % 32 = (row + col) % 32
```

Now when threads read column `threadIdx.y`:

```
Thread 0 reads:  tile[0][0]   -> bank (0+0)%32  = 0
Thread 1 reads:  tile[1][0]   -> bank (1+0)%32  = 1
Thread 2 reads:  tile[2][0]   -> bank (2+0)%32  = 2
...
Thread 31 reads: tile[31][0]  -> bank (31+0)%32 = 31

Result: 32 different banks = fully parallel (1 cycle)
```

The +1 padding costs 32 extra floats (128 bytes) per tile but eliminates
all bank conflicts.

## Step 3: Performance Comparison

| Approach           | Kernel Time | Bandwidth | Improvement |
| ------------------ | ----------- | --------- | ----------- |
| Naive global       | ~100 us     | 1.8 GB/s  | (baseline)  |
| Shared + conflicts | ~40 us      | 4.5 GB/s  | 2.5x        |
| Shared + padding   | ~20 us      | 9.0 GB/s  | 5.0x        |

The progression:

1. **Naive -> Shared**: Fixes uncoalesced global writes. 2.5x improvement.
2. **Conflicts -> Padded**: Fixes shared memory bank conflicts. Another 2x.
3. **Total**: 5x improvement from two targeted fixes.

## Step 4: Diagnosing Bank Conflicts with Nsight

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt --profile nsight \
    --gtest_filter="*SharedWithBankConflicts*" --cycles 100
'
```

Nsight Compute reports:

```
Shared Memory Bank Conflicts: 32-way (maximum)
Shared Memory Throughput: 12.5%  (1/32 of peak)
```

After padding:

```
Shared Memory Bank Conflicts: None
Shared Memory Throughput: 100% of peak
```

## Bank Conflict Quick Reference

| Stride          | Bank Conflict | Fix             |
| --------------- | ------------- | --------------- |
| 1 (consecutive) | None          | Already optimal |
| 2               | 2-way         | Pad +1          |
| 4               | 4-way         | Pad +1          |
| 8               | 8-way         | Pad +1          |
| 16              | 16-way        | Pad +1          |
| 32 (column)     | 32-way        | Pad +1          |

The +1 padding fix works for all power-of-2 strides because it breaks
the alignment between stride and bank count.

## Key Takeaways

- Matrix transpose is the classic shared memory problem
- Shared memory converts uncoalesced global accesses into coalesced ones
- Bank conflicts serialize shared memory accesses within a warp
- 32-way bank conflict = 32x slowdown on shared memory reads
- Adding +1 padding per row eliminates all power-of-2 bank conflicts
- Cost: 128 bytes extra per 32x32 tile (negligible)
- Use Nsight Compute to detect bank conflicts ("Shared Memory Bank Conflicts")
- Two fixes (coalescing + bank conflicts) compound for 5x total improvement

## Further Reading

- `docs/GPU_GUIDE.md` -- Shared memory patterns and bank conflict avoidance
- Demo 10 (GPU Basic) -- Framework fundamentals and transfer overhead
- Demo 11 (Nsight) -- Memory coalescing for global memory access patterns
