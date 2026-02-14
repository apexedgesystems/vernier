# Demo 06: Thread Scaling and Lock Contention

## Overview

Demonstrates the framework's `contentionRun()` API for multi-threaded
benchmarking. Shows how mutex lock contention serializes parallel work
and how atomic operations restore true parallelism.

## Prerequisites

```bash
make compose-debug
make tools-rust
```

## Step 1: Run with Multiple Threads

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_06_ThreadScaling --threads 4 --quick --csv /tmp/thread_demo.csv
  ./bin/tools/rust/bench summary /tmp/thread_demo.csv
'
```

Expected output:

```
ThreadScaling.MutexContention      ~500 us/call    ~2.0K calls/s
ThreadScaling.AtomicLockFree        ~25 us/call   ~40.0K calls/s
ThreadScaling.SingleThreadBaseline  ~15 us/call   ~66.7K calls/s
```

The mutex version is 20x slower than atomic under contention.

## Step 2: Scaling Analysis

Run at different thread counts to see scaling behavior:

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_06_ThreadScaling --threads 1 --quick --csv /tmp/t1.csv
  ./bin/ptests/BenchDemo_06_ThreadScaling --threads 2 --quick --csv /tmp/t2.csv
  ./bin/ptests/BenchDemo_06_ThreadScaling --threads 4 --quick --csv /tmp/t4.csv
  ./bin/ptests/BenchDemo_06_ThreadScaling --threads 8 --quick --csv /tmp/t8.csv
'
```

Expected scaling pattern for mutex:

| Threads | Mutex Latency | Atomic Latency |
| ------- | ------------- | -------------- |
| 1       | ~15 us        | ~15 us         |
| 2       | ~100 us       | ~18 us         |
| 4       | ~500 us       | ~25 us         |
| 8       | ~2000 us      | ~40 us         |

Mutex contention grows super-linearly with thread count. Atomic scales
sub-linearly due to CAS contention but remains much faster.

## Step 3: Diagnose

### Why mutex is slow under contention

When 4 threads compete for one mutex, only 1 thread runs at a time.
The other 3 are blocked in the kernel scheduler. Each lock acquisition
involves:

1. Syscall to futex (if contended)
2. Context switch to sleeping thread
3. Cache line invalidation of the counter

### Why atomic is fast

`fetch_add` with relaxed ordering compiles to a single `LOCK XADD`
instruction. No syscalls, no context switches, no kernel involvement.
The hardware handles the atomic operation at the cache coherency level.

## What the Code Demonstrates

- `PERF_CONTENTION` semantic macro (documents test intent)
- `contentionRun()` API (spawns N threads, measures aggregate throughput)
- `--threads N` flag (configurable thread count)
- Single-thread baseline for scaling efficiency calculation

## Key Takeaways

- Lock contention is the most common source of multi-threaded performance bugs
- `contentionRun()` makes it easy to measure contended throughput
- Atomic operations provide 10-50x improvement over mutexes for simple counters
- Always establish a single-thread baseline to calculate scaling efficiency
- Scaling efficiency = (N-thread throughput) / (1-thread throughput \* N)
- For complex shared state, consider lock-free data structures or sharding

## Further Reading

- `docs/CPU_GUIDE.md` -- Multi-threaded benchmarking patterns
- Demo 09 (bpftrace) -- Trace kernel-level scheduling overhead
