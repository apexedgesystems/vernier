# Demo 02: Linux perf Profiler

## Overview

Demonstrates using the Linux `perf` profiler to identify cache-hostile memory
access patterns. Shows how stride-512 access causes constant cache misses and
how sequential access lets the hardware prefetcher eliminate them.

## Prerequisites

```bash
make compose-debug
make tools-rust
# perf is pre-installed in the dev-cuda container
```

## Step 1: Baseline Measurement

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_02_PerfProfiler --quick --csv /tmp/perf_demo.csv
  ./bin/tools/rust/bench summary /tmp/perf_demo.csv
'
```

Expected output (your numbers will vary):

```
PerfProfiler.StridedAccess     ~450 us/call    CV=2%    ~2.2K calls/s
PerfProfiler.SequentialAccess   ~80 us/call    CV=1%   ~12.5K calls/s
```

The sequential version is ~5x faster. But why? Profiling reveals the answer.

## Step 2: Profile the Slow Path

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_02_PerfProfiler --profile perf \
    --gtest_filter="*StridedAccess*" --cycles 1000
'
```

Expected perf output includes:

```
Performance counter stats:
    cpu-cycles:           ~500,000,000
    instructions:          ~50,000,000
    L1-dcache-load-misses: ~8,000,000    <-- HIGH
    LLC-load-misses:       ~2,000,000    <-- HIGH
```

## Step 3: Diagnose

The strided access pattern (stride=512 bytes = 8 cache lines) means every
single memory access misses in L1 cache. The hardware prefetcher cannot
detect the pattern because the stride exceeds its lookahead window.

Key indicators:

- **High L1-dcache-load-misses**: Each access loads a new cache line
- **High LLC-load-misses**: Data set (4 MB) exceeds L2 but fits in L3
- **Low IPC (instructions per cycle)**: CPU stalls waiting for memory

## Step 4: Profile the Fast Path

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_02_PerfProfiler --profile perf \
    --gtest_filter="*SequentialAccess*" --cycles 1000
'
```

Expected perf output:

```
Performance counter stats:
    cpu-cycles:            ~100,000,000
    instructions:          ~50,000,000
    L1-dcache-load-misses:    ~60,000    <-- 100x FEWER
    LLC-load-misses:           ~5,000    <-- NEGLIGIBLE
```

## Step 5: Compare

The cache miss counts tell the story:

| Metric     | Strided | Sequential | Ratio |
| ---------- | ------- | ---------- | ----- |
| L1 misses  | ~8M     | ~60K       | 133x  |
| LLC misses | ~2M     | ~5K        | 400x  |
| Cycles     | ~500M   | ~100M      | 5x    |

The 5x speedup maps directly to the 133x reduction in cache misses.

## Key Takeaways

- `perf` reveals hardware-level bottlenecks invisible to source-level inspection
- L1-dcache-load-misses is the most important counter for memory-bound code
- Sequential access enables hardware prefetching; strided access defeats it
- Cache line size is 64 bytes -- strides larger than this waste prefetch bandwidth
- Always profile before optimizing: "measure, don't guess"

## Further Reading

- `docs/CPU_GUIDE.md` -- CPU benchmarking patterns
- Demo 04 (Cache Friendly) -- AoS vs SoA layout optimization
