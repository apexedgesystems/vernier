# Demo 04: Cache-Friendly Data Layout (AoS vs SoA)

## Overview

Demonstrates the most impactful data layout optimization in performance
engineering: transforming Array-of-Structs (AoS) to Struct-of-Arrays (SoA).
When your workload only touches a few fields per struct, AoS wastes most of
each cache line loading unused data.

## Prerequisites

```bash
make compose-debug
make tools-rust
```

## Step 1: Baseline Measurement

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_04_CacheFriendly --quick --csv /tmp/cache_demo.csv
  ./bin/tools/rust/bench summary /tmp/cache_demo.csv
'
```

Expected output:

```
CacheFriendly.ArrayOfStructs    ~250 us/call    ~4.0K calls/s
  Memory bandwidth: 2560.0 MB/s (6.4 MB read)
  Estimated efficiency: 21.3% of theoretical peak

CacheFriendly.StructOfArrays     ~50 us/call   ~20.0K calls/s
  Memory bandwidth: 2400.0 MB/s (1.2 MB read)
  Estimated efficiency: 80.0% of theoretical peak
```

The SoA version is ~5x faster. The bandwidth numbers tell the story.

## Step 2: Diagnose

### AoS Layout (slow)

```
ParticleAoS: 128 bytes per particle (2 cache lines)
  [x][y][z][vx][vy][vz][mass][padding * 9]
  ^^^^^^^^^ 24 bytes used / 128 bytes loaded = 18.75% utilization
```

For 50K particles, the CPU loads 6.4 MB but only 1.2 MB is useful position data.
The remaining 81% is wasted cache bandwidth on velocity, mass, and padding.

### SoA Layout (fast)

```
ParticleSoA: separate arrays
  x: [x0][x1][x2]...[x49999]    400 KB
  y: [y0][y1][y2]...[y49999]    400 KB
  z: [z0][z1][z2]...[z49999]    400 KB
  Total: 1.2 MB, 100% utilization
```

Every byte loaded into cache is useful data. The hardware prefetcher sees
clean sequential access and operates at peak efficiency.

## Step 3: Compare with bench

```bash
bench compare /tmp/aos_baseline.csv /tmp/soa_optimized.csv
```

| Metric           | AoS     | SoA    | Improvement |
| ---------------- | ------- | ------ | ----------- |
| Median latency   | ~250 us | ~50 us | 5x          |
| Cache efficiency | ~19%    | ~80%   | 4.2x        |
| Data loaded      | 6.4 MB  | 1.2 MB | 5.3x less   |

## When to Use SoA

SoA is the right choice when:

- Your hot loop touches only a subset of struct fields
- The struct has many fields but each pass uses only a few
- Data set is large relative to cache size
- You are iterating over many elements in sequence

AoS is the right choice when:

- Each operation uses most or all fields of a struct
- You frequently access one object at a time (not iterating arrays)
- The struct is small (<64 bytes, fits in one cache line)

## Key Takeaways

- AoS vs SoA is often the single highest-leverage optimization for data-heavy code
- MemoryProfile reveals actual bandwidth utilization vs theoretical peak
- 128-byte structs waste 81% of cache bandwidth when only 24 bytes are needed
- SoA enables vectorization (SIMD) as a bonus -- contiguous doubles can be loaded 4-at-a-time
- The same principle applies to database column stores vs row stores

## Further Reading

- `docs/CPU_GUIDE.md` -- Memory access patterns
- Demo 02 (perf) -- Verify the cache miss reduction with hardware counters
