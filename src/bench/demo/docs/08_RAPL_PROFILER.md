# Demo 08: Intel RAPL Energy Measurement

## Overview

Demonstrates using Intel RAPL (Running Average Power Limit) to measure
energy consumption per operation. Shows that optimized code not only runs
faster but also consumes less total energy -- a critical metric for
battery-powered and thermally-constrained systems.

## Prerequisites

```bash
make compose-debug
make tools-rust
# Requires Intel CPU (Haswell or later)
# MSR module must be loaded: sudo modprobe msr
# May require root for MSR access
```

## Step 1: Run with RAPL Profiling

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_08_RaplProfiler --profile rapl --quick \
    --csv /tmp/rapl_demo.csv
  ./bin/tools/rust/bench summary /tmp/rapl_demo.csv
'
```

Expected output:

```
RaplProfiler.NaiveDotProduct       ~180 us/call    ~5.6K calls/s
  Energy: 0.0054 J total, 0.54 mJ/op, 12.3W avg power

RaplProfiler.VectorizedDotProduct   ~35 us/call   ~28.6K calls/s
  Energy: 0.0015 J total, 0.15 mJ/op, 14.1W avg power
```

## Step 2: Diagnose

### Why the naive version uses more energy

The naive dot product has a data dependency (`if (sum > 1e18)`) that
prevents auto-vectorization. The CPU processes one element per cycle
instead of 4 (SSE) or 8 (AVX). More cycles = more time at power draw.

### Why the vectorized version is more efficient

`std::inner_product` with doubles and 0.0 init allows the compiler to
emit SIMD instructions. Multiple multiply-adds execute per cycle.
Fewer cycles at similar power draw = less total energy.

## Step 3: Compare Energy Efficiency

| Metric    | Naive   | Vectorized | Improvement          |
| --------- | ------- | ---------- | -------------------- |
| Latency   | ~180 us | ~35 us     | 5.1x faster          |
| Energy/op | 0.54 mJ | 0.15 mJ    | 3.6x less energy     |
| Avg power | 12.3 W  | 14.1 W     | Higher (but shorter) |

Note: The vectorized version draws slightly more power (SIMD units active)
but for much less time, resulting in lower total energy per operation.

## When RAPL Matters

- **Embedded systems**: Battery life depends on energy per operation
- **Data centers**: Power budget per rack limits throughput
- **Thermal throttling**: Sustained power draw triggers clock reduction
- **Sustainability**: Lower energy = lower environmental impact

## Key Takeaways

- RAPL measures actual package energy (not estimated from cycles)
- Faster code usually means less total energy (despite potentially higher power)
- Energy-per-operation (mJ/op) is the key metric for efficiency
- Auto-vectorization is one of the highest-leverage energy optimizations
- RAPL requires Intel CPU (Haswell+) and MSR access

## Further Reading

- `docs/CPU_GUIDE.md` -- RAPL profiling section
- Demo 02 (perf) -- Use hardware counters alongside RAPL
