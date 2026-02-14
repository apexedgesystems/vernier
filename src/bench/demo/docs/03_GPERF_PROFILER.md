# Demo 03: gperftools CPU Profiler

## Overview

Demonstrates using gperftools to identify function-level hotspots. When a
single function dominates execution time, gperftools pinpoints it immediately.
Shows the classic "replace O(n^2) with O(n log n)" optimization.

## Prerequisites

```bash
make compose-debug
make tools-rust
# gperftools is pre-installed in the dev-cuda container
```

## Step 1: Baseline Measurement

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_03_GperfProfiler --quick --csv /tmp/gperf_demo.csv
  ./bin/tools/rust/bench summary /tmp/gperf_demo.csv
'
```

Expected output:

```
GperfProfiler.BubbleSortHotspot    ~800 us/call    ~1.2K calls/s
GperfProfiler.StdSortOptimized      ~15 us/call   ~66.7K calls/s
```

The optimized version is ~50x faster for 500 elements.

## Step 2: Profile the Slow Path

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_03_GperfProfiler --profile gperf \
    --gtest_filter="*BubbleSortHotspot*" --cycles 1000
'
```

This generates a `.prof` file in the artifact directory.

## Step 3: Analyze the Profile

```bash
google-pprof --text ./BenchDemo_03_GperfProfiler *.prof
```

Expected output:

```
Total: 1234 samples
    1180  95.6%  95.6%    1180  95.6%  vernier::demo::bubbleSort
      30   2.4%  98.0%      30   2.4%  std::vector::operator[]
      24   1.9% 100.0%    1234 100.0%  [test lambda]
```

The `bubbleSort` function consumes 95.6% of all CPU time. This is the classic
single-function hotspot that gperftools excels at finding.

For a visual flamegraph:

```bash
google-pprof --pdf ./BenchDemo_03_GperfProfiler *.prof > hotspot.pdf
```

## Step 4: Profile the Fast Path

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_03_GperfProfiler --profile gperf \
    --gtest_filter="*StdSortOptimized*" --cycles 10000
'
```

```bash
google-pprof --text ./BenchDemo_03_GperfProfiler *.prof
```

Expected output:

```
Total: 234 samples
      78  33.3%  33.3%     78  33.3%  __introsort_loop
      45  19.2%  52.6%     45  19.2%  __insertion_sort
      34  14.5%  67.1%     34  14.5%  __unguarded_partition
      ...
```

No single function dominates -- time is distributed across the introsort
implementation. This is the hallmark of well-optimized code.

## Step 5: Compare

```bash
bench compare /tmp/baseline.csv /tmp/optimized.csv
```

| Test              | Baseline | Optimized | Speedup |
| ----------------- | -------- | --------- | ------- |
| Sort 500 elements | ~800 us  | ~15 us    | ~53x    |

The O(n^2) to O(n log n) improvement gives 53x for n=500.

## Key Takeaways

- gperftools shows **which function** is the bottleneck (not just where in memory)
- `--profile gperf` generates `.prof` files for post-mortem analysis
- `google-pprof --text` gives a quick text summary
- `google-pprof --pdf` generates visual call graph flamegraphs
- When one function dominates (>50% time), replacing its algorithm is the fix
- After optimization, look for the next hotspot -- there is always one

## Further Reading

- `docs/CPU_GUIDE.md` -- CPU profiling patterns
- Demo 07 (Callgrind) -- For deterministic A/B comparison without noise
