# Demo 07: Valgrind Callgrind Profiler

## Overview

Demonstrates using Valgrind's callgrind tool for deterministic instruction
counting. Unlike sampling profilers (perf, gperf), callgrind simulates every
instruction -- producing identical results on every run. This makes it ideal
for precise A/B comparisons where measurement noise is unacceptable.

## What is callgrind?

`callgrind` is one of Valgrind's tools. It runs your program on a
synthetic CPU and _counts every instruction executed_ per function, so
the numbers are exact and reproducible -- two runs of the same code
give identical results.

- **Best for:** A/B optimization comparisons where you need an exact
  ratio (1.7x or 2.1x?) instead of "5% +/- 3%". Also great for proving
  an algorithmic change actually reduced instruction count.
- **How it works:** dynamic binary instrumentation. Valgrind translates
  your binary into its IR, executes it on a simulated CPU, and counts.
- **Overhead:** 20-50x slower than native. Use `--cycles 1` and
  `--repeats 1` -- the results are deterministic, so repetition adds
  nothing.
- **Skip it for:** wall-clock measurement (the simulated CPU isn't real
  silicon), production workloads (too slow), and anything I/O-heavy
  (syscall timing is meaningless).

**In vernier:** `--profile callgrind` is wrap-externally -- run the
binary under `valgrind --tool=callgrind ...`. Per-test artifacts land in
`<TestName>.callgrind/callgrind.out`; visualize with
`callgrind_annotate` or `kcachegrind`.

## Prerequisites

```bash
make compose-debug
make tools-rust
# Valgrind is pre-installed in the dev container
```

## Step 1: Baseline Measurement (Normal Speed)

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_07_CallgrindProfiler --quick --csv /tmp/callgrind_demo.csv
  ./bin/tools/rust/bench summary /tmp/callgrind_demo.csv
'
```

Expected output:

```
CallgrindProfiler.LinearSearch   ~500 us/call     ~2.0K calls/s
CallgrindProfiler.BinarySearch     ~5 us/call   ~200.0K calls/s
```

## Step 2: Profile with Callgrind

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_07_CallgrindProfiler --profile callgrind \
    --gtest_filter="*LinearSearch*" --cycles 100 --repeats 1
'
```

Note: `--repeats 1` because callgrind is deterministic (no need for
statistical sampling). Execution is 20-50x slower under valgrind.

## Step 3: Analyze Callgrind Output

```bash
callgrind_annotate callgrind.out.*
```

Expected output for linear search:

```
Ir          file:function
-----------
500,000,000  demo::linearSearch     <-- 500M instructions for 100 searches
    100,000  demo::binarySearch     <-- 100K instructions (not measured here)
```

For binary search:

```
Ir          file:function
-----------
    100,000  demo::binarySearch     <-- 100K instructions for 100 searches
```

## Step 4: Compare Instruction Counts

| Algorithm | Instructions per Search | Complexity |
| --------- | ----------------------- | ---------- |
| Linear    | ~5,000,000 (n/2 \* ops) | O(n)       |
| Binary    | ~1,000 (log2(n) \* ops) | O(log n)   |
| Ratio     | 5000x                   |            |

The instruction ratio exactly reflects the algorithmic improvement:
n/2 = 5000 vs log2(10000) = ~13, giving ~385x theoretical ratio.
The actual 5000x includes constant factors from the loop body.

## When to Use Callgrind vs Sampling Profilers

| Feature        | Callgrind              | perf/gperf             |
| -------------- | ---------------------- | ---------------------- |
| Noise          | Zero (deterministic)   | Some (statistical)     |
| Speed          | 20-50x slower          | Native speed           |
| Granularity    | Every instruction      | Sampling (~1ms)        |
| Use case       | Precise A/B comparison | Hotspot identification |
| Repeats needed | 1                      | 10-20+                 |

## Key Takeaways

- Callgrind counts every instruction deterministically (zero variance)
- Perfect for A/B optimization comparisons where you need exact ratios
- Trade-off: 20-50x runtime overhead means fewer cycles and repeats
- `--repeats 1` is sufficient since results are deterministic
- Instruction counts directly reflect algorithmic complexity
- Use callgrind for precision, use perf/gperf for speed

## See Also

- `docs/CPU_GUIDE.md` -- Profiler comparison table
- Demo 03 (gperf) -- For faster but noisier function-level profiling
