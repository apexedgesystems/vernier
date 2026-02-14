# Demo 01: Basic Benchmarking Workflow

## Overview

This demo teaches the fundamental measure-export-analyze cycle that forms the
foundation of all performance work. You will learn to measure throughput,
export CSV results, and analyze them with the bench and bench-plot tools.

## Prerequisites

```bash
make compose-debug    # Build framework + demos
make tools-rust       # Build bench tool
```

## Step 1: Run the Demo

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_01_BasicWorkflow --quick --csv /tmp/demo01.csv
'
```

Expected output:

```
[BasicWorkflow.SimpleThroughput]  327.433 us/call  CV=0.7%  ~3.1K calls/s  (p10=326 p90=331 sd=2.4)
  Memory bandwidth: 2443.2 MB/s (0.8 MB read, 0.0 MB written per call)
  Estimated efficiency: 20.4% of theoretical peak (~12000 MB/s)

[BasicWorkflow.AccumulateVsManualLoop]  327.693 us/call  CV=0.6%  ~3.1K calls/s
[BasicWorkflow.AccumulateVsManualLoop]  166.090 us/call  CV=1.6%  ~6.0K calls/s

[BasicWorkflow.QuickModeIteration]  32.917 us/call  CV=0.5%  ~30.4K calls/s
```

Your numbers will vary depending on hardware.

## Step 2: Analyze with bench summary

```bash
bench summary /tmp/demo01.csv
```

Expected output:

```
Test                                   Median (us)       P10       P90        CV   Calls/sec  Stable
------------------------------------  ------------  --------  --------  --------  ----------  ------
BasicWorkflow.AccumulateVsManualLoop     162.947    161.991   168.732      2.2%        6137  yes
BasicWorkflow.QuickModeIteration          32.454     32.395    32.561      0.3%       30813  yes
BasicWorkflow.SimpleThroughput           323.158    321.683   330.853      1.5%        3094  yes
```

## Step 3: A/B Comparison

Run the demo twice (simulating before/after optimization):

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_01_BasicWorkflow --quick --csv /tmp/baseline.csv
  ./bin/ptests/BenchDemo_01_BasicWorkflow --quick --csv /tmp/candidate.csv
  ./bin/tools/rust/bench compare /tmp/baseline.csv /tmp/candidate.csv
'
```

Since both runs are the same code, you will see all tests marked "Neutral"
(delta within noise). In a real workflow, you would modify code between runs.

## Step 4: Visualize (Optional)

If bench-plot is installed (`make tools-py`):

```bash
bench-plot plot /tmp/demo01.csv --output /tmp/demo01_plots/
```

Generates interactive charts showing measurement distributions.

## What the Code Does

The demo exercises three patterns:

1. **SimpleThroughput** -- Basic `throughputLoop()` with `MemoryProfile` for bandwidth analysis
2. **AccumulateVsManualLoop** -- A/B test: `std::accumulate` vs pointer-arithmetic loop
3. **QuickModeIteration** -- Smaller dataset for fast iteration

Key framework features demonstrated:

- `PERF_THROUGHPUT` semantic test macro
- `UB_PERF_GUARD(perf)` scoped guard with auto-profiler attachment
- `perf.warmup()` for cache priming
- `perf.throughputLoop()` with MemoryProfile for bandwidth analysis
- `volatile` sink pattern to prevent dead-code elimination
- `--csv` flag for automated result export
- `--quick` mode for fast development iteration

## Key Takeaways

- Always use `warmup()` before measurement to prime caches and branch predictors
- The `volatile` sink pattern prevents the compiler from eliminating your workload
- MemoryProfile enables automatic bandwidth and efficiency calculations
- `--quick` mode trades statistical confidence for speed during development
- Export CSV and use `bench summary` for clean, machine-parseable results
- Use `bench compare` for statistical A/B comparisons

## Next Steps

- Try `--repeats 20` for more stable measurements
- Try `--cycles 50000` for higher precision per sample
- Profile with `--profile perf` (see Demo 02)
- Run `bench compare` with `--fail-on-regression` for CI integration
