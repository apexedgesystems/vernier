# Demo 14: Valgrind Massif Heap Profiler

## Overview

Massif samples heap usage over time. Where gperf / callgrind / perf show
where CPU cycles go, massif shows where *memory* goes -- which allocation
sites dominate peak heap, when allocations happen, and how the profile
changes after a fix.

Two variants in this demo:

- **Slow:** `Massif.SmallChurn` -- a fresh 8 MB allocation per iteration
- **Fast:** `Massif.PooledReuse` -- one allocation, reused across iterations

The story: a `new[]` inside a hot loop looks innocent at the source level
but dominates the heap profile. Massif's allocation-site timeline pins
the cost; the fix is a one-line move.

## Prerequisites

```bash
make compose-debug
# valgrind ships in the dev container (used by callgrind, massif, memcheck)
```

## Step 1: Profile the Slow Path

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=massif --massif-out-file=/tmp/slow.massif \
    ./bin/ptests/BenchDemo_11_MassifProfiler \
    --profile massif --cycles 1 --gtest_filter="Massif.SmallChurn"
  ms_print /tmp/slow.massif | head -40
'
```

Expected output: a sawtooth heap profile, with peak around 8 MB and the
allocation attributed to `std::make_unique<double[]>` from
`11_MassifProfiler_Demo.cpp:65`.

```
    MB
8.476^##       ::::::::::::@:::::@::@:::::::::::::::@::::@::::::::::::::@:::::
     |#                                                                  ...
```

## Step 2: Profile the Fast Path

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=massif --massif-out-file=/tmp/fast.massif \
    ./bin/ptests/BenchDemo_11_MassifProfiler \
    --profile massif --cycles 1 --gtest_filter="Massif.PooledReuse"
  ms_print /tmp/fast.massif | head -40
'
```

Now the heap profile is flat at the single up-front allocation; the inner
loop no longer contributes new peaks. Same arithmetic, no churn.

## Step 3: Diff

A 1-line change (move the allocation out of the inner loop) eliminates
the heap sawtooth entirely. The benchmark also speeds up because
allocator overhead is gone -- but the heap profile shows the *root
cause* that perf / gperf would have shown only as glibc-malloc time.

## When to Use Massif

- Allocation-heavy code where the *bytes* matter, not just the time.
- Long-running services where peak heap drives OOM behavior.
- Refactors that swap allocators (jemalloc, mimalloc, custom pools) --
  massif quantifies the heap-side improvement.

## Overhead

~10x slower under valgrind. Use `--cycles 1` (or 2) for usable wall-time.
