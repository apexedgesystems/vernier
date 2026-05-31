# Demo 21: heaptrack -- Low-Overhead Allocation-Site Profiling

## Overview

heaptrack records every allocation with its call stack at low overhead. Where
massif samples heap _size_ over time, heaptrack gives you a ranked
allocation-_site_ view: which lines allocate the most often, the most bytes,
and the most short-lived temporaries. That ranking points straight at the fix.

Two variants in this demo:

- **Slow:** `Heaptrack.PerIterAlloc` -- a fresh, unreserved `std::vector` grown each iteration
- **Fast:** `Heaptrack.PooledReserve` -- one vector reserved up-front, cleared and reused

The story: building a vector with `push_back` and no `reserve` allocates once
per iteration _and_ reallocates as it grows. heaptrack's "most allocations" /
"most temporaries" view pins the cost to the `push_back` site; reserving once
and reusing the buffer removes it.

## What is heaptrack?

`heaptrack` is a heap-memory profiler that hooks every `malloc` / `new` (and
their frees) and records the allocating call stack. Unlike Valgrind tools it
uses native execution plus stack unwinding, so overhead is modest (typically
single-digit x, not 10-100x). The recorded trace is analyzed offline with
`heaptrack_print` (CLI) or `heaptrack_gui`.

- **Best for:** "which line allocates the most?" -- allocation count, peak
  heap, leaked bytes, and especially _temporary_ allocations (allocate-then-
  immediately-free churn) that massif's size-over-time view does not rank.
- **How it works:** an injected preload library intercepts the allocator and
  appends `(stack, size)` records to a compressed trace; analysis happens after
  the run, not during it.
- **Overhead:** low -- usually a few x, dominated by stack unwinding. Safe to
  run with realistic `--cycles` (50+), unlike the Valgrind family.
- **Skip it for:** CPU/wall-clock profiling (perf / gperf), thread races
  (helgrind), and allocations that bypass the C allocator (custom `mmap`
  arenas heaptrack does not see).

**In vernier:** `--profile heaptrack` is wrap-externally -- run the binary
under `heaptrack -o <dir>/run.heaptrack`. Per-test artifacts land in
`<TestName>.heaptrack/`; the trace is `run.heaptrack.zst`, analyzed with
`heaptrack_print <file>.zst` or `heaptrack_gui`.

## Prerequisites

```bash
make compose-debug
# heaptrack ships in the dev container; install heaptrack-gui locally for the UI.
```

## Step 1: Profile the Slow Path

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  heaptrack -o /tmp/slow.heaptrack \
    ./bin/ptests/BenchDemo_15_HeaptrackProfiler \
    --profile heaptrack --quick --cycles 50 \
    --gtest_filter="Heaptrack.PerIterAlloc"
  heaptrack_print /tmp/slow.heaptrack.zst | head -40
'
```

Expected: a large "total number of allocations" and "temporary allocations"
count, with the top allocating site resolving to the `push_back` in
`15_HeaptrackProfiler_Demo.cpp`.

```
total runtime: 0.42s.
calls to allocation functions: 5100000 (...)
temporary allocations: 4950000 (...)
peak heap memory consumption: 390.62K
...
MOST CALLS TO ALLOCATION FUNCTIONS
  5.1M calls   std::vector<unsigned int>::push_back(...)
               at 15_HeaptrackProfiler_Demo.cpp:67
```

## Step 2: Profile the Fast Path

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  heaptrack -o /tmp/fast.heaptrack \
    ./bin/ptests/BenchDemo_15_HeaptrackProfiler \
    --profile heaptrack --quick --cycles 50 \
    --gtest_filter="Heaptrack.PooledReserve"
  heaptrack_print /tmp/fast.heaptrack.zst | head -40
'
```

Now the allocation count and temporaries collapse to roughly one allocation
(the up-front `reserve`): `clear()` keeps capacity, so the inner loop reuses the
existing buffer and allocates nothing. Same final contents, no churn.

## Step 3: Diff

A 2-line change (`reserve` once, `clear` instead of re-creating) removes
millions of allocations and temporaries. The benchmark also speeds up because
allocator and reallocation overhead is gone -- but the heaptrack ranking shows
the _root cause_ that perf / gperf would have shown only as malloc time.

## When to Use heaptrack

- Allocation-heavy code where you need the _site ranking_, not just the size
  timeline (heaptrack ranks; massif graphs).
- Hunting temporary-allocation churn (allocate-then-free in a hot loop) that
  doesn't move peak heap but burns the allocator.
- Any case where Valgrind's 10-100x is too slow -- heaptrack's low overhead
  lets you profile a realistic workload.

## Overhead

Low -- typically a few x native, dominated by stack unwinding. Run with
realistic `--cycles` (50+); you do not need to shrink the workload the way the
Valgrind tools require.

## Key Takeaways

- heaptrack answers "which line allocates the most?" with a ranked
  allocation-site view -- the complement to massif's size-over-time graph.
- It uniquely ranks _temporary_ allocations, the cheapest churn to miss.
- Low overhead means you can profile realistic `--cycles`, not toy runs.
- Reserve-once + clear-and-reuse removes per-iteration allocations entirely.

## See Also

- [Demo 14 (Massif)](14_MASSIF_PROFILER.md) -- heap _size over time_ (the
  graph; heaptrack is the ranked site list)
- [Demo 22 (jemalloc)](22_JEMALLOC_PROFILER.md) -- sampled allocation profiling
  via the jemalloc runtime
