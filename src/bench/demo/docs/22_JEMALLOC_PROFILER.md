# Demo 22: jemalloc prof -- Sampled Allocation Hotspots

## Overview

jemalloc's built-in heap profiler _samples_ allocations (sampled by bytes, so
it stays cheap even on allocation-heavy code) and attributes each sample to a
call stack. `jeprof` then ranks the sites by sampled bytes -- the same way a CPU
profiler ranks compute hotspots, but for the heap.

Two variants in this demo:

- **Slow:** `Jemalloc.ChurningStrings` -- a fresh `std::string` built per iteration
- **Fast:** `Jemalloc.ReusedBuffer` -- one string reserved and reused per iteration

The story: building a transient string by repeated `append` allocates and grows
on the heap every iteration. jemalloc's sampled profile ranks the string growth
as the dominant site; reusing one reserved buffer collapses the sampled bytes.

## What is jemalloc prof?

`jemalloc` is a general-purpose allocator with an opt-in statistical heap
profiler. Enabled via the `MALLOC_CONF=prof:true` runtime knob, it captures a
backtrace for a sampled fraction of allocations and dumps periodic / at-exit
heap files. `jeprof` (shipped with jemalloc) reads those files and renders a
ranked text report, a graph, or a flame-style breakdown.

- **Best for:** allocation hotspots in long-running services that already link
  jemalloc, and any "which call stack owns the most live/allocated bytes?"
  question where sampling (not full instrumentation) is the right trade-off.
- **How it works:** the allocator itself records a backtrace on a sampled
  subset of allocations (interval set by `lg_prof_sample`); analysis is done
  offline with `jeprof` against the binary + `.heap` files.
- **Overhead:** low and tunable -- sampling means cost scales with the sample
  rate, not the allocation count. Far cheaper than the Valgrind family.
- **Skip it for:** exact per-allocation accounting (sampling is statistical --
  use heaptrack for exact counts), CPU profiling, and thread races.

**In vernier:** `--profile jemalloc` is wrap-externally -- preload libjemalloc
and enable prof via the environment, pointing `prof_prefix` at the per-test
artifact dir. Artifacts land in `<TestName>.jemalloc/` as `jeprof.*.heap`,
analyzed with `jeprof --text <binary> <heap>`.

## Prerequisites

```bash
make compose-debug
# libjemalloc2 + jeprof ship in the dev container.
# Locally: apt install libjemalloc2 libjemalloc-dev
```

`bench doctor` reports whether both the runtime (`libjemalloc.so`) and the
analyzer (`jeprof`) are present.

## Step 1: Profile the Slow Path

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  LD_PRELOAD=$(ldconfig -p | awk "/libjemalloc.so / {print \$4; exit}") \
  MALLOC_CONF=prof:true,prof_prefix:/tmp/jeprof.slow \
    ./bin/ptests/BenchDemo_16_JemallocProfiler \
    --profile jemalloc --quick --cycles 200 \
    --gtest_filter="Jemalloc.ChurningStrings"
  jeprof --text ./bin/ptests/BenchDemo_16_JemallocProfiler /tmp/jeprof.slow.*.heap | head -20
'
```

Expected: the top entries by sampled bytes resolve to `std::string::append` /
string growth invoked from `16_JemallocProfiler_Demo.cpp`.

```
Using local file ./bin/ptests/BenchDemo_16_JemallocProfiler.
Total: 312.4 MB
   245.1  78.5%  78.5%    245.1  78.5%  std::string::_M_append
    61.0  19.5%  98.0%    306.1  98.0%  Jemalloc_ChurningStrings_Test::TestBody
```

## Step 2: Profile the Fast Path

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  LD_PRELOAD=$(ldconfig -p | awk "/libjemalloc.so / {print \$4; exit}") \
  MALLOC_CONF=prof:true,prof_prefix:/tmp/jeprof.fast \
    ./bin/ptests/BenchDemo_16_JemallocProfiler \
    --profile jemalloc --quick --cycles 200 \
    --gtest_filter="Jemalloc.ReusedBuffer"
  jeprof --text ./bin/ptests/BenchDemo_16_JemallocProfiler /tmp/jeprof.fast.*.heap | head -20
'
```

Now `Total` sampled bytes drop sharply: the single up-front `reserve` plus
`clear()`-keeps-capacity means the loop reuses one buffer and allocates nothing,
so the dominant site disappears from the profile.

## Step 3: Diff

Reserve-once + reuse removes the per-iteration string allocation. `jeprof
--text` shows the dominant site collapsing; `jeprof --gv` (or `--svg`) renders
the same as a call graph. The improvement that perf would show only as malloc
time is here pinned to the allocation site.

## When to Use jemalloc prof

- Services that already link jemalloc -- profiling is just two environment
  variables, no rebuild.
- Sampled allocation hotspots where exact counts are unnecessary and low
  overhead matters more.
- Comparing live-heap ownership across versions: `jeprof --base` diffs two
  heap dumps to show what a change added or removed.

## Overhead

Low and tunable via the sample rate (`lg_prof_sample`). Because it samples
rather than instruments every allocation, cost scales with the sample rate, not
the allocation volume -- run realistic `--cycles`.

## Key Takeaways

- jemalloc prof answers "which stack owns the most allocated bytes?" via
  statistical sampling -- cheap enough for realistic workloads.
- Enabled purely through `LD_PRELOAD` + `MALLOC_CONF`; no rebuild required.
- `jeprof` ranks sites (text), graphs them (`--gv`/`--svg`), and diffs heap
  dumps (`--base`).
- Sampling trades exactness for low overhead -- use heaptrack when you need
  exact per-allocation counts.

## See Also

- [Demo 21 (heaptrack)](21_HEAPTRACK_PROFILER.md) -- exact (non-sampled)
  allocation-site profiling
- [Demo 14 (Massif)](14_MASSIF_PROFILER.md) -- heap size-over-time graph
