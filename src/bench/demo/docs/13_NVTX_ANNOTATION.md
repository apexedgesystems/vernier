# Demo 13: NVTX Annotation for Nsight Timelines

## Overview

NVTX (NVIDIA Tools Extension) is *instrumentation*, not a profiler. It emits
labeled ranges that any Nsight tool picks up automatically -- the same
benchmark goes from an unlabeled wall of GPU activity to a timeline with
named phases. Vernier exposes `BENCH_NVTX_SCOPE("name")` (RAII range) and
`BENCH_NVTX_MARK("name")` (instantaneous marker). NsightProfiler also wraps
each test's measured window in an NVTX range named after the test, so even
unannotated benchmarks get per-test grouping in nsys for free.

## Prerequisites

```bash
make compose-debug
# In the dev-cuda container; nvtx3 ships with the CUDA toolkit.
```

## Step 1: Baseline -- Unannotated

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  nsys profile -o /tmp/nvtx_baseline -t cuda,nvtx --force-overwrite=true \
    ./bin/ptests/BenchDemo_10_NvtxAnnotation \
    --quick --gtest_filter="Nvtx.PhasedWorkload"
  nsys stats --report nvtx_pushpop_sum /tmp/nvtx_baseline.nsys-rep | head -20
'
```

Without annotations, the report would show one big `Nvtx.PhasedWorkload`
range (auto-injected by NsightProfiler). Useful but coarse.

## Step 2: Annotated

The demo wraps four conceptual phases:

```cpp
{ BENCH_NVTX_SCOPE("phase_gen");       /* fill */    }
{ BENCH_NVTX_SCOPE("phase_transform"); /* x^2-x+1 */ }
{ BENCH_NVTX_SCOPE("phase_accumulate"); /* sum */    }
{ BENCH_NVTX_SCOPE("phase_finalize");  /* sink */    }
```

Re-run with the same nsys command; the `nvtx_pushpop_sum` report now
breaks down per-phase:

```
 ** NVTX Push-Pop Summary **
 Time (%)  Total Time (ns)  Avg (ns)   Count  Range Name
 --------  ---------------  --------   -----  ----------
 40.1      245,123,000      ...        N      phase_transform
 35.6      217,012,000      ...        N      phase_accumulate
 18.3      111,890,000      ...        N      phase_gen
  6.0       36,872,000      ...        N      phase_finalize
```

(Numbers will vary; ranking order is the durable signal.)

## Step 3: Takeaway

The per-phase breakdown immediately localizes optimization work: in the
example above, `phase_transform` dominates. Without annotation, every
optimization candidate looks equally attractive.

## When to Use

- Multi-stage pipelines where a single test traverses several phases.
- Multi-kernel benchmarks where `nsys` would otherwise show generic
  `kernel_launch` ranges without phase context.
- A/B comparisons where two timelines are easier to diff with named
  regions than as wall-of-activity exports.

## Overhead

NVTX push/pop is a userspace ringbuffer write. On runs where profiling
is *not* active (no nsys attach), the cost is a near-zero branch -- safe
to leave in production code.
