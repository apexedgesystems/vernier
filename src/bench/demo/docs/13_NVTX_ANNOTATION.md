# Demo 13: NVTX Annotation for Nsight Timelines

## Overview

NVTX (NVIDIA Tools Extension) is _instrumentation_, not a profiler. It emits
labeled ranges that any Nsight tool picks up automatically -- the same
benchmark goes from an unlabeled wall of GPU activity to a timeline with
named phases. Vernier exposes `BENCH_NVTX_SCOPE("name")` (RAII range) and
`BENCH_NVTX_MARK("name")` (instantaneous marker). NsightProfiler also wraps
each test's measured window in an NVTX range named after the test, so even
unannotated benchmarks get per-test grouping in nsys for free.

## What is NVTX?

NVTX is NVIDIA's _instrumentation_ API -- header-only, ships with the
CUDA toolkit, no library to link. You wrap a region of code with a
named range; any NVIDIA profiler attached to the process (Nsight
Systems, Nsight Compute, even VTune) renders those ranges as labeled
bars on its timeline.

- **Best for:** turning an opaque kernel-launch wall in nsys into a
  timeline grouped by _your_ concepts -- per-stage, per-frame,
  per-request -- without changing the profiler.
- **How it works:** push/pop into a user-space ringbuffer. Near-zero
  cost when no profiler is attached; the profiler reads the ringbuffer
  out-of-band.
- **Overhead:** essentially free when nothing is listening. Safe to
  leave in production builds.
- **Note:** NVTX does not collect or report anything itself. It only
  _labels_ events for some other tool to display.

**In vernier:** `BENCH_NVTX_SCOPE("name")` opens an RAII range;
`BENCH_NVTX_MARK("name")` drops an instantaneous marker. Compiles to
no-op on CPU-only builds where the CUDA toolkit isn't present, so the
same code builds everywhere. `NsightProfiler` also auto-injects a
top-level range named after each test so nsys can group launches
per-test without any source changes.

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
is _not_ active (no nsys attach), the cost is a near-zero branch -- safe
to leave in production code.

## Key Takeaways

- NVTX is instrumentation, not a profiler -- it labels events for any
  Nsight tool to render.
- `BENCH_NVTX_SCOPE("name")` is the workhorse: RAII range, scope-bound.
- `BENCH_NVTX_MARK("name")` drops an instantaneous marker at a point.
- `NsightProfiler` auto-injects a per-test range, so even unannotated
  tests get per-test grouping in nsys timelines.
- Safe to leave in production: the macros are no-op when NVTX headers
  are absent and near-zero cost when no profiler is attached.

## See Also

- [Demo 11 (Nsight Profiler)](11_NSIGHT_PROFILER.md) -- the tool that
  renders the ranges
- [Demo 19 (CUPTI)](19_CUPTI_KERNEL_METRICS.md) -- in-process kernel
  metrics that complement NVTX timelines
