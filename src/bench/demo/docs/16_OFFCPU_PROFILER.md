# Demo 16: Off-CPU Profiling -- See Where Threads Stop Running

## Overview

On-CPU profilers (perf, gperf, callgrind, rapl) measure where threads run.
None of them shows where threads spend time *blocked*: sleep, mutex wait,
I/O wait, scheduler delay. Off-CPU profiling is the complementary view, and
the only way to localize lock contention or I/O-bound code from a profile.

Two variants in this demo:

- **Slow:** `OffCpu.MutexCounter` -- two threads contend for a single mutex
- **Fast:** `OffCpu.AtomicCounter` -- same workload using `std::atomic`

The story: an on-CPU profile of `MutexCounter` shows a mix of
`futex_wait`, `pthread_mutex_lock`, and the actual counter increment.
The off-CPU profile shows where the threads *blocked* -- pinning the
lock as the cause. Replacing the mutex with `std::atomic::fetch_add`
eliminates the blocked time entirely.

## What is off-CPU profiling?

Off-CPU profiling is the *inverse* of every other profiler in vernier:
instead of sampling threads that are running, it captures stacks every
time a thread *stops* running. The result is a flame graph of where
threads went to sleep, weighted by how long they stayed asleep.

- **Best for:** lock contention, I/O wait, scheduler delay, condition
  variables, blocking syscalls -- anywhere a thread isn't burning CPU
  but is still blocking your progress.
- **How it works:** a bpftrace probe attaches to
  `finish_task_switch()` in the kernel; on every context switch off the
  CPU, it captures the user-space stack and the time until the thread
  runs again. Aggregated, that becomes the off-CPU flame graph.
- **Overhead:** per-context-switch kernel probe + ustack collection.
  Busy systems pay more; vernier limits the trace to measured windows
  to keep cost bounded.
- **Requires:** root (kprobes), kernel BTF, and -- in Docker -- a
  tracefs bind-mount + `--privileged`.
- **Skip it for:** CPU-bound code (use perf / gperf), heap (massif),
  GPU work, or anywhere you can't get root.

**In vernier:** `--profile offcpu` runs the bpftrace off-CPU script
around the measured window. The text-format flame graph lands in
`<TestName>.offcpu/offcpu.txt`.

## Prerequisites

```bash
make compose-debug
# bpftrace ships in the dev container; root is required for kprobes.
```

**Docker note:** the default dev container does not mount tracefs.
For the off-CPU backend to actually attach kprobes inside Docker,
re-run the container with `--mount type=bind,source=/sys/kernel/tracing,target=/sys/kernel/tracing,readonly=false`
(and `--privileged`). On bare metal, sudo is sufficient.

## Step 1: Profile the Slow Path

```bash
docker compose run --rm -T --privileged dev bash -c '
  cd build/native-linux-debug
  sudo ./bin/ptests/BenchDemo_13_OffCpuProfiler \
    --profile offcpu --quick --cycles 50 \
    --gtest_filter="OffCpu.MutexCounter"
  head -30 OffCpu.MutexCounter.offcpu/offcpu.txt
'
```

Expected (top of `offcpu.txt`): stacks attributing the bulk of blocked
nanoseconds to `pthread_mutex_lock` / `__lll_lock_wait` from the
`std::lock_guard` in `13_OffCpuProfiler_Demo.cpp`.

## Step 2: Profile the Fast Path

```bash
docker compose run --rm -T --privileged dev bash -c '
  cd build/native-linux-debug
  sudo ./bin/ptests/BenchDemo_13_OffCpuProfiler \
    --profile offcpu --quick --cycles 50 \
    --gtest_filter="OffCpu.AtomicCounter"
  head -30 OffCpu.AtomicCounter.offcpu/offcpu.txt
'
```

`offcpu.txt` is much shorter / sparser: the atomic version doesn't go
off-CPU on the contended counter, so the only stacks present are
scheduler-noise (thread creation, exit), not application paths.

## Step 3: Cross-Reference with On-CPU

Run gperf or callgrind on the same two variants. The on-CPU profile
also shows the mutex calls -- but as *time spent in the call*, not
*where the thread blocked*. Off-CPU complements rather than replaces.

## When to Use

- Any code with `std::mutex`, `std::condition_variable`, or `pthread`
  primitives where contention is suspected.
- I/O-bound paths where you suspect a blocking syscall is the bottleneck.
- Workloads where on-CPU profilers show mostly sleep / wait stacks
  (`__nptl_death_event`, `pthread_cond_signal`); the off-CPU view
  shows the *callers* of those waits.

## Overhead

Per-context-switch kprobe + ustack collection. On busy systems the
overhead is non-trivial during the trace, but the trace only runs
during measured windows. Use `--quick` to keep run-time bounded.

## Key Takeaways

- Off-CPU is the *inverse* view of every other profiler -- it shows
  where threads stopped running, not where they ran.
- Single highest-leverage diagnostic for lock contention, blocking
  I/O, scheduler delay, and condition-variable waits.
- Output is a flame graph weighted by blocked time per stack.
- Requires root + tracefs; containers need `--privileged` or a
  tracefs bind-mount.

## See Also

- [Demo 9 (bpftrace)](09_BPFTRACE_PROFILER.md) -- the same eBPF
  machinery, different probe targets (syscalls vs context switches)
- [Demo 6 (Thread Scaling)](06_THREAD_SCALING.md) -- on-CPU view of
  the same mutex-vs-atomic comparison
