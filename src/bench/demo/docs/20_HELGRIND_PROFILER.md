# Demo 20: Valgrind Helgrind / DRD -- Catch Data Races in Parallel Code

## Overview

Helgrind is a correctness tool, not a perf tool. The value here is running it
_after_ a parallel optimization -- if a refactor introduced a data race, a
lock-ordering bug, or POSIX-threads misuse, Helgrind flags it before it ships
(and before it becomes a heisenbug in production).

Two variants in this demo:

- **Buggy:** `Helgrind.RacyCounter` -- threads increment a shared unguarded `long`
- **Safe:** `Helgrind.AtomicCounter` -- same workload via `std::atomic`, race-free

The story: an unsynchronized `counter += 1` looks fine and even "works" at low
thread counts, but it is a read-modify-write race. Helgrind shows the two
colliding stacks and the exact source line; the `std::atomic` fix is reported
clean.

## What is helgrind?

`helgrind` is one of Valgrind's tools and the de-facto C/C++ thread-error
detector. It tracks the happens-before relationships between threads and the
locks that protect shared memory, then reports any access to shared data that
is not ordered by a synchronization operation -- along with the two stacks that
race.

- **Best for:** data races, lock-ordering inversions (potential deadlocks),
  and misuse of the pthreads API -- anywhere two threads touch the same memory
  without synchronization. Most valuable as a _post-optimization_ gate: prove a
  parallel refactor didn't introduce a race.
- **How it works:** binary instrumentation through Valgrind's IR; every memory
  access and every lock/unlock is tracked to build the happens-before graph.
- **Overhead:** ~20-100x slower than native. Use a small `--cycles` count
  (5 is plenty) so the threads still overlap but the run stays usable.
- **Skip it for:** performance measurement (Helgrind says nothing about speed),
  single-threaded code, and lock-free code that Helgrind cannot model (atomics
  on some platforms produce false positives -- prefer DRD or annotations there).

### DRD: the alternate detector

DRD is Valgrind's second thread-error tool. It detects the same class of bugs
with a different (lower-memory, per-access) algorithm, and is often better at
condition-variable and `std::atomic` reasoning. In vernier, DRD is selected
through the same backend via `--profile-args drd`; everything else is identical.

**In vernier:** `--profile helgrind` is wrap-externally -- run the binary under
`valgrind --tool=helgrind` (or `--tool=drd`). The log lands in
`<TestName>.helgrind/`. Add `--error-exitcode=1` to fail CI on a detected race.

## Prerequisites

```bash
make compose-debug
# valgrind ships in the dev container (used by callgrind, massif, memcheck, helgrind)
```

## Step 1: Buggy Workload -- Expect a Race Caught

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=helgrind --error-exitcode=0 \
    --log-file=/tmp/racy.helgrind \
    ./bin/ptests/BenchDemo_14_HelgrindProfiler \
    --profile helgrind --quick --cycles 5 \
    --gtest_filter="Helgrind.RacyCounter"
  grep -A8 "Possible data race" /tmp/racy.helgrind | head -20
'
```

Expected: a "Possible data race" report naming a write of size 8, with two
stacks that both resolve to the increment in `14_HelgrindProfiler_Demo.cpp`.

```
==N== Possible data race during write of size 8 at 0x.... by thread #3
==N==    at 0x....: Helgrind_RacyCounter_Test::TestBody()::$_0::operator() ()
==N==    at 14_HelgrindProfiler_Demo.cpp:84
==N== This conflicts with a previous write of size 8 by thread #2
==N==    at 14_HelgrindProfiler_Demo.cpp:84
```

## Step 2: Safe Workload -- Expect No Race

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=helgrind --error-exitcode=0 \
    --log-file=/tmp/atomic.helgrind \
    ./bin/ptests/BenchDemo_14_HelgrindProfiler \
    --profile helgrind --quick --cycles 5 \
    --gtest_filter="Helgrind.AtomicCounter"
  grep -c "Possible data race" /tmp/atomic.helgrind
'
```

Expected: `0` data-race reports. The `std::atomic` increment is a single
indivisible operation, so there is no unsynchronized window for Helgrind to
flag.

## Step 3: Run the DRD Detector

Same backend, alternate tool. DRD is selected via `--profile-args drd`, and you
wrap with `valgrind --tool=drd`:

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=drd --error-exitcode=0 \
    --log-file=/tmp/racy.drd \
    ./bin/ptests/BenchDemo_14_HelgrindProfiler \
    --profile helgrind --profile-args drd --quick --cycles 5 \
    --gtest_filter="Helgrind.RacyCounter"
  grep -A6 "Conflicting" /tmp/racy.drd | head -16
'
```

DRD reports the same race as a "Conflicting load/store" with the two stacks.
DRD is often clearer on condition-variable and atomic-heavy code; Helgrind is
the default. Try both when one is noisy.

## Step 4: Use Helgrind as a CI Gate

After a parallelization pass, run Helgrind on the changed library's ptest. A
race that "passes" today because of timing will be flagged deterministically
here. Add `--error-exitcode=1` to turn a detected race into a CI failure.

## Overhead

~20-100x slower under valgrind. Use `--cycles 5` (or fewer) and a narrow
`--gtest_filter`; you only need a few overlapping thread runs for the detector
to observe the conflict -- do not run Helgrind across a full suite.

## Key Takeaways

- Helgrind/DRD are correctness tools -- a post-optimization gate for threaded
  code, not a perf measurement.
- The value is the _pair of stacks_: a data race is pinned to the two source
  lines that collide.
- `--profile helgrind` runs Helgrind; add `--profile-args drd` to run DRD
  through the same backend.
- `--error-exitcode=1` turns a detected race into a CI failure.
- ~20-100x overhead; always use a small `--cycles` and a narrow filter.

## See Also

- [Demo 16 (Off-CPU)](16_OFFCPU_PROFILER.md) -- where contended threads block
  (the _perf_ side of the same mutex-vs-atomic story)
- [Demo 15 (Memcheck)](15_MEMCHECK_PROFILER.md) -- the memory-error sibling in
  Valgrind's tool family
