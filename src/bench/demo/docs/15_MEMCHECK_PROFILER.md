# Demo 15: Valgrind Memcheck -- Catch Bugs Introduced by Optimization

## Overview

Memcheck is a correctness tool, not a perf tool. The value here is running
memcheck _after_ an optimization pass -- if a refactor introduced a leak,
UAF, or uninitialized read, memcheck will flag it before it ships.

Two variants in this demo:

- **Clean:** `Memcheck.CleanWorkload` -- RAII unique_ptr, zero leaks
- **Buggy:** `Memcheck.WithDeliberateLeak` -- raw `new[]` without `delete[]`

The story: a benchmark that "works" can still leak. Memcheck shows the
exact source line and stack trace.

## What is memcheck?

`memcheck` is Valgrind's default tool and the de-facto C/C++ memory
error detector. It instruments every load/store to track which bytes
have been allocated and initialized, and reports any access that
violates the rules along with a stack trace pointing at the source line.

- **Best for:** finding memory bugs that don't crash but corrupt
  results -- leaks, use-after-free, uninitialized reads, double-frees,
  out-of-bounds heap access. Most valuable as a _post-optimization_
  gate: prove the refactor didn't introduce any of the above.
- **How it works:** binary instrumentation through Valgrind's IR;
  every memory operation is checked against shadow metadata.
- **Overhead:** ~20-30x slower than native. Use `--cycles 1` and run
  only the test you're checking.
- **Skip it for:** performance measurement (memcheck has nothing to
  say about speed), GPU memory (use compute-sanitizer), and code
  that's heavy on custom allocators (memcheck only sees the standard
  ones unless you annotate).

**In vernier:** `--profile memcheck` is wrap-externally -- run the
binary under `valgrind --tool=memcheck --leak-check=full`. The log
file lands in `<TestName>.memcheck/`. Add `--error-exitcode=1` to
fail CI on memory errors.

## Prerequisites

```bash
make compose-debug
# valgrind ships in the dev container
```

## Step 1: Clean Workload -- Expect Zero Leaks

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=memcheck --leak-check=full --error-exitcode=0 \
    --log-file=/tmp/clean.memcheck \
    ./bin/ptests/BenchDemo_12_MemcheckProfiler \
    --profile memcheck --quick --cycles 1 \
    --gtest_filter="Memcheck.CleanWorkload"
  grep -A2 "LEAK SUMMARY" /tmp/clean.memcheck
'
```

Expected:

```
==N== LEAK SUMMARY:
==N==    definitely lost: 0 bytes in 0 blocks
==N==    indirectly lost: 0 bytes in 0 blocks
```

## Step 2: Buggy Workload -- Expect Leaks Caught

```bash
docker compose run --rm -T dev bash -c '
  cd build/native-linux-debug
  valgrind --tool=memcheck --leak-check=full --error-exitcode=0 \
    --log-file=/tmp/leaky.memcheck \
    ./bin/ptests/BenchDemo_12_MemcheckProfiler \
    --profile memcheck --quick --cycles 1 \
    --gtest_filter="Memcheck.WithDeliberateLeak"
  grep -B2 -A6 "definitely lost" /tmp/leaky.memcheck | head -15
'
```

Expected: a stack trace pinning the leak to `new double[WORK]` in
`12_MemcheckProfiler_Demo.cpp:88`, plus a "definitely lost: N bytes in M
blocks" summary scaling with `--cycles`.

```
==N== 800,000,000 bytes in 1000 blocks are definitely lost in loss record 3 of 3
==N==    at 0x48485C3: operator new[](unsigned long)
==N==    by 0x12794D: Memcheck_WithDeliberateLeak_Test::TestBody()::$_1::operator() ()
==N==    at 12_MemcheckProfiler_Demo.cpp:88
```

## Step 3: Use Memcheck as a CI Gate

After an optimization pass, re-run memcheck on the changed library's
ptest. If a refactor accidentally introduced a leak / UAF / uninit read,
the diff against the pre-optimization baseline will surface it before
merge.

Add `--error-exitcode=1` to make memcheck-failures fail the CI job.

## Overhead

~20x slower under valgrind. Use `--cycles 1` (or 2) and run only the
specific test you're checking; do not run memcheck across a full suite.

## Key Takeaways

- Memcheck is a correctness tool -- a post-optimization gate, not a
  perf measurement.
- The value is the _stack trace_: it pins every leak / UAF / uninit
  read to a source line.
- `--error-exitcode=1` turns memcheck violations into CI failures.
- ~20x overhead; always use `--cycles 1` and a narrow `--gtest_filter`.

## See Also

- [Demo 14 (Massif)](14_MASSIF_PROFILER.md) -- heap _profile_ (sizes,
  sites, timeline)
- [Demo 7 (Callgrind)](07_CALLGRIND_PROFILER.md) -- instruction counts
  (the perf-side of Valgrind)
