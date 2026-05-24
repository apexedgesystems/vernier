# Demo 15: Valgrind Memcheck -- Catch Bugs Introduced by Optimization

## Overview

Memcheck is a correctness tool, not a perf tool. The value here is running
memcheck *after* an optimization pass -- if a refactor introduced a leak,
UAF, or uninitialized read, memcheck will flag it before it ships.

Two variants in this demo:

- **Clean:** `Memcheck.CleanWorkload` -- RAII unique_ptr, zero leaks
- **Buggy:** `Memcheck.WithDeliberateLeak` -- raw `new[]` without `delete[]`

The story: a benchmark that "works" can still leak. Memcheck shows the
exact source line and stack trace.

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

## See Also

- [Demo 11 (Massif)](14_MASSIF_PROFILER.md) -- heap *profile* (sizes/sites)
- [Demo 7 (Callgrind)](07_CALLGRIND_PROFILER.md) -- instruction counts
