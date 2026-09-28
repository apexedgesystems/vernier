# Demo 15: Valgrind Memcheck

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads)), plus a deliberately wrong join private to the demo
**Captured:** 2026-09-27 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; valgrind 3.24.0

## Overview

A timer says how fast a function is, and a unit test says its answer is
right. Neither says whether it writes past the end of its buffer. This
walkthrough takes the `join` example from
[walkthrough 01](01_BASIC_WORKFLOW.md), adds a third join that returns the
right string and writes one byte past its buffer on every call, and runs it
under memcheck. Memcheck names the write, the line it is on, the block it
missed and the line that allocated that block; the same run on `joinV1`
reports nothing.

## What is memcheck?

Memcheck is Valgrind's default tool, its memory error detector. Valgrind runs
the program on a synthetic CPU, and memcheck puts its own `malloc`,
`operator new` and their relatives in place of the program's, keeping a
guard zone around every block and a record of which bytes the program may
touch and which it has initialised. Every load and store is checked against
that record. One that misses is reported with the stack that made it, the
heap block its address is nearest to, and the stack that allocated that
block. At exit memcheck prints a heap summary and, with `--leak-check=full`,
what the program still held and whether anything still pointed at it.

- **Best for:** reads and writes outside a block or after its release,
  uninitialised values that decide a branch or reach a system call, and
  leaks: memory nothing points at any more.
- **Overhead:** large, and it depends on what the code does. In step 4 one
  call of `joinV1` took 2.4 ms under memcheck against 21 us in step 1, about
  115 times as long. Run one test at a time under it, with few cycles, and
  never read its timings as measurements.
- **Not for:** time (perf, gperftools and callgrind, walkthroughs
  [02](02_PERF_PROFILER.md), [03](03_GPERF_PROFILER.md) and
  [07](07_CALLGRIND_PROFILER.md)), the heap's size and owners (massif and
  heaptrack, walkthroughs [14](14_MASSIF_PROFILER.md) and
  [21](21_HEAPTRACK_PROFILER.md)), or GPU memory (Compute Sanitizer,
  [walkthrough 17](17_COMPUTE_SANITIZER.md)).

**In Vernier:** `--profile memcheck` selects the memcheck backend, which does
not start valgrind itself. `bench run <binary> --profile memcheck` builds the
wrap and prints it on its first line:
`valgrind --tool=memcheck --leak-check=full --error-exitcode=0 --log-file=<dir>/<binary>.memcheck/memcheck.log <binary> --profile memcheck ...`,
where `<dir>` is `--profile-output-dir` (`bench-out` when you give none).
Memcheck writes one log for the whole process, so profile one test per run
with `--gtest_filter`; no per-test folder is created. Run by hand instead,
with `--profile memcheck` on the binary's own command line, a test that
measures (`Memcheck.JoinV1` here) creates the folder
`Memcheck.JoinV1.memcheck/` in the working directory, which stays empty
unless `--log-file` points into it; run without valgrind, the backend prints
a command that does exactly that. `Memcheck.JoinOffByOne` measures nothing
and creates no folder. Because the wrap passes `--error-exitcode=0`, a
`bench run` with memory errors still exits 0: the log carries the finding.
To fail a job on one, run valgrind yourself
([below](#failing-a-job-on-a-memory-error)).

**Needs:** valgrind, from the rig document's
[package list](../../docs/rigs/RIG_PI4.md#2-one-time-setup); no privileges.
[`bench doctor`](../../docs/rigs/RIG_PI4.md#6-verify-your-rig) reports it as:

```
  [OK]   memcheck   valgrind available (memcheck is the default tool)
```

## The Example

`joinV1`, the version that measures the answer first, allocates once and
appends in place, is the correct join this page ends on:

```cpp
std::string joinV1(const std::vector<std::string>& parts, char sep) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }

  std::string out;
  out.reserve(total); // One allocation, the final size
  for (const std::string& part : parts) {
    out += part;
    out += sep;
  }
  return out;
}
```

The wrong join takes the step someone might take next, building the result
in a raw buffer, and makes the classic mistake with it:

```cpp
std::string joinOffByOne(const std::vector<std::string>& parts, char sep) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }

  // Room for the characters, none for the terminator: the bug
  char* buf = new char[total];
  char* at = buf;
  for (const std::string& part : parts) {
    std::memcpy(at, part.data(), part.size());
    at += part.size();
    *at++ = sep;
  }
  *at = '\0'; // One byte past the end of buf

  // Reads up to the terminator, one byte past the end again; the answer
  // comes out right, so no test of the result can tell
  std::string out(buf);
  delete[] buf;
  return out;
}
```

The buffer has room for the characters and none for the terminator, so the
terminator is written one byte past the end, and constructing the string
from the C string reads it back from there. The buffer is freed and the
right string returned: under memcheck every call for this page returned
exactly what `joinV1` returns, and so did a plain run on an x86-64 laptop,
where the byte past the end landed in what the C library's allocator keeps
after a block of that size. Nothing that checks the answer can tell, and
nothing that times the call can either.

It is not a version of the shared example. The example's versions are held
to the same answers by its unit tests and are safe to run anywhere; this one
is not, so it lives with the demo
([`12_MemcheckProfiler_OffByOne.cpp`](../cpu/12_MemcheckProfiler_OffByOne.cpp)),
compiled with debug information whatever the build type, so memcheck's
report can name its lines. It is `[[gnu::noinline]]` for the same reason the
example's versions are: a function folded into its caller vanishes from the
report.

The demo ([`12_MemcheckProfiler_Demo.cpp`](../cpu/12_MemcheckProfiler_Demo.cpp))
measures `joinV0` and `joinV1` in `Memcheck.JoinV0` and `Memcheck.JoinV1`,
one CSV row each, as walkthrough 01 does. The wrong join has a case of its
own that measures nothing:

```cpp
PERF_TEST(Memcheck, JoinOffByOne) {
  DEMO_SKIP_UNLESS_UNDER_VALGRIND();

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  const std::string EXPECTED = demo::joinV1(PARTS, SEPARATOR);
  for (long call = 0; call < OFF_BY_ONE_CALLS; ++call) {
    EXPECT_EQ(wrong::joinOffByOne(PARTS, SEPARATOR), EXPECTED) << "call " << call;
  }
}
```

It calls the wrong join three times (`OFF_BY_ONE_CALLS`, beside the number
of words and the seed in
[`12_MemcheckProfiler_Workload.hpp`](../cpu/12_MemcheckProfiler_Workload.hpp))
and checks each answer against `joinV1`'s, which passes. Its first line is
the demo helper
[`SkipUnlessUnderValgrind.hpp`](../helpers/SkipUnlessUnderValgrind.hpp): a
case that makes a memory error on purpose must not run in an ordinary test
run, so unless the process is under valgrind the case skips itself and says
how to run it. That is all the demo holds. What this page shows is checked
apart from it, by a test program that runs the demo binary under memcheck
([What Keeps This Page True](#what-keeps-this-page-true)).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_12_MemcheckProfiler \
  --target-time 50ms --repeats 10 --csv run.csv
```

Captured output, with the checkout's path in the skip line shortened to
`...`:

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from Memcheck
[ RUN      ] Memcheck.JoinV0
[target-time] 50.000 ms -> cycles=54 (calibrated 920.0000 us/call, batch of 2)
[Memcheck.JoinV0]  921.213 us/call  CV=0.2%  ~1.1K calls/s  (p10=920.676 p90=926.059 sd=2.305)
[       OK ] Memcheck.JoinV0 (506 ms)
[ RUN      ] Memcheck.JoinV1
[target-time] 50.000 ms -> cycles=2459 (calibrated 20.3281 us/call, batch of 64)
[Memcheck.JoinV1]  20.821 us/call  CV=2.5%  ~48.0K calls/s  (p10=19.919 p90=21.052 sd=0.520)
[       OK ] Memcheck.JoinV1 (508 ms)
[ RUN      ] Memcheck.JoinOffByOne
.../src/bench/demo/cpu/12_MemcheckProfiler_Demo.cpp:86: Skipped
this case makes a memory error for valgrind to find, so it runs only under valgrind: run this binary under `valgrind --tool=memcheck`, or through `bench run --profile memcheck`

[  SKIPPED ] Memcheck.JoinOffByOne (0 ms)
[----------] 3 tests from Memcheck (1015 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (1015 ms total)
[  PASSED  ] 2 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] Memcheck.JoinOffByOne

=================================================================
Test              Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------
Memcheck.JoinV0       921.213    0.2%        1.1K  OK
Memcheck.JoinV1        20.821    2.5%       48.0K  OK
-----------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

`joinV0` takes 921.2 us per call and `joinV1` 20.8 us, 44 times less, with
CVs of 0.2% and 2.5%: walkthrough 01's measurement under this demo's names.
`Memcheck.JoinOffByOne` reports `SKIPPED` with the helper's message, because
the run is not under valgrind. The CSV holds the two timing rows.

## Step 2: Run the Wrong Join Under Memcheck

```bash
bench run ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --profile memcheck \
  --profile-output-dir memcheck-offbyone -- --gtest_filter='Memcheck.JoinOffByOne'
```

Captured output:

```
Running: valgrind --tool=memcheck --leak-check=full --error-exitcode=0 --log-file=memcheck-offbyone/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --profile memcheck --profile-output-dir memcheck-offbyone --gtest_filter=Memcheck.JoinOffByOne
Note: Google Test filter = Memcheck.JoinOffByOne
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Memcheck
[ RUN      ] Memcheck.JoinOffByOne
[       OK ] Memcheck.JoinOffByOne (56 ms)
[----------] 1 test from Memcheck (65 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (141 ms total)
[  PASSED  ] 1 test.
```

The `Running:` line is the wrap `bench run` built. The case ran, because
under valgrind the helper lets it, and it passed: memcheck reports and does
not stop the program, and the answers were right. The one file the run
wrote is the log, in a folder named for the binary and the tool:

```bash
ls memcheck-offbyone/BenchDemo_12_MemcheckProfiler.memcheck
```

```
memcheck.log
```

## Step 3: Read the Report

```bash
cat memcheck-offbyone/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log
```

Captured output. GoogleTest's and `main`'s frames are cut from each stack
and marked `...`, the vector's template arguments are shortened to
`std::vector<...>`, and the binary's directory to `...`:

```
==124554== Memcheck, a memory error detector
==124554== Copyright (C) 2002-2024, and GNU GPL'd, by Julian Seward et al.
==124554== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==124554== Command: ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --profile memcheck --profile-output-dir memcheck-offbyone --gtest_filter=Memcheck.JoinOffByOne
==124554== Parent PID: 124553
==124554==
==124554== Invalid write of size 1
==124554==    at 0x11C424: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:32)
==124554==    by 0x10FE9B: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==124554==  Address 0x4f23ab2 is 0 bytes after a block of size 7,490 alloc'd
==124554==    at 0x488722C: operator new[](unsigned long) (vg_replace_malloc.c:729)
==124554==    by 0x11C3EB: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:25)
==124554==    by 0x10FE9B: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==124554==
==124554== Invalid read of size 1
==124554==    at 0x488E764: strlen (vg_replace_strmem.c:505)
==124554==    by 0x11C433: length (char_traits.h:391)
==124554==    by 0x11C433: basic_string<> (basic_string.h:653)
==124554==    by 0x11C433: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:36)
==124554==    by 0x10FE9B: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==124554==  Address 0x4f23ab2 is 0 bytes after a block of size 7,490 alloc'd
==124554==    at 0x488722C: operator new[](unsigned long) (vg_replace_malloc.c:729)
==124554==    by 0x11C3EB: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:25)
==124554==    by 0x10FE9B: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==124554==
==124554==
==124554== HEAP SUMMARY:
==124554==     in use at exit: 72 bytes in 1 blocks
==124554==   total heap usage: 260 allocs, 259 frees, 209,812 bytes allocated
==124554==
==124554== LEAK SUMMARY:
==124554==    definitely lost: 0 bytes in 0 blocks
==124554==    indirectly lost: 0 bytes in 0 blocks
==124554==      possibly lost: 0 bytes in 0 blocks
==124554==    still reachable: 72 bytes in 1 blocks
==124554==         suppressed: 0 bytes in 0 blocks
==124554== Reachable blocks (those to which a pointer was found) are not shown.
==124554== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==124554==
==124554== For lists of detected and suppressed errors, rerun with: -s
==124554== ERROR SUMMARY: 6 errors from 2 contexts (suppressed: 0 from 0)
```

Every line begins with the process id. Two errors are reported, each with
the stack that made it and the stack that allocated the block it touched:

| Line                                                                | What it says                                                                                                                                          |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Invalid write of size 1`                                           | a one-byte store to memory the program may not touch                                                                                                  |
| `at ... joinOffByOne(...) (12_MemcheckProfiler_OffByOne.cpp:32)`    | the store, and its source line: `*at = '\0';`                                                                                                         |
| `by ... Memcheck_JoinOffByOne_Test::TestBody()`                     | who called it; the frames cut below it are GoogleTest's                                                                                               |
| `Address 0x4f23ab2 is 0 bytes after a block of size 7,490 alloc'd`  | where the byte is: right after a heap block of 7,490 bytes, the joined string's length                                                                |
| `at ... operator new[](unsigned long) (vg_replace_malloc.c:729)`    | the block came from `new[]`, memcheck's own                                                                                                           |
| `by ... joinOffByOne(...) (12_MemcheckProfiler_OffByOne.cpp:25)`    | and the line that asked for it: `char* buf = new char[total];`                                                                                        |
| `Invalid read of size 1`, `at ... strlen (vg_replace_strmem.c:505)` | the second error: `strlen`, memcheck's replacement of the C library's, read the same byte back for `std::string out(buf);` on line 36                 |
| `ERROR SUMMARY: 6 errors from 2 contexts`                           | the case called the join three times, and each call made one write and one read; memcheck counts every occurrence and prints each distinct place once |

The two lines to act on are the ones with a file and line number: the write
at line 32 is the bug, and the allocation at line 25 is where the buffer is
one byte short. The read is the same bug seen a second time.

What not to conclude:

- **That the `LEAK SUMMARY` reports a leak.** `definitely lost` is 0 bytes.
  The 72 bytes `still reachable` are one block that gperftools' profiler
  library, which every benchmark on this rig links, allocates as it loads
  (`ProfileHandler::Init`) and holds until exit; memcheck lists it because
  it was in use at exit, and counts no error for it. On an x86-64 laptop
  whose build links no gperftools, the same run ends with
  `All heap blocks were freed -- no leaks are possible` and prints no
  `LEAK SUMMARY` at all. What every run prints is the `ERROR SUMMARY` line:
  read that one.
- **That the addresses mean anything.** `0x11C424`, `0x4f23ab2` and the rest
  belong to this build and this run.
- **That the line numbers travel.** They are this revision's
  `12_MemcheckProfiler_OffByOne.cpp`; a build of an edited file names other
  lines. The names travel: `joinOffByOne`, `operator new[]`, `strlen`.

## Step 4: Confirm the Fix

```bash
bench run ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --profile memcheck \
  --cycles 1 --repeats 1 --profile-output-dir memcheck-v1 -- --gtest_filter='Memcheck.JoinV1'
cat memcheck-v1/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log
```

Captured output of the run:

```
Running: valgrind --tool=memcheck --leak-check=full --error-exitcode=0 --log-file=memcheck-v1/BenchDemo_12_MemcheckProfiler.memcheck/memcheck.log ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --cycles 1 --repeats 1 --profile memcheck --profile-output-dir memcheck-v1 --gtest_filter=Memcheck.JoinV1
Note: Google Test filter = Memcheck.JoinV1
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Memcheck
[ RUN      ] Memcheck.JoinV1
[Memcheck.JoinV1]  2385.000 us/call  CV=0.0%  ~419 calls/s  (p10=2385.000 p90=2385.000 sd=0.000)
[       OK ] Memcheck.JoinV1 (260 ms)
[----------] 1 test from Memcheck (269 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (345 ms total)
[  PASSED  ] 1 test.
```

and of the log:

```
==124558== Memcheck, a memory error detector
==124558== Copyright (C) 2002-2024, and GNU GPL'd, by Julian Seward et al.
==124558== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==124558== Command: ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --cycles 1 --repeats 1 --profile memcheck --profile-output-dir memcheck-v1 --gtest_filter=Memcheck.JoinV1
==124558== Parent PID: 124557
==124558==
==124558==
==124558== HEAP SUMMARY:
==124558==     in use at exit: 72 bytes in 1 blocks
==124558==   total heap usage: 272 allocs, 271 frees, 189,627 bytes allocated
==124558==
==124558== LEAK SUMMARY:
==124558==    definitely lost: 0 bytes in 0 blocks
==124558==    indirectly lost: 0 bytes in 0 blocks
==124558==      possibly lost: 0 bytes in 0 blocks
==124558==    still reachable: 72 bytes in 1 blocks
==124558==         suppressed: 0 bytes in 0 blocks
==124558== Reachable blocks (those to which a pointer was found) are not shown.
==124558== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==124558==
==124558== For lists of detected and suppressed errors, rerun with: -s
==124558== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

`ERROR SUMMARY: 0 errors from 0 contexts`: memcheck watched three calls of
`joinV1` (the check of the answer, the warmup, one measured call) and found
nothing to report. The 72 reachable bytes are the profiler library's, as
above. The result line is one call under memcheck, 2.4 ms against step 1's
20.8 us; it says what memcheck costs, not how fast `joinV1` is.

## Failing a Job on a Memory Error

`bench run` wraps with `--error-exitcode=0`, so it exits 0 whatever memcheck
found. A job that should stop on a memory error runs valgrind itself, with
an exit code for errors:

```bash
valgrind --tool=memcheck --leak-check=full --error-exitcode=1 --log-file=offbyone.log \
  ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --gtest_filter='Memcheck.JoinOffByOne'
echo "exit $?"
```

```
Note: Google Test filter = Memcheck.JoinOffByOne
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Memcheck
[ RUN      ] Memcheck.JoinOffByOne
[       OK ] Memcheck.JoinOffByOne (55 ms)
[----------] 1 test from Memcheck (64 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (140 ms total)
[  PASSED  ] 1 test.
exit 1
```

The test passed and valgrind exited 1: the errors are in `offbyone.log`, and
the job fails. The same on `joinV1`:

```bash
valgrind --tool=memcheck --leak-check=full --error-exitcode=1 --log-file=v1.log \
  ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --cycles 1 --repeats 1 --gtest_filter='Memcheck.JoinV1'
echo "exit $?"
```

```
Note: Google Test filter = Memcheck.JoinV1
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Memcheck
[ RUN      ] Memcheck.JoinV1
[Memcheck.JoinV1]  2678.000 us/call  CV=0.0%  ~373 calls/s  (p10=2678.000 p90=2678.000 sd=0.000)
[       OK ] Memcheck.JoinV1 (235 ms)
[----------] 1 test from Memcheck (245 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (321 ms total)
[  PASSED  ] 1 test.
exit 0
```

Without `--error-exitcode`, valgrind exits with the program's own status,
which is 0 in both cases here.

## What Should Reproduce

| Reading                  | On this rig                                                                                                          | Elsewhere                                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| the write                | `Invalid write of size 1`, once per call, `0 bytes after a block of size 7,490`, at line 32 of the wrong join's file | should match: it depends on the code, not the machine; the same write after the same block on an x86-64 laptop with GCC 11.4 and with clang 21 |
| the read                 | `Invalid read of size 1` in `strlen`, once per call, at the same address                                             | the same wherever memcheck replaces the C library's `strlen`, as it did on x86-64; a `strlen` it does not replace may read more bytes at once  |
| errors and contexts      | 6 errors from 2 contexts                                                                                             | 6 errors; 6 contexts where the compiler unrolled the case's loop (clang 21 on x86-64 did), because each call is then its own call site         |
| what `joinV1` reports    | 0 errors from 0 contexts, in every run for this page and in five more taken for it                                   | should match; [If It Does Not Match](#if-it-does-not-match) has one report that is not the program's                                           |
| `joinV0` / `joinV1` time | 44x here; 44.2x to 48.3x over the session's twelve runs of step 1                                                    | tens of times in an optimized build                                                                                                            |
| memcheck's cost          | one call of `joinV1` 115 times its time without memcheck                                                             | tens to hundreds of times; depends on the machine                                                                                              |
| absolute times           | 921.2 and 20.8 us/call in step 1; 915.8 to 1013.0 and 19.8 to 21.1 over twelve runs                                  | will differ                                                                                                                                    |

The twelve runs are step 1's command run twelve times in one session on
this rig, the reference capture and this page's run among them. That range
describes those runs; it is not a bound a run has to meet, and another run
of the same command can land outside it. The memcheck lines are the readings
to carry elsewhere.

## If It Does Not Match

- **`Memcheck.JoinOffByOne` reports `SKIPPED` in step 2.** The run was not
  under valgrind. The binary run on its own, with or without
  `--profile memcheck`, prints:

  ```
  Note: Google Test filter = Memcheck.JoinOffByOne
  [==========] Running 1 test from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 1 test from Memcheck
  [ RUN      ] Memcheck.JoinOffByOne
  .../src/bench/demo/cpu/12_MemcheckProfiler_Demo.cpp:86: Skipped
  this case makes a memory error for valgrind to find, so it runs only under valgrind: run this binary under `valgrind --tool=memcheck`, or through `bench run --profile memcheck`

  [  SKIPPED ] Memcheck.JoinOffByOne (0 ms)
  [----------] 1 test from Memcheck (0 ms total)

  [----------] Global test environment tear-down
  [==========] 1 test from 1 test suite ran. (0 ms total)
  [  PASSED  ] 0 tests.
  [  SKIPPED ] 1 test, listed below:
  [  SKIPPED ] Memcheck.JoinOffByOne
  ```

  Run it through `bench run --profile memcheck`, as step 2 does, or under
  `valgrind --tool=memcheck` yourself.

- **`bench run` stops before the test starts:**

  ```
  Error: tool not found: 'valgrind' is not on PATH; --profile memcheck runs the benchmark under it. Install valgrind, or run `bench doctor` to see which profilers this machine can use
  ```

  valgrind is not installed, or not on `PATH`; the doctor reports
  `[FAIL] memcheck   valgrind binary not found on PATH`. Install the package
  from the rig document's one-time setup.

- **`Memcheck.FindsTheOffByOne` reports `SKIPPED`, quoting valgrind giving
  up.** A valgrind older than the compiler that built the binary can fail to
  read its debug information before the program runs; the skip quotes
  valgrind's own two lines,
  `Valgrind: debuginfo reader: Possibly corrupted debuginfo file.` and
  `Valgrind: I can't recover.  Giving up.  Sorry.`, which valgrind 3.18.1
  printed for a clang 21 Debug build on an x86-64 laptop. Use a newer
  valgrind, or a build the installed one can read.

- **Memcheck's report names no function of the binary, only `???`, and
  `Memcheck.FindsTheOffByOne` reports `SKIPPED`, quoting valgrind.** valgrind
  could not read the binary's symbols and said so before the program ran:
  for a GCC 11.4 Release build linked by mold 1.0.3 on an x86-64 laptop,
  valgrind 3.18.1 printed `Can't make sense of .rodata section mapping` about
  the binary. Memcheck still found the write and the read, 6 errors from 2
  contexts, 0 bytes after a block of size 7,490, with `???` for every frame
  in the binary. The check still asserts the count, the offset, the block
  size and `joinV1`'s clean run, and skips only once they pass, quoting
  valgrind's lines. The build used mold because the project links with it
  when it finds it (`VERNIER_USE_FAST_LINKER`, on by default); configured
  with `-DVERNIER_USE_FAST_LINKER=OFF`, the same build was linked by GNU ld,
  the compiler's default, and the same valgrind named `joinOffByOne` at lines
  32, 25 and 36.

- **`joinV1`'s log reports one error that names no line of the program:**
  `Syscall param write(buf) points to uninitialised byte(s)`, in `libunwind`
  under `libprofiler`, from `_dl_init`. That is gperftools' profiler library
  probing memory as it loads, before `main`: on x86-64 the probe writes the
  bytes to a pipe (the `write(buf)` of the report) to see whether they can be
  read, and when they are uninitialised memcheck reports the write's
  argument. The bytes are never used. It appeared in a clang 21 Debug build
  in this project's Ubuntu 24.04 container and never on this rig, whose
  profiler library does not load libunwind. `Memcheck.FindsTheOffByOne`
  suppresses that one report in its own runs; to do the same by hand, put

  ```
  {
     gperftools start-up memory probe
     Memcheck:Param
     write(buf)
     ...
     obj:*libunwind*
  }
  ```

  in a file and pass `--suppressions=<file>` to valgrind.

- **`6 errors from 6 contexts` instead of 2.** The compiler unrolled the
  case's three-call loop, so each call is a different call site and memcheck
  files its write and its read separately. The counts are the same: three
  writes, three reads.

- **`Memcheck.FindsTheOffByOne` and the helper's valgrind test report
  `SKIPPED` in a build with the address or the thread sanitizer.** valgrind
  cannot check either. With clang 21's address sanitizer and valgrind 3.22,
  in this project's container, the binary ran but valgrind's libraries were
  not mapped into it (`LD_PRELOAD` empty, no `vgpreload` line in its maps),
  so memcheck saw none of its allocations and the helper saw no valgrind:
  the wrong join's case skipped and the check would fail for the wrong
  reason. valgrind 3.18.1 gave up reading such a binary before it ran. A
  thread-sanitizer build never reached `main` under valgrind 3.22. Both
  tests know at compile time which sanitizer the build has and skip with
  `this binary is built with the address or the thread sanitizer, whose builds valgrind cannot check; run this test in a build without either`.
  A build with the undefined-behaviour sanitizer is not among them: it ran
  under valgrind as an ordinary build, and the tests do not skip there.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/15_memcheck_profiler.csv run.csv
```

Output for step 1's `run.csv`, printed by the `bench` CLI built from the tree
this page ships in (`bench 1.0.3`):

```

Test                 Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
---------------  ------------  ------------  ----------  --------  --------  --------  ------------
Memcheck.JoinV0     959.29600     921.21300   -38.08300     -4.0%      0.1%      0.2%  neutral
Memcheck.JoinV1      20.88860      20.82130    -0.06730     -0.3%      1.8%      2.5%  neutral

  2 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary on this board immediately
before step 1's run (the two CSVs' timestamps are one second apart), and
nothing changed between them: `joinV0` came out -4.0% against it and
`joinV1` came out -0.3% against it, both inside the 5% threshold the labels
are drawn at, so both rows say `neutral`. That is this pair of runs, not a
property of the code. The CSVs' `hostname` column records the name visible
to the process that captured each: this reference was captured in a UTS
namespace named `pi4` on the same rig, so it records `pi4`, while the rig's
ordinary captures, step 1's `run.csv` among them, record `raspberrypi`. As
the note under the table says, a label compares two runs' medians; it is
not a significance test, and `Base CV` and `Cand CV` are each run's own
spread, which says nothing about the spread between runs: 0.1% and 0.2% for
`joinV0` here. Over the session's twelve runs of step 1,
`joinV0`'s median ranged from 915.8 to 1013.0 us, 10.6%, and `joinV1`'s from
19.8 to 21.1 us, 6.7%, with nothing changed, so a run of unchanged code can
land on either side of the threshold; `joinV0`'s median is the noisiest
measurement on this page. Read the medians and the CVs, and compare the
ratio. The CSV holds times only; what memcheck found is checked by
`Memcheck.FindsTheOffByOne`, below.

## What Keeps This Page True

Three things check what this page shows, and all fail loudly:

- `Memcheck.FindsTheOffByOne`, a test program of its own beside the demo
  ([`12_MemcheckProfiler_uTest.cpp`](../cpu/utst/12_MemcheckProfiler_uTest.cpp),
  built as `TestDemoMemcheck`), runs the demo binary under memcheck as
  `bench run --profile memcheck` wraps it, plus valgrind's error list and an
  exit code for errors, on `Memcheck.JoinOffByOne` and on `Memcheck.JoinV1`,
  and reads the two logs. It fails unless each case it selects runs to its
  end, memcheck reports the write once per call, `0 bytes after a block` the
  size of the joined string, naming `joinOffByOne` in the write's stack and
  in the block's, with valgrind exiting with the code it was given; and
  unless `joinV1`'s log counts no error and valgrind exits 0. It skips only
  in a build with the address or the thread sanitizer (which valgrind cannot
  check), where valgrind is not installed, where valgrind gives up reading
  the demo binary, and where valgrind cannot read the demo binary's symbols,
  once everything but the names has passed; those two skips quote valgrind's
  own lines, and the last needs the report to agree, with the write's own
  frame in the demo binary and unnamed. A wrong join made right fails it,
  whether or not valgrind can read the symbols: with room for the
  terminator, memcheck reports nothing, and the test says the wrong join has
  stopped being wrong. Beside it, `MemcheckLogTest` holds the log reading
  those skips rest on to its cases (a warning about a library, another of
  valgrind's reasons, a frame valgrind named, a run that crashed). All of
  them are registered with `ctest` under the `demo` and `memcheck` labels;
  `ctest --test-dir build -L memcheck` runs them alone.
- The helper's own tests,
  `SkipUnlessUnderValgrindTest.PlainRunSkipsTheProbe` and
  `SkipUnlessUnderValgrindTest.ValgrindRunRunsTheProbe`, run their binary as
  a child on a probe case that uses the helper, plainly and under memcheck,
  and fail if the probe runs in the plain run or skips under valgrind. The
  probe itself, `SkipUnlessUnderValgrindProbe.RunsOnlyUnderValgrind`, is
  reported as skipped by `ctest` in an ordinary run, which is the helper
  working.
- The example's unit tests hold `joinV0` and `joinV1` to the same answers,
  under the same label.

An ordinary test run includes all of them, and every test it runs should
pass:

```bash
ctest --test-dir build -L demo
```

The demo's two timing tests are not registered: what they measure belongs to
the machine they run on. This repository has no continuous-integration lane
on the reference board, so before a release the page's commands are run on
the rig by hand, and the page and its reference CSV are re-captured when
what they show changes.

## See Also

- [Demo 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example,
  measured and compared
- [Demo 14: Massif](14_MASSIF_PROFILER.md) -- the heap's size and owners,
  under the same valgrind
- [Demo 21: heaptrack](21_HEAPTRACK_PROFILER.md) -- who allocates, and how
  often
- [Demo 17: Compute Sanitizer](17_COMPUTE_SANITIZER.md) -- the same kind of
  check for GPU memory
- [CPU guide: memory correctness with memcheck](../../docs/CPU_GUIDE.md#memory-correctness-with-memcheck)
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
