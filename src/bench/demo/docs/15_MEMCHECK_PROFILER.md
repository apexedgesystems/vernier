# Demo 15: Valgrind Memcheck

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads)), plus a deliberately wrong join private to the demo
**Captured:** 2026-09-26 (UTC), written for the Vernier 1.0.4 release; captured
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

It calls the wrong join three times (`OFF_BY_ONE_CALLS`) and checks each
answer against `joinV1`'s, which passes. Its first line is the demo helper
[`SkipUnlessUnderValgrind.hpp`](../helpers/SkipUnlessUnderValgrind.hpp): a
case that makes a memory error on purpose must not run in an ordinary test
run, so unless the process is under valgrind the case skips itself and says
how to run it. A fourth test, `Memcheck.FindsTheOffByOne`, is the demo's
own check of what this page shows: it runs the binary under memcheck on
`Memcheck.JoinOffByOne` and on `Memcheck.JoinV1` and reads the two logs
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
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from Memcheck
[ RUN      ] Memcheck.JoinV0
[target-time] 50.000 ms -> cycles=49 (calibrated 1009.0000 us/call, batch of 1)
[Memcheck.JoinV0]  983.184 us/call  CV=0.1%  ~1.0K calls/s  (p10=982.345 p90=984.622 sd=1.027)
[       OK ] Memcheck.JoinV0 (489 ms)
[ RUN      ] Memcheck.JoinV1
[target-time] 50.000 ms -> cycles=2480 (calibrated 20.1562 us/call, batch of 64)
[Memcheck.JoinV1]  20.953 us/call  CV=2.0%  ~47.7K calls/s  (p10=20.035 p90=21.081 sd=0.424)
[       OK ] Memcheck.JoinV1 (517 ms)
[ RUN      ] Memcheck.JoinOffByOne
.../src/bench/demo/cpu/12_MemcheckProfiler_Demo.cpp:131: Skipped
this case makes a memory error for valgrind to find, so it runs only under valgrind: run this binary under `valgrind --tool=memcheck`, or through `bench run --profile memcheck`

[  SKIPPED ] Memcheck.JoinOffByOne (0 ms)
[ RUN      ] Memcheck.FindsTheOffByOne
[Memcheck.FindsTheOffByOne]  JoinOffByOne: 6 errors from 2 contexts; invalid write of size 1: 3 times in 1 contexts, 0 bytes after a block of size 7490
[Memcheck.FindsTheOffByOne]  JoinV1: 0 errors from 0 contexts
[       OK ] Memcheck.FindsTheOffByOne (3710 ms)
[----------] 4 tests from Memcheck (4717 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (4717 ms total)
[  PASSED  ] 3 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] Memcheck.JoinOffByOne

=================================================================
Test              Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------
Memcheck.JoinV0       983.184    0.1%        1.0K  OK
Memcheck.JoinV1        20.953    2.0%       47.7K  OK
-----------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

`joinV0` takes 983.2 us per call and `joinV1` 21.0 us, 47 times less, with
CVs of 0.1% and 2.0%: walkthrough 01's measurement under this demo's names.
`Memcheck.JoinOffByOne` reports `SKIPPED` with the helper's message, because
the run is not under valgrind. `Memcheck.FindsTheOffByOne` ran the binary
under memcheck twice, on the wrong join and on `joinV1`, and printed what the
two logs said: six errors from two contexts, the write three times, no bytes
after a block of 7,490 bytes; nothing for `joinV1`. That took 3.7 s. The CSV
holds the two timing rows.

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
[       OK ] Memcheck.JoinOffByOne (55 ms)
[----------] 1 test from Memcheck (64 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (142 ms total)
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
==97793== Memcheck, a memory error detector
==97793== Copyright (C) 2002-2024, and GNU GPL'd, by Julian Seward et al.
==97793== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==97793== Command: ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --profile memcheck --profile-output-dir memcheck-offbyone --gtest_filter=Memcheck.JoinOffByOne
==97793== Parent PID: 97792
==97793==
==97793== Invalid write of size 1
==97793==    at 0x124684: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:32)
==97793==    by 0x1116FF: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==97793==  Address 0x4f23db2 is 0 bytes after a block of size 7,490 alloc'd
==97793==    at 0x488722C: operator new[](unsigned long) (vg_replace_malloc.c:729)
==97793==    by 0x12464B: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:25)
==97793==    by 0x1116FF: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==97793==
==97793== Invalid read of size 1
==97793==    at 0x488E764: strlen (vg_replace_strmem.c:505)
==97793==    by 0x124693: length (char_traits.h:391)
==97793==    by 0x124693: basic_string<> (basic_string.h:653)
==97793==    by 0x124693: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:36)
==97793==    by 0x1116FF: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==97793==  Address 0x4f23db2 is 0 bytes after a block of size 7,490 alloc'd
==97793==    at 0x488722C: operator new[](unsigned long) (vg_replace_malloc.c:729)
==97793==    by 0x12464B: vernier::bench::demo::memcheck_demo::joinOffByOne(std::vector<...> const&, char) (12_MemcheckProfiler_OffByOne.cpp:25)
==97793==    by 0x1116FF: Memcheck_JoinOffByOne_Test::TestBody() (in .../BenchDemo_12_MemcheckProfiler)
...
==97793==
==97793==
==97793== HEAP SUMMARY:
==97793==     in use at exit: 72 bytes in 1 blocks
==97793==   total heap usage: 265 allocs, 264 frees, 210,224 bytes allocated
==97793==
==97793== LEAK SUMMARY:
==97793==    definitely lost: 0 bytes in 0 blocks
==97793==    indirectly lost: 0 bytes in 0 blocks
==97793==      possibly lost: 0 bytes in 0 blocks
==97793==    still reachable: 72 bytes in 1 blocks
==97793==         suppressed: 0 bytes in 0 blocks
==97793== Reachable blocks (those to which a pointer was found) are not shown.
==97793== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==97793==
==97793== For lists of detected and suppressed errors, rerun with: -s
==97793== ERROR SUMMARY: 6 errors from 2 contexts (suppressed: 0 from 0)
```

Every line begins with the process id. Two errors are reported, each with
the stack that made it and the stack that allocated the block it touched:

| Line                                                                | What it says                                                                                                                                          |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Invalid write of size 1`                                           | a one-byte store to memory the program may not touch                                                                                                  |
| `at ... joinOffByOne(...) (12_MemcheckProfiler_OffByOne.cpp:32)`    | the store, and its source line: `*at = '\0';`                                                                                                         |
| `by ... Memcheck_JoinOffByOne_Test::TestBody()`                     | who called it; the frames cut below it are GoogleTest's                                                                                               |
| `Address 0x4f23db2 is 0 bytes after a block of size 7,490 alloc'd`  | where the byte is: right after a heap block of 7,490 bytes, the joined string's length                                                                |
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
- **That the addresses mean anything.** `0x124684`, `0x4f23db2` and the rest
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
[Memcheck.JoinV1]  2401.000 us/call  CV=0.0%  ~416 calls/s  (p10=2401.000 p90=2401.000 sd=0.000)
[       OK ] Memcheck.JoinV1 (260 ms)
[----------] 1 test from Memcheck (269 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (346 ms total)
[  PASSED  ] 1 test.
```

and of the log:

```
==97797== Memcheck, a memory error detector
==97797== Copyright (C) 2002-2024, and GNU GPL'd, by Julian Seward et al.
==97797== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==97797== Command: ./build/bin/ptests/BenchDemo_12_MemcheckProfiler --cycles 1 --repeats 1 --profile memcheck --profile-output-dir memcheck-v1 --gtest_filter=Memcheck.JoinV1
==97797== Parent PID: 97796
==97797==
==97797==
==97797== HEAP SUMMARY:
==97797==     in use at exit: 72 bytes in 1 blocks
==97797==   total heap usage: 277 allocs, 276 frees, 190,039 bytes allocated
==97797==
==97797== LEAK SUMMARY:
==97797==    definitely lost: 0 bytes in 0 blocks
==97797==    indirectly lost: 0 bytes in 0 blocks
==97797==      possibly lost: 0 bytes in 0 blocks
==97797==    still reachable: 72 bytes in 1 blocks
==97797==         suppressed: 0 bytes in 0 blocks
==97797== Reachable blocks (those to which a pointer was found) are not shown.
==97797== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==97797==
==97797== For lists of detected and suppressed errors, rerun with: -s
==97797== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

`ERROR SUMMARY: 0 errors from 0 contexts`: memcheck watched three calls of
`joinV1` (the check of the answer, the warmup, one measured call) and found
nothing to report. The 72 reachable bytes are the profiler library's, as
above. The result line is one call under memcheck, 2.4 ms against step 1's
21.0 us; it says what memcheck costs, not how fast `joinV1` is.

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
[       OK ] Memcheck.JoinOffByOne (56 ms)
[----------] 1 test from Memcheck (65 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (143 ms total)
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
[Memcheck.JoinV1]  2400.000 us/call  CV=0.0%  ~417 calls/s  (p10=2400.000 p90=2400.000 sd=0.000)
[       OK ] Memcheck.JoinV1 (224 ms)
[----------] 1 test from Memcheck (233 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (310 ms total)
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
| `joinV0` / `joinV1` time | 47x here; 43.5x to 46.9x over the session's seven runs of step 1                                                     | tens of times in an optimized build                                                                                                            |
| memcheck's cost          | one call of `joinV1` 115 times its time without memcheck                                                             | tens to hundreds of times; depends on the machine                                                                                              |
| absolute times           | 983.2 and 21.0 us/call in step 1; 924.6 to 983.2 and 20.4 to 21.4 over seven runs                                    | will differ                                                                                                                                    |

The seven runs are step 1's command run seven times in one session on this
rig, the reference capture and this page's run among them. That range
describes those runs; it is not a bound a run has to meet. An eighth run of
the same command, taken an hour later to qualify the board at the final
revision of this page's code, read 963.5 and 20.2 us per call, 47.6 times:
`joinV1` below the range and the ratio above it, with nothing changed. The
memcheck lines are the readings to carry elsewhere.

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
  .../src/bench/demo/cpu/12_MemcheckProfiler_Demo.cpp:131: Skipped
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
  `SKIPPED` in a sanitizer build.** valgrind does not run a binary built
  with a sanitizer as it runs an ordinary one. With clang 21's address
  sanitizer and valgrind 3.22, in this project's container, the binary ran
  but valgrind's libraries were not mapped into it (`LD_PRELOAD` empty, no
  `vgpreload` line in its maps), so memcheck saw none of its allocations and
  the helper saw no valgrind: the wrong join's case skipped and the check
  would fail for the wrong reason. valgrind 3.18.1 gave up reading such a
  binary before it ran. Both tests know at compile time that the build has a
  sanitizer and skip with
  `this binary is built with a sanitizer, and valgrind does not run such a binary as it runs an ordinary one (its libraries are not mapped into the process, so memcheck sees nothing); use a build without a sanitizer`.

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
Memcheck.JoinV0     935.71700     983.18400   +47.46700     +5.1%      0.1%      0.1%  REGRESSION
Memcheck.JoinV1      21.40070      20.95300    -0.44770     -2.1%      1.8%      2.0%  neutral

  1 regression(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary on this board 28 seconds
before step 1's run, and nothing changed between them. `joinV0` still came
out 5.1% slower than the reference, past the 5% threshold the labels are
drawn at, so its row says `REGRESSION`. As the note under the table says, a
label compares two runs' medians; it is not a significance test, and
`Base CV` and `Cand CV` are each run's own spread, 0.1% here, which says
nothing about the spread between runs. Over the session's seven runs of
step 1, `joinV0`'s median ranged from 924.6 to 983.2 us, 6.3%, and
`joinV1`'s from 20.4 to 21.4 us, 4.8%, with nothing changed, so a run of
unchanged code can land on either side of the threshold: the eighth run
above came out 5.5% faster than the reference on `joinV1` and was labelled
`IMPROVEMENT`. Read the medians and the CVs, and compare the ratio. The CSV
holds times only; what memcheck found is checked by the demo's own test,
below.

## What Keeps This Page True

Three things check what this page shows, and all fail loudly:

- `Memcheck.FindsTheOffByOne`, in the demo binary, runs the binary under
  memcheck as `bench run --profile memcheck` wraps it, plus valgrind's error
  list and an exit code for errors, on `Memcheck.JoinOffByOne` and on
  `Memcheck.JoinV1`, and reads the two logs. It fails unless memcheck reports
  the write once per call, `0 bytes after a block` the size of the joined
  string, naming `joinOffByOne` in the write's stack and in the block's,
  with valgrind exiting with the code it was given; and unless `joinV1`'s
  log counts no error and valgrind exits 0. It skips only in a build with a
  sanitizer (which valgrind does not run as an ordinary binary), where
  valgrind is not installed, under `--profile` (it runs memcheck itself), and
  where valgrind gives up reading the binary, in valgrind's own words. A
  wrong join made
  right fails it: with room for the terminator, memcheck reports nothing,
  and the test says the wrong join has stopped being wrong. It is registered
  with `ctest` under the `demo` and `memcheck` labels.
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
