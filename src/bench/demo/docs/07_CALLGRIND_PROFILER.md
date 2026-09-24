# Demo 07: Callgrind Profiler

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-24, written for the Vernier 1.0.4 release; captured from
the development tree at project version 1.0.3, whose CLI reported `bench 1.0.3`.
The `bench compare` output was produced from the CSVs that session saved, by
the CLI of a later development tree, which reports `bench 1.0.3` too.

## Overview

Walkthrough 01 timed the two versions of `join` and found one of them tens of
times slower. A timing says how long something took; it does not say what the
slower version spends the time on, and it moves a little from one run to the
next. Callgrind counts the instructions a program executes, function by
function and source line by source line, and counts the same work the same way
on every run. This walkthrough counts both versions of `join`, reads where the
slow version's instructions go, and counts again to see the numbers repeat. It
also shows what callgrind gets wrong on this board, and how to read around it.

## What is Callgrind?

Callgrind is one of Valgrind's tools. It runs the program on a simulated CPU,
counts every instruction it executes, and charges each one to the function and
the source line it came from. It also reconstructs which function called which,
to add up what a call costs including everything it calls. The counts do not
depend on how busy the machine is or which core the program runs on. The price
is speed: on this rig the profiled calls ran about 50 and 94 times slower than
natively (step 2). And an instruction count is not a time: it says nothing about
waiting for memory, which a timing includes.

**In Vernier:** `bench run <binary> --profile callgrind` runs the benchmark
under `valgrind --tool=callgrind` and writes one profile of the whole run to
`bench-out/<binary>.callgrind/callgrind.out` (`--profile-output-dir` replaces
`bench-out`). The CSV's `profileDir` column names that folder, and
`callgrind_annotate` reads the file. Running the binary with
`--profile callgrind` without `bench run` collects nothing; see
[If It Does Not Match](#if-it-does-not-match).

**Needs:** valgrind, from the rig document's
[package list](../../docs/rigs/RIG_PI4.md#2-one-time-setup). Counts by source
line need line tables, which a Release build leaves out, so the join example is
compiled with `-g` in every build type. `-g` adds debug information and changes
no generated code: the example's object file disassembles identically with and
without it, with this rig's g++ 14 and with clang 21 on x86.

## The Example

The code is walkthrough 01's `join`, whose [example section](01_BASIC_WORKFLOW.md#the-example)
shows both versions in full. In one line each: `joinV0` builds the result with
`out = out + part + sep;`, which copies everything joined so far once per part,
and `joinV1` reserves the final size once and appends in place.

The demo binary has four tests:

- `CallgrindProfiler.JoinV0` and `CallgrindProfiler.JoinV1` measure one version
  each over 1,000 words, as demo 01 does, one CSV row each. Step 2 profiles
  these two.
- `CallgrindProfiler.InstructionCounts` counts each version's instructions per
  call under callgrind, and fails unless V0 executes more than five times V1's
  and a second run counts the same (step 4).
- `CallgrindProfiler.CountCalls` calls one version a given number of times for
  `InstructionCounts`, and skips itself in any other run.

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_07_CallgrindProfiler \
  --gtest_filter='CallgrindProfiler.JoinV*' --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
Note: Google Test filter = CallgrindProfiler.JoinV*
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from CallgrindProfiler
[ RUN      ] CallgrindProfiler.JoinV0
[target-time] 50.000 ms -> cycles=48 (calibrated 1035.0000 us/call, batch of 1)
[CallgrindProfiler.JoinV0]  1004.542 us/call  CV=0.1%  ~995 calls/s  (p10=1003.746 p90=1006.081 sd=1.207)
[       OK ] CallgrindProfiler.JoinV0 (489 ms)
[ RUN      ] CallgrindProfiler.JoinV1
[target-time] 50.000 ms -> cycles=2584 (calibrated 19.3438 us/call, batch of 64)
[CallgrindProfiler.JoinV1]  19.817 us/call  CV=2.4%  ~50.5K calls/s  (p10=19.064 p90=20.263 sd=0.476)
[       OK ] CallgrindProfiler.JoinV1 (512 ms)
[----------] 2 tests from CallgrindProfiler (1002 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (1002 ms total)
[  PASSED  ] 2 tests.

==========================================================================
Test                       Median (us)     CV%     Calls/s  Status
--------------------------------------------------------------------------
CallgrindProfiler.JoinV0      1004.542    0.1%         995  OK
CallgrindProfiler.JoinV1        19.817    2.4%       50.5K  OK
--------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

V0 takes 1004.5 us per call and V1 19.8 us, 51 times less, against
repeat-to-repeat spreads (`CV`) of 0.1% and 2.4%. The filter selects the two
measuring tests; the other two run callgrind themselves and come later. Across
seven runs of this command in one session on this rig, V0's median ranged from
949.3 to 1004.5 us and V1's from 19.8 to 22.0 us, and the ratio from 43.8x to
50.7x.

## Step 2: Count the Instructions

```bash
bench run ./build/bin/ptests/BenchDemo_07_CallgrindProfiler --profile callgrind \
  --cycles 10 --repeats 1 -- --gtest_filter='CallgrindProfiler.JoinV*'
```

Captured output, trimmed where marked:

```
Running: valgrind --tool=callgrind --callgrind-out-file=bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out ./build/bin/ptests/BenchDemo_07_CallgrindProfiler --cycles 10 --repeats 1 --profile callgrind --gtest_filter=CallgrindProfiler.JoinV*
...
[ RUN      ] CallgrindProfiler.JoinV0
[CallgrindProfiler.JoinV0]  50712.800 us/call  CV=0.0%  ~20 calls/s  (p10=50712.800 p90=50712.800 sd=0.000)

=== Callgrind Profile ===
Output: bench-out/BenchDemo_07_CallgrindProfiler.callgrind
   Run with --profile-analyze for automatic annotation
   Or manually: callgrind_annotate bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out
   Or: kcachegrind bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out

[       OK ] CallgrindProfiler.JoinV0 (819 ms)
[ RUN      ] CallgrindProfiler.JoinV1
[CallgrindProfiler.JoinV1]  1862.900 us/call  CV=0.0%  ~537 calls/s  (p10=1862.900 p90=1862.900 sd=0.000)
...
==41217== Events    : Ir
==41217== Collected : 29943007
==41217==
==41217== I   refs:      29,943,007
```

`bench run` put `valgrind --tool=callgrind` in front of the binary and told the
benchmark where the profile goes, so the backend reports that folder after each
measured test: one file for the whole run, written when the process exits. The
last lines are valgrind's own: this run executed 29,943,007 instructions.

`--cycles 10 --repeats 1` because a count needs no repeats to average: the same
work counts the same when you run it again (step 5). It also keeps the run
short. Under callgrind V0 took 50,712.8 us per call against 1004.5 us natively,
50 times slower, and V1 94 times slower; at that speed the default of 10,000
calls per repeat would take eight and a half minutes per repeat of V0. Each test
calls its version twelve times: once to check the answer, once to warm up
(`--warmup` defaults to 1) and the ten measured calls.

## Step 3: Read the Report

```bash
callgrind_annotate bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out
```

The report opens with the program's total and then lists functions by the
instructions each executed itself (its self count, callees excluded). Captured
output, the first lines of that list, trimmed where marked:

```
29,943,007 (100.0%)  PROGRAM TOTALS

--------------------------------------------------------------------------------
Ir                   file:function
--------------------------------------------------------------------------------
11,799,891 (39.41%)  ./string/../sysdeps/aarch64/multiarch/../memcpy.S:__GI_memcpy [/usr/lib/aarch64-linux-gnu/libc.so.6]
 2,764,820 ( 9.23%)  ./malloc/./malloc/malloc.c:_int_malloc'2 [/usr/lib/aarch64-linux-gnu/libc.so.6]
 2,320,065 ( 7.75%)  ./malloc/./malloc/malloc.c:_int_malloc [/usr/lib/aarch64-linux-gnu/libc.so.6]
 1,520,597 ( 5.08%)  ./malloc/./malloc/malloc.c:free'2 [/usr/lib/aarch64-linux-gnu/libc.so.6]
 1,133,278 ( 3.78%)  ./malloc/./malloc/malloc.c:_int_free_merge_chunk [/usr/lib/aarch64-linux-gnu/libc.so.6]
 1,020,144 ( 3.41%)  ./malloc/./malloc/malloc.c:malloc [/usr/lib/aarch64-linux-gnu/libc.so.6]
   784,852 ( 2.62%)  ./malloc/./malloc/malloc.c:unlink_chunk.isra.0 [/usr/lib/aarch64-linux-gnu/libc.so.6]
   733,971 ( 2.45%)  ./malloc/./malloc/malloc.c:_int_free_create_chunk [/usr/lib/aarch64-linux-gnu/libc.so.6]
   709,240 ( 2.37%)  ./elf/./elf/dl-lookup.c:do_lookup_x'2 [/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1]
   695,758 ( 2.32%)  /usr/include/c++/14/bits/char_traits.h:vernier::bench::demo::joinV0(...) [...]
   624,596 ( 2.09%)  ./malloc/./malloc/malloc.c:_int_free_chunk [/usr/lib/aarch64-linux-gnu/libc.so.6]
...
```

The run's instructions are copying and allocating: `memcpy` executed 39.41% of
them, and the functions of glibc's allocator together 37.9% (the list continues
past the lines shown). That is V0's work:
its twelve calls, at step 4's count per call, come to 85% of the run, and each
one copies the string joined so far once per part, allocating a new one each
time. The column is `Ir`, instructions executed; percentages are of the program
total.

Further down, the same report annotates source files line by line. The join
example's section, trimmed to `joinV0`:

```
-- Auto-annotated source: src/bench/demo/examples/join/src/Join.cpp
--------------------------------------------------------------------------------
Ir

...
    96 ( 0.00%)  std::string joinV0(const std::vector<std::string>& parts, char sep) {
     .             std::string out;
48,084 ( 0.16%)    for (const std::string& part : parts) {
     .               // Two temporaries per part, and a copy of everything joined so far
     .               out = out + part + sep;
     .             }
     .             return out;
    72 ( 0.00%)  }
```

The line that does the work shows no instructions of its own. g++ inlined
`operator+`, and the copies and allocations it makes, into `joinV0`, and
callgrind charges each instruction to the source line the debug information
names for it: here, lines of the standard library headers. That is why the
function list shows `joinV0` under `char_traits.h`. The copy itself is on this
line of `char_traits.h`:

```
-- Auto-annotated source: /usr/include/c++/14/bits/char_traits.h
...
1,236,228 ( 4.13%)      return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
14,967,647,846 (49987.1%)  => /usr/include/c++/14/bits/basic_string.h:vernier::bench::demo::joinV0(...)
5,705,616 (19.05%)  => ./string/../sysdeps/aarch64/multiarch/../memcpy.S:__GI_memcpy (11,964x)
```

The first number is the line's own count. The lines starting with `=>` are calls
made from that line, each with what the call cost including everything it
called, and they come from callgrind's reconstruction of calls and returns. On
this board that reconstruction is wrong for this program: a call cannot cost
49987.1% of the program it is part of. The same profile credits `joinV0` with
tens of thousands of calls where the run made twelve, shows functions as
`free'2` and `operator new'2` (`'2` is callgrind's mark for a function it
believes is running inside itself), and charges code from the string headers
to `free'2`. So on this rig read the self counts: the function list and the
counts on source lines. Leave the `=>` lines, the inclusive costs
(`--inclusive=yes`) and the call counts alone. An x86 build profiled the same
way gives a consistent call graph, with twelve calls of each version.

## Step 4: Confirm the Fix

The count per call comes from the demo's own check:

```bash
./build/bin/ptests/BenchDemo_07_CallgrindProfiler --gtest_filter=CallgrindProfiler.InstructionCounts
```

Captured output:

```
Note: Google Test filter = CallgrindProfiler.InstructionCounts
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from CallgrindProfiler
[ RUN      ] CallgrindProfiler.InstructionCounts
[CallgrindProfiler.InstructionCounts]  V0 2128633.0 instr/call  V1 63577.0 instr/call  33.5x  second run: identical
[       OK ] CallgrindProfiler.InstructionCounts (8816 ms)
[----------] 1 test from CallgrindProfiler (8816 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (8816 ms total)
[  PASSED  ] 1 test.
```

V0 executes 2,128,633 instructions per call and V1 63,577: 33.5 times fewer for
the same string. The test does not use the call graph. It runs `CountCalls`
under callgrind with 10 and with 20 calls of each version and divides the
difference between the two program totals by 10: starting the program, making
the input and everything else a run does are the same in both, and cancel. It
repeats the 10-call runs and requires the same totals.

The fix removed 97% of the instructions and 98% of the time: step 1 measured
51 times less time for 33.5 times fewer instructions, so V0's instructions take
longer on average than V1's, 0.47 ns against 0.31 ns on this rig. The
instruction count says how much work the code does; the timing still has to say
what that work costs.

## Step 5: Count Again

Move the profile aside, repeat step 2's command exactly, and compare the two
function lists (`--auto=no` leaves out the source annotation):

```bash
mv bench-out first
bench run ./build/bin/ptests/BenchDemo_07_CallgrindProfiler --profile callgrind \
  --cycles 10 --repeats 1 -- --gtest_filter='CallgrindProfiler.JoinV*'
diff <(callgrind_annotate --auto=no first/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out) \
  <(callgrind_annotate --auto=no bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out)
```

Captured output of the `diff`, trimmed where marked:

```
2c2
< Profile data file 'first/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out' (creator: callgrind-3.24.0)
---
> Profile data file 'bench-out/BenchDemo_07_CallgrindProfiler.callgrind/callgrind.out' (creator: callgrind-3.24.0)
7c7
< Timerange: Basic block 0 - 5690258
---
> Timerange: Basic block 0 - 5690238
9c9
< Profiled target:  ./build/bin/ptests/BenchDemo_07_CallgrindProfiler ... (PID 41217, part 1)
---
> Profiled target:  ./build/bin/ptests/BenchDemo_07_CallgrindProfiler ... (PID 41238, part 1)
21c21
< 29,943,007 (100.0%)  PROGRAM TOTALS
---
> 29,942,938 (100.0%)  PROGRAM TOTALS
106c106
<      6,444 ( 0.02%)  ./stdio-common/./stdio-common/printf_fp.c:__printf_fp_buffer_1.isra.0'2 [/usr/lib/aarch64-linux-gnu/libc.so.6]
---
>      6,418 ( 0.02%)  ./stdio-common/./stdio-common/printf_fp.c:__printf_fp_buffer_1.isra.0'2 [/usr/lib/aarch64-linux-gnu/libc.so.6]
```

Every function in the two lists counted the same number of instructions but
one (the lists stop at 99% of the total, the report's `Thresholds` line):
`__printf_fp_buffer_1`, glibc formatting a floating-point number. The harness
prints the times it measured, and different times take different arithmetic to
print. Compared over all 1,138 functions in the two profiles, five differ: four
of glibc's number-formatting functions and one atomic operation
(`__aarch64_cas4_rel`), 69 instructions of 29.9 million between them. `memcpy`,
the allocator and the join functions count the same.

That is why `InstructionCounts` runs `CountCalls`, which measures and prints no
time, and why it can require identical totals. It also runs valgrind with
`--sim-hints=fallback-llsc`, valgrind's alternative handling of load- and
store-exclusive instruction pairs (MIPS and ARM64 only): on this board, three
identical counting runs of V1 gave totals up to 10 instructions apart without
that hint, and the same total three times with it. On x86 the hint changes
nothing.

## What Should Reproduce

| Reading                                   | On this rig                                        | Elsewhere                                                                          |
| ----------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| V0 / V1 instructions per call (step 4)    | 33.5x: 2,128,633 against 63,577                    | same direction; 29.2x on an x86 laptop (Release), 6.9x in an unoptimized x86 build |
| the same count, run again                 | identical                                          | identical for the same binary, machine and environment                             |
| where V0's instructions go (step 3)       | `memcpy` 39%, glibc's allocator 38%                | the same functions; the shares depend on the C library and CPU                     |
| a profile run again (step 5)              | identical but for number formatting and one atomic | the same                                                                           |
| V0 / V1 time (step 1)                     | 51x here; 43.8x to 50.7x over seven runs           | tens of times in an optimized build                                                |
| slowdown under callgrind                  | 50x (V0) and 94x (V1)                              | tens of times                                                                      |
| call graph (`=>` lines, inclusive, calls) | wrong for this program                             | consistent on x86                                                                  |

The counts are exact for this binary: a different compiler, C library or CPU
executes different instructions, so only the ratio and the functions it points
at carry over. The environment matters a little too: a profile taken with one
more environment variable changed the counts of 16 functions, among them
`getenv`, glibc's and GoogleTest's start-up code and the time formatting, and
of none of `memcpy`, the allocator or the join functions.

## If It Does Not Match

- **`[callgrind] not running under valgrind; instrumentation skipped.`** The
  binary was run with `--profile callgrind` directly, and nothing counted it.
  Use step 2's `bench run`, or run the valgrind command that message prints.
  Under that command the benchmark switches counting on only around each
  measured window, so the profile holds the measured calls and the harness's
  work around them and not the rest of the process: for `JoinV0`, 21,332,272
  instructions for its ten measured calls.
- **Functions named `???` and no source lines.** The binary has no line tables:
  it was stripped, or the example was built without its `-g`.
- **A line in step 3 reports more than 100%.** That is callgrind's call graph on
  an Arm board; read self counts, as step 3 explains.
- **The counts differ from this page.** Expected on another compiler, C library
  or CPU; only the ratio carries over. On this rig and build, a difference of
  tens of instructions in a program total comes from formatting measured times,
  and a few instructions of it from an atomic operation, as in step 5.
- **valgrind stops with `Possibly corrupted debuginfo file`.** The valgrind is
  older than the compiler's debug information (seen with valgrind 3.18 and a
  binary built by clang 21 with `-g` throughout). `InstructionCounts` skips in
  that case; a newer valgrind reads it.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/07_callgrind_profiler.csv run1.csv
```

Output, for the captured session's `run1.csv`:

```
Test                          Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
CallgrindProfiler.JoinV0     961.81400    1004.54000   +42.72600     +4.4%      0.2%      0.1%  neutral
CallgrindProfiler.JoinV1      21.95140      19.81660    -2.13480     -9.7%      1.4%      2.4%  IMPROVEMENT

  1 improvement(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference is step 1's command run once more in the same session, in a
namespace whose host name is `pi4`, so that the CSV's `hostname` column names
the rig rather than the board. The labels compare each median change with the
5% threshold: V0's median moved by 4.4%, less than that, so its row is
`neutral`; V1's moved by 9.7% in the faster direction, more than that, so its
row reads `IMPROVEMENT`. That is all the label says: the two runs are the same
binary on the same board, about six minutes apart. Both rows moved by more
than their own spread (the `Base CV` and `Cand CV` columns, 0.1% to 2.4%): a
fresh process measures a little differently, and over the seven runs V0's
median moved by 5.8% and V1's by 10.8%, as walkthrough 01 shows at length for
its own runs. A timing moves between runs; the instruction counts of steps 4
and 5 do not.

## What Keeps This Page True

- `CallgrindProfiler.InstructionCounts`, in the demo binary, fails unless V0
  executes more than five times V1's instructions per call and a second
  counting run gives the same totals. It is registered with `ctest`, because an
  instruction count does not depend on what else the machine is doing, so every
  build that runs `ctest` checks it (CI's configuration is a Debug x86 build in
  the dev image, where the ratio is 6.9x). It skips where valgrind is not
  installed or cannot read the binary.
- The same `ctest` label runs the two tests that hold the callgrind backend to
  its hint: run the way the hint says, the profile holds the measured calls and
  none of the work before or after them, and under `bench run` it holds the
  whole process.

  ```bash
  ctest --test-dir build -L callgrind
  ```

  ```
  1/3 Test #135: CallgrindWindowHintTest ..............   Passed    1.83 sec
  2/3 Test #136: CallgrindWindowRunnerTest ............   Passed    1.35 sec
  3/3 Test #137: CallgrindDemoInstructionCountsTest ...   Passed    8.88 sec

  100% tests passed, 0 tests failed out of 3
  ```

- The example's unit tests hold both versions to the same answers
  (`ctest --test-dir build -L demo`).

The timings of step 1 are not in `ctest`; they are re-run on the rig with the
commands above before a release, and the reference CSV is re-captured when they
move.

## See Also

- [Walkthrough 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example,
  timed and compared
- [Reference rigs](../../docs/rigs/README.md) -- the board these numbers come
  from, and what reproduces elsewhere
- [Walkthrough 02: perf](02_PERF_PROFILER.md) and
  [walkthrough 03: gperftools](03_GPERF_PROFILER.md) -- hardware counters and
  sampling, the timing side of the same question
- [Demos README](../README.md) -- every demo, and the contract each walkthrough
  meets
