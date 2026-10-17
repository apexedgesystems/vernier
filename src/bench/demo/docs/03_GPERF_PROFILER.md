# Demo 03: gperftools CPU Profiler

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-29 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`. The doctor's `gperf` row in Needs comes from this rig on
2026-10-03 (UTC), a later tree at the same version; the skip and the row
quoted under If It Does Not Match, from a Debug build of that tree without
gperftools on an x86-64 laptop.

## Overview

A timing says how long a function takes, not where its time goes. This
walkthrough asks a sampling profiler, gperftools, where the time of the slow
`join` goes. The profile names `joinV0`, and then shows that almost none of the
time is in `joinV0`'s own code: it is in the copying and allocating that
`joinV0` asks the C library to do. After the fix, the profile moves the way
the timing does.

## What is gperftools?

gperftools' CPU profiler is a sampling profiler. A timer interrupts the program
100 times per second of CPU time (gperftools' default), and each interrupt
records where the program was: the instruction it was running and the chain of
calls that led there. Afterwards `google-pprof` counts the samples per
function. A function that holds 60% of the samples was running about 60% of the
time, give or take: with about two hundred samples a share is not exact, and
ten profiles of V1 taken in this session put `joinV1`'s own share anywhere from
55.5% to 64.0%.

At that rate the profiler costs little. Over five alternating runs of each
version on this rig, the profiled medians averaged 1.0% (V0) and 0.9% (V1)
above the unprofiled ones, while the unprofiled runs alone spread over 2.7%
(V0) and 5.1% (V1).

It is not the tool for exact counts of instructions or calls (Callgrind), for
hardware events such as cache misses (perf), or for counting allocations
(heaptrack). It says where the time was, to the function.

**In Vernier:** `--profile gperf` starts the profiler just before a test's
measured repeats and stops it after them, so warmup and the `--target-time`
calibration are not in the profile. The profile is written to
`<Suite.Case>.gperf/cpu.prof` under the working directory
(`--profile-output-dir DIR` moves the root), and
`google-pprof --text <binary> <profile>` reads it.

**Needs:** gperftools' development package when Vernier is built,
`google-pprof` to read the profile, and debug symbols matching the installed C
library (`libc6-dbg` on Debian), which give the library's internal functions
the names this page shows. The rig's
[one-time setup](../../docs/rigs/RIG_PI4.md#2-one-time-setup) installs all
three, and `bench doctor` lists the `gperf` backend as
`[OK]   gperf      gperftools profiles cpu (built: cpu); analyzer /usr/bin/google-pprof`
when it is compiled in and the reader is installed. Without the debug symbols, a
report names each of the library's internal functions after the nearest name
the library exports; [If It Does Not Match](#if-it-does-not-match) shows what
that looks like.

## The Example

The same two functions as [walkthrough 01](01_BASIC_WORKFLOW.md#the-example):

```cpp
std::string joinV0(const std::vector<std::string>& parts, char sep) {
  std::string out;
  for (const std::string& part : parts) {
    // Two temporaries per part, and a copy of everything joined so far
    out = out + part + sep;
  }
  return out;
}

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

Both are declared `[[gnu::noinline]]` in
[`Join.hpp`](../examples/join/inc/Join.hpp). A sample belongs to the function
whose instructions were running, and a function the optimizer had copied into
its caller would have no row of its own in the report.

The demo, [`cpu/03_GperfProfiler_Demo.cpp`](../cpu/03_GperfProfiler_Demo.cpp),
runs each version over the same 1,000 words in a test of its own,
`GperfProfiler.JoinV0` and `GperfProfiler.JoinV1`, one CSV row each; those are
the tests `--profile gperf` wraps below. That is the whole demo: it measures,
and the profiler reads it. The check that the profile keeps saying what this
page says is a performance test of its own beside the example; see
[What Keeps This Page True](#what-keeps-this-page-true).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_03_GperfProfiler \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from GperfProfiler
[ RUN      ] GperfProfiler.JoinV0
[target-time] 50.000 ms -> cycles=46 (calibrated 1084.0000 us/call, batch of 1)
[GperfProfiler.JoinV0]  1004.489 us/call  CV=0.3%  ~996 calls/s  (p10=1000.502 p90=1006.850 sd=2.567)
[       OK ] GperfProfiler.JoinV0 (469 ms)
[ RUN      ] GperfProfiler.JoinV1
[target-time] 50.000 ms -> cycles=2517 (calibrated 19.8594 us/call, batch of 64)
[GperfProfiler.JoinV1]  19.654 us/call  CV=1.7%  ~50.9K calls/s  (p10=18.892 p90=19.815 sd=0.332)
[       OK ] GperfProfiler.JoinV1 (494 ms)
[----------] 2 tests from GperfProfiler (964 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (964 ms total)
[  PASSED  ] 2 tests.

======================================================================
Test                   Median (us)     CV%     Calls/s  Status
----------------------------------------------------------------------
GperfProfiler.JoinV0      1004.489    0.3%         996  OK
GperfProfiler.JoinV1        19.654    1.7%       50.9K  OK
----------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

`joinV0` takes 1004.5 us per call and `joinV1` 19.7 us, 51 times less, and each
median is steady within its run (CV 0.3% and 1.7%). The next three steps take
one profile per version and read it.

## Step 2: Profile the Slow Version

```bash
bench run ./build/bin/ptests/BenchDemo_03_GperfProfiler --taskset 3 --profile gperf -- \
  --gtest_filter=GperfProfiler.JoinV0 --target-time 200ms --repeats 10
```

Captured output:

```
Running: taskset -c 3 ./build/bin/ptests/BenchDemo_03_GperfProfiler --profile gperf --gtest_filter=GperfProfiler.JoinV0 --target-time 200ms --repeats 10
Note: Google Test filter = GperfProfiler.JoinV0
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from GperfProfiler
[ RUN      ] GperfProfiler.JoinV0
[target-time] 200.000 ms -> cycles=203 (calibrated 982.0000 us/call, batch of 2)
PROFILE: interrupts/evictions/bytes = 196/23/11488
[GperfProfiler.JoinV0]  969.766 us/call  CV=0.2%  ~1.0K calls/s  (p10=968.965 p90=973.921 sd=1.969)
[       OK ] GperfProfiler.JoinV0 (1990 ms)
[----------] 1 test from GperfProfiler (1990 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1990 ms total)
[  PASSED  ] 1 test.
```

`bench run` starts the binary with `--profile gperf`, as its `Running:` line
shows; gperftools runs inside the benchmark's own process, so nothing wraps it.
The `PROFILE:` line is gperftools reporting as it stops: 196 samples. The
profile is `GperfProfiler.JoinV0.gperf/cpu.prof`, under the directory
`bench run` ran from.

196 samples for 203 calls x 10 repeats x 969.766 us = 1.97 s of measured time
is 100 per second, and the profile says so itself: the fourth number in its
header is the sampling period, in microseconds.

```bash
od -A d -t d8 -N 40 GperfProfiler.JoinV0.gperf/cpu.prof
```

```
0000000                    0                    3
0000016                    0                10000
0000032                    0
0000040
```

## Step 3: Read the Report

```bash
google-pprof --text ./build/bin/ptests/BenchDemo_03_GperfProfiler GperfProfiler.JoinV0.gperf/cpu.prof
```

Captured output, cut after the last function with samples of its own. The
rows below that have none of their own, only samples in what they called:
`_int_free`, whose samples are in the free-path rows above, the chain of
callers out to `_start`, and more `(inline)` rows. Two of them are kept:

```
Using local file ./build/bin/ptests/BenchDemo_03_GperfProfiler.
Using local file GperfProfiler.JoinV0.gperf/cpu.prof.
Total: 196 samples
     127  64.8%  64.8%      127  64.8% __memcpy_generic
      17   8.7%  73.5%       20  10.2% _int_malloc
       8   4.1%  77.6%        8   4.1% _int_free_create_chunk
       5   2.6%  80.1%       15   7.7% _int_free_merge_chunk
       4   2.0%  82.1%        4   2.0% tcache_try_malloc
       4   2.0%  84.2%        4   2.0% unlink_chunk
       3   1.5%  85.7%       24  12.2% __GI___libc_malloc
       3   1.5%  87.2%        3   1.5% _int_free_chunk
       3   1.5%  88.8%        3   1.5% std::__cxx11::basic_string::_M_data (inline)
       3   1.5%  90.3%      130  66.3% std::char_traits::copy (inline)
       2   1.0%  91.3%        2   1.0% _init
       2   1.0%  92.3%        2   1.0% alloc_perturb
       2   1.0%  93.4%        2   1.0% std::__once_callable@@GLIBCXX_3.4.11
       2   1.0%  94.4%        2   1.0% tag_at
       1   0.5%  94.9%        1   0.5% __GI___libc_free
       1   0.5%  95.4%        1   0.5% _int_free_maybe_consolidate
       1   0.5%  95.9%        1   0.5% arena_for_chunk
       1   0.5%  96.4%        1   0.5% checked_request2size
       1   0.5%  96.9%       33  16.8% operator new@@GLIBCXX_3.4
       1   0.5%  97.4%        1   0.5% std::__cxx11::basic_string::_Alloc_hider::_Alloc_hider (inline)
       1   0.5%  98.0%        1   0.5% std::__cxx11::basic_string::_M_is_local (inline)
       1   0.5%  98.5%        1   0.5% std::__cxx11::basic_string::operator= (inline)
       1   0.5%  99.0%        1   0.5% std::char_traits::assign (inline)
       1   0.5%  99.5%        1   0.5% tcache_free
       1   0.5% 100.0%        1   0.5% tcache_get_n
...
       0   0.0% 100.0%      181  92.3% std::operator+ (inline)
...
       0   0.0% 100.0%      192  98.0% vernier::bench::demo::joinV0
```

Each row is one function. `flat` counts the samples taken while that
function's own instructions were running, and `flat%` is its share of the
total; `sum%` adds up `flat%` down the list; `cum` and `cum%` count the samples
taken while the function was running itself or anything it had called. The rows
are sorted by `flat`.

A row marked `(inline)` is code the compiler copied into the function that
calls it. The join example is compiled with line tables (`-g`), which
walkthroughs 07 and 14 need to name its source lines, and from them pprof tells
the copied code apart and gives it a row under the name it has in the source.
Its samples are still the calling function's machine code. So
`vernier::bench::demo::joinV0` has no samples in its own row, while the six
`(inline)` rows with samples, the string code the compiler placed inside it
(`basic_string::_M_data`, `char_traits::copy` and the others), hold 10:
`joinV0`'s machine code holds 10 of the 196 samples, 5.1%.

Read it from the function you wrote:

- `vernier::bench::demo::joinV0` is on the stack of 192 samples (98.0%), and
  its machine code holds 10 (5.1%). Nearly all of V0's time is spent in
  calls that `joinV0` makes, not in its loop.
- The top row is the C library's `memcpy` (`__memcpy_generic` on this rig), at
  64.8%: copying. `out + part` copies everything joined so far, once per part.
- Most of the other rows are the allocator: `operator new` and the C library's
  `malloc` internals under it (`operator new`'s `cum` is 16.8%), and the free
  path (`__GI___libc_free`, the `_int_free` rows, `unlink_chunk`). The
  allocator's rows hold 55 samples between them, 28.1%: two temporary strings
  are made and thrown away for every part.
- The `(inline)` rows without samples of their own say which part of the
  source led there: `std::operator+ (inline)` is on the stack of 92.3% of the
  samples, the `out + part + sep` of `joinV0`'s loop.

So the profile names the function to change, and says what its time goes on:
copying and allocating, about two to one.

Two rows are not functions. `_init` (1.0%) is the last name before the binary's
procedure linkage table, the stubs through which it calls into the C library,
and pprof gives a sample the nearest name below its address. The two samples
in `std::__once_callable`, a variable in the C++ library, are the same
stand-in for code there that has no name of its own.

## Step 4: Confirm the Fix

Profile `joinV1` the same way and read its report:

```bash
bench run ./build/bin/ptests/BenchDemo_03_GperfProfiler --taskset 3 --profile gperf -- \
  --gtest_filter=GperfProfiler.JoinV1 --target-time 200ms --repeats 10
google-pprof --text ./build/bin/ptests/BenchDemo_03_GperfProfiler GperfProfiler.JoinV1.gperf/cpu.prof
```

Captured output of the first command:

```
Running: taskset -c 3 ./build/bin/ptests/BenchDemo_03_GperfProfiler --profile gperf --gtest_filter=GperfProfiler.JoinV1 --target-time 200ms --repeats 10
...
[target-time] 200.000 ms -> cycles=9523 (calibrated 21.0000 us/call, batch of 64)
PROFILE: interrupts/evictions/bytes = 201/64/11800
[GperfProfiler.JoinV1]  21.272 us/call  CV=1.1%  ~47.0K calls/s  (p10=20.826 p90=21.428 sd=0.232)
...
```

And of the second, cut after the last function with samples of its own:

```
Using local file ./build/bin/ptests/BenchDemo_03_GperfProfiler.
Using local file GperfProfiler.JoinV1.gperf/cpu.prof.
Total: 201 samples
      67  33.3%  33.3%       67  33.3% __memcpy_generic
      38  18.9%  52.2%       38  18.9% std::__cxx11::basic_string::_M_data (inline)
      28  13.9%  66.2%       28  13.9% std::__cxx11::basic_string::size (inline)
      19   9.5%  75.6%       19   9.5% std::char_traits::assign (inline)
      15   7.5%  83.1%       38  18.9% std::__cxx11::basic_string::capacity (inline)
       8   4.0%  87.1%      193  96.0% vernier::bench::demo::joinV1
       7   3.5%  90.5%        7   3.5% _init
       7   3.5%  94.0%       74  36.8% std::char_traits::copy (inline)
       5   2.5%  96.5%        5   2.5% std::__cxx11::basic_string::_M_length (inline)
       4   2.0%  98.5%      121  60.2% std::__cxx11::basic_string::_M_append (inline)
       1   0.5%  99.0%        1   0.5% __GI___libc_free
       1   0.5%  99.5%        2   1.0% __GI___libc_malloc
       1   0.5% 100.0%        1   0.5% _int_malloc
...
```

The report moved the way the timing did. `joinV1`'s own row holds 8 samples,
4.0%, and the `(inline)` rows inside it hold 116 more (`basic_string::_M_data`,
`size`, `char_traits::assign`, `capacity` and the rest): its machine code holds
124 of the 201 samples, 61.7%, and it is on the stack of 96.0%. `memcpy` holds
33.3%, now copying each part once, into place, and the allocator is down to 3
samples, 1.5%, for one allocation and one free per call.

Which `(inline)` rows belong to which function, the stacks say. `--stacks`
lists every sampled call stack with its sample count, innermost frame first,
and names that frame after the function whose machine code the sample hit,
with the source line of the code inlined there:

```bash
google-pprof --text --stacks ./build/bin/ptests/BenchDemo_03_GperfProfiler GperfProfiler.JoinV1.gperf/cpu.prof
```

Captured output, cut to its first stack, and that to the frames in the
benchmark:

```
Using local file ./build/bin/ptests/BenchDemo_03_GperfProfiler.
Using local file GperfProfiler.JoinV1.gperf/cpu.prof.
Total: 201 samples
Stacks:

7        (000000558bc93e0c) .../Join.cpp:39:vernier::bench::demo::joinV1
         (000000558bc8621f) 03_GperfProfiler_Demo.cpp:0:std::_Function_handler::_M_invoke
         (000000558bc87fbb) ??:0:std::_Function_handler::_M_invoke
         (000000558bc8ce33) ??:0:vernier::bench::PerfCase::measured
         (000000558bc87877) ??:0:GperfProfiler_JoinV1_Test::TestBody
         ...
```

Seven samples caught code compiled from line 39, `out += part;`, inside
`joinV1`. Counted this way, 124 of the 201 samples have `joinV1` as their
innermost frame: the 61.7% above.

The percentages are shares of each run's own time, so to compare two runs,
multiply by the time per call: V0 spent about 628 us of its 969.8 us per call
in `memcpy` (64.8%), V1 about 7.1 us of its 21.3 us (33.3%).

## Profiling Your Own Code

1. Give the function a test of its own, shaped like the demo's, so that
   `--gtest_filter` can select it alone.
2. Profile that test:

   ```bash
   bench run ./build/bin/ptests/<YourBenchmark> --profile gperf -- \
     --gtest_filter=<Suite.Case> --target-time 200ms --repeats 10
   ```

   Two seconds of measured calls is about 200 samples at the default rate.

3. Read it:

   ```bash
   google-pprof --text ./build/bin/ptests/<YourBenchmark> <Suite.Case>.gperf/cpu.prof
   ```

   Find your function's row: `cum` says how much of the time was spent in it
   or below it, `flat` how much in its own code. A large `cum` over a small
   `flat` means the time is in what it calls, and the rows above it say what.

4. A function the compiler inlined into its caller has no row; its samples
   count as the caller's. The example's two versions are declared
   `[[gnu::noinline]]` so that each keeps its row. And when your code carries
   line tables, code inlined into your function gets `(inline)` rows of its
   own: the function's machine code is its row plus those, and `--stacks`
   shows which rows are whose.

## What Should Reproduce

| Reading                  | On this rig                                                                              | Elsewhere                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| V1 against V0            | 51x faster here; 44.0x to 51.1x over twelve runs                                         | tens of times faster in an optimized build; 38.5x to 38.9x on x86                                                      |
| `joinV0` in V0's profile | 5.1% own, 98.0% on the stack; 2.5% to 6.0% and 98.0% to 99.5% in the check's ten runs    | should match: 1.0% to 6.5% and 97.0% to 100.0% on x86                                                                  |
| `joinV1` in V1's profile | 61.7% own, 96.0% on the stack; 55.5% to 64.0% and 95.5% to 99.0% in the check's ten runs | a large own share: 67.0% to 86.5% on x86, lower on its efficiency cores (67.0% to 76.5%) than on its performance cores |
| the rest of V0's samples | `memcpy` 64.8%, the allocator 28.1%                                                      | the same kinds of function; their names depend on the C library and its debug symbols                                  |
| sampling rate            | 100 per second of CPU time                                                               | the same default                                                                                                       |
| absolute times           | V0 1004.5 and V1 19.7 us/call; 932.9 to 1032.0 and 19.7 to 21.3 over twelve runs         | will differ                                                                                                            |

"Own" is the function's machine code, its row and the `(inline)` rows inside
it, counted from the stacks as step 4 does. The twelve runs are step 1's
command run in one session on this rig, the reference capture and the page's
run among them; the check's ten runs are `GperfProfiler.ProfileAttribution`
run ten times in the same session. Those ranges describe those runs; they are
not a bound a run has to meet, and another run can land outside them. The x86
figures come from an x86 laptop with performance and efficiency cores (clang 21,
gperftools 2.15, Release) that was busy with other work: the ratio from five
runs of step 1's command on a performance core, the profile shares from ten
runs of the check on each kind of core.

## If It Does Not Match

- **Names like `__xpg_strerror_r` or `timer_settime` at the top of V0's
  report.** Without the C library's debug symbols, pprof knows only the names
  the library exports, and gives each sample the nearest one below its address.
  With `libc6-dbg`'s files hidden, this rig's V0 profile from step 3 showed
  `__xpg_strerror_r@@GLIBC_2.17` at 64.8%, the row that is `__memcpy_generic`
  above, and the allocator as `__default_morecore`, `timer_settime`, `malloc`
  and `__libc_free`. The `joinV0` row and the `(inline)` rows did not change,
  because the benchmark's own names are in its binary. Install the C library's
  debug symbols (`libc6-dbg` on Debian) and read the same profile again.
- **No `(inline)` rows, and `joinV1`'s own row holds most of V1's samples.**
  The join example was built without its line tables, or stripped. The machine
  code is the same and so is the split between the functions; the report only
  folds the inlined code into its caller's row. On x86, a build of the example
  without `-g` put 74.7% to 79.4% of V1's samples in `joinV1`'s row, with no
  `(inline)` rows.
- **`GperfProfiler.ProfileAttribution` is skipped.**
  `gperf backend unavailable: missing: gperftools headers were not present when libbench was built`,
  as a build without gperftools printed it on an x86-64 laptop, means Vernier
  was built without gperftools; that build's doctor row reads
  `[FAIL] gperf      missing: gperftools headers were not present when libbench was built`.
  Install its development package, reconfigure and rebuild, and
  `bench doctor build/bin/ptests/BenchDemo_03_GperfProfiler` should list the
  row quoted in Needs. `google-pprof is not on PATH; it reads the profiles`
  means the reader is missing (`google-perftools` on Debian). The test also
  skips under `--profile` (`runs its own profiles; run it without --profile`),
  because it takes profiles of its own.
- **Too few samples.** The profiler's default is 100 samples per second of CPU
  time, and `--profile-frequency` does not change it as Vernier applies it: the
  gperf backend sets `CPUPROFILE_FREQUENCY` from the flag just before it starts
  the profiler, and by then the installed gperftools has already read the
  variable, so profiles taken with `--profile-frequency 1000` on this rig still
  record a 10,000-microsecond period. Setting the variable before the process
  starts does change the rate: `CPUPROFILE_FREQUENCY=250 bench run ...`
  recorded a 4,000-microsecond period and took 475 samples in 1.9 s here. Or
  size the profiled run: the steps above use `--target-time 200ms --repeats 10`
  for about 200 samples. The rate asked for is not always the rate taken: this
  rig's kernel ticks 250 times a second, and asked for 1,000 samples a second,
  gperftools took 249 a second.
- **`joinV0 [clone .constprop.0]` instead of `joinV0`.** A build with
  link-time optimization lets GCC specialize the function for the one
  separator every caller passes. It is the same function under a longer name.

## Check Against the Reference

A capture from this rig is committed with the demo, so you can compare a run of
your own against it:

```bash
bench compare src/bench/demo/reference/pi4/03_gperf_profiler.csv run1.csv
```

Output for step 1's `run1.csv`, printed by the `bench` CLI built from the tree
this page ships in (`bench 1.0.3`):

```
Test                      Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
--------------------  ------------  ------------  ----------  --------  --------  --------  ------------
GperfProfiler.JoinV0     942.18300    1004.49000   +62.30700     +6.6%      0.2%      0.3%  REGRESSION
GperfProfiler.JoinV1      19.77560      19.65380    -0.12180     -0.6%      0.3%      1.7%  neutral

  1 regression(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary seconds before step 1, and
nothing changed between the two runs, yet V0's row is labelled `REGRESSION`:
its median came out 6.6% above the reference's, past the 5% threshold the
labels are drawn at, while V1's moved 0.6% and is `neutral`. `Base CV` and
`Cand CV` are each run's spread across its own repeats. Over the twelve runs of
step 1's command in this session, V0's median ranged from 932.9 to 1032.0
us/call, 10.6%, and V1's from 19.7 to 21.3 us/call, 8.6%, with nothing
changed, which is more than either run's CV says; V0's median is the noisiest
reading on this page. The CSVs' `hostname` column records the name visible to
the process that captured each: this reference was captured in a UTS
namespace named `pi4` on the same rig, so it records `pi4`, while the rig's
ordinary captures, step 1's `run1.csv` among them, record `raspberrypi`. What
should hold is the ratio between the two rows, and the profile of each
version.

## What Keeps This Page True

Two things run against this example, and both fail loudly:

- `GperfProfiler.ProfileAttribution` profiles each version for two seconds of
  CPU time through the same backend as `--profile gperf`, and reads the
  profiles with `google-pprof --text --stacks`, counting a sample as a
  function's own when its stack's innermost frame is that function, as step 4
  does. It fails if either function is on the stack of fewer than 80% of its
  version's samples (the profile no longer names it, which is what happens to
  a function the optimizer copies into its caller), if `joinV0`'s own code
  holds more than a quarter of V0's samples, or if `joinV1`'s own code holds
  less than a quarter of V1's. It is a performance test beside the join
  example,
  [`JoinProfileAttribution_pTest.cpp`](../examples/join/ptst/JoinProfileAttribution_pTest.cpp),
  built as `JoinProfileAttribution`; a profile's shares are samples, so it is
  not registered with `ctest`, and it is run on the rig by hand:

  ```bash
  taskset -c 3 ./build/bin/ptests/JoinProfileAttribution
  ```

  Captured on this rig:

  ```
  [==========] Running 1 test from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 1 test from GperfProfiler
  [ RUN      ] GperfProfiler.ProfileAttribution
  PROFILE: interrupts/evictions/bytes = 201/74/18544
  PROFILE: interrupts/evictions/bytes = 200/110/16000
  [GperfProfiler.ProfileAttribution]  V0: 201 samples, joinV0 3.0% self, 98.5% total
  [GperfProfiler.ProfileAttribution]  V1: 200 samples, joinV1 60.5% self, 98.0% total
  [       OK ] GperfProfiler.ProfileAttribution (6024 ms)
  [----------] 1 test from GperfProfiler (6024 ms total)

  [----------] Global test environment tear-down
  [==========] 1 test from 1 test suite ran. (6024 ms total)
  [  PASSED  ] 1 test.
  ```

  In ten runs in this session it passed every time, with the shares in the
  table above. It fails when `joinV0` is made to reserve like `joinV1` (its
  own code then holds 55.0% of V0's samples), when `joinV1` copies like
  `joinV0` (4.5% of V1's), and when both versions are forced inline (neither
  name appears). It reads the stacks because the flat column moves with the
  line tables: profiled the same way and read from `google-pprof --text`'s
  flat column, `joinV1`'s share was 5.0% to 11.0% in ten runs in this session,
  the rest sitting in the `(inline)` rows of step 3.

- The example's unit tests hold both versions to the same answers and are
  registered with `ctest` under the `demo` label, beside the other shared
  examples' tests, so ordinary CI runs them:

  ```bash
  ctest --test-dir build -L demo
  ```

  Every test it runs should pass. The ones that hold `joinV0` and `joinV1` to
  the same string are `JoinTest.KnownAnswer` and
  `*/JoinSizesTest.VersionsAgree/*`, one CTest entry that runs the comparison at
  every input size the example's tests use.

This repository has no continuous-integration lane on the reference board, so
nothing runs the demo or its check automatically. Before a release they are
run on the rig by hand, with the commands above, and the reference CSV is
re-captured when the numbers move.

## See Also

- [Walkthrough 01: basic workflow](01_BASIC_WORKFLOW.md) -- the `join` example,
  measured and compared
- [Walkthrough 07: Callgrind](07_CALLGRIND_PROFILER.md) -- exact instruction
  counts instead of samples
- [Walkthrough 21: heaptrack](21_HEAPTRACK_PROFILER.md) -- counting the
  allocations this profile found
- [API reference: ProfilerGperf](../../docs/API_REFERENCE.md#profilergperf) --
  the backend's CPU and heap modes
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
