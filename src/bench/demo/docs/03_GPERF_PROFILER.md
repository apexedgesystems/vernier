# Demo 03: gperftools CPU Profiler

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-24 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`

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
the ten profiles of V1 taken by step 1's command in this session put `joinV1`'s
own share anywhere from 53% to 65%.

At that rate the profiler costs little. Over five alternating runs of each
version on this rig, the profiled medians averaged 0.4% (V0) and 1.4% (V1)
above the unprofiled ones, while the unprofiled V0 runs alone spread over 4.6%.

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
`gperftools linked: cpu` when it is compiled in. Without the debug symbols, a
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
the tests `--profile gperf` wraps below. A third test,
`GperfProfiler.ProfileAttribution`, profiles both versions itself and fails if
the profile stops saying what this page says; see
[What Keeps This Page True](#what-keeps-this-page-true).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_03_GperfProfiler \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from GperfProfiler
[ RUN      ] GperfProfiler.JoinV0
[target-time] 50.000 ms -> cycles=50 (calibrated 999.5000 us/call, batch of 2)
[GperfProfiler.JoinV0]  970.270 us/call  CV=0.1%  ~1.0K calls/s  (p10=969.934 p90=972.436 sd=1.292)
[       OK ] GperfProfiler.JoinV0 (494 ms)
[ RUN      ] GperfProfiler.JoinV1
[target-time] 50.000 ms -> cycles=2507 (calibrated 19.9375 us/call, batch of 64)
[GperfProfiler.JoinV1]  20.913 us/call  CV=1.3%  ~47.8K calls/s  (p10=20.396 p90=20.948 sd=0.268)
[       OK ] GperfProfiler.JoinV1 (524 ms)
[ RUN      ] GperfProfiler.ProfileAttribution
PROFILE: interrupts/evictions/bytes = 201/88/20680
PROFILE: interrupts/evictions/bytes = 200/105/14984
[GperfProfiler.ProfileAttribution]  V0: 201 samples, joinV0 7.5% self, 97.0% total
[GperfProfiler.ProfileAttribution]  V1: 200 samples, joinV1 58.0% self, 97.5% total
[       OK ] GperfProfiler.ProfileAttribution (5758 ms)
[----------] 3 tests from GperfProfiler (6777 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (6777 ms total)
[  PASSED  ] 3 tests.

======================================================================
Test                   Median (us)     CV%     Calls/s  Status
----------------------------------------------------------------------
GperfProfiler.JoinV0       970.270    0.1%        1.0K  OK
GperfProfiler.JoinV1        20.913    1.3%       47.8K  OK
----------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

`joinV0` takes 970.3 us per call and `joinV1` 20.9 us, 46 times less, and each
median is steady within its run (CV 0.1% and 1.3%). The third test's two lines
come from profiles it took itself: in V0's, `joinV0` is on the stack of 97.0% of
the samples while its own code holds only 7.5%; in V1's, `joinV1`'s own code
holds 58.0%. The next three steps take one profile per version by hand and read
it.

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
[target-time] 200.000 ms -> cycles=191 (calibrated 1042.5000 us/call, batch of 2)
PROFILE: interrupts/evictions/bytes = 185/16/9856
[GperfProfiler.JoinV0]  970.770 us/call  CV=0.1%  ~1.0K calls/s  (p10=970.088 p90=971.548 sd=0.775)
[       OK ] GperfProfiler.JoinV0 (1875 ms)
[----------] 1 test from GperfProfiler (1875 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1875 ms total)
[  PASSED  ] 1 test.
```

`bench run` starts the binary with `--profile gperf`, as its `Running:` line
shows; gperftools runs inside the benchmark's own process, so nothing wraps it.
The `PROFILE:` line is gperftools reporting as it stops: 185 samples. The
profile is `GperfProfiler.JoinV0.gperf/cpu.prof`, under the directory
`bench run` ran from.

185 samples for 191 calls x 10 repeats x 970.77 us = 1.85 s of measured time is
100 per second, and the profile says so itself: the fourth number in its header
is the sampling period, in microseconds.

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

Captured output, cut after the last function with samples of its own. The rows
below the cut have none: `_int_free`, whose 15 samples are all in the free-path
rows above, and the chain of callers that led to `joinV0`, from
`std::_Function_handler::_M_invoke` out to `_start`:

```
Using local file ./build/bin/ptests/BenchDemo_03_GperfProfiler.
Using local file GperfProfiler.JoinV0.gperf/cpu.prof.
Total: 185 samples
     132  71.4%  71.4%      132  71.4% __memcpy_generic
      17   9.2%  80.5%       18   9.7% _int_malloc
       4   2.2%  82.7%        5   2.7% _int_free_create_chunk
       4   2.2%  84.9%       10   5.4% _int_free_merge_chunk
       4   2.2%  87.0%      181  97.8% vernier::bench::demo::joinV0
       3   1.6%  88.6%        3   1.6% _init
       3   1.6%  90.3%        3   1.6% _int_free_chunk
       3   1.6%  91.9%        3   1.6% unlink_chunk
       2   1.1%  93.0%        2   1.1% __GI___libc_free
       2   1.1%  94.1%       21  11.4% __GI___libc_malloc
       2   1.1%  95.1%        2   1.1% _int_free_maybe_consolidate
       2   1.1%  96.2%        2   1.1% checked_request2size
       2   1.1%  97.3%        2   1.1% tcache_available
       1   0.5%  97.8%       28  15.1% operator new@@GLIBCXX_3.4
       1   0.5%  98.4%        1   0.5% std::__once_callable@@GLIBCXX_3.4.11
       1   0.5%  98.9%        1   0.5% tag_new_usable
       1   0.5%  99.5%        1   0.5% tcache_get_n
       1   0.5% 100.0%        1   0.5% tcache_try_malloc
...
```

Each row is one function. `flat` counts the samples taken while that
function's own instructions were running, and `flat%` is its share of the
total; `sum%` adds up `flat%` down the list; `cum` and `cum%` count the samples
taken while the function was running itself or anything it had called. The rows
are sorted by `flat`.

Read it from the function you wrote:

- `vernier::bench::demo::joinV0` holds 4 samples of its own (2.2%) and is on the
  stack of 181 (97.8%). Nearly all of V0's time is spent in calls that
  `joinV0` makes, not in its loop.
- The top row is the C library's `memcpy` (`__memcpy_generic` on this rig), at
  71.4%: copying. `out + part` copies everything joined so far, once per part.
- Most of the other rows are the allocator: `operator new` and the C library's
  `malloc` internals under it (`operator new`'s `cum` is 15.1%), and the free
  path (`__GI___libc_free`, the `_int_free` rows, `unlink_chunk`). The
  allocator's rows hold 45 samples between them, 24%: two temporary strings
  are made and thrown away for every part.

So the profile names the function to change, and says what its time goes on:
copying and allocating, about three to one.

Two rows are not functions. `_init` (1.6%) is the last name before the binary's
procedure linkage table, the stubs through which it calls into the C library,
and pprof gives a sample the nearest name below its address. The one sample in
`std::__once_callable`, a variable in the C++ library, is the same stand-in for
code there that has no name of its own.

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
[target-time] 200.000 ms -> cycles=10007 (calibrated 19.9844 us/call, batch of 64)
PROFILE: interrupts/evictions/bytes = 209/81/14128
[GperfProfiler.JoinV1]  20.938 us/call  CV=2.0%  ~47.8K calls/s  (p10=20.419 p90=21.485 sd=0.417)
...
```

And of the second, cut the same way:

```
Using local file ./build/bin/ptests/BenchDemo_03_GperfProfiler.
Using local file GperfProfiler.JoinV1.gperf/cpu.prof.
Total: 209 samples
     128  61.2%  61.2%      203  97.1% vernier::bench::demo::joinV1
      73  34.9%  96.2%       73  34.9% __memcpy_generic
       5   2.4%  98.6%        5   2.4% _init
       2   1.0%  99.5%        2   1.0% _int_malloc
       1   0.5% 100.0%        1   0.5% _int_free_merge_chunk
...
```

The report moved the way the timing did. `joinV1`'s own loop holds most of the
samples, 61.2%; `memcpy` holds 34.9%, now copying each part once, into place;
and the allocator is down to 3 samples, 1.4%, for one allocation and one free
per call.

The percentages are shares of each run's own time, so to compare two runs,
multiply by the time per call: V0 spent about 690 us of its 970.77 us per call
in `memcpy` (71.4%), V1 about 7.3 us of its 20.938 us (34.9%).

## What Should Reproduce

| Reading                  | On this rig                                                                                    | Elsewhere                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| V1 against V0            | 46x faster here; 43.6x to 49.6x over ten runs                                                  | tens of times faster in an optimized build; 30x to 33x on x86                         |
| `joinV0` in V0's profile | 2.2% own, 97.8% on the stack; 2.0% to 9.0% and 96.5% to 100.0% in the ten runs' own profiles   | should match: 1% to 5% and 95% to 100% on x86                                         |
| `joinV1` in V1's profile | 61.2% own, 97.1% on the stack; 53.0% to 65.0% and 94.5% to 98.5% in the ten runs' own profiles | a large own share: 58% to 65% on x86                                                  |
| the rest of V0's samples | `memcpy` 71.4%, the allocator 24%                                                              | the same kinds of function; their names depend on the C library and its debug symbols |
| sampling rate            | 100 per second of CPU time                                                                     | the same default                                                                      |
| absolute times           | V0 970.3 and V1 20.9 us/call; 923.1 to 997.7 and 20.0 to 21.7 over ten runs                    | will differ                                                                           |

The ten runs are step 1's command run ten times in one session, the reference
capture among them; each carries its own profiles, from
`GperfProfiler.ProfileAttribution`. The x86 figures are five runs of the same
command on an x86 laptop (clang 21, gperftools 2.15, Release) that was busy
with other work, so its timings are noisier than the rig's.

## If It Does Not Match

- **Names like `__xpg_strerror_r` or `timer_settime` at the top of V0's
  report.** Without the C library's debug symbols, pprof knows only the names
  the library exports, and gives each sample the nearest one below its address.
  With `libc6-dbg`'s files hidden, this rig's V0 profile from step 3 showed
  `__xpg_strerror_r@@GLIBC_2.17` at 71.4%, the row that is `__memcpy_generic`
  above, and the allocator as `__default_morecore`, `malloc` and
  `timer_settime`. The `joinV0` row did not change, because the benchmark's own
  names are in its binary. Install the C library's debug symbols (`libc6-dbg`
  on Debian) and read the same profile again.
- **`GperfProfiler.ProfileAttribution` is skipped.** "gperf backend
  unavailable: gperftools headers not present at build time" means Vernier was
  built without gperftools: install its development package, reconfigure and
  rebuild, and `bench doctor build/bin/ptests/BenchDemo_03_GperfProfiler`
  should list `gperf` as `gperftools linked: cpu`. "google-pprof is not on
  PATH" means the reader is missing (`google-perftools` on Debian). The test
  also skips under `--profile`, because it takes profiles of its own.
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
GperfProfiler.JoinV0     937.24500     970.27000   +33.02500     +3.5%      0.0%      0.1%  neutral
GperfProfiler.JoinV1      21.01740      20.91320    -0.10420     -0.5%      1.3%      1.3%  neutral

  2 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary seconds before step 1. V0 came
out 3.5% slower than the reference and V1 0.5% faster, and both rows are
labelled neutral because both medians moved by less than the 5% threshold.
`Base CV` and `Cand CV` are each run's spread across its own repeats. Over the
ten runs of step 1's command in this session, V0's median ranged from 923.1 to
997.7 us/call and V1's from 20.0 to 21.7 us/call, 8.1% each, with nothing
changed, which is more than either run's CV says. What should hold is the ratio
between the two rows, and the profile of each version.

## What Keeps This Page True

Two things run against this example, and both fail loudly:

- `GperfProfiler.ProfileAttribution`, in the demo binary, profiles each version
  for two seconds of CPU time through the same backend as `--profile gperf`,
  and reads the profiles with `google-pprof --text`, as steps 2 to 4 do. It
  fails if either function is on the stack of fewer than 80% of its version's
  samples (the profile no longer names it, which is what happens to a function
  the optimizer copies into its caller), if `joinV0`'s own code holds more than
  a quarter of V0's samples, or if `joinV1`'s own code holds less than a quarter
  of V1's. On this rig it fails when `joinV0` is made to reserve like `joinV1`
  (its own code then holds 57.5% of V0's samples), and when both versions are
  forced inline (neither name appears).
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
nothing runs the demo itself automatically. Before a release it is run on the
rig by hand, with the command in step 1, and the reference CSV is re-captured
when the numbers move.

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
