# Demo 02: Linux perf Hardware Counters

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) and [`filter`](../examples/filter/inc/Filter.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-26 (UTC), written for the Vernier 1.0.4 release; captured from
the development tree at project version 1.0.3, whose CLI reported `bench 1.0.3`;
perf 6.18.39

## Overview

A timing says how long a call takes. The processor's own counters say what
it did in that time: how many instructions it retired, how many cycles it
spent, how often it guessed a branch wrong, how often a load missed the
cache. This walkthrough reads those counters through `perf stat` for two
examples. The `join` from [walkthrough 01](01_BASIC_WORKFLOW.md) shows the
first kind of slowness, more work: the slow version retires about 33 times
the instructions of the fast one per call. A filter that copies the values
above a threshold shows the second kind, the same work done slowly: on
random input it retires the same instructions as on sorted input and takes
3.6 times the cycles, and the branch-misses column says why.

## What is perf?

perf is the Linux kernel's profiler. Its `perf stat` mode asks the
processor's performance monitoring unit to count events (cycles retired,
instructions retired, branches mispredicted, cache lines missed) while a
program runs, and prints the totals when it stops. Nothing is sampled and
nothing is slowed down: the counters run beside the program at full speed,
and a run under `perf stat` takes as long as one without it. What it cannot
do is say where in the program the events happened; that is `perf record`,
below, and the sampling profiler of [walkthrough 03](03_GPERF_PROFILER.md).

The counts are exact for the events they count, and the events are the
processor's: which ones exist, and what each one means, differ between
processors. This rig's Cortex-A72 has no event for perf's generic
`branches`, so that line reads `<not supported>` on every report below, and
its `cache-misses` reports the same count as its `L1-dcache-load-misses`
(the run in [Adding an Event](#adding-an-event) shows both), so it is the
level-1 data cache's misses here. On the x86 laptop this branch also ran on,
`branches` counts, and `cache-misses` is a different, far rarer event. Read
the events per machine, and compare counts between runs on the same machine.

**In Vernier:** `--profile perf` starts `perf stat -e
cpu-cycles,instructions,branches,branch-misses,cache-misses -p <pid>` on the
benchmark's own process just before a test's measured repeats and stops it
after them, so warmup and the `--target-time` calibration are not counted.
perf's report is written to `<Suite.Case>.perf/stat.txt` under the working
directory (`--profile-output-dir DIR` moves the root), and that is the only
file stat mode writes. The totals are for the whole measured window, so a
per-call figure is a total divided by the number of measured calls,
`--cycles` times `--repeats`; the steps below pass both explicitly so the
division is in plain sight. Two flags change what runs: `--profile-args
"record -g"` runs `perf record` instead and writes `perf.data` (see
[Record Mode](#record-mode)), and any other `--profile-args` text is
appended to the `perf stat` command (see [Adding an Event](#adding-an-event)).

**Needs:** perf installed (`linux-perf` on Debian, in the rig's
[one-time setup](../../docs/rigs/RIG_PI4.md#2-one-time-setup)), and permission
to count. With `kernel.perf_event_paranoid` at 2, the kernel's default and
this rig's, a user may count the user-space events of its own processes,
which is what the backend asks for; every event then carries a
`:u` suffix in the report. At 3 or above the kernel refuses, and `stat.txt`
holds perf's refusal instead of counts; [If It Does Not Match](#if-it-does-not-match)
shows it. No root is needed on the rig.

## The Example

The same two functions as [walkthrough 01](01_BASIC_WORKFLOW.md#the-example):
`joinV0` builds the result with `out = out + part + sep`, two temporaries and
a copy of everything joined so far per part, and `joinV1` measures, reserves
once and appends in place. Both are declared `[[gnu::noinline]]` in
[`Join.hpp`](../examples/join/inc/Join.hpp).

The demo, [`cpu/02_PerfProfiler_Demo.cpp`](../cpu/02_PerfProfiler_Demo.cpp),
runs each version over the same 1,000 words in a test of its own,
`PerfProfiler.JoinV0` and `PerfProfiler.JoinV1`, one CSV row each; those are
the tests `--profile perf` wraps in steps 2 and 4. Three more tests measure
the filter of [the second example](#the-second-example-a-filter), and two
tests count the events themselves and fail if the counts stop saying what
this page says: see [What Keeps This Page True](#what-keeps-this-page-true).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_02_PerfProfiler \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from PerfProfiler
[ RUN      ] PerfProfiler.JoinV0
[target-time] 50.000 ms -> cycles=48 (calibrated 1039.0000 us/call, batch of 1)
[PerfProfiler.JoinV0]  1001.740 us/call  CV=0.1%  ~998 calls/s  (p10=1000.798 p90=1003.163 sd=1.251)
[       OK ] PerfProfiler.JoinV0 (488 ms)
[ RUN      ] PerfProfiler.JoinV1
[target-time] 50.000 ms -> cycles=2463 (calibrated 20.2969 us/call, batch of 64)
[PerfProfiler.JoinV1]  21.342 us/call  CV=3.8%  ~46.9K calls/s  (p10=20.251 p90=22.242 sd=0.818)
[       OK ] PerfProfiler.JoinV1 (529 ms)
[ RUN      ] PerfProfiler.FilterBranchyRandom
[target-time] 50.000 ms -> cycles=56 (calibrated 889.0000 us/call, batch of 2)
[PerfProfiler.FilterBranchyRandom]  882.661 us/call  CV=0.1%  ~1.1K calls/s  (p10=882.195 p90=883.788 sd=0.662)
[       OK ] PerfProfiler.FilterBranchyRandom (502 ms)
[ RUN      ] PerfProfiler.FilterBranchySorted
[target-time] 50.000 ms -> cycles=204 (calibrated 244.0000 us/call, batch of 8)
[PerfProfiler.FilterBranchySorted]  242.145 us/call  CV=0.6%  ~4.1K calls/s  (p10=241.650 p90=243.366 sd=1.478)
[       OK ] PerfProfiler.FilterBranchySorted (515 ms)
[ RUN      ] PerfProfiler.FilterBranchless
[target-time] 50.000 ms -> cycles=316 (calibrated 158.1250 us/call, batch of 8)
[PerfProfiler.FilterBranchless]  158.351 us/call  CV=0.4%  ~6.3K calls/s  (p10=157.898 p90=159.022 sd=0.556)
[       OK ] PerfProfiler.FilterBranchless (508 ms)
[ RUN      ] PerfProfiler.JoinInstructions
[PerfProfiler.JoinInstructions]  V0 2150574 instructions/call  V1 63616 instructions/call  33.8x
[       OK ] PerfProfiler.JoinInstructions (17 ms)
[ RUN      ] PerfProfiler.FilterBranchMisses
[PerfProfiler.FilterBranchMisses]  branchy random 50010  branchy sorted 3  branchless 1  branch-misses/call
[       OK ] PerfProfiler.FilterBranchMisses (34 ms)
[----------] 7 tests from PerfProfiler (2597 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 1 test suite ran. (2597 ms total)
[  PASSED  ] 7 tests.

==================================================================================
Test                               Median (us)     CV%     Calls/s  Status
----------------------------------------------------------------------------------
PerfProfiler.JoinV0                   1001.740    0.1%         998  OK
PerfProfiler.JoinV1                     21.342    3.8%       46.9K  OK
PerfProfiler.FilterBranchyRandom       882.661    0.1%        1.1K  OK
PerfProfiler.FilterBranchySorted       242.145    0.6%        4.1K  OK
PerfProfiler.FilterBranchless          158.351    0.4%        6.3K  OK
----------------------------------------------------------------------------------
5 tests | 5 stable | 0 unstable
```

`joinV0` takes 1,001.7 us per call and `joinV1` 21.3 us, 47 times less. The
three filter rows are one function twice and its branchless twin once: 882.7 us
per call on values in random order, 242.1 us on the same values in ascending
order, 158.4 us with no branch on the data. Each median is steady within its
run (CV 0.1% to 3.8%). The two lines without a time are the demo's counter
checks: over ten calls each, `joinV0` retired 2,150,574 instructions per call
and `joinV1` 63,616, 33.8 times fewer; the branchy filter mispredicted 50,010
branches per call on random input and 3 on sorted input, and the branchless
filter 1. The next steps take those readings one test at a time, through
`perf stat`, and read them.

## Step 2: Profile the Slow Version

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf -- \
  --gtest_filter=PerfProfiler.JoinV0 --cycles 100 --repeats 10
```

Captured output:

```
Running: taskset -c 3 ./build/bin/ptests/BenchDemo_02_PerfProfiler --profile perf --gtest_filter=PerfProfiler.JoinV0 --cycles 100 --repeats 10
Note: Google Test filter = PerfProfiler.JoinV0
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from PerfProfiler
[ RUN      ] PerfProfiler.JoinV0
[PerfProfiler.JoinV0]  998.165 us/call  CV=0.1%  ~1.0K calls/s  (p10=997.523 p90=999.099 sd=1.065)
[       OK ] PerfProfiler.JoinV0 (2214 ms)
[----------] 1 test from PerfProfiler (2214 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2215 ms total)
[  PASSED  ] 1 test.
```

`bench run` starts the binary pinned to core 3 with `--profile perf`, as its
`Running:` line shows. Just before the measured repeats the backend starts
`perf stat` on the benchmark's own process, and just after them it stops it;
nothing else changes, and the timing, 998.2 us per call, is step 1's.
`--cycles 100 --repeats 10` makes the measured window exactly 1,000 calls, the
number every count in the next step is divided by. perf's report is
`PerfProfiler.JoinV0.perf/stat.txt`, under the directory `bench run` ran from.

## Step 3: Read the Report

```bash
cat PerfProfiler.JoinV0.perf/stat.txt
```

Captured output:

```
 Performance counter stats for process id '99835':

     1,796,101,901      cpu-cycles:u
     2,129,197,013      instructions:u                   #    1.19  insn per cycle
   <not supported>      branches:u
         8,439,901      branch-misses:u
        19,703,861      cache-misses:u

       1.170691455 seconds time elapsed
```

Each line is one event: its total over the window, and perf's name for it
with `:u` for user space. Divided by the 1,000 calls:

- `cpu-cycles`: 1,796,101,901, so 1,796,102 cycles per call. At this rig's
  1.8 GHz that is 997.8 us; the counter and the clock agree, and the core
  ran at full speed for the whole window.
- `instructions`: 2,129,197,013, so 2,129,197 instructions per call, 1.19 of
  them retired per cycle (the `insn per cycle` figure perf adds). This is
  V0's problem in one number: joining 1,000 words takes two million
  instructions, because `out + part + sep` copies everything joined so far
  once per part and builds two temporary strings to do it.
- `branches`: `<not supported>`. This processor has no event perf can map
  that name to, so the column is empty here; it counts on an x86 processor.
- `branch-misses`: 8,439,901, so 8,440 per call: one mispredicted branch
  for every 250 instructions.
- `cache-misses`: 19,703,861, so 19,704 per call. On this rig the event
  reports the same count as `L1-dcache-load-misses` (see
  [Adding an Event](#adding-an-event)), so read it as the level-1 data
  cache's misses here: V0 streams its growing result through the cache once
  per part.

The `time elapsed` line is perf's own clock, 1.17 s: the 1,000 calls of about
1 ms, plus the 200 ms the backend gives perf to attach before the loop.

## Step 4: Confirm the Fix

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf -- \
  --gtest_filter=PerfProfiler.JoinV1 --cycles 5000 --repeats 10
cat PerfProfiler.JoinV1.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '99843':

     1,769,722,061      cpu-cycles:u
     3,182,248,287      instructions:u                   #    1.80  insn per cycle
   <not supported>      branches:u
        16,116,574      branch-misses:u
         8,064,963      cache-misses:u

       1.136171592 seconds time elapsed
```

This window is 50,000 calls (`--cycles 5000 --repeats 10`), so the per-call
figures are the totals divided by 50,000: 35,394 cycles (19.7 us at 1.8 GHz;
the run itself measured 19.7 us per call), 63,645 instructions, 322 branch
misses, 161 cache misses, and 1.80 instructions per cycle.

Per call, V1 retires 33.5 times fewer instructions than V0 (63,645 against
2,129,197) and spends 50.7 times fewer cycles (35,394 against 1,796,102). The
cycles ratio is the timing ratio; the instructions ratio says what most of
that time was: work, two million instructions of copying and allocating that
V1 does not ask for. The cache misses fall further still, 122 times, because
V1 writes each part once into a buffer reserved at the final size instead of
copying the result so far. V1 also gets more done per cycle, 1.80 against
1.19; that remaining gap is the memory system's, not the instruction
count's. The demo's own check reads the same relation on ten calls through
the same kernel interface: 2,150,574 against 63,616 instructions per call,
33.8 times.

## The Second Example: a Filter

The join shows slowness that is more work. The filter shows slowness that is
the same work done slowly. [`Filter.hpp`](../examples/filter/inc/Filter.hpp)
copies the values above a threshold into an output buffer, two ways:

```cpp
std::size_t filterBranchy(std::span<const double> values, double threshold, std::span<double> out) {
  std::size_t kept = 0;
  for (const double value : values) {
    if (value > threshold) {
      // A conditional store: the branch around it survives optimization
      out[kept++] = value;
    }
  }
  return kept;
}

std::size_t filterBranchless(std::span<const double> values, double threshold,
                             std::span<double> out) {
  std::size_t kept = 0;
  for (const double value : values) {
    out[kept] = value; // Always stored; a rejected value is overwritten by the next
    kept += static_cast<std::size_t>(value > threshold);
  }
  return kept;
}
```

Both leave the same values, in the same order, in `out`, and return the same
count; the example's unit tests hold them to that at every size, against the
standard library's `copy_if`. `filterBranchy` tests each value and stores it
only when it passes. On random input the test goes either way with even
odds, in an order no predictor can learn, so the processor guesses wrong
about half the time and throws away what it had started along the wrong
path. On the same values in ascending order the test is false for the first
half and true for the second, and the branch turns once per call.
`filterBranchless` stores every value, kept or not, over the last rejected
one, and advances the cursor by the test's result: there is nothing to
predict.

The store in `filterBranchy` is conditional on purpose. A compiler may not
invent a store the source does not make, so it has to keep the branch. A
conditional sum has no such protection: an optimizer that turns it into a
conditional select makes the two versions the same machine code, and the
three cases below time the same. The demo measures the three cases over the
same 100,000 values in `PerfProfiler.FilterBranchyRandom`,
`PerfProfiler.FilterBranchySorted` and `PerfProfiler.FilterBranchless`, and
the next three steps count each one over 1,000 calls.

## Step 5: Count the Branchy Filter on Random Input

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf -- \
  --gtest_filter=PerfProfiler.FilterBranchyRandom --cycles 100 --repeats 10
cat PerfProfiler.FilterBranchyRandom.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '99852':

     1,586,848,004      cpu-cycles:u
       750,850,433      instructions:u                   #    0.47  insn per cycle
   <not supported>      branches:u
        50,128,904      branch-misses:u
           664,055      cache-misses:u

       1.035762420 seconds time elapsed
```

Per call, over the 1,000 calls: 1,586,848 cycles (881.6 us at 1.8 GHz; the
run measured 883.1 us), 750,850 instructions, 50,129 branch misses, 664 cache
misses, and 0.47 instructions per cycle. Per value: 7.5 instructions, 15.9
cycles, and 0.50 mispredictions. The branch is guessed wrong for one value in
two, as a coin toss deserves, and the core retires less than half an
instruction per cycle while it recovers each time.

## Step 6: The Same Filter on Sorted Input

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf -- \
  --gtest_filter=PerfProfiler.FilterBranchySorted --cycles 100 --repeats 10
cat PerfProfiler.FilterBranchySorted.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '99860':

       443,523,711      cpu-cycles:u
       750,849,455      instructions:u                   #    1.69  insn per cycle
   <not supported>      branches:u
            12,997      branch-misses:u
           637,128      cache-misses:u

       0.398783228 seconds time elapsed
```

The same code on the same values in a different order: 750,849 instructions
per call against 750,850, one instruction apart, and 13 branch misses per
call in place of 50,129. The cycles fall from 1,586,848 to 443,524 per call,
3.58 times, and the instructions per cycle rise from 0.47 to 1.69. Nothing
else moved: 637 cache misses per call against 664. Divide the difference:
1,143,324 cycles per call over 50,116 fewer mispredictions is 22.8 cycles for
each one, the price of a wrong guess on this core, whose pipeline is flushed
and refilled from the right path. The whole gap between the two runs is in
the `branch-misses` line.

## Step 7: The Branchless Filter

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf -- \
  --gtest_filter=PerfProfiler.FilterBranchless --cycles 100 --repeats 10
cat PerfProfiler.FilterBranchless.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '99868':

       288,326,152      cpu-cycles:u
       600,583,131      instructions:u                   #    2.08  insn per cycle
   <not supported>      branches:u
            12,097      branch-misses:u
           925,502      cache-misses:u

       0.324278189 seconds time elapsed
```

No branch on the data, and no order to care about: 12 branch misses per
call on the random input, as few as the sorted case had. It retires fewer
instructions too, 600,583 per call, 6.0 per value against 7.5, because the
loop has no test-and-jump and no separate path for a kept value, and it
retires more of them per cycle, 2.08, because nothing waits on a guess. The
result is 288,326 cycles per call (160.2 us at 1.8 GHz; the run measured
159.1 us): 5.5 times faster than the branchy filter on random input, and on
this rig 1.5 times faster than the branchy filter on sorted input as well.
Its cache misses are the highest of the three, 926 per call, because it
writes every value and so doubles the output traffic. That is the trade: an
unconditional store per value for no misprediction, ever. Which of the
sorted and the branchless cases is faster depends on the processor; on an
x86 laptop this branch's own runs measured the sorted case faster (see
[What Should Reproduce](#what-should-reproduce)). The demo's check reads the
same three counts on ten calls each: 50,010, 3 and 1 branch misses per call
in step 1's run.

## Record Mode

`perf stat` says how many; `perf record` says where. `--profile-args "record
-g"` makes the backend run `perf record -g` (sampling, with call stacks)
instead of `perf stat`, writing `perf.data` and perf's own messages,
`record.err.txt`, into the same folder; the guides' `--target-time 250ms`
sizes the run so that perf has time to attach and sample (see the
[CPU guide](../../docs/CPU_GUIDE.md#cpu-profiling-with-perf)). On this rig,
250 ms per repeat gave 9,912 samples:

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf \
  --profile-args "record -g" --target-time 250ms -- --gtest_filter=PerfProfiler.JoinV0
perf report -i PerfProfiler.JoinV0.perf/perf.data --stdio --no-children
```

Captured output of the second command, cut to the first two functions and,
within the first, to the two frames that matter; the rows below them are the
chain of callers out to `_start`:

```
    64.57%  BenchDemo_02_Pe  libc.so.6                  [.] __memcpy_generic
            |
            ---__memcpy_generic
               vernier::bench::demo::joinV0(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, char)
               ...
    13.63%  BenchDemo_02_Pe  libc.so.6                  [.] _int_malloc
...
```

The report puts 64.6% of the samples in the C library's `memcpy` and 13.6%
in the allocator's `_int_malloc`, both under `joinV0`: the reading
[walkthrough 03](03_GPERF_PROFILER.md) gets from gperftools, from perf.

One thing to know: the benchmark does not wait for `perf record` to finish
writing. It stops perf when the measured repeats end, then prints its result
and exits, and with call stacks to write perf finished about 0.7 s after that
on this rig. `record.err.txt` gets perf's `Captured and wrote` line when the
file is complete; read `perf.data` once that line is there, or `perf report`
says the file's data size is 0 and asks whether perf record was terminated
properly.

## Adding an Event

In stat mode, `--profile-args` text that does not start with `record` is
appended to the `perf stat` command, so `-e <event>` adds an event to the
five. The value starts with a hyphen, which the `bench` CLI's parser takes
for an option of its own unless the value is attached with `=`:

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf \
  --profile-args="-e L1-dcache-load-misses" -- --gtest_filter=PerfProfiler.JoinV0 --cycles 100 --repeats 10
cat PerfProfiler.JoinV0.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '100084':

     1,788,932,858      cpu-cycles:u
     2,129,195,813      instructions:u                   #    1.19  insn per cycle
   <not supported>      branches:u
         8,373,639      branch-misses:u
        22,470,747      cache-misses:u
        22,470,747      L1-dcache-load-misses:u

       1.146819608 seconds time elapsed
```

The sixth line counts the level-1 data cache's load misses, and on this rig it
equals `cache-misses` to the count, 22,470,747: that is what perf's generic
event is on this processor. `perf list` shows the events perf knows on a
machine. Run without `bench run`, the binary takes
`--profile-args "-e L1-dcache-load-misses"` with a space.

## What Should Reproduce

| Reading                                       | On this rig                                                                                                                                                              | Elsewhere                                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| V1 against V0                                 | 47x faster here; 45.6x to 50.1x over ten runs                                                                                                                            | tens of times faster in an optimized build; 36x on the x86 laptop                                                          |
| instructions per call, V0 against V1          | 33.5x in the reports (2,129,197 against 63,645); 33.8x in the demo's check, the same to the instruction in all ten runs                                                  | should match within a few times: 20.5x on the x86 laptop (the demo's check, 1,373,778 against 67,096)                      |
| filter, random against sorted input           | 3.65x slower here; 3.56x to 3.65x over ten runs; 3.58x in cycles                                                                                                         | slower on random input everywhere, by the core's own misprediction penalty: 9.4x on the x86 laptop (499.8 against 53.4 us) |
| filter, random against branchless             | 5.6x slower here; 5.49x to 5.88x over ten runs                                                                                                                           | slower everywhere; 6.3x on the x86 laptop (499.8 against 79.9 us)                                                          |
| filter, sorted against branchless             | branchless 1.5x faster here; 1.51x to 1.61x over ten runs                                                                                                                | either way: the sorted case was 1.5x faster on the x86 laptop (53.4 against 79.9 us)                                       |
| branch misses per value, random input         | 0.50                                                                                                                                                                     | about 0.5 on any processor, 0.48 on the x86 laptop: a coin toss cannot be predicted                                        |
| branch misses per call, sorted and branchless | 13 and 12 in the reports; 3 to 4 and 1 to 2 in the demo's check over ten runs                                                                                            | a handful: 3 and 2 on the x86 laptop                                                                                       |
| instructions, random against sorted input     | equal to within one per call                                                                                                                                             | equal: the same code runs                                                                                                  |
| `branches`                                    | `<not supported>`                                                                                                                                                        | counts on x86: 299,495 per call of V0 on the laptop                                                                        |
| `cache-misses`                                | the level-1 data cache's misses, 19,704 per call of V0                                                                                                                   | a different event elsewhere: 66 per call of V0 on the x86 laptop                                                           |
| absolute times                                | V0 1,001.7 and V1 21.3 us per call, 962.9 to 1,031.6 and 20.2 to 21.5 over ten runs; filter 882.7, 242.1 and 158.4 us, 881.0 to 882.9, 241.5 to 248.0 and 150.0 to 160.6 | will differ                                                                                                                |

The ten runs are step 1's command run ten times in one session, the
reference capture just before them. They describe those runs, not a bound a
run has to meet: the noisiest readings, V0's median and the branchless
filter's, moved 7% across the ten with nothing changed, more than any run's
own CV says. The x86 figures come from this branch's own runs on a laptop
with a hybrid x86 processor (performance and efficiency cores): GCC 11 and
Release for the timings, clang 21 as root in the development container for
the counter checks and the `perf stat` reports. The laptop was busy with
other work, so its timings are noisier than the rig's.

## If It Does Not Match

- **`stat.txt` holds a refusal instead of counts.** With
  `kernel.perf_event_paranoid` at 3 or above (some distributions default to
  it, and a locked-down host can be at 4), perf prints
  `Access to performance monitoring and observability operations is limited`
  with the setting, and counts nothing.
  `bench doctor ./build/bin/ptests/BenchDemo_02_PerfProfiler` reports
  the perf backend's view; on this rig its row reads
  `[WARN] perf       perf_event_paranoid=2 (kernel profiling blocked; userspace counters still work)`,
  a warning because kernel-side events are off limits at 2, which this page
  never needs. Lower the setting (`sudo sysctl -w kernel.perf_event_paranoid=2`),
  grant `CAP_PERFMON` to the binary, or run as root. The demo's counter
  checks skip with the same reason, and say so.
- **`<not supported>` on other lines.** A processor without the event, or a
  virtual machine without a performance monitoring unit. The count stays
  empty and the run is otherwise unaffected; `perf list` says which events
  exist there.
- **Two rows per event, `cpu_atom/...` reading `<not counted>`.** A hybrid
  x86 processor with two kinds of core: perf opens each event on both, and a
  test pinned to a performance core counts on `cpu_core` only. Read the
  `cpu_core` rows.
- **`perf report` says the data size field is 0.** perf record was still
  writing: see [Record Mode](#record-mode).
- **`bench run` rejects `--profile-args "-e ..."`.** Attach the value with
  `=`: see [Adding an Event](#adding-an-event).
- **Different per-call figures.** Check the call count: the totals cover
  `--cycles` times `--repeats` calls, and with `--target-time` the calibrated
  cycle count is the one the `[target-time]` line prints.

## Check Against the Reference

A capture from this rig is committed with the demo, so you can compare a run
of your own against it:

```bash
bench compare src/bench/demo/reference/pi4/02_perf_profiler.csv run1.csv
```

Output for step 1's `run1.csv`, printed by the `bench` CLI built from the tree
this page ships in (`bench 1.0.3`):

```
Test                                  Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
--------------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
PerfProfiler.FilterBranchless        158.75500     158.35100    -0.40400     -0.3%      1.2%      0.4%  neutral
PerfProfiler.FilterBranchyRandom     881.68800     882.66100    +0.97300     +0.1%      0.2%      0.1%  neutral
PerfProfiler.FilterBranchySorted     244.96700     242.14500    -2.82200     -1.2%      0.3%      0.6%  neutral
PerfProfiler.JoinV0                 1020.11000    1001.74000   -18.37000     -1.8%      0.1%      0.1%  neutral
PerfProfiler.JoinV1                   20.98360      21.34230    +0.35870     +1.7%      2.5%      3.8%  neutral

  5 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary seconds before step 1. Every
row moved by less than 2% and is labelled neutral against the 5% threshold.
`Base CV` and `Cand CV` are each run's spread across its own repeats. Over
the ten runs of step 1's command in this session, V0's median ranged from
962.9 to 1,031.6 us per call and V1's from 20.2 to 21.5, about 7% each, with
nothing changed, while the filter's random case stayed within 0.2%. What
should hold is each ratio, and each report's counts per call. The
`hostname` column holds the board's hostname as the capture recorded it.

## What Keeps This Page True

Three things run against these examples, and all fail loudly:

- `PerfProfiler.JoinInstructions`, in the demo binary, counts the
  instructions each join version retires over ten calls through the kernel's
  perf interface, the interface `perf stat` reads, and fails unless V0
  retires at least ten times V1's. On this rig it reads 33.8; with `joinV1`
  given `joinV0`'s body it reads 1.0 and fails.
- `PerfProfiler.FilterBranchMisses` counts the branch misses per call of
  the three filter cases the same way, and fails unless the random case
  mispredicts at least a tenth of a branch per value and at least ten times
  as often as the sorted case and as the branchless case. On this rig it
  reads 50,010, 3 and 1. With `filterBranchy` given `filterBranchless`'s body
  it reads 1, 2 and 1 and fails; with `filterBranchless` given a branch it
  reads 50,085, 3 and 50,069 and fails.
- The example's unit tests hold both filter versions to the same answers as
  the standard library's `copy_if` at every size, and
  `FilterBranchTest.ConditionalStoreKeepsItsBranch` counts the branchy
  filter's mispredictions on random and on sorted input and fails when the
  branch is gone: it is what keeps the conditional store from being
  simplified into a select. All of these are registered with `ctest` under
  the `demo` label, beside the join example's tests:

  ```bash
  ctest --test-dir build -L demo
  ```

  Every test it runs should pass. Where the counter cannot be opened (a
  container's default seccomp profile, `perf_event_paranoid` above 2, a
  processor without the event), or where it was not on the PMU for the
  calls it should have counted (a hybrid processor counts an event on one
  kind of core only, so an unpinned run that lands on the other kind reads
  nothing; a PMU with more events open than counters takes turns), the
  three counting tests skip and say why, and CTest reports them as skipped,
  not passed. Each reading carries the share of the thread's time its
  counter was running, as `perf stat` does, and a count taken part of the
  time is scaled and printed with that share.

This repository has no continuous-integration lane on the reference board, so
nothing runs the demo itself automatically. Before a release it is run on the
rig by hand, with the command in step 1, and the reference CSV is re-captured
when the numbers move.

## See Also

- [Walkthrough 01: basic workflow](01_BASIC_WORKFLOW.md) -- the `join`
  example, measured and compared
- [Walkthrough 03: gperftools](03_GPERF_PROFILER.md) -- which function has
  the time, by sampling
- [Walkthrough 07: Callgrind](07_CALLGRIND_PROFILER.md) -- exact instruction
  counts, line by line, without hardware counters
- [CPU guide: CPU profiling with perf](../../docs/CPU_GUIDE.md#cpu-profiling-with-perf)
  -- the backend's stat and record modes in the framework's own words
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
