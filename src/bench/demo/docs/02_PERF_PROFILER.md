# Demo 02: Linux perf Hardware Counters

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) and [`filter`](../examples/filter/inc/Filter.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-27 (UTC), written for the Vernier 1.0.4 release; captured from
the development tree at project version 1.0.3, whose CLI reported `bench 1.0.3`;
perf 6.18.39. Step 2's console output, the record step's check and the
doctor's rows under If It Does Not Match come from this rig on 2026-10-03
(UTC), a later tree at the same version; the reports of step 3 and of Record
Mode are the first session's. The refused run quoted under If It Does Not Match comes from
the project's development container on an x86-64 laptop whose kernel sets
`perf_event_paranoid` to 4.

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
processor's performance monitoring unit to count events (cycles, instructions
retired, branches mispredicted, cache lines missed) while a program runs, and
prints the totals when it stops. Nothing is sampled: the processor keeps the
counts beside the program, so counting's overhead is generally low, though
not zero. What it cannot do is say where in the program the events happened;
that is `perf record`, below, and the sampling profiler of
[walkthrough 03](03_GPERF_PROFILER.md).

A count is the processor's own tally of an event, and what it tallies depends
on the processor and on scheduling. Which events exist, and what each one
means, differ between processors. This rig's Cortex-A72 has no event for
perf's generic `branches`, so that line reads `<not supported>` on every
report below, and its `cache-misses` reports the same count as its
`L1-dcache-load-misses` (the run in [Adding an Event](#adding-an-event) shows
both), so it is the level-1 data cache's misses here. On the x86 laptop this
tree was also run on, `branches` counts, and `cache-misses` is a different,
far rarer event. And a processor has a fixed number of counters: when more
events are open than it has, perf takes turns among them and scales each count
up from the share of the time it was counted, printing that share at the end
of its line, and such a count is an estimate. None of the lines on this rig
carries a share. Read the events per machine, and compare counts between runs
on the same machine.

**In Vernier:** `--profile perf` starts `perf stat -e
cpu-cycles,instructions,branches,branch-misses,cache-misses -p <pid>` on the
benchmark's own process just before a test's measured repeats and stops it
after them, so warmup and the `--target-time` calibration are not counted. It
starts the measured calls once perf answers that it is counting, and after
them stops perf and waits until perf has written its report: the per-call
timings cover the measured calls alone, and a profiled test takes perf's
start and stop longer than its calls. perf's report is
written to `<Suite.Case>.perf/stat.txt` under the working directory
(`--profile-output-dir DIR` moves the root), and that is the only file stat
mode writes. The totals are for the whole measured window, so a per-call
figure is a total divided by the number of measured calls, `--cycles` times
`--repeats`; the steps below pass both explicitly so the division is in plain
sight. Two flags change what runs: `--profile-args "record -g"` runs `perf
record` instead and writes `perf.data` (see [Record Mode](#record-mode)), and
any other `--profile-args` text is appended to the `perf stat` command (see
[Adding an Event](#adding-an-event)).

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
the filter of [the second example](#the-second-example-a-filter). That is the
whole demo: it measures, and perf reads the counters. The tests that check
the counters still say what this page says belong to the examples' own unit
tests: see [What Keeps This Page True](#what-keeps-this-page-true).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_02_PerfProfiler \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from PerfProfiler
[ RUN      ] PerfProfiler.JoinV0
[target-time] 50.000 ms -> cycles=52 (calibrated 948.0000 us/call, batch of 2)
[PerfProfiler.JoinV0]  959.577 us/call  CV=0.1%  ~1.0K calls/s  (p10=959.325 p90=960.410 sd=1.075)
[       OK ] PerfProfiler.JoinV0 (508 ms)
[ RUN      ] PerfProfiler.JoinV1
[target-time] 50.000 ms -> cycles=2476 (calibrated 20.1875 us/call, batch of 64)
[PerfProfiler.JoinV1]  21.933 us/call  CV=0.8%  ~45.6K calls/s  (p10=21.530 p90=21.963 sd=0.175)
[       OK ] PerfProfiler.JoinV1 (544 ms)
[ RUN      ] PerfProfiler.FilterBranchyRandom
[target-time] 50.000 ms -> cycles=56 (calibrated 881.0000 us/call, batch of 2)
[PerfProfiler.FilterBranchyRandom]  882.768 us/call  CV=0.1%  ~1.1K calls/s  (p10=881.759 p90=883.671 sd=0.990)
[       OK ] PerfProfiler.FilterBranchyRandom (502 ms)
[ RUN      ] PerfProfiler.FilterBranchySorted
[target-time] 50.000 ms -> cycles=200 (calibrated 249.6250 us/call, batch of 8)
[PerfProfiler.FilterBranchySorted]  243.050 us/call  CV=0.4%  ~4.1K calls/s  (p10=241.875 p90=243.922 sd=0.933)
[       OK ] PerfProfiler.FilterBranchySorted (506 ms)
[ RUN      ] PerfProfiler.FilterBranchless
[target-time] 50.000 ms -> cycles=317 (calibrated 157.6250 us/call, batch of 8)
[PerfProfiler.FilterBranchless]  153.891 us/call  CV=0.8%  ~6.5K calls/s  (p10=153.542 p90=155.049 sd=1.161)
[       OK ] PerfProfiler.FilterBranchless (496 ms)
[----------] 5 tests from PerfProfiler (2559 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (2559 ms total)
[  PASSED  ] 5 tests.

==================================================================================
Test                               Median (us)     CV%     Calls/s  Status
----------------------------------------------------------------------------------
PerfProfiler.JoinV0                    959.577    0.1%        1.0K  OK
PerfProfiler.JoinV1                     21.933    0.8%       45.6K  OK
PerfProfiler.FilterBranchyRandom       882.768    0.1%        1.1K  OK
PerfProfiler.FilterBranchySorted       243.050    0.4%        4.1K  OK
PerfProfiler.FilterBranchless          153.891    0.8%        6.5K  OK
----------------------------------------------------------------------------------
5 tests | 5 stable | 0 unstable
```

`joinV0` takes 959.6 us per call and `joinV1` 21.9 us, 44 times less. The
three filter rows are one function twice and its branchless twin once: 882.8 us
per call on values in random order, 243.1 us on the same values in ascending
order, 153.9 us with no branch on the data. Each median is steady within its
run (CV 0.1% to 0.8%). The next steps count what the processor did in each
test, one test at a time, through `perf stat`, and read the counts.

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

[WARN] Profiler 'perf': perf stat counts this process, but branches:u <not supported> here; those columns stay empty; kernel.perf_event_paranoid=2 limits this user to user-space events

[PerfProfiler.JoinV0]  919.675 us/call  CV=0.1%  ~1.1K calls/s  (p10=919.083 p90=920.424 sd=1.169)
[       OK ] PerfProfiler.JoinV0 (1161 ms)
[----------] 1 test from PerfProfiler (1162 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1162 ms total)
[  PASSED  ] 1 test.
```

`bench run` starts the binary pinned to core 3 with `--profile perf`, as its
`Running:` line shows. Just before the measured repeats the backend starts
`perf stat` on the benchmark's own process, and just after them it stops it.
The `[WARN]` line is the perf backend's readiness report, printed once per
run: perf counts this process, but this processor has no `branches:u` event,
and `perf_event_paranoid` 2 keeps the counting to user space, which is all
this page needs. The timing covers the measured calls alone: 919.7 us per
call, a little below the twelve runs of step 1 (926.9 to 1,002.6). The test
took 1,161 ms for about 920 ms of calls; the rest is perf's start, on which
the backend waits until perf answers that it is counting, perf's stop, on
which it waits until perf has written its report, and the test's setup and
warmup.
`--cycles 100 --repeats 10` makes the measured window exactly 1,000 calls, the
number every count in the next step is divided by. perf's report is
`PerfProfiler.JoinV0.perf/stat.txt`, under the directory `bench run` ran from.

## Step 3: Read the Report

```bash
cat PerfProfiler.JoinV0.perf/stat.txt
```

Captured output:

```
 Performance counter stats for process id '115164':

     1,700,528,786      cpu-cycles:u
     2,129,196,778      instructions:u                   #    1.25  insn per cycle
   <not supported>      branches:u
         6,631,451      branch-misses:u
        16,642,608      cache-misses:u

       1.093726453 seconds time elapsed
```

Each line is one event: its total over the window, and perf's name for it
with `:u` for user space. Divided by the 1,000 calls:

- `cpu-cycles`: 1,700,528,786, so 1,700,529 cycles per call. At this rig's
  1.8 GHz that is 944.7 us; the counter and the clock agree, and the core
  ran at full speed for the whole window.
- `instructions`: 2,129,196,778, so 2,129,197 instructions per call, 1.25 of
  them retired per cycle (the `insn per cycle` figure perf adds). This is
  V0's problem in one number: joining 1,000 words takes two million
  instructions, because `out + part + sep` copies everything joined so far
  once per part and builds two temporary strings to do it.
- `branches`: `<not supported>`. This processor has no event perf can map
  that name to, so the column is empty here; it counts on an x86 processor.
- `branch-misses`: 6,631,451, so 6,631 per call: one mispredicted branch
  for every 321 instructions.
- `cache-misses`: 16,642,608, so 16,643 per call. On this rig the event
  reports the same count as `L1-dcache-load-misses` (see
  [Adding an Event](#adding-an-event)), so read it as the level-1 data
  cache's misses here: V0 streams its growing result through the cache once
  per part.

The `time elapsed` line is perf's own clock, 1.09 s: the 1,000 calls of about
0.95 ms, plus most of the 200 ms that the tree this report comes from waited
for perf before the loop. The backend now starts the loop once perf answers
that it is counting, so a current run's window holds little more than the
calls.

## Step 4: Confirm the Fix

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf -- \
  --gtest_filter=PerfProfiler.JoinV1 --cycles 5000 --repeats 10
cat PerfProfiler.JoinV1.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '115172':

     1,751,316,001      cpu-cycles:u
     3,182,198,527      instructions:u                   #    1.82  insn per cycle
   <not supported>      branches:u
        16,157,909      branch-misses:u
         9,684,373      cache-misses:u

       1.129955419 seconds time elapsed
```

This window is 50,000 calls (`--cycles 5000 --repeats 10`), so the per-call
figures are the totals divided by 50,000: 35,026 cycles (19.5 us at 1.8 GHz;
the run itself measured 19.6 us per call), 63,644 instructions, 323 branch
misses, 194 cache misses, and 1.82 instructions per cycle.

Per call, V1 retires 33.5 times fewer instructions than V0 (63,644 against
2,129,197) and spends 48.6 times fewer cycles (35,026 against 1,700,529). The
cycles ratio is the timing ratio; the instructions ratio says what most of
that time was: work, two million instructions of copying and allocating that
V1 does not ask for. The cache misses fall further still, 86 times, because
V1 writes each part once into a buffer reserved at the final size instead of
copying the result so far. V1 also gets more done per cycle, 1.82 against
1.25; that remaining gap is the memory system's, not the instruction
count's.

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

The store in `filterBranchy` is conditional on purpose. A compiler may not add
a store the source does not make, which leaves it fewer ways to remove the
branch than a conditional sum gives it: an optimizer can turn a conditional
sum into a conditional select, and then the two versions are the same machine
code and the three cases below time the same. In every build this page reports
on, `filterBranchy` kept its branch: GCC 14 on this rig, where steps 5 to 7
count its mispredictions; clang 21 on the x86 laptop, where the example's unit
tests count them; and GCC 11 on the laptop, whose machine code for it has the
branch. A build that loses the branch fails those unit tests (see [What Keeps
This Page True](#what-keeps-this-page-true)). The demo measures the three
cases over the same 100,000 values in `PerfProfiler.FilterBranchyRandom`,
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
 Performance counter stats for process id '115180':

     1,588,661,544      cpu-cycles:u
       750,849,761      instructions:u                   #    0.47  insn per cycle
   <not supported>      branches:u
        50,095,324      branch-misses:u
           663,131      cache-misses:u

       1.032246674 seconds time elapsed
```

Per call, over the 1,000 calls: 1,588,662 cycles (882.6 us at 1.8 GHz; the
run measured 883.8 us), 750,850 instructions, 50,095 branch misses, 663 cache
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
 Performance counter stats for process id '115188':

       440,905,132      cpu-cycles:u
       750,849,358      instructions:u                   #    1.70  insn per cycle
   <not supported>      branches:u
            13,163      branch-misses:u
           636,025      cache-misses:u

       0.405215426 seconds time elapsed
```

The same code on the same values in a different order: 750,849 instructions
per call against 750,850, one instruction apart, and 13 branch misses per
call in place of 50,095. The cycles fall from 1,588,662 to 440,905 per call,
3.60 times, and the instructions per cycle rise from 0.47 to 1.70. Nothing
else moved: 636 cache misses per call against 663. Divide the difference:
1,147,756 cycles per call over 50,082 fewer mispredictions is 22.9 cycles for
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
 Performance counter stats for process id '115196':

       281,458,881      cpu-cycles:u
       600,582,636      instructions:u                   #    2.13  insn per cycle
   <not supported>      branches:u
            11,964      branch-misses:u
           900,322      cache-misses:u

       0.316425860 seconds time elapsed
```

No branch on the data, and no order to care about: 12 branch misses per
call on the random input, as few as the sorted case had. It retires fewer
instructions too, 600,583 per call, 6.0 per value against 7.5, because the
loop has no test-and-jump and no separate path for a kept value, and it
retires more of them per cycle, 2.13, because nothing waits on a guess. The
result is 281,459 cycles per call (156.4 us at 1.8 GHz; the run measured
155.9 us): 5.6 times faster than the branchy filter on random input, and on
this rig 1.6 times faster than the branchy filter on sorted input as well.
Its cache misses are the highest of the three, 900 per call, because it
writes every value and so doubles the output traffic. That is the trade: an
unconditional store per value for no misprediction, ever. Which of the
sorted and the branchless cases is faster depends on the processor; on an
x86 laptop runs of this tree measured the sorted case faster (see
[What Should Reproduce](#what-should-reproduce)).

## Record Mode

`perf stat` says how many; `perf record` says where. `--profile-args "record
-g"` makes the backend run `perf record -g` (sampling, with call stacks)
instead of `perf stat`, writing `perf.data` and perf's own messages,
`record.err.txt`, into the same folder; the guides' `--target-time 250ms`
sizes the run so that perf has time to attach and sample (see the
[CPU guide](../../docs/CPU_GUIDE.md#cpu-profiling-with-perf)). Capture:

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf \
  --profile-args "record -g" --target-time 250ms -- --gtest_filter=PerfProfiler.JoinV0
```

When the measured repeats end, the backend stops perf and waits, up to 5 s,
until perf has written `perf.data` before it reports the test; a perf that
has not finished by then fails the run. So the file is complete when
`bench run` returns, and perf's last line in `record.err.txt` says so:

```bash
grep "Captured and wrote" PerfProfiler.JoinV0.perf/record.err.txt
```

Run straight after `bench run` on this rig, it printed:

```
[ perf record: Captured and wrote 1.857 MB ./PerfProfiler.JoinV0.perf/perf.data (9868 samples) ]
```

Then read the report:

```bash
perf report -i PerfProfiler.JoinV0.perf/perf.data --stdio --no-children
```

Captured output, cut to the first two functions and, within the first, to
the two frames that matter; the rows below them are the chain of callers out
to `_start`:

```
    66.15%  BenchDemo_02_Pe  libc.so.6                  [.] __memcpy_generic
            |
            ---__memcpy_generic
               vernier::bench::demo::joinV0(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, char)
               ...
    12.47%  BenchDemo_02_Pe  libc.so.6                  [.] _int_malloc
...
```

The report puts 66.2% of the samples in the C library's `memcpy` and 12.5%
in the allocator's `_int_malloc`, both under `joinV0`: the reading
[walkthrough 03](03_GPERF_PROFILER.md) gets from gperftools, from perf.

The check, not a fixed wait, is what says the file is complete. In a second
run on this rig, `perf report` run straight after `bench run`, with perf still
writing, printed

```
WARNING: The PerfProfiler.JoinV0.perf/perf.data file's data size field is 0 which is unexpected.
Was the 'perf record' command properly terminated?
Error:
failed to process sample
```

and exited with status 0, so its own status does not tell you either.

## Adding an Event

In stat mode, `--profile-args` text that does not start with `record` is
appended to the `perf stat` command, so `-e <event>` adds an event to the
five. `bench run` takes the value attached with `=`, as below, or after a
space:

```bash
bench run ./build/bin/ptests/BenchDemo_02_PerfProfiler --taskset 3 --profile perf \
  --profile-args="-e L1-dcache-load-misses" -- --gtest_filter=PerfProfiler.JoinV0 --cycles 100 --repeats 10
cat PerfProfiler.JoinV0.perf/stat.txt
```

Captured output of the second command:

```
 Performance counter stats for process id '115248':

     1,705,579,008      cpu-cycles:u
     2,129,196,564      instructions:u                   #    1.25  insn per cycle
   <not supported>      branches:u
         6,583,317      branch-misses:u
        16,232,627      cache-misses:u
        16,232,627      L1-dcache-load-misses:u

       1.096468732 seconds time elapsed
```

The sixth line counts the level-1 data cache's load misses, and on this rig it
equals `cache-misses` to the count, 16,232,627: that is what perf's generic
event is on this processor. `perf list` shows the events perf knows on a
machine. Run without `bench run`, the binary takes
`--profile-args "-e L1-dcache-load-misses"` with a space.

## What Should Reproduce

| Reading                                       | On this rig                                                                                                                                                               | Elsewhere                                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| V1 against V0                                 | 44x faster here; 43.8x to 49.4x over twelve runs                                                                                                                          | tens of times faster in an optimized build; 30x on the x86 laptop                                                          |
| instructions per call, V0 against V1          | 33.5x in the reports (2,129,197 against 63,644)                                                                                                                           | should match within a few times: about 20x on the x86 laptop (19.8x to 20.6x in the join's unit test)                      |
| filter, random against sorted input           | 3.63x slower here; 3.60x to 3.69x over twelve runs; 3.60x in cycles                                                                                                       | slower on random input everywhere, by the core's own misprediction penalty: 9.5x on the x86 laptop (325.2 against 34.2 us) |
| filter, random against branchless             | 5.7x slower here; 5.27x to 5.85x over twelve runs                                                                                                                         | slower everywhere; 6.8x on the x86 laptop (325.2 against 47.6 us)                                                          |
| filter, sorted against branchless             | branchless 1.6x faster here; 1.44x to 1.61x over twelve runs                                                                                                              | either way: the sorted case was 1.4x faster on the x86 laptop (34.2 against 47.6 us)                                       |
| branch misses per value, random input         | 0.50                                                                                                                                                                      | about 0.5 on any processor, 0.45 on the x86 laptop: a coin toss cannot be predicted                                        |
| branch misses per call, sorted and branchless | 13 and 12 in the reports                                                                                                                                                  | a handful: 2 to 5 and 1 to 2 in the filter's unit tests on the x86 laptop                                                  |
| instructions, random against sorted input     | equal to within one per call                                                                                                                                              | equal: the same code runs                                                                                                  |
| `branches`                                    | `<not supported>`                                                                                                                                                         | counts on x86: 299,453 per call of V0 on the laptop                                                                        |
| `cache-misses`                                | the level-1 data cache's misses; per call of V0, 16,643 in step 3's run and 16,233 and 20,513 in the two runs behind [Adding an Event](#adding-an-event)                  | a different event elsewhere: 76 per call of V0 on the x86 laptop                                                           |
| absolute times                                | V0 959.6 and V1 21.9 us per call, 926.9 to 1,002.6 and 20.2 to 21.9 over twelve runs; filter 882.8, 243.1 and 153.9 us, 880.9 to 884.1, 239.8 to 245.6 and 151.0 to 167.3 | will differ                                                                                                                |

The twelve runs are step 1's command in one session on this rig: one a minute
before the reference capture, the reference capture, and ten more, the first
of them step 1's. They describe those runs, not a bound a run has to meet: the
noisiest readings, the branchless filter's median and both joins', moved 8% to
11% across the twelve with nothing changed, more than any run's own CV says,
and another run can land outside them. The x86 figures come from runs of this
tree on a laptop with a hybrid x86 processor (performance and efficiency
cores): GCC 11 and Release for the timings, clang 21 as root in the
development container for the unit tests' counts and the `perf stat` reports.
Other work was running on the laptop, so its timings are noisier than the
rig's.

## If It Does Not Match

- **The run fails with `denied: perf stat cannot open the counters as this
user`.** With `kernel.perf_event_paranoid` at 3 or above (some
  distributions default to it, and a locked-down host can be at 4), perf
  refuses to count for a user that is not root. The perf backend checks
  before the measured phase, so the test runs without perf, no `stat.txt` is
  written, and the run exits with status 4 after a report of the request:

  ```
  [FAIL] Profiler 'perf': denied: perf stat cannot open the counters as this user: Access to performance monitoring and observability operations is limited.
  ...
  [profile] --profile perf failed; the run exits with status 4:
  ```

  `bench doctor ./build/bin/ptests/BenchDemo_02_PerfProfiler` reports the
  perf backend's view, and with `--profile perf` added it reports the request
  this page runs on a `Selected request` row as well. On this rig both rows
  read
  `[WARN] perf       perf stat counts this process, but branches:u <not supported> here; those columns stay empty; kernel.perf_event_paranoid=2 limits this user to user-space events`,
  a warning because this core has no `branches:u` event and because
  kernel-side events are off limits at 2, which this page never needs. Lower
  the setting (`sudo sysctl -w kernel.perf_event_paranoid=2`), grant
  `CAP_PERFMON` to the binary, or run as root. The examples' counting tests
  skip with the same reason, and say so.

- **`<not supported>` on other lines.** A processor without the event, or a
  virtual machine without a performance monitoring unit. The count stays
  empty and the run is otherwise unaffected; `perf list` says which events
  exist there.
- **Two rows per event, `cpu_atom/...` reading `<not counted>`.** A hybrid
  x86 processor with two kinds of core: perf opens each event on both, and a
  test pinned to a performance core counts on `cpu_core` only. Read the
  `cpu_core` rows.
- **`perf report` says the data size field is 0.** perf record had not
  finished writing: the file was read during the run, or it came from a
  build of Vernier older than this page, which did not wait for perf to
  finish (see [Record Mode](#record-mode)).
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
PerfProfiler.FilterBranchless        151.04200     153.89100    +2.84900     +1.9%      1.1%      0.8%  neutral
PerfProfiler.FilterBranchyRandom     883.80000     882.76800    -1.03200     -0.1%      0.1%      0.1%  neutral
PerfProfiler.FilterBranchySorted     239.75600     243.05000    +3.29400     +1.4%      0.4%      0.4%  neutral
PerfProfiler.JoinV0                  926.87500     959.57700   +32.70200     +3.5%      0.1%      0.1%  neutral
PerfProfiler.JoinV1                   20.93900      21.93260    +0.99360     +4.7%      0.9%      0.8%  neutral

  5 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary seconds before step 1. Every
row is labelled neutral against the 5% threshold; the joins moved most, V1 by
4.7% and V0 by 3.5%. `Base CV` and `Cand CV` are each run's spread across its
own repeats. Over the twelve runs of step 1's command in this session, V0's
median ranged from 926.9 to 1,002.6 us per call and V1's from 20.2 to 21.9,
8% and 9%, with nothing changed, while the filter's random case stayed within
0.4%. What should hold on this rig is the relationships the steps read: V0
retiring about 33 times V1's instructions per call; the branchy filter
retiring the same instructions on both inputs while mispredicting about once
every two values on random input and a handful of times per call on sorted
input; the branchless filter mispredicting as rarely as the sorted case. The
times and the cache-miss counts move from run to run. The `hostname` column
holds the board's hostname as the capture recorded it.

## What Keeps This Page True

The demo only measures. What this page reads from the counters is checked by
the examples' own unit tests, in the examples' test binary, so that the demo
stays the program you would write; each of them fails loudly:

- `JoinInstructionTest.V0RetiresFarMoreThanV1` counts the instructions each
  join version retires per call over ten calls at the demo's 1,000 words,
  through the kernel's perf interface that `perf stat` reads, and fails
  unless V0 retires at least ten times V1's. On this rig it read 33.0 to 35.1
  in the runs kept for this page (its binary counts allocations with an
  `operator new` of its own, so its counts differ a little from the
  reports'); with `joinV1` given `joinV0`'s body it reads 1.0 and fails, and
  so do the join's allocation tests.
- `FilterBranchTest.ConditionalStoreKeepsItsBranch` counts the branchy
  filter's mispredictions per call on random and on sorted input and fails
  unless the random input mispredicts at least a tenth of a branch per value
  and at least ten times as often as the sorted input: it is what keeps the
  conditional store from being simplified into a select. On this rig it read
  50,100 to 50,194 against 3.
- `FilterBranchTest.BranchlessFormRemovesTheMisses` counts the branchy and
  the branchless filter on the same random input and fails unless the
  branchy one mispredicts at least ten times as often: 50,042 to 50,147
  against 1 to 2 on this rig. With `filterBranchy` given `filterBranchless`'s
  body both filter tests fail (1 against 1, and 2 against 1); with
  `filterBranchless` given a branch the second fails (50,164 against 50,057).
- The examples' other unit tests hold both filter versions to the same
  answers as the standard library's `copy_if` at every size.

All of them are registered with `ctest` under the `demo` label, beside the
join example's other tests, so an ordinary test run includes them:

```bash
ctest --test-dir build -L demo
```

To run only the three counting tests, pin them to one core, so that a hybrid
processor keeps them on the kind of core whose counter they read:

```bash
taskset -c 3 ./build/bin/tests/TestDemoExamples --gtest_filter='JoinInstructionTest.*:FilterBranchTest.*'
```

Every test should pass. Where the counter cannot be opened (a container's
default seccomp profile, `perf_event_paranoid` above 2, a processor without
the event), or where it was not on the PMU for the calls it should have
counted (a hybrid processor counts an event on one kind of core only, so an
unpinned run that lands on the other kind reads nothing; a PMU with more
events open than counters takes turns), the three counting tests skip and
say why, and CTest reports them as skipped, not passed. Only the counter's
own state makes them skip: a counter that ran and counted no mispredictions
on random input fails the two filter tests, as a lost branch should. Each
reading carries the share of the thread's time its counter was running, as
`perf stat` does, and a count taken part of the time is scaled and printed
with that share.

The demo's timing tests are not registered: what they measure belongs to the
machine they run on. This repository has no continuous-integration lane on
the reference board, and a hosted test run is not this page's evidence:
where its machine cannot count, it reports the three counting tests skipped.
Before a release the page's commands and the counting tests are run on the
rig by hand, and the page and its reference CSV are re-captured when what
they show changes.

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
