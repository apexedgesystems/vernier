# Demo 06: Thread Scaling and Lock Contention

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp), called from several threads at once (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-28 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`

## Overview

A timing taken on one thread says what a call costs when nothing else is
making it. In a program with threads, the same call can cost more: other
threads make it at the same moment, and something they share makes them wait
for each other. `contentionRun()` measures a call while other threads make it
too. This walkthrough starts threads that each join the `join` example's 1,000
words and add the length to one total, first under one lock held for the whole
call, then with nothing shared while they run. On one thread the two cost the
same. On three threads, the version with the lock gets through calls no faster
than one thread does, and the version that shares nothing gets through them
2.6 times as fast as one thread.

## What contentionRun Measures

`perf.contentionRun(worker, label)` is the threaded counterpart of
`throughputLoop()`. For each repeat it starts `--threads` threads, holds them
at a start line until all of them are running, lets each one call `worker`
`--cycles` times, and joins them. It divides the repeat's wall time by every
call the threads made, `--threads` times `--cycles`, and summarizes the
repeats the way the other loops do: a median with its spread, and one CSV row
per test. So the number it prints is the wall time the group needed per call:

- one call's time, when the calls take turns;
- one call's time divided by the number of threads, when no thread waits for
  another.

Multiplied by the thread count, it is the repeat's wall time divided by the
calls each thread made: how long a call took, on average, for the thread that
made it.

The repeat's clock starts before the threads are created and stops after they
are joined, so each repeat also pays for starting and joining them, and for the
start line itself: see [Which Cores](#which-cores).

**In Vernier:** `PERF_CONTENTION(Suite, Name)` declares the test (it is
GoogleTest's `TEST`, named for what it measures), and `PERF_GUARD(perf)` gives
it the harness. `--threads N` sets how many threads `contentionRun()` starts,
and each test's CSV row records that number in its `threads` column. Only
`contentionRun()` starts threads: a test in the same binary that measures with
`throughputLoop()` runs on the calling thread alone, and its row records
`--threads` all the same. The worker makes one call: `contentionRun()`
supplies the loop, so a worker that loops `perf.cycles()` times itself makes
`--cycles` times too many calls.

**Needs:** a core for each thread and one more (see
[Which Cores](#which-cores)), and a machine held still: the
[rig document](../../docs/rigs/RIG_PI4.md)'s build and governor recipe,
sections 3 and 4. Nothing to install.

## The Example

Both versions join the same 1,000 words with `joinV1`, the reserving join of
[walkthrough 01](01_BASIC_WORKFLOW.md#the-example), and add the length to a
total. They differ in what the threads share while they do it.

The first holds one lock for the whole call:

```cpp
struct SharedTotal {
  std::mutex lock;
  std::size_t value = 0;
};

inline void addUnderCoarseLock(SharedTotal& total, const std::vector<std::string>& parts) {
  std::lock_guard<std::mutex> guard(total.lock);
  total.value += joinV1(parts, SEPARATOR).size();
}
```

The total needs a lock: it is one number that every thread adds to. But the
lock is held while `joinV1` runs, and the join needs none: it reads the words,
which no thread changes, and builds a string of its own. Only the add needs the
lock. While one thread joins, the others wait for it.

The second shares nothing while the threads run:

```cpp
inline std::atomic<std::size_t> finishedThreadsTotal{0};

struct ThreadTotal {
  std::size_t value = 0;

  ThreadTotal() = default;
  ThreadTotal(const ThreadTotal&) = delete;
  ThreadTotal& operator=(const ThreadTotal&) = delete;
  ~ThreadTotal() { finishedThreadsTotal += value; }
};

inline thread_local ThreadTotal threadTotal;

inline void addToThreadTotal(const std::vector<std::string>& parts) {
  threadTotal.value += joinV1(parts, SEPARATOR).size();
}
```

`thread_local` gives each thread its own `threadTotal`, which no other thread
reads or writes, so adding to it needs no lock. It is destroyed when its
thread ends, and the destructor adds it to `finishedThreadsTotal`, once per
thread. `contentionRun()` joins every thread it started before it returns, so
by then every thread's total is in. The threads still share the words they
read, and each call still allocates a string; what they do not share is a total
to write to while they work. A lock held only around the add would also let the
joins run at the same time, and leave one short shared write per call.

Both versions are in
[`06_ThreadScaling_Totals.hpp`](../cpu/06_ThreadScaling_Totals.hpp), which the
demo and its check share. The demo
([`06_ThreadScaling_Demo.cpp`](../cpu/06_ThreadScaling_Demo.cpp)) measures each
in a test of its own, one CSV row each:

```cpp
PERF_CONTENTION(ThreadScaling, CoarseLock) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  perf.warmup([&] { addUnderCoarseLock(total, PARTS); });
  perf.contentionRun([&] { addUnderCoarseLock(total, PARTS); }, "coarse_lock");
}
```

`ThreadScaling.NoSharing` is the same with `addToThreadTotal(PARTS)` and no
`SharedTotal`. The warm-up (`--warmup`, one call by default) runs on the
test's own thread before any thread starts, and is not timed.

## Step 1: One Thread

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 2,3 ./build/bin/ptests/BenchDemo_06_ThreadScaling --threads 1 --repeats 10 --csv one.csv
```

Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from ThreadScaling
[ RUN      ] ThreadScaling.CoarseLock
[ThreadScaling.CoarseLock]  18.962 us/call  CV=2.1%  ~52.7K calls/s  (p10=18.735 p90=19.625 sd=0.400)
[       OK ] ThreadScaling.CoarseLock (1915 ms)
[ RUN      ] ThreadScaling.NoSharing
[ThreadScaling.NoSharing]  18.784 us/call  CV=2.2%  ~53.2K calls/s  (p10=18.466 p90=19.526 sd=0.417)
[       OK ] ThreadScaling.NoSharing (1891 ms)
[----------] 2 tests from ThreadScaling (3806 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (3806 ms total)
[  PASSED  ] 2 tests.

==========================================================================
Test                       Median (us)     CV%     Calls/s  Status
--------------------------------------------------------------------------
ThreadScaling.CoarseLock        18.962    2.1%       52.7K  OK
ThreadScaling.NoSharing         18.784    2.2%       53.2K  OK
--------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

Both versions take the same time per call: 19.0 us with the lock and 18.8 us
without, a difference inside either run's spread (CVs of 2.1% and 2.2%). With
one thread nobody else wants the lock, and taking and releasing it costs
nothing the timer can see next to a join of 1,000 words. This is what one call
costs alone. Each test took about 1.9 seconds: ten repeats of 10,000 calls,
the default `--cycles`. Of the two cores `taskset -c 2,3` gives the test, one
is for the thread `contentionRun()` starts and one for the test's own thread;
[Which Cores](#which-cores) shows why the second matters.

## Step 2: Three Threads

```bash
taskset -c 0-3 ./build/bin/ptests/BenchDemo_06_ThreadScaling --threads 3 --repeats 10 --csv three.csv
```

Captured output. The harness prints a progress line about every two seconds
while a test measures, and erases it when the test ends; this is what the
terminal shows at the end:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from ThreadScaling
[ RUN      ] ThreadScaling.CoarseLock
[ThreadScaling.CoarseLock]  20.883 us/call  CV=2.5%  ~47.9K calls/s  (p10=20.619 p90=21.743 sd=0.519)
[       OK ] ThreadScaling.CoarseLock (6319 ms)
[ RUN      ] ThreadScaling.NoSharing
[ThreadScaling.NoSharing]  7.163 us/call  CV=2.9%  ~139.6K calls/s  (p10=6.866 p90=7.339 sd=0.204)
[       OK ] ThreadScaling.NoSharing (2134 ms)
[----------] 2 tests from ThreadScaling (8454 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (8454 ms total)
[  PASSED  ] 2 tests.

==========================================================================
Test                       Median (us)     CV%     Calls/s  Status
--------------------------------------------------------------------------
ThreadScaling.CoarseLock        20.883    2.5%       47.9K  OK
ThreadScaling.NoSharing          7.163    2.9%      139.6K  OK
--------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

With three threads, the version with the lock takes 20.9 us per call, 10% more
than step 1's 19.0: three threads got through calls no faster than one. Only
one of them can be joining at a time, so their calls take turns. Multiplied by
three, 62.6 us: that is how long each thread's call took, on average, most of
it spent waiting for the other two.

The version that shares nothing takes 7.2 us per call, 2.9 times less than the
lock's: the three threads joined at the same time. Multiplied by three,
21.5 us: each thread's call took 14% longer than one call takes alone in
step 1.

The lock's test ran for 6.3 seconds and the other for 2.1: both made three
threads times 10,000 calls per repeat, one taking turns and the other not.

## Step 3: Read the Thread Count in the CSV

The two files hold the same test names; what tells their rows apart is how the
measurement was taken. Five of the columns:

```bash
cut -d, -f1,2,5,10,19 one.csv three.csv
```

```
test,cycles,threads,wallMedian,wallCV
ThreadScaling.CoarseLock,10000,1,18.9622,0.0209258
ThreadScaling.NoSharing,10000,1,18.7842,0.0220643
test,cycles,threads,wallMedian,wallCV
ThreadScaling.CoarseLock,10000,3,20.8825,0.0246394
ThreadScaling.NoSharing,10000,3,7.1632,0.0286138
```

Each row records the thread count its test ran with: `threads` is 1 in
`one.csv` and 3 in `three.csv`, for both tests. `cycles` is the calls each
thread made per repeat, not their total. `wallMedian` and `wallCV` are the
median and its spread as the console printed them, the CV as a fraction.

## Step 4: Compare the Two Runs

```bash
bench compare one.csv three.csv
```

Captured output; the CLI prints the header in bold and the labels in colour,
which is not reproduced here:

```

Test                          Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
ThreadScaling.CoarseLock      18.96220      20.88250    +1.92030    +10.1%      2.1%      2.5%  REGRESSION
ThreadScaling.NoSharing       18.78420       7.16320   -11.62100    -61.9%      2.2%      2.9%  IMPROVEMENT

  1 regression(s)  1 improvement(s)  0 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

`bench compare` puts rows with the same test name side by side and labels
each median change against the 5% threshold. It does not read `threads`: that
the two runs differ in their thread count, and in the cores they were given,
is something you know. Read that way, the labels say what three threads did to
each version: the lock's time per call went up 10.1%, past the threshold, and
its row reads `REGRESSION`; the other's fell 61.9%, and its row reads
`IMPROVEMENT`. The code is the same in both files.

## Which Cores

Step 1 gave its one thread two cores, and step 2 its three threads four: a
core for each thread, and one for the thread that runs the test.
`contentionRun()` holds its threads at a start line until the last one has
started. They wait there by spinning, and the thread that started them spins
too, until it sees the last one arrive and lets them go. A spinning thread
does not give up its core, so when there are more threads than cores, the one
still to arrive cannot run until the scheduler takes a core away from one that
is spinning; on this rig those waits lasted whole ticks of the scheduler's
clock, which ticks every 4 ms. Both commands below make one call per thread,
so a repeat is little more than that start:

```bash
taskset -c 0-3 ./build/bin/ptests/BenchDemo_06_ThreadScaling --threads 3 --cycles 1 --repeats 40
taskset -c 0-3 ./build/bin/ptests/BenchDemo_06_ThreadScaling --threads 4 --cycles 1 --repeats 40
```

Captured result lines, the first two from the first command:

```
[ThreadScaling.CoarseLock]  65.000 us/call  CV=26.8%  ~15.4K calls/s  (p10=63.633 p90=68.900 sd=18.340) [UNSTABLE]
[ThreadScaling.NoSharing]  50.333 us/call  CV=17.8%  ~19.9K calls/s  (p10=49.000 p90=58.500 sd=9.474) [UNSTABLE]
[ThreadScaling.CoarseLock]  1999.500 us/call  CV=1.2%  ~500 calls/s  (p10=1996.400 p90=2002.675 sd=23.346)
[ThreadScaling.NoSharing]  1987.375 us/call  CV=27.6%  ~503 calls/s  (p10=1010.225 p90=1999.125 sd=455.983) [UNSTABLE]
```

With three threads on the four cores, a repeat took 195 us with the lock and
151 us without: the result lines' 65.0 and 50.3 us per call, times three
calls. With four threads on the same four cores, it took 8.0 and 7.9 ms at the
median, 1999.5 and 1987.4 us per call times four: two ticks. Some of the
no-sharing test's repeats waited one tick instead: its `p10` is 4.0 ms. Repeats
this short are uneven, and the harness marks three of the four results
`[UNSTABLE]`, for CVs of 17.8% to 27.6%; here the unevenness is what is being
shown. One thread needs the spare core as well: the same one-call command with
`--threads 1`, run in the same session, took 72 and 74 us per repeat on cores 2
and 3, and 8.0 ms on core 3 alone.

The wait comes at the start of every repeat, so its weight depends on how long
the repeat is. Next to step 2's repeats, 626 ms with the lock and 215 ms
without, 8 ms is 1% and 4%.

## What Should Reproduce

| Reading                                                   | On this rig                                                              | Elsewhere                                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| one thread: the lock's time per call over no sharing's    | 1.01x here; 0.99x to 1.02x over eleven runs: equal                       | equal                                                                                                                       |
| three threads, the lock: time per call                    | 20.6 to 21.2 us over twelve runs, against 18.7 to 19.2 us for one thread | no lower than one thread's                                                                                                  |
| three threads: the lock's time per call over no sharing's | 2.9x here; 2.80x to 3.22x over twelve runs                               | near the thread count, with a core for each thread and one more; 2.84x to 3.06x with three threads on four x86 laptop cores |
| a repeat of one call per thread                           | 195 and 151 us with a core to spare; 8.0 and 7.9 ms without              | whole scheduler ticks without a spare core; how long a tick is depends on the kernel                                        |
| absolute times                                            | 19.0 and 18.8 us per call in step 1, 20.9 and 7.2 us in step 2           | will differ                                                                                                                 |

The ranges come from one session on this rig: step 1's and step 2's commands
ten times each, alternating, then step 2's once more for the reference, then
both again as this page's steps 1 and 2, eleven runs of step 1's command and
twelve of step 2's in all. They describe those runs; they are not a bound a run
has to meet, and another run of the same command can land outside them. The
x86 figure comes from three runs on a laptop's efficiency cores with other work
running; it says the direction holds, not how far.

## If It Does Not Match

- **Every thread on one core.** Pinned to one core, three threads take turns
  whatever the code does:

  ```bash
  taskset -c 3 ./build/bin/ptests/BenchDemo_06_ThreadScaling --threads 3 --repeats 10
  ```

  ```
  [==========] Running 2 tests from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 2 tests from ThreadScaling
  [ RUN      ] ThreadScaling.CoarseLock
  [ThreadScaling.CoarseLock]  21.570 us/call  CV=0.8%  ~46.4K calls/s  (p10=21.343 p90=21.760 sd=0.174)
  [       OK ] ThreadScaling.CoarseLock (6476 ms)
  [ RUN      ] ThreadScaling.NoSharing
  [ThreadScaling.NoSharing]  21.288 us/call  CV=1.3%  ~47.0K calls/s  (p10=20.966 p90=21.412 sd=0.286)
  [       OK ] ThreadScaling.NoSharing (6365 ms)
  [----------] 2 tests from ThreadScaling (12841 ms total)

  [----------] Global test environment tear-down
  [==========] 2 tests from 1 test suite ran. (12841 ms total)
  [  PASSED  ] 2 tests.

  ==========================================================================
  Test                       Median (us)     CV%     Calls/s  Status
  --------------------------------------------------------------------------
  ThreadScaling.CoarseLock        21.570    0.8%       46.4K  OK
  ThreadScaling.NoSharing         21.288    1.3%       47.0K  OK
  --------------------------------------------------------------------------
  2 tests | 2 stable | 0 unstable
  ```

  Both versions take a little more than one call's time per call, 21.6 and
  21.3 us against step 1's 19.0 and 18.8: the version that shares nothing gains
  nothing, because nothing runs at the same time. Give each thread a core, and
  the test's own thread one more.

- **One thread reads slower than in step 1, and the two versions differ.** The
  test's thread and the thread it started share a core. Step 1's command pinned
  to core 3 alone read 21.0 to 22.6 us per call over five runs, with the lock
  1% to 7% slower than without; on cores 2 and 3 it read 18.5 to 19.3 us over
  eleven runs, the versions within 2% of each other. That is more than the
  8 ms wait at the start of each repeat ([Which Cores](#which-cores)), 0.8 us
  per call, accounts for. Use two cores, as step 1 does.
- **CV above a few percent.** The governor is not pinned to `performance`, the
  test is not pinned to its cores, or something else is running. The rig
  document's measurement section has the governor recipe, and
  `vcgencmd get_throttled` should read `0x0` before and after.
- **A thread's calls look `--cycles` times too slow.** The worker loops
  `perf.cycles()` times itself; `contentionRun()` already calls it that many
  times on each thread.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/06_thread_scaling.csv three.csv
```

Output for step 2's `three.csv`, printed by the `bench` CLI built from the tree
this page ships in (`bench 1.0.3`):

```

Test                          Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
ThreadScaling.CoarseLock      20.78120      20.88250    +0.10130     +0.5%      1.6%      2.5%  neutral
ThreadScaling.NoSharing        6.60145       7.16320    +0.56175     +8.5%      2.2%      2.9%  REGRESSION

  1 regression(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference is step 2's command run once more by the same binary on this
board, in the same session, just before this page's steps 1 and 2: its rows'
timestamps are twelve and thirteen seconds before `three.csv`'s. Nothing
changed between the two runs, yet the version that shares nothing came out
8.5% slower, past the 5% threshold, so its row reads `REGRESSION`; the lock's
moved 0.5% and reads `neutral`. That is this pair of runs, not a property of
the code. The CSVs' `hostname` column records the name visible to the process
that captured each: this reference and `three.csv` were captured in the rig's
ordinary session, not in a UTS namespace named `pi4` as some of the rig's other
references were, so both record the board's host name, `raspberrypi`. As the
note under the table says, a label compares two runs' medians; it is not a
significance test, and `Base CV` and `Cand CV` are each run's own spread, which
says nothing about the spread between runs: 2.2% and 2.9% for the version that
shares nothing here. Over the session's twelve runs of step 2's command, that
version's median ranged from 6.5 to 7.4 us, 13.6%, and the lock's from 20.6 to
21.2 us, 2.5%, with nothing changed: the no-sharing median is the noisiest
measurement on this page, and a run of unchanged code can land on either side
of the threshold. Read the medians and their ratio, not the label.

## What Keeps This Page True

- `ThreadScalingTotalsTest`, a test program of its own beside the demo
  ([`06_ThreadScaling_uTest.cpp`](../cpu/utst/06_ThreadScaling_uTest.cpp),
  built as `TestDemoThreadScaling`), runs each version through
  `contentionRun()` as the demo does, with four threads, 25 calls per thread
  and three repeats, and fails unless the total holds every call's joined
  length: the shared total, and the per-thread totals handed over by the time
  `contentionRun()` returns. Without the hand-over, or with one total for
  every thread instead of one each, it fails. It counts, so a busy machine
  does not change its answer, and it is registered with `ctest` under the
  `demo` label.
- The example's unit tests hold `joinV1` to `joinV0`'s answers and
  `joinedSize()` to both, under the same label.

An ordinary test run includes them, and every test it runs should pass:

```bash
ctest --test-dir build -L demo
```

The demo's two timing tests are not registered: what they measure belongs to
the machine they run on. This repository has no continuous-integration lane on
the reference board, so before a release the page's commands are run on the
rig by hand, and the page and its reference CSV are re-captured when what they
show changes.

## See Also

- [Walkthrough 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same `join`,
  measured on one thread and compared
- [Walkthrough 16: off-CPU profiling](16_OFFCPU_PROFILER.md) -- where threads
  wait, when they wait
- [Walkthrough 20: Helgrind](20_HELGRIND_PROFILER.md) -- data races, the
  mistake that removing a lock without removing the sharing makes
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
