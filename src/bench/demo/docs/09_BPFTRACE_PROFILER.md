# Demo 09: bpftrace

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** the [`join`](../examples/join/inc/Join.hpp) example's words (see [Shared Workloads](../README.md#5-shared-workloads)), written out as lines and joined by threads, in a workload private to the demo
**Captured:** 2026-10-03 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; bpftrace 0.23.2 on Linux 6.18

## Overview

A timer says how long a call takes. It does not say what the kernel did for
the call meanwhile: how many system calls it made and how long each took, or
how often one of its threads slept until another woke it. bpftrace answers
that kind of question with a short script of probes that run in the kernel.
This walkthrough runs two of the scripts Vernier ships against two pairs of
tests, then a script of your own. Writing the `join` example's 1,000 words as
lines, one `write()` per line, takes about 1,000 times as long per call on
this rig as writing them all with one `write()`, and `write_latency.bt` shows
why: 1,000 system calls per call against one, each as quick as the other.
Three threads that hold one lock for a whole join wake each other about twice
in every three calls, which `wakeup_latency.bt` records, and threads that
share nothing almost never do. A script of your own, a few lines long, then
tells the two write versions apart by the size of each write.

## What is bpftrace?

bpftrace is a tracing language for Linux. A script names probes, places in the
kernel or in a program where something happens, and gives each one an action:
a few statements that run every time the probe fires. This page uses the
kernel's tracepoints: `syscalls:sys_enter_write` fires when a thread enters
`write()` and `syscalls:sys_exit_write` when it returns, `sched:sched_waking`
when a thread asks for another to be woken, `sched:sched_switch` when a CPU
switches from one thread to another. The actions record into maps, such as a
count, a histogram, or a timestamp kept for a later probe to read, and
bpftrace prints the maps when it ends. It compiles the script into BPF
programs, which the kernel checks and attaches; the traced program is neither
changed nor restarted.

- **Best for:** what the kernel did for a program, and how long it took:
  system calls, wakeups, switches between threads, moves between CPUs.
- **Overhead:** paid per event, by the thread that fires the probe, and it
  depends on the script. On this rig a `write()` traced by
  `write_latency.bt` cost about 1.5 us more than an untraced one, and one
  traced by step 6's one-probe script about 0.2 us more
  ([Step 3](#step-3-read-the-report)); a probe that seldom fires costs
  nothing a timer can see.
- **Not for:** where a function spends its CPU time (perf, gperftools and
  callgrind, walkthroughs [02](02_PERF_PROFILER.md),
  [03](03_GPERF_PROFILER.md) and [07](07_CALLGRIND_PROFILER.md)), heap
  ([14](14_MASSIF_PROFILER.md), [21](21_HEAPTRACK_PROFILER.md)) or memory
  errors ([15](15_MEMCHECK_PROFILER.md)). Where threads block, with their
  stacks, is the off-CPU backend's ([16](16_OFFCPU_PROFILER.md)).

**In Vernier:** `--profile bpftrace` runs one bpftrace per selected script
around each test that has the profiler guard, `PERF_GUARD(perf)`, as all four
of this demo's tests do. `--bpf` selects the scripts, comma-separated;
without it, `write_latency` and `fsync_latency` run. For each script, the
backend writes a copy into the test's capture folder, `<Suite.Case>.bpf/`
under `--profile-output-dir` (under the working directory when that is not
given), with `{{PID}}` replaced by the benchmark's process id and a short
program appended, the capture window, and starts bpftrace on the copy. The
test's warm-up has run by then. The measured repeats start only once every
tracer has printed `bpftrace armed <pid> <tid>`, which shows that its probes
are attached and that it sees the benchmark's threads; after them, each
tracer prints `bpftrace disarmed <pid> <tid>`, is stopped with SIGINT, and
prints its maps. The capture folder holds three files per script, named after
the script's file: `<name>.tmp.bt`, the copy bpftrace ran; `<name>.out.text`,
what it printed; and `<name>.err.txt`, its errors. A clean capture prints
nothing on the console. A capture that is not prints a `[bpftrace]` line
saying why ([If It Does Not Match](#if-it-does-not-match)).
[BPF_SCRIPTS.md](../../docs/BPF_SCRIPTS.md) has the appended program, what its
two lines prove and what they do not, and the rules for names and paths.

**Needs:** bpftrace, which the rig document's
[one-time setup](../../docs/rigs/RIG_PI4.md#2-one-time-setup) installs, and
root: bpftrace refuses every other user. Run the benchmark as root, or set
`BENCH_SUDO=1`, and the backend runs bpftrace, and the `kill` that stops it,
through `sudo -n`; the test itself and its files stay yours. A sudoers grant
must then allow `bpftrace` with the run's script arguments and `kill` with
`-2`, `-15` and `-9`. This rig's user may run anything through `sudo -n`
(`(ALL) NOPASSWD: ALL`), so this page shows the opt-in working, not a grant
scoped to those two programs. The binary's `--profile-check` says whether a
request can run:

```bash
BENCH_SUDO=1 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace --bpf write_latency --profile-check
```

Captured output, the row for this request; the check first reports the build
and every backend's default mode, cut here as `...`:

```
...
  Selected request: --profile bpftrace --bpf write_latency
  [OK]   bpftrace   write_latency: a probe copy with a 5 s self-exit stayed running for 1000 ms through sudo -n (BENCH_SUDO=1) and stopped on SIGINT through sudo -n kill (probe with /usr/bin/bpftrace); not checked: the grant for the run's own command (/usr/bin/bpftrace -q -B none <capture folder>/write_latency.tmp.bt), SIGTERM and SIGKILL through sudo, and the run's capture
...
```

The check ran a copy of the script through `sudo -n` until it had attached, a
second at least, and stopped it. Its copy lives in a temporary directory, so it could not try the run's own
command, which names the capture folder: with a grant scoped to fixed
arguments, the run's start shows whether the grant allows that command.

## The Example

The writes come first. Both versions write the same text to `/dev/null`: the
`join` example's 1,000 words (`makeParts(1000, 42)`, words of 3 to 10
letters), each followed by a newline, 7,490 bytes in all. One writes it the
way a logger that writes out every line as it comes does, one `write()` per
line:

```cpp
inline std::size_t writeEachLine(int fd, const std::vector<std::string>& lines) {
  std::size_t written = 0;
  for (const std::string& line : lines) {
    const ssize_t RESULT = ::write(fd, line.data(), line.size());
    written += RESULT > 0 ? static_cast<std::size_t>(RESULT) : 0;
  }
  return written;
}
```

The other writes the lines end to end, `joinV1(words, '\n')`, with one
`write()`:

```cpp
inline std::size_t writeBatched(int fd, const std::string& text) {
  const ssize_t RESULT = ::write(fd, text.data(), text.size());
  return RESULT > 0 ? static_cast<std::size_t>(RESULT) : 0;
}
```

They write the same bytes; they differ in how many system calls they make for
them, 1,000 against one.

The threads come second, in the shape [walkthrough 06](06_THREAD_SCALING.md)
times: each call joins the 1,000 words with `joinV1` and adds the length to a
total. `addUnderCoarseLock()` holds one mutex for the whole call, so the joins
take turns, and a thread that finds the mutex held sleeps until the thread
that holds it wakes it. `addToThreadTotal()` adds to a `thread_local` total
that its thread hands over once, when it ends, so the threads share nothing
while they run. Both are in
[`09_BpftraceProfiler_Workload.hpp`](../cpu/09_BpftraceProfiler_Workload.hpp),
with the two write versions.

The demo, [`09_BpftraceProfiler_Demo.cpp`](../cpu/09_BpftraceProfiler_Demo.cpp),
measures each version in a test of its own, one CSV row each:

```cpp
PERF_IO(BpftraceProfiler, WritePerLine) {
  PERF_GUARD(perf);

  const auto LINES = linesOf(demo::makeParts(PART_COUNT, PART_SEED));
  const int FD = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
  ASSERT_GE(FD, 0) << "cannot open /dev/null";

  perf.warmup([&] { writeEachLine(FD, LINES); });
  perf.throughputLoop([&] { writeEachLine(FD, LINES); }, "write_per_line");
  ::close(FD);
}
```

`BpftraceProfiler.WriteBatched` is the same with the joined text and
`writeBatched()`. `BpftraceProfiler.CoarseLock` and
`BpftraceProfiler.NoSharing` call their version through `contentionRun()`,
which starts `--threads` threads and has each make `--cycles` calls per repeat;
walkthrough 06 explains what its time per call means. `PERF_IO` and
`PERF_CONTENTION` are GoogleTest's `TEST`, named for what the test measures.
Nothing in the demo mentions bpftrace: `--profile bpftrace` on the command line
is all it takes.

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler \
  --gtest_filter='BpftraceProfiler.Write*' --target-time 50ms --repeats 10 --csv writes.csv
```

Run it as the rig document's
[measurement section](../../docs/rigs/RIG_PI4.md#4-running-a-measurement)
says, governor and all. Captured output:

```
Note: Google Test filter = BpftraceProfiler.Write*
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from BpftraceProfiler
[ RUN      ] BpftraceProfiler.WritePerLine
[target-time] 50.000 ms -> cycles=88 (calibrated 564.5000 us/call, batch of 2)
[BpftraceProfiler.WritePerLine]  566.989 us/call  CV=0.2%  ~1.8K calls/s  (p10=566.912 p90=567.807 sd=1.349)
[       OK ] BpftraceProfiler.WritePerLine (505 ms)
[ RUN      ] BpftraceProfiler.WriteBatched
[target-time] 50.000 ms -> cycles=88048 (calibrated 0.5679 us/call, batch of 2048)
[BpftraceProfiler.WriteBatched]  0.569 us/call  CV=0.2%  ~1.8M calls/s  (p10=0.567 p90=0.570 sd=0.001)
[       OK ] BpftraceProfiler.WriteBatched (503 ms)
[----------] 2 tests from BpftraceProfiler (1009 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (1009 ms total)
[  PASSED  ] 2 tests.

===============================================================================
Test                            Median (us)     CV%     Calls/s  Status
-------------------------------------------------------------------------------
BpftraceProfiler.WritePerLine       566.989    0.2%        1.8K  OK
BpftraceProfiler.WriteBatched         0.569    0.2%        1.8M  OK
-------------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

The per-line version took 567.0 us per call and the batched one 0.569 us: 996
times as long, for the same 7,490 bytes. One `write()` of the whole text costs
about what one `write()` of a line does: the cost is the system call, entering
the kernel and leaving it, not the bytes, which `/dev/null` takes without
copying. `--target-time 50ms` sized the repeats from a calibration, as the
`[target-time]` lines show: 88 calls per repeat of the per-line version and
88,048 of the batched one, with CVs of 0.2%.

Over the session's twelve runs of this command, the per-line median ranged
from 531.9 to 567.0 us, the batched one from 0.528 to 0.569 us, and their ratio
from 991 to 1,009. This run was the slowest of the twelve in both tests, by
about the same 6.5%, so the ratio held. Those ranges describe those runs; they
are not a bound a run has to meet, and another run can land outside them.

The threads need cores of their own: three threads, and one more for the
test's own thread ([walkthrough 06](06_THREAD_SCALING.md#which-cores) says
why):

```bash
taskset -c 0-3 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler \
  --gtest_filter='BpftraceProfiler.CoarseLock:BpftraceProfiler.NoSharing' --threads 3 \
  --target-time 50ms --repeats 10 --csv threads.csv
```

Captured output:

```
Note: Google Test filter = BpftraceProfiler.CoarseLock:BpftraceProfiler.NoSharing
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from BpftraceProfiler
[ RUN      ] BpftraceProfiler.CoarseLock
[target-time] 50.000 ms -> cycles=2509 (calibrated 19.9219 us/call, batch of 64)
[BpftraceProfiler.CoarseLock]  20.688 us/call  CV=2.1%  ~48.3K calls/s  (p10=20.146 p90=21.400 sd=0.438)
[       OK ] BpftraceProfiler.CoarseLock (1569 ms)
[ RUN      ] BpftraceProfiler.NoSharing
[target-time] 50.000 ms -> cycles=2496 (calibrated 20.0312 us/call, batch of 64)
[BpftraceProfiler.NoSharing]  7.180 us/call  CV=2.3%  ~139.3K calls/s  (p10=6.888 p90=7.306 sd=0.165)
[       OK ] BpftraceProfiler.NoSharing (536 ms)
[----------] 2 tests from BpftraceProfiler (2105 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (2105 ms total)
[  PASSED  ] 2 tests.

=============================================================================
Test                          Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------------------
BpftraceProfiler.CoarseLock        20.688    2.1%       48.3K  OK
BpftraceProfiler.NoSharing          7.180    2.3%      139.3K  OK
-----------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

On three threads, the version with the lock took 20.7 us per call and the one
that shares nothing 7.2 us: 2.9 times as long, and 2.80 to 3.07 times over the
session's twelve runs. Walkthrough 06 reads these numbers. Here they are what
step 5 traces: under the lock the threads take turns, and a thread that finds
the lock held sleeps until the holder wakes it.

## Step 2: Trace the Writes

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace \
  --profile-output-dir bpf-writes --taskset 3 --cycles 20 --repeats 2 \
  -- --bpf write_latency --gtest_filter=BpftraceProfiler.WritePerLine
```

`--cycles 20 --repeats 2` fixes the measured calls at 40, so the report can be
checked against them: 40 calls of 1,000 lines are 40,000 `write()` calls. It
also keeps the traced run short. Captured output:

```
Running: taskset -c 3 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --cycles 20 --repeats 2 --profile bpftrace --profile-output-dir bpf-writes --bpf write_latency --gtest_filter=BpftraceProfiler.WritePerLine
Note: Google Test filter = BpftraceProfiler.WritePerLine
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from BpftraceProfiler
[ RUN      ] BpftraceProfiler.WritePerLine
[BpftraceProfiler.WritePerLine]  2075.825 us/call  CV=0.5%  ~482 calls/s  (p10=2067.725 p90=2083.925 sd=10.125)
[       OK ] BpftraceProfiler.WritePerLine (2561 ms)
[----------] 1 test from BpftraceProfiler (2561 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2561 ms total)
[  PASSED  ] 1 test.
```

`bench run` printed the command it ran: `--taskset 3` became `taskset -c 3` in
front, and its other options and those after `--` went to the binary. The test
passed, and the backend printed nothing else: the capture was clean. The
traced calls took 2,075.8 us each, against step 1's 567.0 us untraced; step 3
says what the difference is. The test's files are in a capture folder named
after it, under `bpf-writes/`:

```bash
ls bpf-writes/BpftraceProfiler.WritePerLine.bpf
```

```
write_latency.err.txt
write_latency.out.text
write_latency.tmp.bt
```

`write_latency.tmp.bt` is the copy bpftrace ran: the bundled script with the
process id in place of `{{PID}}`, then the capture window.

```bash
cat bpf-writes/BpftraceProfiler.WritePerLine.bpf/write_latency.tmp.bt
```

Captured output, with the script's opening comment cut to `...`:

```
...
tracepoint:syscalls:sys_enter_write /pid == 128551/ {
  @start[tid] = nsecs;
}

tracepoint:syscalls:sys_exit_write /@start[tid]/ {
  @write_latency_us = hist((nsecs - @start[tid]) / 1000);
  delete(@start[tid]);
}

tracepoint:sched:sched_process_exit /tid == 128551/ {
  exit();
}

// Added by Vernier: the capture window (BPF_SCRIPTS.md). The measured
// repeats run between its arm line and its stop line.
tracepoint:sched:sched_switch {
  if (pid == 128551 && (args->prev_state == 1 || args->prev_state == 2)) {
    if (tid == 128572 && comm == "vernier-arm") {
      if (@vernier_window == 0) {
        @vernier_window = 1;
        printf("bpftrace armed %d %d\n", pid, tid);
      }
    } else if (tid == 128551 && comm == "vernier-stop") {
      if (@vernier_window == 1) {
        @vernier_window = 2;
        printf("bpftrace disarmed %d %d\n", pid, tid);
        clear(@vernier_window);
      }
    }
  }
}
```

The first probe fires when a thread of process 128551 enters `write()` and
keeps the time under the thread's id, `tid`; the second fires when that thread
returns from it, adds the time the call took to a histogram of microseconds,
`@write_latency_us`, and drops the kept time. The third ends the trace if the
process's main thread exits first, whose thread id is the process id. The
window's program arms when thread 128572, which the backend started under the
name `vernier-arm`, goes to sleep, and closes when the main thread, which takes
the name `vernier-stop` after the measured repeats, does.

## Step 3: Read the Report

```bash
cat bpf-writes/BpftraceProfiler.WritePerLine.bpf/write_latency.out.text
```

```
bpftrace armed 128551 128572
bpftrace disarmed 128551 128551




@write_latency_us:
[0]                39977 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[1]                   12 |                                                    |
[2, 4)                 2 |                                                    |
[4, 8)                 6 |                                                    |
[8, 16)                3 |                                                    |
```

The first two lines are the capture window's. `bpftrace armed 128551 128572`:
the tracer saw thread 128572 of process 128551, the one the backend started
to arm it, go to sleep, and the measured repeats started only then.
`bpftrace disarmed 128551 128551`: after them, it saw the thread that ran the
test, the main thread, whose id is the process's. Both name the threads the
copy binds, so this report is this run's. Then come a few empty lines of
bpftrace's own and the maps it printed when it stopped.

`@write_latency_us` counts each `write()` the process made while traced, in
buckets of microseconds: `[0]` for under one, `[1]` for one to two, `[2, 4)`
for two up to four, and so on, each bucket twice as wide as the one before.
The counts add up to 40,000: the 40 measured calls' 1,000 lines each, every
one of them traced, and no other `write()` in that time. 39,977 took under a
microsecond in the kernel, 12 one to two, and 11 from 2 to 16.

What not to conclude:

- **That tracing is free.** The traced calls took 2,075.8 us, against 567.0
  us untraced in step 1: about 1.5 us more per `write()`, the price of two
  probes, a map entry kept and deleted and a histogram update around each one.
  Time with the profiler off, as step 1 does, and use the trace to count and
  to see where the time goes.
- **That a `write()` takes under a microsecond in your program.** These write
  to `/dev/null`, which takes the bytes without copying them; a write to a
  file, a pipe or a terminal does work these do not.
- **That the count is always the measured calls' exactly.** The tracer counts
  every `write()` the process makes between its start and its stop. This run
  wrote its output to a file, where the harness's result line waits in a
  buffer until the test ends. Run in a terminal, the same command counts one
  more: the harness prints the result line before it stops the tracer (40,001
  on an x86-64 laptop with bpftrace 0.14.0, against 40,000 to a file).

## Step 4: Confirm the Fix

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace \
  --profile-output-dir bpf-writes --taskset 3 --cycles 20 --repeats 2 \
  -- --bpf write_latency --gtest_filter=BpftraceProfiler.WriteBatched
cat bpf-writes/BpftraceProfiler.WriteBatched.bpf/write_latency.out.text
```

Captured output of the run:

```
Running: taskset -c 3 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --cycles 20 --repeats 2 --profile bpftrace --profile-output-dir bpf-writes --bpf write_latency --gtest_filter=BpftraceProfiler.WriteBatched
Note: Google Test filter = BpftraceProfiler.WriteBatched
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from BpftraceProfiler
[ RUN      ] BpftraceProfiler.WriteBatched
[BpftraceProfiler.WriteBatched]  2.375 us/call  CV=15.8%  ~421.1K calls/s  (p10=2.075 p90=2.675 sd=0.375) [UNSTABLE]
[       OK ] BpftraceProfiler.WriteBatched (2421 ms)
[----------] 1 test from BpftraceProfiler (2421 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2421 ms total)
[  PASSED  ] 1 test.
```

and of the report:

```
bpftrace armed 128586 128596
bpftrace disarmed 128586 128586




@write_latency_us:
[0]                   39 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[1]                    0 |                                                    |
[2, 4)                 0 |                                                    |
[4, 8)                 1 |@                                                   |
```

The batched version made 40 `write()` calls in its 40 calls, one each, where
the per-line version made 40,000 for the same bytes. Each took about as long
in the kernel as a per-line write: 39 under a microsecond, one 4 to 8 us. That
is the whole of what step 1 measured: one version pays for 1,000 system calls
per call, the other for one. The traced time per call, 2.375 us, is marked
`[UNSTABLE]`: 20 calls of about 2 us make repeats of about 50 us, too short for
a steady median, and the per-call time of this version is step 1's to give.

## Step 5: Wakeups Under a Lock

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace \
  --profile-output-dir bpf-wakeups --taskset 0-3 --cycles 200 --repeats 3 \
  -- --bpf wakeup_latency --threads 3 \
  --gtest_filter='BpftraceProfiler.CoarseLock:BpftraceProfiler.NoSharing'
cat bpf-wakeups/BpftraceProfiler.CoarseLock.bpf/wakeup_latency.out.text
cat bpf-wakeups/BpftraceProfiler.NoSharing.bpf/wakeup_latency.out.text
```

Captured output of the run:

```
Running: taskset -c 0-3 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --cycles 200 --repeats 3 --profile bpftrace --profile-output-dir bpf-wakeups --bpf wakeup_latency --threads 3 --gtest_filter=BpftraceProfiler.CoarseLock:BpftraceProfiler.NoSharing
Note: Google Test filter = BpftraceProfiler.CoarseLock:BpftraceProfiler.NoSharing
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from BpftraceProfiler
[ RUN      ] BpftraceProfiler.CoarseLock
[BpftraceProfiler.CoarseLock]  21.633 us/call  CV=1.5%  ~46.2K calls/s  (p10=21.503 p90=22.105 sd=0.324)
[       OK ] BpftraceProfiler.CoarseLock (2391 ms)
[ RUN      ] BpftraceProfiler.NoSharing
[BpftraceProfiler.NoSharing]  7.105 us/call  CV=2.3%  ~140.7K calls/s  (p10=7.097 p90=7.381 sd=0.165)
[       OK ] BpftraceProfiler.NoSharing (827 ms)
[----------] 2 tests from BpftraceProfiler (3219 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (3219 ms total)
[  PASSED  ] 2 tests.

=============================================================================
Test                          Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------------------
BpftraceProfiler.CoarseLock        21.633    1.5%       46.2K  OK
BpftraceProfiler.NoSharing          7.105    2.3%      140.7K  OK
-----------------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

and of the two reports, the lock's first:

```
bpftrace armed 128610 128620
bpftrace disarmed 128610 128610



@wakeup_latency_us:
[4, 8)              1204 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[8, 16)                7 |                                                    |
[16, 32)               3 |                                                    |
```

```
bpftrace armed 128610 128640
bpftrace disarmed 128610 128610



@wakeup_latency_us:
[4, 8)                 7 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[8, 16)                3 |@@@@@@@@@@@@@@@@@@@@@@                              |
[16, 32)               2 |@@@@@@@@@@@@@@                                      |
[32, 64)               0 |                                                    |
[64, 128)              1 |@@@@@@@                                             |
```

`wakeup_latency.bt` records each wakeup a thread of the process asks for, at
the `sched:sched_waking` tracepoint, which runs in the thread that asks, and
times it until the woken thread starts to run (`sched:sched_switch`). Each
test made 1,800 calls: 3 threads, 200 calls each, 3 repeats. Under the lock
they made 1,214 wakeups, about two for every three calls: a thread that finds
the lock held goes to sleep, and the thread that holds it wakes it when it
lets go. Sharing nothing, the same 1,800 calls made 13: the threads never wait
for each other while they run. The two tests ran on the same four cores, each
under a tracer of its own running the same script.

What the latency is: 1,204 of the lock's 1,214 wakeups took 4 to 8 us from the
request to the woken thread running. That spans the wakeup's trip to the CPU
the woken thread runs on and the switch to it there. It is not how long the
thread waited for the lock, which started when the thread went to sleep, well
before another thread asked for it to be woken. Tracing cost little here: the
lock's calls took 21.6 us traced against 20.7 untraced in step 1.

## Step 6: A Script of Your Own

The bundled scripts are examples; a script of your own runs the same way.
This one, `write_sizes.bt`, records something none of them does: a histogram
of the size each `write()` asks for, `args->count`, the system call's third
argument as the `sys_enter_write` tracepoint names it:

```
// Histogram of write() sizes (bytes) for a specific PID. {{PID}} is replaced
// by the backend before the run.
tracepoint:syscalls:sys_enter_write /pid == {{PID}}/ {
  @write_bytes = hist(args->count);
}

tracepoint:sched:sched_process_exit /tid == {{PID}}/ {
  exit();
}
```

`{{PID}}` is the backend's one convention: in its copy, it becomes the
benchmark's process id. bpftrace's `pid` is the process id of the thread a
probe fires in, so the first probe counts every thread of the benchmark. The
second ends the trace if the benchmark's main thread exits first, whose thread
id, `tid`, is the process id; the bundled scripts end the same way. The
backend stops the tracer with SIGINT before that, when the measured repeats
end.

Give `--bpf` the script's path, from the working directory, to run it on both
write tests:

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace \
  --profile-output-dir bpf-sizes --taskset 3 --cycles 20 --repeats 2 \
  -- --bpf ./write_sizes.bt --gtest_filter='BpftraceProfiler.Write*'
cat bpf-sizes/BpftraceProfiler.WritePerLine.bpf/write_sizes.out.text
cat bpf-sizes/BpftraceProfiler.WriteBatched.bpf/write_sizes.out.text
```

Captured output of the run, with GoogleTest's lines cut to `...`:

```
Running: taskset -c 3 ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --cycles 20 --repeats 2 --profile bpftrace --profile-output-dir bpf-sizes --bpf ./write_sizes.bt --gtest_filter=BpftraceProfiler.Write*
...
[BpftraceProfiler.WritePerLine]  751.425 us/call  CV=0.3%  ~1.3K calls/s  (p10=749.485 p90=753.365 sd=2.425)
...
[BpftraceProfiler.WriteBatched]  0.950 us/call  CV=21.1%  ~1.1M calls/s  (p10=0.790 p90=1.110 sd=0.200) [UNSTABLE]
...
```

and of the two reports:

```
bpftrace armed 128661 128671
bpftrace disarmed 128661 128661



@write_bytes:
[4, 8)             19680 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  |
[8, 16)            20320 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
```

```
bpftrace armed 128661 128682
bpftrace disarmed 128661 128661



@write_bytes:
[4K, 8K)              40 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
```

The per-line version's 40 calls asked for 40,000 writes of 4 to 15 bytes:
19,680 of 4 to 7 and 20,320 of 8 to 15, one per word and its newline. The
batched version's 40 calls asked for 40 writes of 4K to 8K, each the
7,490-byte text. The sizes alone tell the two versions apart. One probe and a
histogram cost the per-line version about 0.2 us per `write()`: 751.4 us per
call, against 567.0 untraced, where step 2's script cost 1.5 us.

To keep your scripts in a directory of their own and name them as the bundled
ones are named, give `--bpf-scripts` the directory; `--bpf write_sizes` then
runs `myscripts/write_sizes.bt`:

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace \
  --profile-output-dir bpf-sizes-dir --taskset 3 --cycles 20 --repeats 2 \
  -- --bpf-scripts myscripts --bpf write_sizes --gtest_filter=BpftraceProfiler.WritePerLine
cat bpf-sizes-dir/BpftraceProfiler.WritePerLine.bpf/write_sizes.out.text
```

```
bpftrace armed 128694 128715
bpftrace disarmed 128694 128694



@write_bytes:
[4, 8)             19680 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  |
[8, 16)            20320 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
```

The same writes. `PERF_BPF_SCRIPTS=myscripts` in the environment does what
`--bpf-scripts` does; the flag wins over the variable, and without either, a
name is looked up in the bundled directory.

The same script runs without Vernier's hooks too. Put bpftrace's own `cpid`,
the process id of the command it starts with `-c`, in place of `{{PID}}`, and
let bpftrace start the benchmark:

```bash
sed 's/{{PID}}/cpid/g' write_sizes.bt > write_sizes_cpid.bt
sudo bpftrace -q -c "./build/bin/ptests/BenchDemo_09_BpftraceProfiler --gtest_filter=BpftraceProfiler.WritePerLine --cycles 20 --repeats 2" \
  write_sizes_cpid.bt
```

Captured output:

```
Note: Google Test filter = BpftraceProfiler.WritePerLine
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from BpftraceProfiler
[ RUN      ] BpftraceProfiler.WritePerLine
[BpftraceProfiler.WritePerLine]  767.475 us/call  CV=1.3%  ~1.3K calls/s  (p10=759.495 p90=775.455 sd=9.975)
[       OK ] BpftraceProfiler.WritePerLine (35 ms)
[----------] 1 test from BpftraceProfiler (35 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (35 ms total)
[  PASSED  ] 1 test.


@write_bytes:
[4, 8)             20172 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  |
[8, 16)            20828 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[16, 32)               0 |                                                    |
[32, 64)               5 |                                                    |
[64, 128)              2 |                                                    |
[128, 256)             1 |                                                    |
```

bpftrace started the benchmark and traced it from its start to its exit, so
the histogram holds the warm-up call too, 41 calls of 1,000 lines, and 8
larger writes of 32 to 255 bytes, the test's own output, which GoogleTest and
the harness write a line or two at a time. Nothing marks the measured repeats
in such a trace, and the benchmark ran as root, as bpftrace does; it left
nothing root-owned behind, since it wrote no files. Vernier's hooks are what
confine a capture to the measured repeats. [BPF_SCRIPTS.md](../../docs/BPF_SCRIPTS.md#scripts)
also shows how to attach to a benchmark that is already running.

## What Should Reproduce

| Reading                                     | On this rig                                                                   | Elsewhere                                                                                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| per line over batched, time per call        | 996x here; 991x to 1,009x over twelve runs                                    | about 1,000x where a `write()` to `/dev/null` costs the same at any size; 979x to 997x over three runs on an x86-64 laptop           |
| `write()` calls traced in step 2's 40 calls | 40,000 per line, 40 batched                                                   | the same, wherever the `syscalls` tracepoints exist: they are the code's count (40,000 on the x86-64 laptop, one more on a terminal) |
| `write()` latency                           | 39,977 of 40,000 under 1 us                                                   | depends on the kernel and the CPU                                                                                                    |
| what `write_latency.bt` costs per `write()` | about 1.5 us                                                                  | depends on the machine                                                                                                               |
| wakeups in 1,800 calls on three threads     | 1,214 under the lock, 13 sharing nothing                                      | many under the lock and few without; how many depends on the cores and the scheduler                                                 |
| wakeup latency under the lock               | 1,204 of 1,214 in 4 to 8 us                                                   | depends on the machine                                                                                                               |
| write sizes                                 | per line 19,680 of 4 to 7 bytes and 20,320 of 8 to 15; batched 40 of 4K to 8K | should match: they are the example's                                                                                                 |
| absolute times                              | 567.0 and 0.569 us per call; 20.7 and 7.2 us on three threads                 | will differ                                                                                                                          |

The twelve runs are step 1's commands run twelve times each in one session on
this rig, with the governor pinned, the reference capture first and this
page's runs last. The ranges describe those runs; they are not a bound a run
has to meet, and another run can land outside them. The laptop's figures come
from three runs on one of its cores, with other work running.

## If It Does Not Match

Ask the binary first: `--profile-check` with the same request makes the
decision the run makes before its first test, and runs no test.

- **No privileges.** Without `BENCH_SUDO=1`, and not as root, bpftrace refuses
  to run, and the check says so:

  ```bash
  ./build/bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace --bpf write_latency --profile-check
  ```

  ```
  ...
    Selected request: --profile bpftrace --bpf write_latency
    [FAIL] bpftrace   denied: script 'write_latency' could not attach as the current user: ERROR: bpftrace currently only supports running as the root user.
               Set BENCH_SUDO=1 with a scoped sudoers grant for /usr/bin/bpftrace and /usr/bin/kill, or run as root.
  ...
  ```

  A run without the opt-in prints the same reason once, when its first test
  asks for the profiler, then measures untraced and writes no capture folder:

  ```
  [FAIL] Profiler 'bpftrace': denied: script 'write_latency' could not attach as the current user: ERROR: bpftrace currently only supports running as the root user.
     Set BENCH_SUDO=1 with a scoped sudoers grant for /usr/bin/bpftrace and /usr/bin/kill, or run as root.
     Falling back to no-op (measurements will proceed without profiling).
  ```

  bpftrace runs only as root, whatever the user's capabilities: set
  `BENCH_SUDO=1` with a sudoers grant for `bpftrace` and `kill`, or run the
  benchmark as root.

- **A script name it cannot find.** A name without a `/` is looked up in the
  scripts directory, and the check names the file it looked for, here for
  `--bpf write_latncy`, with the checkout's path cut to `...`:

  ```
    [FAIL] bpftrace   missing: bpftrace script 'write_latncy' not found at .../src/bench/bpf/write_latncy.bt
               --bpf takes a script name, looked up as <name>.bt in .../src/bench/bpf (set by --bpf-scripts DIR or PERF_BPF_SCRIPTS), or a path to a script file, with or without .bt.
  ```

  The directory is the source tree's `src/bench/bpf/`, wherever you run from;
  `--bpf-scripts DIR` or `PERF_BPF_SCRIPTS` points it elsewhere. A library
  installed after its source tree was removed has no bundled scripts to find:
  point `--bpf-scripts` at a copy.

- **The kernel lacks a probe the script uses.** Some vendor kernels leave out
  the `syscalls` tracepoints that `write_latency.bt` and `fsync_latency.bt`
  use. bpftrace refuses a script that names a tracepoint the kernel does not
  have, and the check quotes it; here for a script naming a system call that
  does not exist, `--bpf ./no_such_probe.bt`, with the temporary directory cut
  to `...`:

  ```
    [FAIL] bpftrace   unsupported: script './no_such_probe.bt': .../probe0.bt:1-3: ERROR: tracepoint not found: syscalls:sys_enter_no_such_call
               The kernel lacks a probe the script uses (bpftrace's message names it), or tracefs is not mounted: mount -t tracefs tracefs /sys/kernel/tracing.
  ```

  `wakeup_latency.bt` and `cpu_migrations.bt` use only `sched` tracepoints.

- **A `[bpftrace]` line says the script ended by itself.** A script that runs
  `exit()` before the measured repeats end covers only part of them. Here one
  whose `interval:s:2` probe ends it two seconds after it starts, on a run
  long enough to outlast it (`--cycles 3000 --repeats 2`):

  ```
  [bpftrace] script './ends_itself.bt' ended by itself before the measured repeats finished; its output covers only the part before it ended
  ```

  Its report holds the arm line and no stop line. End a script only on the
  benchmark's exit, as the bundled ones do; the backend stops it when the
  measured repeats end. A script that ends itself before the check stops its
  copy, once the copy has attached and a second after its start at the
  earliest, is refused before the run instead, as one that ends itself too soon
  to be checked.

- **A `[bpftrace]` line says the script printed no data of its own.** The run
  without `--bpf` traces with `write_latency` and `fsync_latency`, and the
  writes make no `fsync()`:

  ```
  [bpftrace] script 'fsync_latency' captured the measured repeats (its tracer acknowledged their start and their end for pid 128729 and flushed its output), but printed no data of its own: its output holds only the capture window's lines
  ```

  The capture is complete, both of the tracer's lines are in the report, and
  the report holds nothing else: no event reached the script's maps, which is
  not a measured zero. Check the script's filters, and that the test makes the
  calls the script traces.

- **The arm line never comes.** In a PID namespace of its own, such as a
  container started without `--pid=host`, bpftrace does not see the
  benchmark's threads under the ids the benchmark has there. Step 2's test,
  run as root inside a namespace of its own
  (`sudo unshare --pid --fork --mount-proc`), prints:

  ```
  [bpftrace] unsupported: script 'write_latency' did not acknowledge its arm probe within 5000 ms: this process runs in PID namespace pid:[4026532587], not the host's, where bpftrace does not see its threads under the ids it knows
  [bpftrace] Run the benchmark on the host, or in a container started with --pid=host.
  ```

  Run the benchmark on the host, or in a container started with
  `--pid=host`; [BPF_SCRIPTS.md](../../docs/BPF_SCRIPTS.md#pid-namespaces) has
  what each bpftrace version sees there.

- **The histogram counts more writes than the test made.** The tracer counts
  every `write()` the process makes between its start and its stop: the
  test's result line on a terminal (step 3), and the harness's progress line
  on a run whose measured repeats last more than two seconds. Keep traced runs
  short, as step 2's `--cycles` and `--repeats` do.

## Check Against the Reference

A capture of step 1's write pair from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/09_bpftrace_profiler.csv writes.csv
```

Output for step 1's `writes.csv`, printed by the `bench` CLI built from the
tree this page ships in (`bench 1.0.3`); the CLI prints the header in bold and
the labels in colour, which is not reproduced here:

```

Test                               Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
-----------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
BpftraceProfiler.WriteBatched       0.53389       0.56864    +0.03475     +6.5%      0.3%      0.2%  REGRESSION
BpftraceProfiler.WritePerLine     532.28000     566.98900   +34.70900     +6.5%      0.1%      0.2%  REGRESSION

  2 regression(s)  0 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference is step 1's command run first in the same session on this
board, by the same binary: its rows' timestamps are eleven seconds before
`writes.csv`'s. Nothing changed between the two runs, yet both of this page's
medians came out 6.5% slower, past the 5% threshold, so both rows read
`REGRESSION`. That is this pair of runs, not a property of the code: of the
session's twelve runs, this page's was the slowest in both tests, while nine
of the other eleven read within 0.5% of the reference's per-line median. As the note under the table says, a label compares two runs'
medians; it is not a significance test, and `Base CV` and `Cand CV` are each
run's own spread, which says nothing about the spread between runs. Over the
twelve runs, the per-line median ranged from 531.9 to 567.0 us, 6.6%, with
nothing changed. Read the ratio between the two tests, which held at 996x
here and 991x to 1,009x over the twelve, not the label. The CSVs' `hostname`
column records the name visible to the process that captured each: both were
captured in the rig's ordinary session, not in a UTS namespace named `pi4`, so
both record the board's host name, `raspberrypi`. The CSV holds times only;
what bpftrace counted is checked by `Bpftrace.CountsEveryWrite`, below.

## What Keeps This Page True

- `TestDemoBpftrace`, a test program of its own beside the demo
  ([`09_BpftraceProfiler_uTest.cpp`](../cpu/utst/09_BpftraceProfiler_uTest.cpp)),
  holds the demo's versions to what this page says they do.
  `BpftraceWritesTest` reads the kernel's own count of the `write()` calls the
  calling thread has made (`syscw` in `/proc/thread-self/io`) around ten calls
  of each write version, and fails unless the per-line version made 10,000 and
  the batched one 10; it also writes each version into a pipe and fails unless
  both wrote the lines end to end, byte for byte. `BpftraceTotalsTest` runs
  each threaded version through `contentionRun()` with four threads, 25 calls
  each and three repeats, and fails unless the total holds every call's joined
  length. These count, so a busy machine cannot change their answers.
- `Bpftrace.CountsEveryWrite`, in the same program, runs the demo binary as
  step 2 does, under `--profile bpftrace --bpf write_latency` with
  `BENCH_SUDO=1`, on `WritePerLine`, 20 calls and 2 repeats, and reads what
  the run left. It fails unless the test passes and the backend prints no
  `[bpftrace]` line, the capture folder holds its three files, the report
  holds one arm line and one stop line, for the demo's process and the very
  threads its copy names, and the histogram counts at least the 40,000 writes
  the measured calls made and fewer than one more call's worth. The measured
  repeats start only once the tracer has armed, so a busy machine makes it
  wait longer, not fail. It skips, saying why, in a PID namespace other than
  the host's, without bpftrace, without root or sudo, and when the run says
  that bpftrace could not trace here, quoting sudo's or bpftrace's own line.
  `BpftraceReportTest` holds the report reading to reports taken from this rig
  and from an x86-64 laptop.
- Vernier's own tests hold the backend and the scripts: `BundledBpfScriptsTest`
  holds every bundled script's exit probe to the main thread
  (`tid == {{PID}}`) and `wakeup_latency.bt`'s recording probe to
  `sched_waking`, and `BpfCheckTest` and `BpfWindowTest` hold the lookup of
  names and paths, the files a run writes, the hints and the capture window to
  their rules, against a stand-in for bpftrace.

All of them are registered with `ctest`. The demo's check runs under the
`demo` label, and its traced test also under the `bpftrace` label, alone:

```bash
ctest --test-dir build -L bpftrace
ctest --test-dir build -L demo
```

On this rig, at this page's revision, `-L bpftrace` ran its 1 test and
`-L demo` its 60, and every test passed; the traced test counted 40,000 of
40,000 writes. The demo's timing tests are not registered: what they measure
belongs to the machine they run on. This repository has no
continuous-integration lane on the reference board, so before a release the
page's commands are run on the rig by hand, and the page and its reference CSV
are re-captured when what they show changes.

## See Also

- [BPF_SCRIPTS.md](../../docs/BPF_SCRIPTS.md) -- the bundled scripts, the
  capture window, script names and paths, running a script by hand
- [Walkthrough 06: thread scaling](06_THREAD_SCALING.md) -- the same threads
  and lock, timed
- [Walkthrough 16: off-CPU profiling](16_OFFCPU_PROFILER.md) -- where threads
  wait, with their stacks
- [Walkthrough 01: basic workflow](01_BASIC_WORKFLOW.md) -- the `join`
  example, measured and compared
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
