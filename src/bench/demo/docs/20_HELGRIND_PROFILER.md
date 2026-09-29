# Demo 20: Valgrind Helgrind

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads)), its result shared between threads, plus a deliberately racy addition private to the demo
**Captured:** 2026-09-28 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; valgrind 3.24.0

## Overview

Threads that share a variable need something to put their accesses to it in
order. Without that, two threads can read the same total, each add to it, and
each write its own sum back, and one of the additions is lost. A test of the
answer can pass while that is possible, and a timer cannot see it at all. This
walkthrough shares the `join` example's result between four threads twice,
once with a mutex around each addition and once without, and runs both under
helgrind. Helgrind names the line that adds without the mutex, and reports
that no lock was held, in runs whose answer came out right; with the mutex it
reports nothing. The mutex is not free: on this rig, four threads that take
turns on it complete fewer calls a second than one thread does alone, and
step 1 measures by how much.

## What is helgrind?

Helgrind is one of Valgrind's tools, its thread-error detector. Valgrind runs
the program on a synthetic CPU, one thread at a time, and helgrind watches
every load and store and every call the program makes into the POSIX threads
library: creating and joining threads, locking and unlocking mutexes, waiting
on condition variables, semaphores and barriers. From those calls it works out
which accesses are ordered, one before the other, and it reports two threads
that touch the same memory, at least one of them writing, with nothing
ordering the two: a data race. It reports the race whether or not the timing
of the run made it do harm. It also records the order in which each thread
takes locks, and reports two locks that different threads take in opposite
orders (`--track-lockorders=yes`, the default), a deadlock waiting for the
wrong timing; and it reports misuse of the threads library, such as unlocking
a mutex that another thread holds. This page's example takes one lock, so it
has no order to get wrong: the race is what it shows.

- **Best for:** data races, locks taken in inconsistent orders, misuse of the
  POSIX threads API.
- **Overhead:** large, and it changes from run to run. Under helgrind,
  `Helgrind.LockedTotal`'s result line read from 51.6 to 268.9 ms per call
  over the eight runs of it for this page, step 4's among them, against
  22.1 us in step 1: thousands of times as long, and five times apart between
  runs of the same command. Valgrind runs one thread at a time, and its manual
  warns that runs of the same program can get very different thread
  scheduling. Run one test at a time under it, with few cycles, and never read
  its timings as measurements.
- **Not for:** time (perf, gperftools and callgrind, walkthroughs
  [02](02_PERF_PROFILER.md), [03](03_GPERF_PROFILER.md) and
  [07](07_CALLGRIND_PROFILER.md)), where threads block and for how long
  ([walkthrough 16](16_OFFCPU_PROFILER.md)), or memory errors (memcheck,
  [walkthrough 15](15_MEMCHECK_PROFILER.md)).
- **What it cannot see:** an ordering built by spinning on atomic variables
  instead of the threads library. Helgrind knows the POSIX primitives, and it
  reports accesses that only such an ordering protects as races. Vernier's own
  start gate, which releases a contention test's threads together, is built
  that way and marks its flags for helgrind ([Step 4](#step-4-confirm-the-fix)).

**In Vernier:** `--profile helgrind` selects the helgrind backend, which does
not start valgrind itself. `bench run <binary> --profile helgrind` builds the
wrap and prints it on its first line:
`valgrind --tool=helgrind --log-file=<dir>/<binary>.helgrind/helgrind.log <binary> --profile helgrind ...`,
where `<dir>` is `--profile-output-dir` (`bench-out` when you give none).
Helgrind writes one log for the whole process, so profile one test per run
with `--gtest_filter`; no per-test folder is created. Run by hand instead, with
`--profile helgrind` on the binary's own command line and no valgrind, a test
that measures (`Helgrind.LockedTotal` here) runs as usual, creates the folder
`Helgrind.LockedTotal.helgrind/` in the working directory, which stays empty,
and prints a `valgrind --tool=helgrind` command that writes the log there. The
wrap passes valgrind no `--error-exitcode`, so valgrind exits with the
program's own status, and a `bench run` whose log reports races still exits 0:
the log carries the finding. To fail a job on one, run valgrind yourself
([below](#failing-a-job-on-a-race)).

**Needs:** valgrind, from the rig document's
[package list](../../docs/rigs/RIG_PI4.md#2-one-time-setup); no privileges.
[`bench doctor`](../../docs/rigs/RIG_PI4.md#6-verify-your-rig) reports it as:

```
  [OK]   helgrind   valgrind available (helgrind + drd thread-error detectors ship with it)
```

To keep Vernier's own start gate out of the report, build the benchmark where
valgrind's header `helgrind.h` is installed, as this rig's build was
([If It Does Not Match](#if-it-does-not-match)).

## The Example

The threads share one result of the `join` example from
[walkthrough 01](01_BASIC_WORKFLOW.md): `joinV1`, the version that measures the
answer first, allocates once and appends in place, joins the same 1,000 words
on every call, and each call adds the joined string's length to one
total that all the threads update. The demo,
[`14_HelgrindProfiler_Demo.cpp`](../cpu/14_HelgrindProfiler_Demo.cpp), does it
correctly in `Helgrind.LockedTotal`:

```cpp
PERF_CONTENTION(Helgrind, LockedTotal) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  std::mutex totalMutex;
  std::size_t total = 0;
  std::size_t calls = 0;
  const auto addJoinedLength = [&] {
    std::lock_guard<std::mutex> lock(totalMutex);
    total += demo::joinV1(PARTS, SEPARATOR).size();
    ++calls;
  };

  perf.warmup(addJoinedLength);
  perf.contentionRun(addJoinedLength, "locked_total");
  EXPECT_EQ(total, calls * demo::joinedSize(PARTS));
}
```

`contentionRun` starts as many threads as `--threads` asks for, releases them
together, has each make the same call `--cycles` times per repeat, and times
the repeat; the test writes one CSV row, as any measured test does. Every call
takes the mutex before it touches the total or the count of calls, so the two
always agree, and the test checks that they do: the total is the number of
calls times the joined length, which the example's `joinedSize()` computes
without joining.

The racy version is the same addition without the lock, in a source of its
own, [`14_HelgrindProfiler_Racy.cpp`](../cpu/14_HelgrindProfiler_Racy.cpp):

```cpp
void addJoinedLength(std::size_t& total, const std::vector<std::string>& parts, char sep) {
  // Read the total, add, write it back, with no lock: two threads can read
  // the same total and each write back its own sum, and one addition is lost
  total += joinV1(parts, sep).size();
}
```

`Helgrind.RacyTotal` runs it on four threads of its own, each adding once:

```cpp
PERF_TEST(Helgrind, RacyTotal) {
  if (!vernier::bench::profiler_env::isRunningUnderValgrind()) {
    GTEST_SKIP() << RACY_SKIP_REASON;
  }

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  std::size_t total = 0;
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < RACY_THREADS; ++t) {
    threads.emplace_back([&] { racy::addJoinedLength(total, PARTS, SEPARATOR); });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(total, RACY_THREADS * demo::joinedSize(PARTS));
}
```

A data race is undefined behaviour in C++, so a case that makes one on
purpose must not run in an ordinary test run. Unless the process runs under
valgrind, `Helgrind.RacyTotal` skips itself and says how to run it; it asks
`isRunningUnderValgrind()`, from [`ProfilerEnv.hpp`](../../inc/ProfilerEnv.hpp),
whether valgrind is running it. It measures nothing: under valgrind a time
means nothing, and a fixed number of additions is what its total is checked
against. That check passed in all seven runs of the case under helgrind for
this page, and the check program below requires it. Valgrind runs one thread
at a time, and in these runs no addition was lost: the total came out right,
the race was there all the same, and that is why a test of the answer cannot
be relied on to find it.

The racy addition is not a version of the shared example: the example's
versions are held to the same answers by its unit tests and are safe to run
anywhere, and this one is not. It lives with the demo, compiled with debug
information whatever the build type, so that helgrind's report can name its
line, and it is `[[gnu::noinline]]`, so that it keeps a frame of its own in
the report. That is all the demo holds. What this page shows is checked apart
from it, by a test program that runs the demo binary under helgrind
([What Keeps This Page True](#what-keeps-this-page-true)).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
./build/bin/ptests/BenchDemo_14_HelgrindProfiler --threads 4 \
  --target-time 50ms --repeats 10 --csv run.csv
```

Run it as the rig document's
[measurement section](../../docs/rigs/RIG_PI4.md#4-running-a-measurement) says,
governor and all, but without its `taskset`: the four threads need the board's
four cores. Captured output, with the checkout's path in the skip line
shortened to `...`:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from Helgrind
[ RUN      ] Helgrind.LockedTotal
[target-time] 50.000 ms -> cycles=2503 (calibrated 19.9688 us/call, batch of 64)
[Helgrind.LockedTotal]  22.110 us/call  CV=1.4%  ~45.2K calls/s  (p10=21.672 p90=22.336 sd=0.308)
[       OK ] Helgrind.LockedTotal (2213 ms)
[ RUN      ] Helgrind.RacyTotal
.../src/bench/demo/cpu/14_HelgrindProfiler_Demo.cpp:96: Skipped
this case makes a data race for helgrind to find, so it runs only under valgrind: run this binary under `valgrind --tool=helgrind`, or through `bench run --profile helgrind`

[  SKIPPED ] Helgrind.RacyTotal (0 ms)
[----------] 2 tests from Helgrind (2213 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (2214 ms total)
[  PASSED  ] 1 test.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] Helgrind.RacyTotal
```

`Helgrind.LockedTotal` completed a call every 22.1 us on four threads, with a
CV of 1.4%. `contentionRun` times a repeat from starting the threads to
joining them and divides by the calls all four made, so this is how often a
call finished, whichever thread made it. The `[target-time]` line is the
harness sizing the repeats from calls made alone before the threads start:
20.0 us each, so 2,503 calls per thread for 50 ms. Four threads make four
times as many calls a repeat, so the test took 2.2 s, where the one-thread run
below took 0.5 s. The same command on one thread:

```bash
./build/bin/ptests/BenchDemo_14_HelgrindProfiler --threads 1 \
  --target-time 50ms --repeats 10 --gtest_filter='Helgrind.LockedTotal'
```

```
Note: Google Test filter = Helgrind.LockedTotal
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Helgrind
[ RUN      ] Helgrind.LockedTotal
[target-time] 50.000 ms -> cycles=2406 (calibrated 20.7812 us/call, batch of 64)
[Helgrind.LockedTotal]  18.914 us/call  CV=2.1%  ~52.9K calls/s  (p10=18.616 p90=19.183 sd=0.396)
[       OK ] Helgrind.LockedTotal (463 ms)
[----------] 1 test from Helgrind (463 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (463 ms total)
[  PASSED  ] 1 test.
```

One thread alone completed a call every 18.9 us, CV 2.1%: 52.9K calls a
second against the four threads' 45.2K. The mutex is held across the whole
call, the join included, so only one join runs at any moment however many
threads there are, and running the calls on four threads cost 3.2 us more per
call than running them on one. That is what this fix costs here: it makes the
answer right, and it gives up everything the threads were meant to gain. A
lock held for less of the call lets more of the work overlap; a timing is how
to see what that wins, and helgrind is how to see that it is still right.
`Helgrind.RacyTotal` reports `SKIPPED` with its message, because the run is
not under valgrind, and it is not timed anywhere: under valgrind a time says
nothing about the code. The CSV holds the one timing row.

Over the session's twelve runs of step 1's command on this rig, the median
ranged from 21.7 to 22.5 us, and over six runs of the one-thread command from
18.9 to 19.2 us: every four-thread run was slower than every one-thread run.
Those ranges describe those runs; they are not a bound a run has to meet, and
another run can land outside them.

## Step 2: Run the Racy Case Under Helgrind

```bash
bench run ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --profile helgrind \
  --profile-output-dir helgrind-racy -- --gtest_filter='Helgrind.RacyTotal'
```

Captured output:

```
Running: valgrind --tool=helgrind --log-file=helgrind-racy/BenchDemo_14_HelgrindProfiler.helgrind/helgrind.log ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --profile helgrind --profile-output-dir helgrind-racy --gtest_filter=Helgrind.RacyTotal
Note: Google Test filter = Helgrind.RacyTotal
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Helgrind
[ RUN      ] Helgrind.RacyTotal
[       OK ] Helgrind.RacyTotal (117 ms)
[----------] 1 test from Helgrind (127 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (180 ms total)
[  PASSED  ] 1 test.
```

The `Running:` line is the wrap `bench run` built. The case ran, because
under valgrind it does not skip, and it passed: helgrind reports and does not
stop the program, and the total came out right. `bench run` exited 0. The log
is the one file in a folder named for the binary and the tool:

```bash
ls helgrind-racy/BenchDemo_14_HelgrindProfiler.helgrind
```

```
helgrind.log
```

## Step 3: Read the Report

```bash
cat helgrind-racy/BenchDemo_14_HelgrindProfiler.helgrind/helgrind.log
```

Captured output. GoogleTest's frames are cut from the stacks that created the
threads and marked `...`, the vector's template arguments are shortened to
`std::vector<...>`, and the binary's directory to `...`:

```
==178549== Helgrind, a thread error detector
==178549== Copyright (C) 2007-2024, and GNU GPL'd, by OpenWorks LLP et al.
==178549== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==178549== Command: ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --profile helgrind --profile-output-dir helgrind-racy --gtest_filter=Helgrind.RacyTotal
==178549== Parent PID: 178548
==178549==
==178549== ---Thread-Announcement------------------------------------------
==178549==
==178549== Thread #3 was created
==178549==    at 0x4E0DC03: clone (clone.S:65)
==178549==    by 0x4DA5A87: create_thread (pthread_create.c:298)
==178549==    by 0x4DA6587: pthread_create@@GLIBC_2.34 (pthread_create.c:841)
==178549==    by 0x4893743: pthread_create_WRK (hg_intercepts.c:445)
==178549==    by 0x4A9B667: std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)()) (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
==178549==    by 0x10F07F: Helgrind_RacyTotal_Test::TestBody() (in .../BenchDemo_14_HelgrindProfiler)
...
==178549==
==178549== ---Thread-Announcement------------------------------------------
==178549==
==178549== Thread #2 was created
==178549==    at 0x4E0DC03: clone (clone.S:65)
==178549==    by 0x4DA5A87: create_thread (pthread_create.c:298)
==178549==    by 0x4DA6587: pthread_create@@GLIBC_2.34 (pthread_create.c:841)
==178549==    by 0x4893743: pthread_create_WRK (hg_intercepts.c:445)
==178549==    by 0x4A9B667: std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)()) (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
==178549==    by 0x10F07F: Helgrind_RacyTotal_Test::TestBody() (in .../BenchDemo_14_HelgrindProfiler)
...
==178549==
==178549== ----------------------------------------------------------------
==178549==
==178549== Possible data race during read of size 8 at 0x1FFF000118 by thread #3
==178549== Locks held: none
==178549==    at 0x11BC08: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, std::vector<...> const&, char) (14_HelgrindProfiler_Racy.cpp:21)
==178549==    by 0x4A9B4DF: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
==178549==    by 0x489391F: mythread_wrapper (hg_intercepts.c:406)
==178549==    by 0x4DA5F37: start_thread (pthread_create.c:448)
==178549==    by 0x4E0DC2B: thread_start (clone.S:82)
==178549==
==178549== This conflicts with a previous write of size 8 by thread #2
==178549== Locks held: none
==178549==    at 0x11BC10: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, std::vector<...> const&, char) (14_HelgrindProfiler_Racy.cpp:21)
==178549==    by 0x4A9B4DF: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
==178549==    by 0x489391F: mythread_wrapper (hg_intercepts.c:406)
==178549==    by 0x4DA5F37: start_thread (pthread_create.c:448)
==178549==    by 0x4E0DC2B: thread_start (clone.S:82)
==178549==  Address 0x1fff000118 is on thread #1's stack
==178549==  in frame #5, created by Helgrind_RacyTotal_Test::TestBody() (???:)
==178549==
==178549== ----------------------------------------------------------------
==178549==
==178549== Possible data race during write of size 8 at 0x1FFF000118 by thread #3
==178549== Locks held: none
==178549==    at 0x11BC10: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, std::vector<...> const&, char) (14_HelgrindProfiler_Racy.cpp:21)
==178549==    by 0x4A9B4DF: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
==178549==    by 0x489391F: mythread_wrapper (hg_intercepts.c:406)
==178549==    by 0x4DA5F37: start_thread (pthread_create.c:448)
==178549==    by 0x4E0DC2B: thread_start (clone.S:82)
==178549==
==178549== This conflicts with a previous write of size 8 by thread #2
==178549== Locks held: none
==178549==    at 0x11BC10: vernier::bench::demo::helgrind_demo::addJoinedLength(unsigned long&, std::vector<...> const&, char) (14_HelgrindProfiler_Racy.cpp:21)
==178549==    by 0x4A9B4DF: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
==178549==    by 0x489391F: mythread_wrapper (hg_intercepts.c:406)
==178549==    by 0x4DA5F37: start_thread (pthread_create.c:448)
==178549==    by 0x4E0DC2B: thread_start (clone.S:82)
==178549==  Address 0x1fff000118 is on thread #1's stack
==178549==  in frame #5, created by Helgrind_RacyTotal_Test::TestBody() (???:)
==178549==
==178549==
==178549== Use --history-level=approx or =none to gain increased speed, at
==178549== the cost of reduced accuracy of conflicting-access information
==178549== For lists of detected and suppressed errors, rerun with: -s
==178549== ERROR SUMMARY: 6 errors from 2 contexts (suppressed: 6 from 2)
```

Every line begins with the process id. Helgrind first introduces the threads
its reports name, then reports two races, each with the access that raced and
the earlier access it conflicts with:

| Line                                                                                            | What it says                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Thread #3 was created`, under `---Thread-Announcement---`                                      | helgrind introduces each thread the first time a report names it, with the stack that created it: a `std::thread` started in `Helgrind_RacyTotal_Test::TestBody()`. Thread #1 is the main thread |
| `Possible data race during read of size 8 at 0x1FFF000118 by thread #3`                         | an eight-byte read, of the `std::size_t` total, that helgrind found nothing ordering against another thread's access                                                                             |
| `Locks held: none`                                                                              | the locks thread #3 held when it read: none                                                                                                                                                      |
| `at ... addJoinedLength(...) (14_HelgrindProfiler_Racy.cpp:21)`                                 | the read, and its source line: `total += joinV1(parts, sep).size();`                                                                                                                             |
| `by 0x4A9B4DF: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)` and the frames below it | the thread's start: a function of the C++ library that valgrind has no name for, helgrind's wrapper, and the C library's thread start                                                            |
| `This conflicts with a previous write of size 8 by thread #2`                                   | the other access: thread #2's write of the same eight bytes, at the same line, with no lock held either                                                                                          |
| `Address 0x1fff000118 is on thread #1's stack`                                                  | where the total is: a local variable of the test body, on the main thread's stack                                                                                                                |
| `Possible data race during write of size 8`                                                     | the second race: thread #3 writing its sum back, against thread #2's write                                                                                                                       |
| `ERROR SUMMARY: 6 errors from 2 contexts (suppressed: 6 from 2)`                                | two places in the code, the read and the write of line 21, and three races at each; the six suppressed are valgrind's own, not this program's                                                    |

The two accesses to act on are both at line 21, on two threads, each with
`Locks held: none`. Helgrind knows of nothing, no lock and no other ordering,
that puts thread #2's write before thread #3's read, so another run could
interleave them and lose an addition. The fix is one lock, held at both
accesses in every thread, which is what `Helgrind.LockedTotal` does. Run with
`-s` (`--show-error-list=yes`), as the log's last lines suggest, and valgrind
lists each place with its count, `3 errors in context 1 of 2` and
`3 errors in context 2 of 2`, and names the suppressions it used: in every run
for this page that printed the list, of either case, one of valgrind's own
suppressions for the C library, `helgrind-glibc2X-005`, and no other.

What not to conclude:

- **That the total came out wrong.** It came out right in every run for this
  page: under valgrind one thread runs at a time, and no addition was lost.
  Helgrind reports accesses that nothing orders, whether or not a run lost an
  addition to them.
- **That the addresses mean anything.** `0x11BC08`, `0x1FFF000118` and the
  rest belong to this build and this run: runs of the same binary with other
  arguments put the total at `0x1FFF000298` and `0x1FFF000218`, and the frame
  number the report gives for it changed between runs of one command,
  `frame #5` in three and `frame #3` in two.
- **That the line numbers travel.** They are this revision's
  `14_HelgrindProfiler_Racy.cpp`; a build of an edited file names other lines.
  The names travel: `addJoinedLength`, `Locks held: none`, `read of size 8`,
  `write of size 8`.
- **That `???` means something is broken.** In `(???:)` after `TestBody()`,
  valgrind has no line for the demo's own source, which a Release build
  compiles without line tables; only the racy source is compiled with `-g`.

## Step 4: Confirm the Fix

```bash
bench run ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --profile helgrind \
  --cycles 2 --repeats 1 --profile-output-dir helgrind-locked \
  -- --gtest_filter='Helgrind.LockedTotal' --threads 4
cat helgrind-locked/BenchDemo_14_HelgrindProfiler.helgrind/helgrind.log
```

Captured output of the run:

```
Running: valgrind --tool=helgrind --log-file=helgrind-locked/BenchDemo_14_HelgrindProfiler.helgrind/helgrind.log ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --cycles 2 --repeats 1 --profile helgrind --profile-output-dir helgrind-locked --gtest_filter=Helgrind.LockedTotal --threads 4
Note: Google Test filter = Helgrind.LockedTotal
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Helgrind
[ RUN      ] Helgrind.LockedTotal
[Helgrind.LockedTotal]  112309.500 us/call  CV=0.0%  ~9 calls/s  (p10=112309.500 p90=112309.500 sd=0.000)
[       OK ] Helgrind.LockedTotal (1103 ms)
[----------] 1 test from Helgrind (1111 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1163 ms total)
[  PASSED  ] 1 test.
```

and of the log:

```
==178557== Helgrind, a thread error detector
==178557== Copyright (C) 2007-2024, and GNU GPL'd, by OpenWorks LLP et al.
==178557== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==178557== Command: ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --cycles 2 --repeats 1 --profile helgrind --profile-output-dir helgrind-locked --gtest_filter=Helgrind.LockedTotal --threads 4
==178557== Parent PID: 178556
==178557==
==178557==
==178557== Use --history-level=approx or =none to gain increased speed, at
==178557== the cost of reduced accuracy of conflicting-access information
==178557== For lists of detected and suppressed errors, rerun with: -s
==178557== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 34 from 8)
```

`ERROR SUMMARY: 0 errors from 0 contexts`: four threads made two calls each
through the mutex, and helgrind found every access to the total ordered by the
lock. `--cycles 2 --repeats 1` keeps the run short, and two calls per thread
are enough for the threads to share the total. The result line, 112.3 ms per
call, is what helgrind costs, not how fast the locked version is.

Helgrind did not report Vernier's own start gate either, which
`contentionRun` uses to release the threads together. The gate spins on two
atomic flags, which helgrind cannot see as synchronisation; where valgrind's
`helgrind.h` is installed when the benchmark is built, as it was here, the
gate marks those two flags for helgrind while it exists, and helgrind leaves
them out. The rest of the program is checked as before: a race between the
workers of a contention test is still reported
([What Keeps This Page True](#what-keeps-this-page-true)). Built without that
header, or with `NVALGRIND`, the same run reports five races in
`contentionRun` ([If It Does Not Match](#if-it-does-not-match)).

## Failing a Job on a Race

`bench run` wraps with no `--error-exitcode`, so it exits 0 whatever helgrind
found. A job that should stop on a race runs valgrind itself, with an exit
code for errors:

```bash
valgrind --tool=helgrind --error-exitcode=1 --log-file=racy.log \
  ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --gtest_filter='Helgrind.RacyTotal'
echo "exit $?"
```

```
Note: Google Test filter = Helgrind.RacyTotal
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Helgrind
[ RUN      ] Helgrind.RacyTotal
[       OK ] Helgrind.RacyTotal (121 ms)
[----------] 1 test from Helgrind (131 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (184 ms total)
[  PASSED  ] 1 test.
exit 1
```

The test passed and valgrind exited 1: the race is in `racy.log`, and the job
fails. The same on the locked version:

```bash
valgrind --tool=helgrind --error-exitcode=1 --log-file=locked.log \
  ./build/bin/ptests/BenchDemo_14_HelgrindProfiler --threads 4 --cycles 2 --repeats 1 \
  --gtest_filter='Helgrind.LockedTotal'
echo "exit $?"
```

```
Note: Google Test filter = Helgrind.LockedTotal
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Helgrind
[ RUN      ] Helgrind.LockedTotal
[Helgrind.LockedTotal]  148877.875 us/call  CV=0.0%  ~7 calls/s  (p10=148877.875 p90=148877.875 sd=0.000)
[       OK ] Helgrind.LockedTotal (1353 ms)
[----------] 1 test from Helgrind (1361 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1414 ms total)
[  PASSED  ] 1 test.
exit 0
```

Without `--error-exitcode`, valgrind exits with the program's own status,
which is 0 in both cases here.

## What Should Reproduce

| Reading                           | On this rig                                                                                                                                                      | Elsewhere                                                                                                                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the race                          | a read and a write of size 8 at line 21 of the racy file, in `addJoinedLength`, each against an earlier write at the same line, `Locks held: none` on both sides | should match: it depends on the code, not the machine; the same on an x86-64 laptop with GCC 11.4 and valgrind 3.18.1, and in this project's container with clang 21, whose report names the file with its directory |
| errors and contexts               | 6 errors from 2 contexts, in all seven runs of the racy case                                                                                                     | 6 from 2 on the laptop and in the container                                                                                                                                                                          |
| suppressed reports                | 6 from 2, valgrind's `helgrind-glibc2X-005`                                                                                                                      | depend on valgrind and the C library: 0 from 0 on the laptop                                                                                                                                                         |
| the racy case's answer            | right in all seven runs                                                                                                                                          | right on the laptop and in the container too; the case never runs without valgrind                                                                                                                                   |
| the locked version under helgrind | 0 errors from 0 contexts, in all eight runs of it                                                                                                                | 0 errors on the laptop and in the container too; five races in `contentionRun` where the start gate is not marked                                                                                                    |
| four threads against one (step 1) | four threads slower per call, 22.1 against 18.9 us, in step 1 and in every run of the session                                                                    | no faster than one thread wherever the lock covers the whole call, since only one call runs at a time; by how much depends on the machine                                                                            |
| helgrind's cost                   | the locked version 51.6 to 268.9 ms per call over eight runs, against 22.1 us                                                                                    | depends on the machine, and changes from run to run                                                                                                                                                                  |
| absolute times                    | 22.1 and 18.9 us per call in step 1; 21.7 to 22.5 over twelve runs and 18.9 to 19.2 over six                                                                     | will differ                                                                                                                                                                                                          |

The twelve runs are step 1's command run twelve times in one session on this
rig, the reference capture and this page's run among them; the six are the
one-thread command in the same session, this page's run among them. Those
ranges describe those runs; they are not a bound a run has to meet, and
another run of the same command can land outside them. The helgrind readings
are the ones to carry elsewhere. The laptop's come from a GCC 11.4 Release
build linked by GNU ld, the container's from a clang 21 Debug build.

## If It Does Not Match

- **`Helgrind.RacyTotal` reports `SKIPPED` in step 2.** The run was not under
  valgrind. The binary run on its own prints:

  ```
  Note: Google Test filter = Helgrind.RacyTotal
  [==========] Running 1 test from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 1 test from Helgrind
  [ RUN      ] Helgrind.RacyTotal
  .../src/bench/demo/cpu/14_HelgrindProfiler_Demo.cpp:96: Skipped
  this case makes a data race for helgrind to find, so it runs only under valgrind: run this binary under `valgrind --tool=helgrind`, or through `bench run --profile helgrind`

  [  SKIPPED ] Helgrind.RacyTotal (0 ms)
  [----------] 1 test from Helgrind (0 ms total)

  [----------] Global test environment tear-down
  [==========] 1 test from 1 test suite ran. (0 ms total)
  [  PASSED  ] 0 tests.
  [  SKIPPED ] 1 test, listed below:
  [  SKIPPED ] Helgrind.RacyTotal
  ```

  Run it through `bench run --profile helgrind`, as step 2 does, or under
  `valgrind --tool=helgrind` yourself.

- **`bench run` stops before the test starts:**

  ```
  Error: tool not found: 'valgrind' is not on PATH; --profile helgrind runs the benchmark under it. Install valgrind, or run `bench doctor` to see which profilers this machine can use
  ```

  valgrind is not installed, or not on `PATH`; the doctor's `helgrind` row then
  reads `valgrind binary not found on PATH`. Install the package from the rig
  document's one-time setup.

- **The locked version reports five races in `contentionRun`.** The log ends
  `ERROR SUMMARY: 5 errors from 2 contexts`: a `write of size 1` by thread #1
  and four `read of size 1` by the workers, in
  `vernier::bench::PerfCase::contentionRun` and in the thread function it
  starts. That is Vernier's start gate, unmarked: the benchmark was built where
  valgrind's `helgrind.h` was not found, or with `-DNVALGRIND`. On this rig,
  the same source built with `-DNVALGRIND` reported exactly that for step 4's
  case. Build where valgrind's headers are installed, without `NVALGRIND`.
  `Helgrind.LockedTotalReportsNothing` skips in such a build, with
  `valgrind's helgrind.h is not available to this build, or NVALGRIND is defined, so contentionRun's start gate is not marked for helgrind and is reported`.

- **`Helgrind.FindsTheRace` reports `SKIPPED`, quoting valgrind.** valgrind
  could not read the demo binary: it gave up reading its debug information
  before the program ran, or it could not read its symbols. For a GCC 11.4
  Release build linked by mold 1.0.3 on an x86-64 laptop, valgrind 3.18.1
  printed `Can't make sense of .rodata section mapping` about the binary, and
  its report gave the racy frame as `???` while still counting 6 errors from 2
  contexts. The check asserts everything else, and the function and the line
  in any frame valgrind did name, then skips, quoting valgrind's lines. The build used mold because the project
  links with it when it finds it (`VERNIER_USE_FAST_LINKER`, on by default);
  configured with `-DVERNIER_USE_FAST_LINKER=OFF`, the same build was linked
  by GNU ld, and the same valgrind named `addJoinedLength` at line 21.

- **The helgrind checks report `SKIPPED` in a build with the address or the
  thread sanitizer.** valgrind cannot check such a build: on the laptop,
  valgrind 3.18.1 stopped on an assertion in its debug-information reader
  before the demo's address-sanitizer build ran. `Helgrind.FindsTheRace` and
  `Helgrind.LockedTotalReportsNothing` skip with
  `this binary is built with the address or the thread sanitizer, whose builds valgrind cannot check; run this test in a build without either`,
  and the harness's two helgrind tests with a reason of their own.

- **The log is helgrind's after `--profile-args drd`.** `bench run` wraps
  with `--tool=helgrind` whatever `--profile-args` holds: with
  `--profile-args drd`, its `Running:` line and the log were still helgrind's,
  `Helgrind, a thread error detector`. The backend reads `drd` only for the
  command it prints when run without valgrind. This page uses helgrind only.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/20_helgrind_profiler.csv run.csv
```

Output for step 1's `run.csv`, printed by the `bench` CLI built from the tree
this page ships in (`bench 1.0.3`):

```

Test                      Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
--------------------  ------------  ------------  ----------  --------  --------  --------  ------------
Helgrind.LockedTotal      22.09600      22.11040    +0.01440     +0.1%      2.8%      1.4%  neutral

  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary on this board immediately
before step 1's run (the two CSVs' timestamps are three seconds apart), and
nothing changed between them: the median came out +0.1% against it, inside
the 5% threshold the labels are drawn at, so the row says `neutral`. That is
this pair of runs, not a property of the code. The CSVs' `hostname` column
records the name visible to the process that captured each: this reference was
captured in a UTS namespace named `pi4` on the same rig, so it records `pi4`,
while the rig's ordinary captures, step 1's `run.csv` among them, record
`raspberrypi`. As the note under the table says, a label compares two runs'
medians; it is not a significance test, and `Base CV` and `Cand CV` are each
run's own spread, which says nothing about the spread between runs: 2.8% and
1.4% here. Over the session's twelve runs of step 1, the median ranged from
21.7 to 22.5 us, 3.9%, with nothing changed: inside the threshold, but that
range describes those runs, and another run can land outside it. The
four-thread median is the noisiest measurement on this page, against 1.5% for
the one-thread median over its six runs. Read the median and the CV, and compare the four-thread time with one
thread's. The CSV holds times only; what helgrind found is checked by
`Helgrind.FindsTheRace`, below.

## What Keeps This Page True

Three things check what this page shows, and all fail loudly:

- `Helgrind.FindsTheRace`, in a test program of its own beside the demo
  ([`14_HelgrindProfiler_uTest.cpp`](../cpu/utst/14_HelgrindProfiler_uTest.cpp),
  built as `TestDemoHelgrind`), runs the demo binary under
  `valgrind --tool=helgrind`, as `bench run --profile helgrind` does, with
  valgrind's error list and an exit code for errors, on `Helgrind.RacyTotal`,
  and reads the log. It fails unless the case runs to its end with the right
  total, helgrind reports at least one race, and every race it reports is a
  read or a write of the total's eight bytes, with `Locks held: none`, in
  `addJoinedLength` at the line of the racy statement, against an earlier
  access with no lock held at the same line, and unless valgrind exits with
  the code it was given. The check finds the statement's line in the racy
  source, so an edit that moves the statement moves what the check looks for;
  the file may carry the directory its debug information records, as a clang
  build's does. Given a lock, the racy addition fails it ("the racy addition
  has stopped racing"), and so does looking for any other line.
  `Helgrind.LockedTotalReportsNothing` runs `Helgrind.LockedTotal` the same
  way, on four threads, two calls each, and fails unless the case runs to its
  end, helgrind counts no error and valgrind exits 0: with the start gate's
  marking taken out of the harness it fails on the gate's five reports.
  `Helgrind.RacyTotalSkipsOutsideValgrind` runs the racy case without valgrind
  and fails unless it reports `SKIPPED`, says how to run it, and does not run.
  The first two skip in a build with the address or the thread sanitizer
  (which valgrind cannot check), where valgrind is not installed, and where
  valgrind gives up reading the demo binary. `FindsTheRace` also skips where
  valgrind cannot read the binary's symbols, once every other check has
  passed: it reads each frame on its own, looks for the function and the line
  in every frame valgrind named, and excuses only a frame left unnamed in that
  binary. `LockedTotalReportsNothing` also skips in a build where the gate is
  not marked. The skips that rest on valgrind quote its lines. Beside them,
  `HelgrindReportTest` holds the report reading to its cases (a lock held, a
  frame at another line or in another function, a file named with its
  directory, a frame valgrind could not name, a wrong frame beside an unnamed
  one, a log without a race).
- `StartGateHelgrindTest.GateReportsNothing` and
  `StartGateHelgrindTest.WorkerRaceIsStillReported`, the harness's own test of
  the gate
  ([`StartGateHelgrind_uTest.cpp`](../../utst/StartGateHelgrind_uTest.cpp),
  built as `TestStartGateHelgrind`), start four threads through
  `contentionRun` under helgrind: workers that add under a mutex must report no
  error, and workers that add without one must still report their race, so the
  marking covers the gate's two flags and a race between the workers is still
  found.
- The example's unit tests hold `joinV0` and `joinV1` to the same answers,
  under the `demo` label.

All of them are registered with `ctest`. The demo's check and the harness's
test carry the `helgrind` label, which runs them alone:

```bash
ctest --test-dir build -L helgrind
```

and the demo's check runs with the example's tests under the `demo` label:

```bash
ctest --test-dir build -L demo
```

Every test they select should pass, but the probes that race or make an
error on purpose, which CTest reports as skipped outside valgrind: this page's
is `StartGateProbe.RacyWorkers`, the racy probe of the harness's test, and
its skip is the probe working. The demo's timing test is not registered: what it
measures belongs to the machine it runs on. This repository has no
continuous-integration lane on the reference board, so before a release the
page's commands are run on the rig by hand, and the page and its reference CSV
are re-captured when what they show changes.

## See Also

- [Demo 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example,
  measured and compared
- [Demo 06: thread scaling](06_THREAD_SCALING.md) -- contention between
  threads
- [Demo 16: off-CPU](16_OFFCPU_PROFILER.md) -- where threads block, and for
  how long
- [Demo 15: Memcheck](15_MEMCHECK_PROFILER.md) -- memory errors, under the
  same valgrind
- [CPU guide: thread-safety with Helgrind](../../docs/CPU_GUIDE.md#thread-safety-with-helgrind--drd)
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
