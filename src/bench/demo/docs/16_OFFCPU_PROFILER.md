# Demo 16: Off-CPU Profiling

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp), called from several threads at once (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-10-03 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; bpftrace 0.23.2

## Overview

A timer says how long a group of threads needed per call. It does not say
which of them waited, where, or for how long, and neither does a CPU sampler,
which finds a thread only while it runs. This walkthrough starts three threads
that each join the `join` example's 1,000 words and add the length to one
total, first under one lock held for the whole call, then with nothing shared
while they run, and captures where the threads went to sleep and how long
each stayed off the CPU. With the lock, the capture holds 20,016 sleeps,
19,997 of them in the lock, under `addUnderCoarseLock`, and the workers that
slept were off the CPU for about 18 or about 34 ms each; with nothing shared,
no worker slept. A CPU sampler's profile of the locked version shows the
join's work and nothing of the waiting.

## What is off-CPU profiling?

A thread is off the CPU whenever it is not running: asleep until a lock is
free, until another thread ends or until I/O completes, or ready to run and
waiting for a CPU. An off-CPU profiler records where threads go to sleep and
how long they stay away. Vernier's runs bpftrace on the kernel's
`sched_switch` tracepoint, which fires every time a CPU switches from one
thread to another and says which thread left, in what state, and which one
takes its place. Each time a thread of the benchmark leaves its CPU asleep,
in state S (interruptible: waiting for a lock, a `join` or a timer) or D
(uninterruptible, as in some I/O), the script counts the stack it was at; a
thread that leaves its CPU still able to run, because it was preempted, or
that exits, is not counted. The time recorded for the thread runs from that
switch to the next one that puts it back on a CPU: its sleep, and then its
wait for a CPU after it was woken. A thread still asleep when the capture
stops is counted, with no time.

- **Best for:** threads that wait: on a lock, on each other, on I/O, on a
  timer.
- **Overhead:** the script runs on every context switch of the machine and
  records a stack only for the benchmark's own sleeps. In this session the
  timed medians under the capture overlapped those without it: 20.2 to
  20.7 us per call with the lock over eleven captures, against 20.2 to
  20.8 us over twelve runs without, and 6.6 to 7.4 us against 6.9 to 7.3 us
  with nothing shared. Its cost is at the start and the stop, outside the
  timed repeats: before its first capture a run checks the request, as the
  doctor does (below), and each capture waits for bpftrace to start and arm,
  then to disarm and stop. Step 2's test took 4.2 s, against 0.6 s for the
  same test in step 1.
- **Not for:** where CPU time goes (gperftools, perf and callgrind,
  walkthroughs [03](03_GPERF_PROFILER.md), [02](02_PERF_PROFILER.md) and
  [07](07_CALLGRIND_PROFILER.md)), or data races (helgrind,
  [walkthrough 20](20_HELGRIND_PROFILER.md)).

**In Vernier:** `--profile offcpu` starts bpftrace with Vernier's off-CPU
script before a test's measured repeats and stops it after them. The script
records nothing until a thread that Vernier starts for the purpose, named
`vernier-arm`, goes to sleep; then it prints `offcpu armed <pid> <tid>`, the
benchmark's process id and that thread's id, and records every sleep of the
benchmark's other threads. When the repeats end, the thread that ran them
goes to sleep under the name `vernier-stop`, and the script prints
`offcpu disarmed <pid> <tid> <n>`, `n` the sleeps it recorded, and records
nothing more. Vernier waits for the first line before the repeats start and
for the second after they end, checks the ids in each against its own, stops
bpftrace, which then prints its maps, and checks that the dump is whole. The
capture lands in `<Suite.Case>.offcpu/` under the working directory
(`--profile-output-dir DIR` moves the root): `offcpu.txt` holds bpftrace's
output, the two lines and the maps, and `offcpu.err.txt` its messages. Each
capture ends with one line from the backend:
`[offcpu] stacks written to <path> (<n> sleeping switch-outs)`; or
`[offcpu] no thread of this process went to sleep while the capture was armed; ...`
for a capture checked from start to stop in which nothing slept; or a line
saying what failed, with the files kept. The thread names `vernier-arm`,
`vernier-wait` (the capture's own waits, which are not recorded) and
`vernier-stop` are reserved: a test's own thread given one of them disturbs
the capture.

**Needs:** bpftrace, from the rig document's
[package list](../../docs/rigs/RIG_PI4.md#2-one-time-setup), and tracefs,
which this rig's kernel mounts at `/sys/kernel/tracing`. bpftrace 0.23.2 runs
only as root: run the benchmark as root, or set `BENCH_SUDO=1` and let your
user run, through `sudo -n` and without a password, the two commands a
capture uses: `<bpftrace> -B none -e <the off-CPU script> <pid>`, and
`<kill> -2|-15|-9 <tracer>`, which stops it (SIGINT, then SIGTERM and SIGKILL
only if bpftrace has not stopped). Only bpftrace is elevated; the test and
its files stay yours. This rig's user may run any command through `sudo -n`
(`(ALL) NOPASSWD: ALL`); a grant limited to those two commands has not been
qualified on it. The benchmark must run in the host's PID namespace
([If It Does Not Match](#if-it-does-not-match)). The doctor reports it as:

```bash
source build/.env     # puts the bench CLI on PATH
BENCH_SUDO=1 bench doctor ./build/bin/ptests/BenchDemo_13_OffCpuProfiler
```

```
  [OK]   offcpu     the off-CPU script, with a 5 s self-exit added, stayed running for the 1500 ms start grace through sudo -n (BENCH_SUDO=1) and stopped on SIGINT through sudo -n kill (probe with /usr/bin/bpftrace), and a copy with unbuffered output armed on this process's arm thread (pid 25522, thread 25594); not checked: the grant for the run's own command (/usr/bin/bpftrace -B none -e <the off-CPU script> <benchmark pid>), SIGTERM and SIGKILL through sudo, and the run's capture
```

The row says what the check did. It ran the off-CPU script twice through
`sudo -n`: once for its 1.5 s start grace, with a 5 s self-exit added, and
stopped it with SIGINT; then as a copy that must report a sleep of a thread
the check starts, under that thread's id and the benchmark's process id
(thread 25594 of process 25522 here): the check that a capture can see the
benchmark's threads. That second run has a cost. On this rig the doctor took
5.9 to 6.0 s over five runs, 2.7 to 2.8 s of it in the offcpu row, of which
the copy accounts for about 0.8 s; the two runs taken before the governor was
pinned took 6.1 s. Adding `--profile offcpu` to the benchmark's own
`--profile-check` checks offcpu once more, for the selected request, so the
script runs four times and the doctor took 8.6 s. A run checks once, before
its first capture.

## The Example

Both versions join the same 1,000 words with `joinV1`, the reserving join of
[walkthrough 01](01_BASIC_WORKFLOW.md#the-example), and add the length to a
total. They differ in what the threads share while they do it, as the two
versions of [walkthrough 06](06_THREAD_SCALING.md#the-example) do. The first
holds one lock for the whole call:

```cpp
struct SharedTotal {
  std::mutex lock;
  std::size_t value = 0;
};

[[gnu::noinline]] inline void addUnderCoarseLock(SharedTotal& total,
                                                 const std::vector<std::string>& parts) {
  std::lock_guard<std::mutex> guard(total.lock);
  total.value += joinV1(parts, SEPARATOR).size();
}
```

The join runs under the lock, so only one thread can be joining at a time,
and a thread that finds the lock taken goes to sleep in it until the lock is
free. `[[gnu::noinline]]` keeps the function a call of its own, so that a
thread asleep in the lock has `addUnderCoarseLock` in its stack between the
lock and the caller: a function the compiler folded into its caller would not
appear there.

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

Each thread adds to its own `threadTotal`, which no other thread touches, and
hands it over to `finishedThreadsTotal` once, when the thread ends.

Both are in [`13_OffCpuProfiler_Totals.hpp`](../cpu/13_OffCpuProfiler_Totals.hpp),
this demo's own copy of walkthrough 06's versions under the same names. The
demo, [`13_OffCpuProfiler_Demo.cpp`](../cpu/13_OffCpuProfiler_Demo.cpp),
measures each in a test of its own, one CSV row each:

```cpp
PERF_CONTENTION(OffCpu, CoarseLock) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  perf.warmup([&] { addUnderCoarseLock(total, PARTS); });
  perf.contentionRun([&] { addUnderCoarseLock(total, PARTS); }, "coarse_lock");
}
```

`OffCpu.NoSharing` is the same with `addToThreadTotal(PARTS)` and no
`SharedTotal`. `contentionRun()` starts `--threads` threads for each repeat,
holds them at a start line until all of them are running, lets each make the
call `--cycles` times, and joins them;
[walkthrough 06](06_THREAD_SCALING.md#what-contentionrun-measures) says what
it measures. Under `--profile offcpu` the capture is armed before the first
repeat and disarmed after the last, so it sees every repeat's threads, and
the test's own thread waiting for them to end.

## Step 1: Measure

```bash
taskset -c 0-3 ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --threads 3 \
  --cycles 1000 --repeats 10 --csv run.csv
```

Run it as the rig document's
[measurement section](../../docs/rigs/RIG_PI4.md#4-running-a-measurement)
says, governor and all, but on the board's four cores instead of one: a core
for each of the three threads and one for the test's own thread, as
[walkthrough 06](06_THREAD_SCALING.md#which-cores) explains. Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from OffCpu
[ RUN      ] OffCpu.CoarseLock
[OffCpu.CoarseLock]  20.530 us/call  CV=1.2%  ~48.7K calls/s  (p10=20.267 p90=20.634 sd=0.254)
[       OK ] OffCpu.CoarseLock (619 ms)
[ RUN      ] OffCpu.NoSharing
[OffCpu.NoSharing]  7.281 us/call  CV=2.6%  ~137.3K calls/s  (p10=6.900 p90=7.345 sd=0.188)
[       OK ] OffCpu.NoSharing (215 ms)
[----------] 2 tests from OffCpu (835 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (835 ms total)
[  PASSED  ] 2 tests.

===================================================================
Test                Median (us)     CV%     Calls/s  Status
-------------------------------------------------------------------
OffCpu.CoarseLock        20.530    1.2%       48.7K  OK
OffCpu.NoSharing          7.281    2.6%      137.3K  OK
-------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

`contentionRun()` times each repeat as a whole and divides it by every call
the threads made, so each result is the wall time the group needed per call:
not how long one call took, and not how long a thread waited
([walkthrough 06](06_THREAD_SCALING.md#what-contentionrun-measures)). With
the lock, the three threads got through a call every 20.5 us; with nothing
shared, every 7.3 us, 2.8 times as often. Each test made 30,000 calls: ten
repeats of three threads with 1,000 calls each. The lock covers the whole
join, so with it only one thread can be joining at any moment; without it,
all three join at once. That is what the timing says the lock costs. It does
not say where the threads spent the difference, or which of them did: the
capture does.

## Step 2: Capture

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --taskset 0-3 --profile offcpu -- \
  --gtest_filter='OffCpu.CoarseLock' --threads 3 --cycles 1000 --repeats 10
```

Captured output:

```
Running: taskset -c 0-3 ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --profile offcpu --gtest_filter=OffCpu.CoarseLock --threads 3 --cycles 1000 --repeats 10
Note: Google Test filter = OffCpu.CoarseLock
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from OffCpu
[ RUN      ] OffCpu.CoarseLock
[offcpu] stacks written to ./OffCpu.CoarseLock.offcpu/offcpu.txt (20016 sleeping switch-outs)
[OffCpu.CoarseLock]  20.217 us/call  CV=1.2%  ~49.5K calls/s  (p10=20.173 p90=20.595 sd=0.242)
[       OK ] OffCpu.CoarseLock (4191 ms)
[----------] 1 test from OffCpu (4191 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (4191 ms total)
[  PASSED  ] 1 test.
```

`bench run` starts the binary pinned to cores 0 to 3 with `--profile offcpu`,
as its `Running:` line shows; the backend runs inside the benchmark and
starts bpftrace through `sudo -n`, which `BENCH_SUDO=1` asks for. The
`[offcpu]` line is the capture's outcome: checked from its arm to its stop,
its dump whole, 20,016 sleeping switch-outs, written to
`./OffCpu.CoarseLock.offcpu/offcpu.txt` under the directory `bench run` ran
from. The result line is step 1's measurement taken again under the capture:
20.2 us per call, against 20.5 us there. The test took 4.2 s against step 1's
0.6 s, for the check and the capture's start and stop
([What is off-CPU profiling?](#what-is-off-cpu-profiling)).

```bash
ls OffCpu.CoarseLock.offcpu
```

```
offcpu.err.txt
offcpu.txt
```

## Step 3: Read the Capture

```bash
cat OffCpu.CoarseLock.offcpu/offcpu.txt
```

Captured output, with three long C++ names shortened to `(...)` and `<...>`,
and the eight GoogleTest frames between each test body and `main` cut and
marked `...`:

```
Attaching 2 probes...
offcpu armed 25675 25695
offcpu disarmed 25675 25675 20011


@armed: 2
@offcpu_blocks[
        __clone3+52
        popen+100
        vernier::bench::captureGitHash[abi:cxx11]()+64
        vernier::bench::PerfCase::contentionRun(...)+6656
        OffCpu_CoarseLock_Test::TestBody()+1576
        ...
        main+128
        __libc_start_call_main+124
        __libc_start_main@GLIBC_2.17+156
        _start+48
, BenchDemo_13_Of]: 1
@offcpu_blocks[
        0x7fa283ef6c
        __syscall_cancel+16
        _IO_new_file_underflow+320
        __GI__IO_default_uflow+56
        _IO_getline_info+212
        _IO_fgets+168
        vernier::bench::captureGitHash[abi:cxx11]()+88
        vernier::bench::PerfCase::contentionRun(...)+6656
        OffCpu_CoarseLock_Test::TestBody()+1576
        ...
        main+128
        __libc_start_call_main+124
        __libc_start_main@GLIBC_2.17+156
        _start+48
, BenchDemo_13_Of]: 1
@offcpu_blocks[
        0x7fa283ef6c
        __GI___futex_abstimed_wait_cancelable64+72
        __pthread_clockjoin_ex+420
        std::thread::join()+36
        vernier::bench::PerfCase::contentionRun(...)+840
        OffCpu_CoarseLock_Test::TestBody()+1576
        ...
        main+128
        __libc_start_call_main+124
        __libc_start_main@GLIBC_2.17+156
        _start+48
, BenchDemo_13_Of]: 17
@offcpu_blocks[
        __lll_lock_wait+80
        __pthread_mutex_lock+264
        vernier::bench::demo::offcpu_demo::addUnderCoarseLock(...)+24
        std::thread::_State_impl<...>::_M_run()+104
        0x7fa2b3b4e0
        start_thread+920
        thread_start+12
, BenchDemo_13_Of]: 19997
@offcpu_ns[25727]: 17823445
@offcpu_ns[25725]: 17975066
@offcpu_ns[25713]: 18066470
@offcpu_ns[25709]: 18066544
@offcpu_ns[25706]: 18148437
@offcpu_ns[25722]: 18161341
@offcpu_ns[25716]: 18193060
@offcpu_ns[25718]: 18265737
@offcpu_ns[25703]: 18303977
@offcpu_ns[25702]: 18489862
@offcpu_ns[25719]: 33614405
@offcpu_ns[25726]: 33658123
@offcpu_ns[25707]: 33664119
@offcpu_ns[25712]: 33677764
@offcpu_ns[25710]: 33719011
@offcpu_ns[25721]: 33756208
@offcpu_ns[25728]: 33936044
@offcpu_ns[25704]: 33956002
@offcpu_ns[25700]: 34516236
@offcpu_ns[25715]: 34558051
@offcpu_ns[25675]: 611641580
@recorded: 20011
```

| Line                                            | What it says                                                                                                                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Attaching 2 probes...`                         | bpftrace attaching the script's two probes: the one on the switch, and one that ends the script if the benchmark exits first                                                                     |
| `offcpu armed 25675 25695`                      | the script saw thread 25695 of process 25675, the thread Vernier started to arm it, go to sleep, and records from here                                                                           |
| `offcpu disarmed 25675 25675 20011`             | after the last repeat, thread 25675, the test's own (a process's first thread has the process's id), went to sleep as `vernier-stop`; the script had recorded 20,011 sleeps, and records no more |
| `@armed: 2`                                     | the window is closed; bpftrace printed its maps when Vernier stopped it, in the order of their names                                                                                             |
| `@offcpu_blocks[<stack>, BenchDemo_13_Of]: <n>` | how many sleeps were taken at that stack, innermost frame first, by threads named `BenchDemo_13_Of`, the program's name cut to 15 characters, which its threads inherit                          |
| `@offcpu_ns[<thread>]: <nanoseconds>`           | per thread, the time off the CPU after its counted sleeps                                                                                                                                        |
| `@recorded: 20011`                              | the script's own count of the sleeps, the number in the `disarmed` line                                                                                                                          |

The four stacks hold every sleep. Read them from the bottom of the dump up:

- **19,997 sleeps in the lock.** `__lll_lock_wait` in `__pthread_mutex_lock`,
  called from `addUnderCoarseLock`, called from the thread function
  `contentionRun()` gives each worker: the workers, asleep in the mutex the
  coarse lock takes. This stack is the finding: it names the function that
  takes the lock, and the count says how often a worker found it taken.
- **17 sleeps in `std::thread::join`**, under `contentionRun()` and the
  test's body: the test's own thread, waiting for each repeat's workers to
  end.
- **One sleep each under `captureGitHash`**, in `popen`, starting
  `git describe`, and in `fgets`, reading its answer: the harness records the
  tree's version for the CSV, once per process, after the last repeat and
  before the stop.

The last two are the harness's and the test's own, not the code under test:
the waits in `join` come from `contentionRun()` itself, which waits for its
workers, and the `git describe` pair from the first capture of a process. Two
frames are addresses: bpftrace found no name for them.

Twenty-one threads have a time in `@offcpu_ns`. Thread 25675, the test's
own, was off the CPU for 611.6 ms, about the whole measured time (ten repeats
of 3,000 calls at the median 20.217 us is 606.5 ms): it waited in `join`
while the workers ran, and for `git describe`. Twenty of the thirty workers
have a row: ten with 17.8 to 18.5 ms and ten with 33.6 to 34.6 ms. The other
ten have no row: they took no sleeping switch-out while the capture was
armed.

`@recorded` is 20,011, five below the stacks' sum of 20,016: the script
counts with an increment that two CPUs can make at the same moment, which
can only make the count come out lower. The `[offcpu]` line reports the
stacks' sum.

## Step 4: Confirm the Fix

```bash
BENCH_SUDO=1 bench run ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --taskset 0-3 --profile offcpu -- \
  --gtest_filter='OffCpu.NoSharing' --threads 3 --cycles 1000 --repeats 10
cat OffCpu.NoSharing.offcpu/offcpu.txt
```

Captured output of the run:

```
Running: taskset -c 0-3 ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --profile offcpu --gtest_filter=OffCpu.NoSharing --threads 3 --cycles 1000 --repeats 10
Note: Google Test filter = OffCpu.NoSharing
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from OffCpu
[ RUN      ] OffCpu.NoSharing
[offcpu] stacks written to ./OffCpu.NoSharing.offcpu/offcpu.txt (23 sleeping switch-outs)
[OffCpu.NoSharing]  6.646 us/call  CV=0.7%  ~150.5K calls/s  (p10=6.631 p90=6.749 sd=0.048)
[       OK ] OffCpu.NoSharing (3752 ms)
[----------] 1 test from OffCpu (3752 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (3752 ms total)
[  PASSED  ] 1 test.
```

and of the capture, cut as in step 3:

```
Attaching 2 probes...
offcpu armed 25747 25767
offcpu disarmed 25747 25747 23


@armed: 2
@offcpu_blocks[
        __clone3+52
        popen+100
        vernier::bench::captureGitHash[abi:cxx11]()+64
        vernier::bench::PerfCase::contentionRun(...)+6656
        OffCpu_NoSharing_Test::TestBody()+1580
        ...
        main+128
        __libc_start_call_main+124
        __libc_start_main@GLIBC_2.17+156
        _start+48
, BenchDemo_13_Of]: 1
@offcpu_blocks[
        0x7f92adef6c
        __syscall_cancel+16
        _IO_new_file_underflow+320
        __GI__IO_default_uflow+56
        _IO_getline_info+212
        _IO_fgets+168
        vernier::bench::captureGitHash[abi:cxx11]()+88
        vernier::bench::PerfCase::contentionRun(...)+6656
        OffCpu_NoSharing_Test::TestBody()+1580
        ...
        main+128
        __libc_start_call_main+124
        __libc_start_main@GLIBC_2.17+156
        _start+48
, BenchDemo_13_Of]: 1
@offcpu_blocks[
        0x7f92adef6c
        __GI___futex_abstimed_wait_cancelable64+72
        __pthread_clockjoin_ex+420
        std::thread::join()+36
        vernier::bench::PerfCase::contentionRun(...)+840
        OffCpu_NoSharing_Test::TestBody()+1580
        ...
        main+128
        __libc_start_call_main+124
        __libc_start_main@GLIBC_2.17+156
        _start+48
, BenchDemo_13_Of]: 21
@offcpu_ns[25747]: 201669430
@recorded: 23
```

Twenty-three sleeps, and none of them a worker's: the test's own thread, 21
times in `join` and twice for `git describe`, the harness's rows of step 3.
Only the test's thread has a time, 201.7 ms, again about the whole measured
time (ten repeats of 3,000 calls at the median 6.646 us is 199.4 ms). The
lock's 19,997 sleeps went with the lock, and with them two thirds of the
time.

Over the eleven captures of this command in the session, this one among
them, up to two workers had a row: one sleep each, of 0.024 to 0.030 ms, in
the C library, when a thread first registered the destructor of its
`thread_local` total, under a lock of the library's
(`__cxa_thread_atexit_impl`, in 7 of the 11 captures), or set up its memory
allocator on its first allocation (`malloc`, in 2). None was under
`addUnderCoarseLock`.

## Step 5: What a Sampler Sees

```bash
bench run ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --taskset 0-3 --profile gperf -- \
  --gtest_filter='OffCpu.CoarseLock' --threads 3 --cycles 1000 --repeats 10
google-pprof --text ./build/bin/ptests/BenchDemo_13_OffCpuProfiler OffCpu.CoarseLock.gperf/cpu.prof
```

Captured output of the run:

```
Running: taskset -c 0-3 ./build/bin/ptests/BenchDemo_13_OffCpuProfiler --profile gperf --gtest_filter=OffCpu.CoarseLock --threads 3 --cycles 1000 --repeats 10
Note: Google Test filter = OffCpu.CoarseLock
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from OffCpu
[ RUN      ] OffCpu.CoarseLock
PROFILE: interrupts/evictions/bytes = 71/37/3264
[OffCpu.CoarseLock]  21.955 us/call  CV=1.3%  ~45.5K calls/s  (p10=21.706 p90=22.299 sd=0.277)
[       OK ] OffCpu.CoarseLock (676 ms)
[----------] 1 test from OffCpu (676 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (676 ms total)
[  PASSED  ] 1 test.
```

and of the report, cut after the last function with samples of its own:

```
Using local file ./build/bin/ptests/BenchDemo_13_OffCpuProfiler.
Using local file OffCpu.CoarseLock.gperf/cpu.prof.
Total: 71 samples
      16  22.5%  22.5%       16  22.5% __memcpy_generic
      15  21.1%  43.7%       15  21.1% std::char_traits::assign (inline)
       9  12.7%  56.3%        9  12.7% std::__cxx11::basic_string::_M_data (inline)
       9  12.7%  69.0%        9  12.7% std::__cxx11::basic_string::size (inline)
       6   8.5%  77.5%       63  88.7% vernier::bench::demo::joinV1
       3   4.2%  81.7%        3   4.2% __GI___lll_lock_wake
       3   4.2%  85.9%        3   4.2% _init
       3   4.2%  90.1%       33  46.5% std::__cxx11::basic_string::_M_append (inline)
       3   4.2%  94.4%       19  26.8% std::char_traits::copy (inline)
       2   2.8%  97.2%        5   7.0% std::__cxx11::basic_string::capacity (inline)
       1   1.4%  98.6%        1   1.4% __aarch64_swp4_acq
       1   1.4% 100.0%        1   1.4% futex_wait
...
```

gperftools' profiler, which [walkthrough 03](03_GPERF_PROFILER.md) reads in
full, samples where the program is 100 times per second of CPU time. It took
71 samples, 0.71 s of CPU time, while the repeats lasted about 0.66 s (ten
repeats of 3,000 calls at 21.955 us): about one thread on a CPU at a time.
The profile is the join's: `joinV1` is on the stack of 63 samples, 88.7%,
copying (`__memcpy_generic`) and appending, and below the cut
`addUnderCoarseLock` is on the stack of 65. The lock's own code took a few,
waking a waiting thread (`__GI___lll_lock_wake`, 3) and in `futex_wait`
(1). Nothing in the profile shows the workers' sleeps in the lock: 19,997 in
step 3's capture, up to 34.6 ms per worker. A sleeping thread is not on a
CPU, and a CPU sampler takes no sample of it.

The same command with `--gtest_filter='OffCpu.NoSharing'`, run twice in the
session, took 58 and 59 samples, 0.58 and 0.59 s of CPU time against the
lock's 0.71 s, while its repeats lasted about 0.22 s: the three threads were
on CPUs at once. The two versions' profiles look alike; their time per call
differs about threefold, and the difference is time the threads spent off the
CPU, which only the off-CPU capture shows.

## What Should Reproduce

| Reading                                            | On this rig                                                                                                                | Elsewhere                                                                                                                                   |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| the lock's sleeps                                  | `__lll_lock_wait` in `__pthread_mutex_lock` under `addUnderCoarseLock`: 19,991 to 21,376 sleeps in each of eleven captures | should match wherever bpftrace's stacks reach the program; on an x86-64 laptop they did not ([If It Does Not Match](#if-it-does-not-match)) |
| the coarse lock's other sleeps                     | 13 to 20: the test's own thread in `join`, and the `git describe` pair                                                     | the same rows; how many `join` sleeps varies                                                                                                |
| no sharing                                         | 18 to 23 sleeps, none under `addUnderCoarseLock`; 0 to 2 of them a worker's, in the C library                              | none under `addUnderCoarseLock` should match; the C library's own waits depend on the library                                               |
| which workers sleep, and for how long              | with the lock, 20 to 23 of the 30 workers had a row, the others none; a worker that slept, 3.9 to 37.4 ms                  | varies with scheduling                                                                                                                      |
| the test's own thread                              | 610.4 to 628.0 ms with the lock, 199.4 to 223.6 ms with nothing shared: about the measured time                            | follows the measured time                                                                                                                   |
| the sampler                                        | 68 to 71 samples on the coarse lock over six runs, 58 and 59 with nothing shared; the join's functions on top              | the same shape where gperftools samples every thread; the counts depend on the machine                                                      |
| step 1: the lock's time per call over no sharing's | 2.8x here; 2.82x to 3.01x over twelve runs                                                                                 | the same direction wherever each thread has a core; how far depends on the machine                                                          |
| the doctor                                         | 5.9 to 6.0 s, its offcpu row 2.7 to 2.8 s                                                                                  | depends on how quickly bpftrace starts                                                                                                      |
| absolute times                                     | 20.5 and 7.3 us per call in step 1; 20.2 to 20.8 and 6.9 to 7.3 us over twelve runs                                        | will differ                                                                                                                                 |

The ranges come from one session on this rig: step 1's command and the two
capture commands ten times each, alternating, then step 1's once more for the
reference, then this page's commands, so twelve runs of step 1's command and
eleven of each capture, six runs of step 5's sampler command and five of the
doctor. They describe those runs; they are not a bound a run has to meet,
and another run of the same command can land outside them.

## If It Does Not Match

- **The doctor and the run say bpftrace supports only the root user.**
  Without `BENCH_SUDO=1`, the doctor's row reads

  ```
    [FAIL] offcpu     denied: the off-CPU script could not attach as the current user: ERROR: bpftrace currently only supports running as the root user.
               Set BENCH_SUDO=1 with a scoped sudoers grant for /usr/bin/bpftrace and /usr/bin/kill, run with CAP_BPF and CAP_PERFMON, or run as root.
  ```

  and step 2's command prints

  ```
  [FAIL] Profiler 'offcpu': denied: the off-CPU script could not attach as the current user: ERROR: bpftrace currently only supports running as the root user.
     Set BENCH_SUDO=1 with a scoped sudoers grant for /usr/bin/bpftrace and /usr/bin/kill, run with CAP_BPF and CAP_PERFMON, or run as root.
     Falling back to no-op (measurements will proceed without profiling).
  ```

  then measures as usual, writes no capture and exits 0. bpftrace 0.23.2
  checks for the root user itself, so of the hint's three ways, the two that
  work here are `BENCH_SUDO=1` with a grant, as in
  [Needs](#what-is-off-cpu-profiling), and running as root.

- **In this project's dev container, the tracepoint is not found.** The dev
  container (`docker compose run dev`: Ubuntu 24.04, bpftrace 0.20.2) mounts
  no tracefs. There the doctor's row reads

  ```
    [FAIL] offcpu     unsupported: the off-CPU script: stdin:1-2: ERROR: tracepoint not found: sched:sched_switch
  ```

  and step 2's command without `--taskset`, on the container's own build,
  prints

  ```
  [FAIL] Profiler 'offcpu': unsupported: the off-CPU script: stdin:1-2: ERROR: tracepoint not found: sched:sched_switch
     The kernel lacks a probe the script uses, or tracefs is not mounted (mount -t tracefs tracefs /sys/kernel/tracing); select a script whose probes exist here.
     Falling back to no-op (measurements will proceed without profiling).
  ```

  measures as usual and writes no capture folder. The hint's last clause is
  the generic bpftrace backend's: offcpu has one script. The dev container is
  privileged, and `sudo mount -t tracefs tracefs /sys/kernel/tracing` inside
  it mounts tracefs; the doctor then names the next obstacle, the container's
  own PID namespace.

- **The script never acknowledges its arm probe, and the line names a PID
  namespace.** In a PID namespace of its own, as in a container started
  without `--pid=host`, bpftrace does not number the benchmark's threads as
  the benchmark does, so the script never sees the arm thread it waits for,
  and the check refuses the request. On this rig, in a namespace made with
  `unshare --pid`, the doctor's row read

  ```
    [FAIL] offcpu     unsupported: the off-CPU script ran through sudo -n (BENCH_SUDO=1) but did not acknowledge its arm probe within 4000 ms; this process runs in PID namespace pid:[4026532587], not in the initial one (pid:[4026531836]), and offcpu traces only from the host's PID view
  ```

  and step 2's command printed the same reason as
  `[FAIL] Profiler 'offcpu': unsupported: ...`, followed by
  `Run the benchmark on the host, or in a container started with --pid=host.`
  and the fallback line, and wrote no capture; the dev container with tracefs
  mounted gives the same lines with its own namespace. Run the benchmark on
  the host, or start the container with `--pid=host`.

- **`no thread of this process went to sleep while the capture was armed`
  instead of `stacks written`.** That is a capture checked from its arm to
  its stop in which no thread of the benchmark slept, not a failure. On this
  rig, demo 01's single-threaded `BasicWorkflow.JoinV1`, measured after
  `BasicWorkflow.JoinV0` in one run under `--profile offcpu`, printed

  ```
  [offcpu] no thread of this process went to sleep while the capture was armed; the tracer's output is in ./BasicWorkflow.JoinV1.offcpu/offcpu.txt
  ```

  and its `offcpu.txt` held the two lines and `@armed: 2`, no stack: the
  join never sleeps. `JoinV0`, the first case of that run, recorded the
  harness's `git describe` (3 sleeps).

- **The stacks stop after a frame or two and name nothing of the demo.**
  bpftrace's user stacks follow frame pointers, and a function built without
  them breaks the walk. On an x86-64 laptop with Ubuntu 22.04, whose C library
  sets up no frame pointer in its lock and futex waits, and with a Release
  build of the demo there that sets up none either (GCC 11.4), step 2's
  command with bpftrace 0.14.0 wrote 53 stacks for 68 sleeps: 49 were two
  addresses bpftrace could not name, three a single frame of the C library,
  such as `__GI___futex_abstimed_wait_cancelable64+231`, and one `__wait4` and
  an address. None reached `addUnderCoarseLock` or the test. The capture
  itself was whole, and the run printed `stacks written`: only the stacks are
  short. On this rig they run from the C library to `_start` and
  `thread_start` (step 3).

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/16_offcpu_profiler.csv run.csv
```

Output for step 1's `run.csv`, printed by the `bench` CLI built from the tree
this page ships in (`bench 1.0.3`); the CLI prints the header in bold and the
labels in colour, which is not reproduced here:

```

Test                   Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
-----------------  ------------  ------------  ----------  --------  --------  --------  ------------
OffCpu.CoarseLock      20.68150      20.53000    -0.15150     -0.7%      1.2%      1.2%  neutral
OffCpu.NoSharing        6.92500       7.28117    +0.35617     +5.1%      9.2%      2.6%  REGRESSION

  1 regression(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference is step 1's command run once more by the same binary on this
board, in the same session, just before this page's commands: its rows'
timestamps are seven seconds before `run.csv`'s. Nothing changed between the
two runs, yet the version that shares nothing came out 5.1% slower, past the
5% threshold, so its row reads `REGRESSION`; the lock's moved -0.7% and reads
`neutral`. That is this pair of runs, not a property of the code. The CSVs'
`hostname` column records the name visible to the process that captured
each: this reference and `run.csv` were captured in the rig's ordinary
session, not in a UTS namespace named `pi4` as some of the rig's other
references were, so both record the board's host name, `raspberrypi`. As the
note under the table says, a label compares two runs' medians; it is not a
significance test, and `Base CV` and `Cand CV` are each run's own spread,
which says nothing about the spread between runs: the reference's 9.2% for
no sharing is the spread of that run's ten repeats. Over the session's twelve
runs of step 1's command, the no-sharing median ranged from 6.89 to 7.30 us,
6.0%, and the lock's from 20.20 to 20.83 us, 3.1%, with nothing changed: the
no-sharing median is the noisiest measurement on this page, and a run of
unchanged code can land on either side of the threshold. Read the medians and
their ratio, not the label. The CSV holds times only; what the capture found
is checked by `OffCpu.CapturesTheLockWait`, below.

## What Keeps This Page True

Three things check what this page shows, and all fail loudly:

- `OffCpu.CapturesTheLockWait`, in a test program of its own beside the demo
  ([`13_OffCpuProfiler_uTest.cpp`](../cpu/utst/13_OffCpuProfiler_uTest.cpp),
  built as `TestDemoOffCpu`), runs the demo binary under `--profile offcpu`
  with `BENCH_SUDO=1`, as steps 2 and 4 do, on each case with
  `--threads 3 --cycles 200 --repeats 3`, and reads the two captures. It
  fails unless each case runs to its end with one `[offcpu]` line,
  `stacks written` (or, for no sharing, the line for a capture in which
  nothing slept); unless the two acknowledgements name the demo's process,
  the arm a thread other than its first and the stop its first thread, the
  arm first; unless the dump holds `@armed: 2` and no `@start` line, with
  `@recorded` equal to the stop's count and no more than the stacks' sum; and
  unless the coarse lock's capture holds sleeps in the lock under
  `addUnderCoarseLock` and the no-sharing capture none. How long anything
  slept, how often and which workers slept are scheduling and are not
  checked. With `addUnderCoarseLock` left for the compiler to inline, it
  fails on this rig: no stack names the function. It skips before anything
  runs in a PID namespace other than the host's (this project's container
  has one), where bpftrace is not installed, and without root or sudo; where
  the run says offcpu could not trace, quoting the run's line, which ends
  with sudo's or bpftrace's own words; and, once both captures have passed
  their checks, where no stack reaches the demo's test body, quoting the
  stacks, as on the x86-64 laptop above. Beside it,
  `OffCpuCaptureReadingTest` holds that reading to two captures and three
  notices from this rig, and `OffCpuTotalsTest` runs each version through
  `contentionRun()` with four threads, 25 calls per thread and three
  repeats, and fails unless the total holds every call's joined length.
- The backend's own tests drive it with a stand-in for bpftrace, in
  `TestBench` (`BpfCheckTest.OffCpu*`, `OffCpuCaptureTest.*`) and through
  the CLI (`ReadinessCli.Offcpu*`): a capture whose tracer never
  acknowledges, acknowledges for another process or thread, ends before the
  stop, cannot be stopped or leaves its dump cut short is reported as
  failed, never as `stacks written`.
- The example's unit tests hold `joinV0` and `joinV1` to the same answers,
  under the `demo` label.

The demo's check runs under the `demo` label, the traced check also under
`bpftrace`, which runs it alone:

```bash
ctest --test-dir build -L demo
ctest --test-dir build -L bpftrace
```

On this rig the first selected 50 tests and the second one, and every one
passed but the memcheck walkthrough's valgrind probe, which `ctest` reports
as skipped outside valgrind, as it should. The demo's two timing tests are
not registered: what they measure belongs to the machine they run on. This
repository has no continuous-integration lane on the reference board, so
before a release the page's commands are run on the rig by hand, and the page
and its reference CSV are re-captured when what they show changes.

## See Also

- [Demo 9 (bpftrace)](09_BPFTRACE_PROFILER.md) -- the same tracer, on
  other kernel events
- [Demo 6 (Thread Scaling)](06_THREAD_SCALING.md) -- what a call
  costs while other threads make it, under one lock held for the whole
  call and with nothing shared, measured with `contentionRun()`
- [Demo 20 (Helgrind)](20_HELGRIND_PROFILER.md) -- data races between
  threads that share a total
- [Demo 3 (gperftools)](03_GPERF_PROFILER.md) -- the CPU sampler of step 5,
  on one thread
- [CPU guide: off-CPU profiling](../../docs/CPU_GUIDE.md#off-cpu-profiling-where-threads-sleep)
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
