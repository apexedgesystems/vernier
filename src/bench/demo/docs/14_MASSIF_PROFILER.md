# Demo 14: Valgrind Massif Heap Profiler

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) at 20,000 words (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-24, written for the Vernier 1.0.4 release; captured from
the development tree at project version 1.0.3, whose CLI reported `bench 1.0.3`

## Overview

Massif answers two questions about a program's heap: how much it held at its
worst moment, and which code was holding it. The example is the `join` from
[walkthrough 01](01_BASIC_WORKFLOW.md), at 20,000 words. Both versions return
the same 149,812-byte string. At its peak, V1's call holds that string and
nothing more; V0's holds five times as much, and massif's report shows it as
two entries in `joinV0`, one for each place that allocated it.

## What Is Massif?

Massif is one of Valgrind's tools. Valgrind runs the program on a synthetic CPU
and puts its own `malloc`, `operator new` and their relatives in place of the
program's, so massif sees every heap allocation and every free, with the call
stack that made it. It takes snapshots of the heap's size as those events
happen, keeps up to 100 of them, and records the largest: the peak. For the
peak, and for one snapshot in ten, it also keeps a tree of the call stacks
that hold the bytes. `ms_print` renders the file massif writes as a text graph
followed by those trees.

- **Best for:** the heap's high-water mark, what it is made of, and which
  allocation sites own it.
- **Overhead:** it depends on what the code does. On this rig a profiled call
  of V0 took 0.88 s against 0.40 s without valgrind, and a call of V1 3.2 ms
  against 0.47 ms. Try one profiled call before sizing a run.
- **Not for:** time (perf, gperftools and callgrind, walkthroughs
  [02](02_PERF_PROFILER.md), [03](03_GPERF_PROFILER.md) and
  [07](07_CALLGRIND_PROFILER.md)), leaks and invalid accesses (memcheck,
  [walkthrough 15](15_MEMCHECK_PROFILER.md)), or memory the program maps itself
  with `mmap`, which massif leaves out unless you add `--pages-as-heap=yes`.

**In Vernier:** `--profile massif` selects the massif backend, which does not
start valgrind itself. Run the binary under `valgrind --tool=massif`, as the
steps below do, or let `bench run --profile massif` build that command
([below](#letting-bench-run-do-the-wrap)). Either way massif writes one file
for the whole process, where `--massif-out-file` says. Run by hand with
`--profile massif`, the binary also creates a folder for each test it runs in
the working directory, `Massif.JoinV0.massif/` in step 2. Massif writes there
only if `--massif-out-file` points into it, which is what the command the
backend prints does when the binary runs without valgrind; with the commands on
this page the folder stays empty. `bench run` creates no such folder.

**Needs:** valgrind, from the rig document's
[package list](../../docs/rigs/RIG_PI4.md#2-one-time-setup);
[`bench doctor`](../../docs/rigs/RIG_PI4.md#6-verify-your-rig) checks for it.

## The Example

The two versions of `join`:

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

Both return the same string, and the example's unit tests
([`Join_uTest.cpp`](../examples/join/utst/Join_uTest.cpp)) hold them to that.
Walkthrough 01 measures what V0's copying costs in time; this page measures
what it costs in memory. V1 allocates once, the size of the result. V0, at the
last word, holds three buffers at once: the string built so far, which `out`
still holds; the temporary `out + part`, built at its exact size; and that
temporary moved into a buffer twice its size when `+ sep` appends the
separator, which is the buffer `out` takes next. The first and the last are
each about twice the joined string and the middle one about once, so V0's call
holds about five times what V1's does. That count is libstdc++'s, the GNU
standard library this rig builds with; libc++ regrows strings differently, and
an x86 build against libc++ 14 measured four times.

The demo measures each version in its own test at 20,000 words, one CSV row
each, as walkthrough 01 does. A third test, `Massif.JoinPeakHeap`, checks the
effect this page is about. It cannot run massif, so it counts bytes another
way: the demo replaces the global `operator new` and `operator delete` with
versions that count what they hand out and take back, and `peakHeapDuring`
returns the most one call held at once beyond what was live before it:

```cpp
template <typename Op> std::size_t peakHeapDuring(Op&& op) {
  const std::size_t BASE = heapLive.load(std::memory_order_relaxed);
  heapPeak.store(BASE, std::memory_order_relaxed);
  op();
  return heapPeak.load(std::memory_order_relaxed) - BASE;
}
```

The test fails unless V0's call holds more than three times what V1's does. It
writes no CSV row. Valgrind puts its own `operator new` in place of the
demo's, so under valgrind the test counts nothing and skips itself.

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_11_MassifProfiler \
  --target-time 50ms --repeats 10 --csv run.csv
```

Captured output:

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from Massif
[ RUN      ] Massif.JoinV0
[target-time] 50.000 ms -> cycles=1 (calibrated 430685.0000 us/call, batch of 1)
[Massif.JoinV0]  398687.000 us/call  CV=0.6%  ~3 calls/s  (p10=394663.700 p90=400840.500 sd=2529.484)
[       OK ] Massif.JoinV0 (5547 ms)
[ RUN      ] Massif.JoinV1
[target-time] 50.000 ms -> cycles=106 (calibrated 471.0000 us/call, batch of 4)
[Massif.JoinV1]  470.807 us/call  CV=0.3%  ~2.1K calls/s  (p10=469.898 p90=472.783 sd=1.551)
[       OK ] Massif.JoinV1 (506 ms)
[ RUN      ] Massif.JoinPeakHeap
[Massif.JoinPeakHeap]  joined 149812 bytes  V0 holds 749048  V1 holds 149816  5.0x
[       OK ] Massif.JoinPeakHeap (710 ms)
[----------] 3 tests from Massif (6765 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (6765 ms total)
[  PASSED  ] 3 tests.

===============================================================
Test            Median (us)     CV%     Calls/s  Status
---------------------------------------------------------------
Massif.JoinV0    398687.000    0.6%           3  OK
Massif.JoinV1       470.807    0.3%        2.1K  OK
---------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

At 20,000 words V0 takes 398.7 ms per call and V1 0.47 ms, about 850 times
less, with CVs of 0.6% and 0.3%: walkthrough 01's copying, at twenty times the
words. `--target-time 50ms` ran V0 once per repeat, since one call outlasts
the target, and V1 106 times.

The line this page is about is `JoinPeakHeap`'s. The joined string is 149,812
bytes; at its peak V0's call held 749,048 bytes and V1's 149,816, five times as
much. These are counts of bytes, not times, and they came out the same in every
run on this rig.

## Step 2: Profile V0

```bash
valgrind --tool=massif --time-unit=B --massif-out-file=massif.v0.out \
  ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif \
  --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV0
ms_print massif.v0.out
```

`--cycles 1 --repeats 1` keeps the run to three calls of `joinV0`: the test's
check of the result's size, the warmup call and the one measured call. Under
massif the test took 2.8 s. `--time-unit=B` is explained
[below](#why---time-unitb).

The profile is `massif.v0.out` in the working directory, the file
`--massif-out-file` names. The run also leaves the empty folder
`Massif.JoinV0.massif/` beside it (see **In Vernier** above). `ms_print`
printed the graph, a table of snapshots, and a tree for each detailed
snapshot. Captured output, cut to the graph and the peak:

```
--------------------------------------------------------------------------------
Command:            ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV0
Massif arguments:   --time-unit=B --massif-out-file=massif.v0.out
ms_print arguments: massif.v0.out
--------------------------------------------------------------------------------


    MB
1.403^                        #
     |                     :@@#                                           :: :
     |                 ::  :@ #               ::::                    :  ::: :
     |            :    :   :@ #               : ::: @::           ::  :  ::: :
     |           ::   ::   :@ #               : ::: @:           ::   :  ::: :
     |       @@  ::   ::   :@ #      :     @  : ::::@:      :   @:: : :  ::: :
     |       @ ::::   ::   :@ #    :::     @  : ::::@:    :::   @:: : :  ::: :
     |    :  @ : ::  ::: :::@ #    : ::  ::@  : ::::@:   :: : ::@:: ::::::::@:
     |    :  @ : ::::::: : :@ #   :: ::@@: @::: ::::@:   :: ::: @:: :::: :::@:
     |  :::::@ : ::: ::: : :@ # :::: ::@ : @: : ::::@:  ::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
     | :: :: @ : ::: ::: : :@ #:: :: ::@ : @: : ::::@: :::: ::: @:: :::: :::@:
   0 +----------------------------------------------------------------------->GB
     0                                                                   24.91

Number of snapshots: 54
 Detailed snapshots: [5, 15, 16 (peak), 23, 25, 32, 41, 51]
...
--------------------------------------------------------------------------------
  n        time(B)         total(B)   useful-heap(B) extra-heap(B)    stacks(B)
--------------------------------------------------------------------------------
 16  8,932,350,264        1,471,304        1,470,433           871            0
99.94% (1,470,433B) (heap allocation functions) malloc/new/new[], --alloc-fns, etc.
->43.50% (640,000B) 0x11E0D3: vernier::bench::demo::makeParts[abi:cxx11](unsigned long, unsigned int) (in .../BenchDemo_11_MassifProfiler)
| ->43.50% (640,000B) 0x10F4B3: Massif_JoinV0_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|   ...
|
->40.60% (597,322B) 0x11DB2F: vernier::bench::demo::joinV0(std::vector<...> const&, char) (in .../BenchDemo_11_MassifProfiler)
| ->40.60% (597,322B) 0x10F4C3: Massif_JoinV0_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|   ...
|
->10.15% (149,335B) 0x11D7D3: vernier::bench::demo::joinV0(std::vector<...> const&, char) (in .../BenchDemo_11_MassifProfiler)
| ->10.15% (149,335B) 0x10F4C3: Massif_JoinV0_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|   ...
|
->05.01% (73,728B) 0x4A44F2B: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
| ->05.01% (73,728B) 0x4004A6B: call_init (dl-init.c:74)
|   ...
|
->00.68% (10,048B) in 1+ places, all below ms_print's threshold (01.00%)
```

## Step 3: Read the Report

**The graph.** Up the side, the heap in MB; along the bottom, because of
`--time-unit=B`, the bytes allocated and freed so far. Each column is a
snapshot: `:` a normal one, `@` a detailed one, `#` the peak. The three teeth
are the three calls: each climbs as V0's string grows and drops back when the
call's result is destroyed.

**The peak.** Snapshot 16 held 1,471,304 bytes: 1,470,433 that the program
asked for (`useful-heap`) and 871 more that the allocator spent on its own
headers and rounding (`extra-heap`). The tree divides the useful heap by the
call stack that allocated it, largest first:

| Bytes   | Allocated in                        | What it is                                                                                           |
| ------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 640,000 | `makeParts`                         | the input: 20,000 strings of 32 bytes each, every word short enough to live inside its string object |
| 597,322 | `joinV0`, 0x11DB2F                  | the buffers `+ sep` moves the temporary into: two are live, the one `out` holds and the new one      |
| 149,335 | `joinV0`, 0x11D7D3                  | the temporary `out + part`, at its exact size                                                        |
| 73,728  | libstdc++ (`???`), from `call_init` | allocated by the standard library while the program loads, before `main`                             |
| 10,048  | below 1%                            | everything else                                                                                      |

So at its peak V0's own buffers are 746,657 bytes, five times the 149,813
bytes V1 allocates for the same result (the 149,812 characters and a
terminating null).

What not to conclude:

- **That the two `joinV0` entries were read off source lines.** A Release build
  carries no debug information, so both print as `joinV0`, told apart by
  their addresses and their sizes. Built with `-g`, the same `Join.cpp`, from
  the same compiler, names them: the larger comes from `append` in the
  `operator+` that adds a character, the smaller from `reserve` in the
  `operator+` that joins two strings.
- **That massif's peak is exact.** It records a peak only when the heap passes
  the last recorded one by 1% (`--peak-inaccuracy`), so its peak snapshot can
  fall a little before the true one: here the temporary is 149,335 bytes, 477
  short of its size at the last word. `JoinPeakHeap` counts every allocation,
  in the sizes the allocator hands back, and read 749,048 bytes for the same
  call.
- **That the total is the comparison.** V0's 1,471,304 bytes against V1's
  874,440 in step 4 is 1.68 times, because the 640,000-byte input and
  libstdc++'s 73,728 bytes are in both. The code's difference is in the join's
  own entries: 746,657 bytes against 149,813.

## Step 4: Confirm the Fix

```bash
valgrind --tool=massif --time-unit=B --massif-out-file=massif.v1.out \
  ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif \
  --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV1
ms_print massif.v1.out
```

Captured output, cut to the graph and the peak:

```
--------------------------------------------------------------------------------
Command:            ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV1
Massif arguments:   --time-unit=B --massif-out-file=massif.v1.out
ms_print arguments: massif.v1.out
--------------------------------------------------------------------------------


    KB
854.0^                                             ::::
     |                            ####    :::::    :
     |                            #       :        :
     |                            #       :        :
     |                       :::::#   :::::    :::::   @:::::::::::::::::::
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |                       :    #   :   :    :   :   @:
     |  @:@:::::::::::::::::::    #   :   :    :   :   @:                  @:
   0 +----------------------------------------------------------------------->MB
     0                                                                   2.345

Number of snapshots: 70
 Detailed snapshots: [6, 23, 32 (peak), 39, 47, 49, 59, 69]
...
--------------------------------------------------------------------------------
  n        time(B)         total(B)   useful-heap(B) extra-heap(B)    stacks(B)
--------------------------------------------------------------------------------
...
 32        956,664          874,440          873,589           851            0
99.90% (873,589B) (heap allocation functions) malloc/new/new[], --alloc-fns, etc.
->73.19% (640,000B) 0x11E0D3: vernier::bench::demo::makeParts[abi:cxx11](unsigned long, unsigned int) (in .../BenchDemo_11_MassifProfiler)
| ->73.19% (640,000B) 0x10FF93: Massif_JoinV1_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|   ...
|
->17.13% (149,813B) 0x11D423: vernier::bench::demo::joinV1(std::vector<...> const&, char) (in .../BenchDemo_11_MassifProfiler)
| ->17.13% (149,813B) 0x10FFA3: Massif_JoinV1_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|   ...
|
->08.43% (73,728B) 0x4A44F2B: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
| ->08.43% (73,728B) 0x4004A6B: call_init (dl-init.c:74)
|   ...
|
->01.15% (10,048B) in 68 places, all below massif's threshold (1.00%)
```

The first step up is the input; the three teeth on top of it are the three
calls, one buffer each. The `#` sits a row below the tallest columns: those are
the third call's, 874,464 bytes against the recorded peak's 874,440, a rise
too small for massif to record as a new peak (it needs 1%) but enough to set
the top of `ms_print`'s scale. At the peak, `joinV1` holds 149,813 bytes, the
result it reserved before appending anything, and the input is the largest
owner of the heap, 73% of it. The heap V1's call owns is a fifth of V0's.

## Why --time-unit=B

Massif snapshots the heap when the program allocates or frees, and `ms_print`
places each snapshot at its time. By default that time is instructions
executed. Step 4 without `--time-unit=B`:

```bash
valgrind --tool=massif --massif-out-file=massif.v1.default.out \
  ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif \
  --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV1
ms_print massif.v1.default.out
```

Captured output, cut to the graph:

```
--------------------------------------------------------------------------------
Command:            ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV1
Massif arguments:   --massif-out-file=massif.v1.default.out
ms_print arguments: massif.v1.default.out
--------------------------------------------------------------------------------


    KB
854.0^                                                                ::::::
     |                                                   ::::::#::::: :
     |                                                   :     #:     :
     |                                                   :     #:     :
     |              ::::::::::::::::::::::::::::::::::::::     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |              :                                    :     #:    ::     @
     |           :@@@                                    :     #:    ::     @@
   0 +----------------------------------------------------------------------->Mi
     0                                                                   13.39

Number of snapshots: 86
 Detailed snapshots: [14, 32, 33, 34, 40, 43, 50, 54 (peak), 64, 74, 84]
```

Building the 20,000 words is the long flat stretch. V1's three calls come
after it, at the right, and they do not read as three: massif keeps at most
100 snapshots and thins them out as the run goes on, and none of those it kept
fell between the first two calls, so those two draw as one block and only the
third stands apart. On the byte axis the clock moves only when memory is
allocated or freed, and every call allocates and frees the same 149,813 bytes,
so each call and each gap between calls get about the same width and the calls
come out as three teeth. V0's graph shows its three calls either way, because
they are most of what the program does, in bytes and in instructions alike.

## Letting bench run Do the Wrap

```bash
bench run ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif \
  --cycles 1 --repeats 1 -- --gtest_filter=Massif.JoinV0
ms_print bench-out/BenchDemo_11_MassifProfiler.massif/massif.out
```

Captured output of the first command:

```
Running: valgrind --tool=massif --massif-out-file=bench-out/BenchDemo_11_MassifProfiler.massif/massif.out ./build/bin/ptests/BenchDemo_11_MassifProfiler --cycles 1 --repeats 1 --profile massif --gtest_filter=Massif.JoinV0
==35985== Massif, a heap profiler
==35985== Copyright (C) 2003-2024, and GNU GPL'd, by Nicholas Nethercote et al.
==35985== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==35985== Command: ./build/bin/ptests/BenchDemo_11_MassifProfiler --cycles 1 --repeats 1 --profile massif --gtest_filter=Massif.JoinV0
==35985==
Note: Google Test filter = Massif.JoinV0
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Massif
[ RUN      ] Massif.JoinV0
[Massif.JoinV0]  1009412.000 us/call  CV=0.0%  ~1 calls/s  (p10=1009412.000 p90=1009412.000 sd=0.000)
[       OK ] Massif.JoinV0 (3181 ms)
[----------] 1 test from Massif (3186 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (3228 ms total)
[  PASSED  ] 1 test.
==35985==
```

`bench run` builds the same wrap and prints it on its first line, without
`--time-unit=B`, which it has no option to pass: its graph's axis is
instructions. The peak's owners and their byte counts are the ones in step 2.
The profile goes to `bench-out/BenchDemo_11_MassifProfiler.massif/massif.out`,
one file for the binary, not one per test, and no per-test folder is created
([Troubleshooting](../../docs/TROUBLESHOOTING.md#profile-artifacts-land-in-cwd-move-immediately)
has the rule for every wrapped tool). A second `bench run` of the same binary
writes the same file: run V1 after V0 and only V1's profile is left. A
different `--profile-output-dir` gives a run a file of its own.

## What Should Reproduce

| Reading                                        | On this rig                                                                                                      | Elsewhere                                                                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| heap V0's call holds at its peak, against V1's | 5.0x (`JoinPeakHeap`); 746,657 against 149,813 bytes in massif's trees                                           | 5.0x with libstdc++, also on an x86 laptop; 4.0x with libc++ 14                                       |
| who holds V0's peak                            | the input 640,000 bytes, `joinV0` 597,322 and 149,335, libstdc++ 73,728                                          | the same owners; the bytes depend on the standard library                                             |
| total heap at the peak                         | 1,471,304 against 874,440 bytes, 1.68x                                                                           | depends on the size of `std::string` (32 bytes here) and on what else the process holds               |
| byte counts from run to run                    | the join's entries identical in every run; the total within 64 bytes, moving with the command line and directory | should hold                                                                                           |
| V0 and V1 per call                             | 398.7 ms and 0.47 ms, about 850x; 771x to 849x over eleven runs                                                  | will differ: V0's time grows with the square of the word count, and memory bandwidth decides the rest |

Over the eleven Step 1 runs taken for this page, V0's median stayed between
390.3 and 405.0 ms, and V1's between 465.6 and 513.0 us. Ten of them kept V1
at or under 493.8 us with CVs of 2.2% or less; the eleventh ran seconds after a
build had finished on the same board, and read 513.0 us with a CV of 4.8%, V0
4.3%. The byte counts did not move in any of them.

## If It Does Not Match

- **`Massif.JoinV0.massif/` is empty.** It is meant to be: the profile is the
  file `--massif-out-file` names (see **In Vernier**).
- **`Massif.JoinPeakHeap` reports `SKIPPED`.** It ran under valgrind, which
  replaces the demo's counting `operator new`; the skip says
  `valgrind replaces this binary's operator new, so the heap is not counted; run this test without valgrind`.
  Run it as in step 1.
- **V1's graph is one block at the right edge.** The run left out
  `--time-unit=B`, or came from `bench run`, which does not pass it.
- **V0 holds about four times V1's heap, not five.** The build uses another
  standard library; libc++ regrows strings differently. The test's floor is
  three.
- **CV above a few percent in step 1.** Something else was running on the
  board, or had just finished: one run for this page, taken seconds after a
  build, read 4.3% and 4.8%. The rig document's measurement section has the
  governor and `taskset` recipe; `vcgencmd get_throttled` should read `0x0`
  before and after.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/14_massif_profiler.csv run.csv
```

Captured with the `bench` CLI built from this tree, which reported
`bench 1.0.3`:

```
Test               Baseline     Candidate       Delta         %   p-value        Result
-------------  ------------  ------------  ----------  --------  --------  ------------
Massif.JoinV0  400400.00000  398687.00000  -1713.00000     -0.4%    0.0757  neutral
Massif.JoinV1     474.74300     470.80700    -3.93600     -0.8%    0.4274  neutral

  2 neutral
```

The reference is the same binary on the same board, captured about two hours
before the run above. Both rows moved by less than 1%, well inside the tool's
5% default threshold and inside the spread above, and both are labelled
`neutral`. The CSV carries times, not heap sizes: the heap is what
`JoinPeakHeap` checks.

## What Keeps This Page True

- `Massif.JoinPeakHeap`, in the demo binary, fails unless V0's call holds more
  than three times the heap V1's call holds at its peak. It counts bytes, so a
  busy machine does not change its answer.
- The example's unit tests hold every version to the same answers and are
  registered with `ctest`, so ordinary CI runs them:

  ```bash
  ctest --test-dir build -L demo
  ```

  ```
  100% tests passed, 0 tests failed out of 8
  ```

This repository has no continuous-integration lane on the reference board, so
nothing runs the demo itself automatically. Before a release it is run on the
rig by hand, with the commands above, and the reference CSV is re-captured
when the numbers move.

## See Also

- [Demo 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example, measured
  for time
- [Demo 21: heaptrack](21_HEAPTRACK_PROFILER.md) -- which call sites allocate,
  and how often
- [Demo 15: memcheck](15_MEMCHECK_PROFILER.md) -- leaks and invalid accesses
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
