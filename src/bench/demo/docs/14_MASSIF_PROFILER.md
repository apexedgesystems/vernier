# Demo 14: Valgrind Massif Heap Profiler

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) at 20,000 words (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-28 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; valgrind 3.24.0

## Overview

Code that builds a large result a piece at a time, a report, a CSV file, a
message assembled in a loop, can hold several times that result while it
works, and the most it holds at once decides whether it fits in memory.
Massif answers two questions about a program's heap: how much it held at its
worst moment, and which code was holding it. The example is the `join` from
[walkthrough 01](01_BASIC_WORKFLOW.md), at 20,000 words. Both versions return
the same 149,812-byte string. At its peak, V1's call holds that string and
nothing more; V0's holds five times as much, and massif's report names the
source line that holds it: the one line of `joinV0` that builds the string,
once for each of the two allocations it makes.

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
  of V0 took 0.88 s against 0.39 s without valgrind, and a call of V1 2.9 ms
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

The demo, [`cpu/11_MassifProfiler_Demo.cpp`](../cpu/11_MassifProfiler_Demo.cpp),
measures each version in a test of its own at 20,000 words, `Massif.JoinV0`
and `Massif.JoinV1`, one CSV row each, as walkthrough 01 does. V0's test:

```cpp
PERF_THROUGHPUT(Massif, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}
```

It builds the words, checks the answer's size against `joinedSize()`, which
computes it without allocating, and measures. That is the whole demo: it
measures, and massif reads its heap. The check that V0 holds more than three
times V1's heap is a test program of its own beside the join example, because
it counts through a replaced `operator new`, and the program massif profiles
should be the one you would write
([What Keeps This Page True](#what-keeps-this-page-true)).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_11_MassifProfiler \
  --target-time 50ms --repeats 10 --csv run.csv
```

Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from Massif
[ RUN      ] Massif.JoinV0
[target-time] 50.000 ms -> cycles=1 (calibrated 382295.0000 us/call, batch of 1)
[Massif.JoinV0]  387532.000 us/call  CV=0.5%  ~3 calls/s  (p10=384834.600 p90=389980.600 sd=2113.136)
[       OK ] Massif.JoinV0 (5293 ms)
[ RUN      ] Massif.JoinV1
[target-time] 50.000 ms -> cycles=108 (calibrated 460.2500 us/call, batch of 4)
[Massif.JoinV1]  469.384 us/call  CV=1.9%  ~2.1K calls/s  (p10=452.618 p90=472.459 sd=8.638)
[       OK ] Massif.JoinV1 (509 ms)
[----------] 2 tests from Massif (5802 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (5803 ms total)
[  PASSED  ] 2 tests.

===============================================================
Test            Median (us)     CV%     Calls/s  Status
---------------------------------------------------------------
Massif.JoinV0    387532.000    0.5%           3  OK
Massif.JoinV1       469.384    1.9%        2.1K  OK
---------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

In this run, at 20,000 words, V0 took 387.5 ms per call and V1 0.47 ms, 826
times less, with CVs of 0.5% and 1.9%: walkthrough 01's copying, at twenty
times the words. `--target-time 50ms` ran V0 once per repeat, since one call
outlasts the target, and V1 108 times.

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
snapshot. Captured output, cut to the graph and the peak. In the peak's tree,
the input's entry is cut between its first frame and the join example's, every
entry is cut below the test's own frame (GoogleTest's frames and `main`), the
vector's template arguments are shortened to `std::vector<...>`, and the
binary's directory to `...`:

```
--------------------------------------------------------------------------------
Command:            ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV0
Massif arguments:   --time-unit=B --massif-out-file=massif.v0.out
ms_print arguments: massif.v0.out
--------------------------------------------------------------------------------


    MB
1.403^                                               ::
     |                      @##                  @@: :                    ::
     |                :     @#                :  @ : :                  ::::
     |            @   :     @#                :  @ :::            ::@:  :::: :
     |         @@ @ :::     @#         ::     :  @ :::         :@:: @::::::: :
     |         @  @ : ::::  @#         :   :  :::@ :::        ::@:: @::::@:: :
     |    :  ::@  @ : :: :  @#     @   :   :: :: @ :::    :   ::@:: @::::@:: :
     |   @:: : @  @ : :: :::@#     @::@:   ::::: @ :::    :   ::@:: @::::@::::
     |   @:: : @ :@:: :: :: @#  :  @: @: @@::::: @ ::: :: :   ::@:: @::::@::::
     |   @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::::::@:: @::::@::::
     | ::@:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
     | : @:::: @ :@:: :: :: @# ::::@: @: @ ::::: @ ::: :::::: ::@:: @::::@::::
   0 +----------------------------------------------------------------------->GB
     0                                                                   25.06

Number of snapshots: 70
 Detailed snapshots: [2, 7, 9, 16, 17 (peak), 22, 24, 26, 33, 45, 50, 60]
...
--------------------------------------------------------------------------------
  n        time(B)         total(B)   useful-heap(B) extra-heap(B)    stacks(B)
--------------------------------------------------------------------------------
 17  8,927,870,112        1,470,720        1,469,864           856            0
99.94% (1,469,864B) (heap allocation functions) malloc/new/new[], --alloc-fns, etc.
->43.52% (640,000B) 0x11CA73: allocate (new_allocator.h:151)
| ...
|             ->43.52% (640,000B) 0x11CA73: vernier::bench::demo::makeParts[abi:cxx11](unsigned long, unsigned int) (Join.cpp:58)
|               ->43.52% (640,000B) 0x10E713: Massif_JoinV0_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|                 ...
|
->40.60% (597,158B) 0x11C4D7: allocate (new_allocator.h:151)
| ->40.60% (597,158B) 0x11C4D7: allocate (allocator.h:196)
|   ->40.60% (597,158B) 0x11C4D7: allocate (alloc_traits.h:515)
|     ->40.60% (597,158B) 0x11C4D7: _S_allocate (basic_string.h:131)
|       ->40.60% (597,158B) 0x11C4D7: _M_create (basic_string.tcc:159)
|         ->40.60% (597,158B) 0x11C4D7: _M_mutate (basic_string.tcc:332)
|           ->40.60% (597,158B) 0x11C4D7: _M_replace_aux (basic_string.tcc:468)
|             ->40.60% (597,158B) 0x11C4D7: append (basic_string.h:1499)
|               ->40.60% (597,158B) 0x11C4D7: operator+<char, std::char_traits<char>, std::allocator<char> > (basic_string.h:3742)
|                 ->40.60% (597,158B) 0x11C4D7: vernier::bench::demo::joinV0(std::vector<...> const&, char) (Join.cpp:25)
|                   ->40.60% (597,158B) 0x10E723: Massif_JoinV0_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|                     ...
|
->10.15% (149,295B) 0x11C17B: allocate (new_allocator.h:151)
| ->10.15% (149,295B) 0x11C17B: allocate (allocator.h:196)
|   ->10.15% (149,295B) 0x11C17B: allocate (alloc_traits.h:515)
|     ->10.15% (149,295B) 0x11C17B: _S_allocate (basic_string.h:131)
|       ->10.15% (149,295B) 0x11C17B: _M_create (basic_string.tcc:159)
|         ->10.15% (149,295B) 0x11C17B: reserve (basic_string.tcc:315)
|           ->10.15% (149,295B) 0x11C17B: __str_concat<std::__cxx11::basic_string<char> > (basic_string.h:3582)
|             ->10.15% (149,295B) 0x11C17B: operator+<char, std::char_traits<char>, std::allocator<char> > (basic_string.h:3604)
|               ->10.15% (149,295B) 0x11C17B: vernier::bench::demo::joinV0(std::vector<...> const&, char) (Join.cpp:25)
|                 ->10.15% (149,295B) 0x10E723: Massif_JoinV0_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|                   ...
|
->05.01% (73,728B) 0x4A44F2B: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
| ->05.01% (73,728B) 0x4004A6B: call_init (dl-init.c:74)
|   ...
|
->00.66% (9,683B) in 1+ places, all below ms_print's threshold (01.00%)
```

## Step 3: Read the Report

**The graph.** Up the side, the heap in MB; along the bottom, because of
`--time-unit=B`, the bytes allocated and freed so far. Each column is a
snapshot: `:` a normal one, `@` a detailed one, `#` the peak. The three teeth
are the three calls: each climbs as V0's string grows and drops back when the
call's result is destroyed. The `#` sits a row below the top: the second call
reached 1,471,536 bytes, 816 more than the recorded peak, a rise too small for
massif to record as a new peak (it needs 1%) but enough to set the top of
`ms_print`'s scale.

**The peak.** Snapshot 17 held 1,470,720 bytes: 1,469,864 that the program
asked for (`useful-heap`) and 856 more that the allocator spent on its own
headers and rounding (`extra-heap`). The tree divides the useful heap by the
call stack that allocated it, largest first. Each entry is one stack, read
down from the allocation to its callers. The lines that share one address are
one call site: the compiler inlined the standard library's functions into the
function that calls them, and massif prints each inlined function as a frame
of its own, with the file and line the debug information gives it. The join
example is compiled with `-g` in every build type, so the frames in its code
carry a line of `Join.cpp`:

| Bytes   | Allocated at                                                  | What it is                                                                                           |
| ------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 640,000 | `makeParts`, `Join.cpp:58`                                    | the input: 20,000 strings of 32 bytes each, every word short enough to live inside its string object |
| 597,158 | `joinV0`, `Join.cpp:25`, through `append`                     | the buffers `+ sep` moves the temporary into: two are live, the one `out` holds and the new one      |
| 149,295 | `joinV0`, `Join.cpp:25`, through `__str_concat` and `reserve` | the temporary `out + part`, at its exact size                                                        |
| 73,728  | libstdc++ (`???`), from `call_init`                           | allocated by the standard library while the program loads, before `main`                             |
| 9,683   | below 1%                                                      | everything else                                                                                      |

Both of `joinV0`'s entries name line 25, `out = out + part + sep;`: one line,
two allocations, told apart by the frames above it. The smaller comes from the
`operator+` that joins two strings: `out + part` reserves the exact size of the
two (`__str_concat`, `reserve`). The larger comes from the `operator+` that
adds a character: `+ sep` appends the separator to that temporary (`append`),
which is already full, so the string moves into a buffer twice its size. So at
its peak V0's own buffers are 746,453 bytes, five times the 149,813 bytes V1
allocates for the same result (the 149,812 characters and a terminating
null).

What not to conclude:

- **That the addresses or the header lines mean anything elsewhere.**
  `0x11C4D7` and the rest belong to this build, and `basic_string.h:3742` to
  this rig's standard library headers. The names travel: `joinV0`, `append`,
  `__str_concat`, `Join.cpp:25`.
- **That massif's peak is exact.** It records a peak only when the heap passes
  the last recorded one by 1% (`--peak-inaccuracy`), so its peak snapshot can
  fall a little before the true one: here the temporary is 149,295 bytes, 517
  short of its size at the last word. `Massif.JoinPeakHeap`, the page's check,
  counts what `operator new` hands out in the sizes the allocator gives back,
  and read 749,048 bytes for one call of V0 on the same input.
- **That the total is the comparison.** V0's 1,470,720 bytes against V1's
  874,064 in step 4 is 1.68 times, because the 640,000-byte input and
  libstdc++'s 73,728 bytes are in both. The code's difference is in the join's
  own entries: 746,453 bytes against 149,813.

## Step 4: Confirm the Fix

```bash
valgrind --tool=massif --time-unit=B --massif-out-file=massif.v1.out \
  ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif \
  --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV1
ms_print massif.v1.out
```

Captured output, cut as in step 2:

```
--------------------------------------------------------------------------------
Command:            ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 --gtest_filter=Massif.JoinV1
Massif arguments:   --time-unit=B --massif-out-file=massif.v1.out
ms_print arguments: massif.v1.out
--------------------------------------------------------------------------------


    KB
853.6^                                             ::::
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
     |  @:::::::::::::::::::::    #   :   :    :   :   @:                  @:
   0 +----------------------------------------------------------------------->MB
     0                                                                   2.344

Number of snapshots: 66
 Detailed snapshots: [6, 32 (peak), 39, 48, 58]
...
--------------------------------------------------------------------------------
  n        time(B)         total(B)   useful-heap(B) extra-heap(B)    stacks(B)
--------------------------------------------------------------------------------
...
 32        956,112          874,064          873,224           840            0
99.90% (873,224B) (heap allocation functions) malloc/new/new[], --alloc-fns, etc.
->73.22% (640,000B) 0x11CA73: allocate (new_allocator.h:151)
| ...
|             ->73.22% (640,000B) 0x11CA73: vernier::bench::demo::makeParts[abi:cxx11](unsigned long, unsigned int) (Join.cpp:58)
|               ->73.22% (640,000B) 0x10F3A7: Massif_JoinV1_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|                 ...
|
->17.14% (149,813B) 0x11BDCB: allocate (new_allocator.h:151)
| ->17.14% (149,813B) 0x11BDCB: allocate (allocator.h:196)
|   ->17.14% (149,813B) 0x11BDCB: allocate (alloc_traits.h:515)
|     ->17.14% (149,813B) 0x11BDCB: _S_allocate (basic_string.h:131)
|       ->17.14% (149,813B) 0x11BDCB: _M_create (basic_string.tcc:159)
|         ->17.14% (149,813B) 0x11BDCB: reserve (basic_string.tcc:315)
|           ->17.14% (149,813B) 0x11BDCB: vernier::bench::demo::joinV1(std::vector<...> const&, char) (Join.cpp:37)
|             ->17.14% (149,813B) 0x10F3B7: Massif_JoinV1_Test::TestBody() (in .../BenchDemo_11_MassifProfiler)
|               ...
|
->08.44% (73,728B) 0x4A44F2B: ??? (in /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33)
| ->08.44% (73,728B) 0x4004A6B: call_init (dl-init.c:74)
|   ...
|
->01.11% (9,683B) in 67 places, all below massif's threshold (1.00%)
```

The first step up is the input; the three teeth on top of it are the three
calls, one buffer each. The `#` sits a row below the tallest columns, as in
step 2: those are the third call's, 874,088 bytes against the recorded peak's
874,064. At the peak, `joinV1` holds 149,813 bytes from one call site, the
`reserve` on line 37 that sizes the result before anything is appended, and
the input is the largest owner of the heap, 73% of it. The heap V1's call owns
is a fifth of V0's.

## Profiling Your Own Code

The same steps work on any function you can call from a test:

1. Give the code a test of its own, shaped like the demo's: build the input
   once, check the answer, then `PERF_GUARD`, `perf.warmup` and
   `perf.throughputLoop` around the call. One version per test lets
   `--gtest_filter` select one at a time.
2. Run that test under massif for one measured call, and print the file:

   ```bash
   valgrind --tool=massif --time-unit=B --massif-out-file=massif.out \
     ./build/bin/ptests/<YourBenchmark> --profile massif \
     --cycles 1 --repeats 1 --gtest_filter=<Suite.Case>
   ms_print massif.out
   ```

3. Find the peak in the `Detailed snapshots` line and read its tree: the
   largest entries first, each down to the first frame in your own code.
   Massif names that frame's file and line only where the code carries debug
   information. The join example is compiled with `-g` for that reason, in
   every build type; for its code `-g` changes no generated instruction
   ([walkthrough 07](07_CALLGRIND_PROFILER.md#what-is-callgrind) compared the
   object files). A Release build of your own target can do the same with
   `target_compile_options(<target> PRIVATE -g)`.
4. Change the code, profile again, and compare the entries your code owns,
   not the total: the input and the libraries' allocations are in both runs.

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
853.6^                                                                ::::::
     |                                                   ::::::#:::::::
     |                                                   :     #:     :
     |                                                   :     #:     :
     |              ::::::::::::::::::::::::::::::::::::::     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |              :                                    :     #:     :     @
     |           :@@:                                    :     #:     :     @@
   0 +----------------------------------------------------------------------->Mi
     0                                                                   13.35

Number of snapshots: 76
 Detailed snapshots: [13, 31, 39, 40, 50 (peak), 60, 70]
```

Building the 20,000 words is the long flat stretch: the vector of words is
allocated first, and every word fits inside its string object, so the heap
stays level while they are made. V1's three calls come after it, at the right,
and they do not read as three. Massif keeps at most 100 snapshots and thins
them out as the run goes on: none of those it kept fell between the first two
calls, so those two draw as one block, and the gap before the third is
narrower than a column, so the third draws as a step up from them. On the
byte axis the clock moves only when memory is allocated or freed, and every
call allocates and frees the same 149,813 bytes, so each call and each gap
between calls get about the same width and the calls come out as three teeth.
V0's graph shows its three calls either way, because they are most of what the
program does, in bytes and in instructions alike.

## Letting bench run Do the Wrap

```bash
bench run ./build/bin/ptests/BenchDemo_11_MassifProfiler --profile massif \
  --cycles 1 --repeats 1 -- --gtest_filter=Massif.JoinV0
ms_print bench-out/BenchDemo_11_MassifProfiler.massif/massif.out
```

Captured output of the first command:

```
Running: valgrind --tool=massif --massif-out-file=bench-out/BenchDemo_11_MassifProfiler.massif/massif.out ./build/bin/ptests/BenchDemo_11_MassifProfiler --cycles 1 --repeats 1 --profile massif --gtest_filter=Massif.JoinV0
==172332== Massif, a heap profiler
==172332== Copyright (C) 2003-2024, and GNU GPL'd, by Nicholas Nethercote et al.
==172332== Using Valgrind-3.24.0 and LibVEX; rerun with -h for copyright info
==172332== Command: ./build/bin/ptests/BenchDemo_11_MassifProfiler --cycles 1 --repeats 1 --profile massif --gtest_filter=Massif.JoinV0
==172332==
Note: Google Test filter = Massif.JoinV0
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Massif
[ RUN      ] Massif.JoinV0
[Massif.JoinV0]  1013325.000 us/call  CV=0.0%  ~1 calls/s  (p10=1013325.000 p90=1013325.000 sd=0.000)
[       OK ] Massif.JoinV0 (3189 ms)
[----------] 1 test from Massif (3194 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (3234 ms total)
[  PASSED  ] 1 test.
==172332==
```

`bench run` builds the same wrap and prints it on its first line, without
`--time-unit=B`, which it has no option to pass: its graph's axis is
instructions. The peak's owners and their byte counts are the ones in step 2;
the total is 16 bytes more.
The profile goes to `bench-out/BenchDemo_11_MassifProfiler.massif/massif.out`,
one file for the binary, not one per test, and no per-test folder is created
([Troubleshooting](../../docs/TROUBLESHOOTING.md#profile-artifacts-land-in-cwd-move-immediately)
has the rule for every wrapped tool). A second `bench run` of the same binary
writes the same file: run V1 after V0 and only V1's profile is left. A
different `--profile-output-dir` gives a run a file of its own.

## What Should Reproduce

| Reading                                        | On this rig                                                                                                | Elsewhere                                                                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| heap V0's call holds at its peak, against V1's | 5.0x (`Massif.JoinPeakHeap`); 746,453 against 149,813 bytes in massif's trees                              | 5.0x with libstdc++, also on an x86 laptop; 4.0x with libc++ 14                                       |
| who holds V0's peak                            | the input 640,000 bytes; `joinV0` 597,158 through `append` and 149,295 through `reserve`; libstdc++ 73,728 | the same owners and the same line of `Join.cpp`; the bytes depend on the standard library             |
| total heap at the peak                         | 1,470,720 against 874,064 bytes, 1.68x                                                                     | depends on the size of `std::string` (32 bytes here) and on what else the process holds               |
| byte counts from run to run                    | the same owners and lines in every profile; the bytes within massif's 1% peak inaccuracy (step 3)          | the same owners and lines; the bytes within the same inaccuracy                                       |
| V0 and V1 per call                             | 387.5 ms and 0.47 ms in step 1's run (826x); the spread is below                                           | will differ: V0's time grows with the square of the word count, and memory bandwidth decides the rest |

The byte counts are the reading this page checks. The check counts exactly,
and its counts repeat: every run of it on this rig printed the same
`Massif.JoinPeakHeap` line. Massif's peak is approximate (step 3). The four
profiles of V0 behind this page held the same entries for the join and the
input, with totals within 16 bytes, but that describes those runs, not a
bound: another run can take its peak snapshot at a slightly different moment
of the call, and its entries then differ within massif's 1%, about 14,700 of
V0's 1,470,720 bytes, with the same owners and lines.

The times vary more. Over the twelve step 1 runs behind this page, V0's median
ranged from 384.6 to 395.8 ms and V1's from 469.4 to 513.7 us, and V0 was 764 to
834 times slower than V1. V1's median is the noisiest reading here: one of the
twelve runs read 513.7 us, 7.5% above the next highest, with a CV of 0.4%,
steady within itself and apart from the others. That range describes those
twelve runs. It is not a bound a run has to meet, and another run of the same
command can land outside it. The times give the context; the byte counts are the
check.

## If It Does Not Match

- **`Massif.JoinV0.massif/` is empty.** It is meant to be: the profile is the
  file `--massif-out-file` names (see **In Vernier**).
- **V1's three calls do not read as three.** The run left out
  `--time-unit=B`, or came from `bench run`, which does not pass it
  ([Why --time-unit=B](#why---time-unitb)).
- **V0 holds about four times V1's heap, not five.** The build uses another
  standard library; libc++ regrows strings differently. The check's floor is
  three.
- **Massif's byte counts differ a little from this page's.** Its peak
  snapshot is recorded only when the heap passes the last one recorded by 1%
  (step 3), so two runs of the same binary can take it at slightly different
  moments of the same call. Compare the owners, the lines and the ratio of the
  join's entries to V1's; `Massif.JoinPeakHeap` counts exactly, and its line
  should not change on this rig.
- **A CV well above a few percent in step 1.** Over the twelve runs behind this
  page, with the governor pinned, V0's CV ranged from 0.4% to 3.7% and V1's
  from 0.4% to 2.2%; V0 runs once per repeat, so each of its repeats is one
  call. Well above that, something else was running on the board, or had just
  finished. The rig document's measurement section has the governor and
  `taskset` recipe; `vcgencmd get_throttled` should read `0x0` before and
  after.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/14_massif_profiler.csv run.csv
```

Captured with the `bench` CLI built from this tree, which reported
`bench 1.0.3`, on the `run.csv` step 1 wrote:

```
Test               Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
-------------  ------------  ------------  ----------  --------  --------  --------  ------------
Massif.JoinV0  390966.00000  387532.00000  -3434.00000     -0.9%      0.7%      0.5%  neutral
Massif.JoinV1     473.45200     469.38400    -4.06800     -0.9%      2.2%      1.9%  neutral

  2 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference is the same binary on the same board, captured immediately
before step 1's run. Both medians moved by 0.9%, inside the 5% threshold the
labels are drawn at, so both rows are `neutral`. As the note under the table
says, a label describes the difference between these two runs; it is not a
significance test, and `Base CV` and `Cand CV` are each run's own spread. The
CSVs' `hostname` column records the name visible to the process that captured
each: this reference was captured in a UTS namespace named `pi4` on the same
rig, so it records `pi4`, while the rig's ordinary captures, step 1's `run.csv`
among them, record `raspberrypi`. The CSV carries times, not heap sizes: the
heap is what `Massif.JoinPeakHeap` checks.

## What Keeps This Page True

- `Massif.JoinPeakHeap` fails unless V0's call holds more than three times
  the heap V1's call holds at its peak, and unless `joinedSize()` holds none.
  It is a test program of its own beside the join example,
  [`JoinPeakHeap_uTest.cpp`](../examples/join/utst/JoinPeakHeap_uTest.cpp),
  built as `TestDemoJoinPeakHeap`. To count, it replaces the global
  `operator new` and `operator delete` with versions that add up the bytes
  handed out and not yet taken back, and it records the most held at once
  while one call runs. The demo, which massif profiles, replaces neither. Nor
  does a build with the thread sanitizer (`-DSANITIZER=tsan`): its runtime
  defines `operator new` itself, and clang's cannot link beside a
  replacement, so there the check counts nothing and skips, saying why. The
  check counts bytes, not time, so a busy machine does not change its answer,
  and it is registered with `ctest` under the `demo` and `massif` labels.
  Run on its own:

  ```bash
  ./build/bin/tests/TestDemoJoinPeakHeap
  ```

  Captured on this rig, with the build's path shortened to `...`:

  ```
  Running main() from .../gtest_main.cc
  [==========] Running 1 test from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 1 test from Massif
  [ RUN      ] Massif.JoinPeakHeap
  [Massif.JoinPeakHeap]  joined 149812 bytes  V0 holds 749048  V1 holds 149816  5.0x
  [       OK ] Massif.JoinPeakHeap (1096 ms)
  [----------] 1 test from Massif (1096 ms total)

  [----------] Global test environment tear-down
  [==========] 1 test from 1 test suite ran. (1096 ms total)
  [  PASSED  ] 1 test.
  ```

  With `joinV0` given `joinV1`'s body, V0 holds 149,816 bytes, the same as
  V1, and the check fails at 1.0x.

- The example's unit tests hold every version to the same answers, under the
  `demo` label. An ordinary test run includes them and the check, and every
  test either label selects should pass:

  ```bash
  ctest --test-dir build -L massif     # the check alone
  ctest --test-dir build -L demo       # with the example's tests and the other demos' checks
  ```

The demo's two timing tests are not registered: what they measure belongs to
the machine they run on. This repository has no continuous-integration lane on
the reference board, so before a release the page's commands are run on the
rig by hand, and the page and its reference CSV are re-captured when what they
show changes.

## See Also

- [Demo 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example, measured
  for time
- [Demo 21: heaptrack](21_HEAPTRACK_PROFILER.md) -- which call sites allocate,
  and how often
- [Demo 15: memcheck](15_MEMCHECK_PROFILER.md) -- leaks and invalid accesses
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
