# Demo 21: heaptrack for Allocation Profiling

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-24 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; heaptrack 1.5.0

## Overview

A timer says how long a function takes. heaptrack says where it asks the
allocator for memory, and how often. This walkthrough profiles the `join`
example from [Demo 01](01_BASIC_WORKFLOW.md): on this rig `joinV0` takes about
50 times as long as `joinV1`, and heaptrack shows how differently the two use
the allocator. `joinV0` calls it 1,995 times per call, `joinV1` once.

## What is heaptrack?

heaptrack is a heap profiler for Linux. It starts your program with a library
preloaded that intercepts the C allocation functions (`malloc`, `free`,
`realloc` and their relatives), records every call with the call stack that
made it, and writes the record to a file. `heaptrack_print`, or the graphical
`heaptrack_gui`, reads the file after the program exits and ranks the places
that allocate: by number of calls, by share of the peak, by temporary
allocations, and by leaks.

Its cost is paid per allocation, because each one is recorded with its call
stack. Code that seldom allocates runs close to full speed; code that
allocates constantly slows down in proportion. In the run below, `joinV0`
took 4,038 us per call under heaptrack against 966 us without it, 4.2 times
as long, and `joinV1` 29.3 us against 19.0 us, 1.5 times. Time the code
without heaptrack, and use heaptrack to count.

It sees only what goes through the malloc family. Memory mapped directly with
`mmap` never reaches it, and neither do C++ allocations served by an allocator
that replaces `operator new` without calling `malloc`, as tcmalloc does: see
[If It Does Not Match](#if-it-does-not-match).

**In Vernier:** `bench run <binary> --profile heaptrack` starts the benchmark
under heaptrack for you, as `heaptrack -o <dir>/<binary>.heaptrack/run
<binary> ...`, where `<dir>` is `--profile-output-dir` (`bench-out` when you
give none). heaptrack writes one recording for the whole process when it
exits: `run.zst` when heaptrack was built with zstd support and the `zstd`
program is installed, as on this rig, and `run.gz` otherwise. Because the
recording covers the whole process, profile one test per run with
`--gtest_filter`.

**Needs:** the `heaptrack` package, which the rig document's
[one-time setup](../../docs/rigs/RIG_PI4.md#2-one-time-setup) installs; no
privileges, since heaptrack runs as your user; and a build without tcmalloc,
which is the default.

## The Example

`joinV0` is the version people write first:

```cpp
std::string joinV0(const std::vector<std::string>& parts, char sep) {
  std::string out;
  for (const std::string& part : parts) {
    // Two temporaries per part, and a copy of everything joined so far
    out = out + part + sep;
  }
  return out;
}
```

`joinV1` measures the answer first, allocates once, and appends in place:

```cpp
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
What differs is how often they ask for memory. `out + part` builds a new
string holding everything joined so far plus the part, and adding `sep` then
needs a bigger buffer again: two allocations for every part, apart from the
first few, whose joined text still fits inside the string object itself.
`joinV1` asks once, for the final size.

The demo measures each version in its own test, over the same 1,000 words as
Demo 01:

```cpp
PERF_THROUGHPUT(Heaptrack, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}
```

Two things about this test matter under heaptrack. It calls only its own
version -- `joinedSize` checks the answer without calling either one -- so a
recording of `Heaptrack.JoinV0` holds none of `joinV1`'s allocations, and the
other way round. And it calls its version a known number of times: once for
the check, once to warm up, then `--cycles` times per repeat. With
`--cycles 100 --repeats 1` that is 102 calls, the number the counts below
divide by.

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from Heaptrack
[ RUN      ] Heaptrack.JoinV0
[target-time] 50.000 ms -> cycles=52 (calibrated 951.0000 us/call, batch of 2)
[Heaptrack.JoinV0]  966.067 us/call  CV=0.1%  ~1.0K calls/s  (p10=965.313 p90=968.242 sd=1.434)
[       OK ] Heaptrack.JoinV0 (511 ms)
[ RUN      ] Heaptrack.JoinV1
[target-time] 50.000 ms -> cycles=2642 (calibrated 18.9219 us/call, batch of 64)
[Heaptrack.JoinV1]  19.048 us/call  CV=0.5%  ~52.5K calls/s  (p10=18.847 p90=19.104 sd=0.103)
[       OK ] Heaptrack.JoinV1 (504 ms)
[----------] 2 tests from Heaptrack (1016 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (1016 ms total)
[  PASSED  ] 2 tests.

==================================================================
Test               Median (us)     CV%     Calls/s  Status
------------------------------------------------------------------
Heaptrack.JoinV0       966.067    0.1%        1.0K  OK
Heaptrack.JoinV1        19.048    0.5%       52.5K  OK
------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

This is Demo 01's measurement under this demo's test names: `joinV0` takes
966.1 us per call and `joinV1` 19.0 us, 51 times less, with the spread inside
the run (`CV`) at 0.1% and 0.5%. Nothing in these lines says why. The next
step asks the allocator.

## Step 2: Profile

Record `joinV0` alone, with a fixed number of calls:

```bash
bench run ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler --profile heaptrack \
  --cycles 100 --repeats 1 --profile-output-dir heaptrack-v0 \
  -- --gtest_filter='Heaptrack.JoinV0'
```

Captured output, with the two lines that name the recording's absolute path
cut:

```
Running: heaptrack -o heaptrack-v0/BenchDemo_15_HeaptrackProfiler.heaptrack/run ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler --cycles 100 --repeats 1 --profile heaptrack --profile-output-dir heaptrack-v0 --gtest_filter=Heaptrack.JoinV0
...
starting application, this might take some time...
Note: Google Test filter = Heaptrack.JoinV0
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Heaptrack
[ RUN      ] Heaptrack.JoinV0
[heaptrack] wrapping detected; heap profile will be written at process
[heaptrack] exit. Artifact directory: heaptrack-v0/BenchDemo_15_HeaptrackProfiler.heaptrack
[Heaptrack.JoinV0]  4038.030 us/call  CV=0.0%  ~248 calls/s  (p10=4038.030 p90=4038.030 sd=0.000)
[       OK ] Heaptrack.JoinV0 (641 ms)
[----------] 1 test from Heaptrack (641 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (644 ms total)
[  PASSED  ] 1 test.
heaptrack stats:
        allocations:            203769
        leaked allocations:     37
        temporary allocations:  23
Heaptrack finished! Now run the following to investigate the data:

  ...
```

The `Running:` line is the command `bench run` built. The benchmark's
heaptrack backend only reports where the recording goes; heaptrack itself
prints its totals as the process exits. The recording is the only file the
run writes:

```
heaptrack-v0/BenchDemo_15_HeaptrackProfiler.heaptrack/run.zst
```

`allocations: 203769` counts the whole process: the 102 calls of `joinV0`,
and whatever GoogleTest and the harness allocated around them. The time on
the result line is one repeat under heaptrack, the 4.2 times from above; it
is not a measurement of `joinV0`.

## Step 3: Read the Report

```bash
heaptrack_print --print-peaks 0 --print-temporary 0 --peak-limit 3 --sub-peak-limit 0 \
  heaptrack-v0/BenchDemo_15_HeaptrackProfiler.heaptrack/run.*
```

The options keep the report to one ranking: `--print-peaks 0` and
`--print-temporary 0` drop the rankings by peak memory and by temporary
allocations, `--peak-limit 3` keeps the top three places, and
`--sub-peak-limit 0` leaves out the call stacks that lead to each. The glob
matches the one recording in the folder: `run.zst` here, `run.gz` where
heaptrack writes that instead. Captured output, with the lines naming the
binary's absolute path cut:

```
reading file "heaptrack-v0/BenchDemo_15_HeaptrackProfiler.heaptrack/run.zst" - please wait, this might take some time...
Debuggee command was: ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler --cycles 100 --repeats 1 --profile heaptrack --profile-output-dir heaptrack-v0 --gtest_filter=Heaptrack.JoinV0
finished reading file, now analyzing data:

MOST CALLS TO ALLOCATION FUNCTIONS
203490 calls to allocation functions with 37.44K peak consumption from
vernier::bench::demo::joinV0(std::vector<> const&, char)
  ...
  and 203490 from 6 other places

55 calls to allocation functions with 0B peak consumption from
testing::Message::Message()
  ...
  and 55 from 54 other places

45 calls to allocation functions with 162B peak consumption from
testing::internal::StringStreamToString(std::__cxx11::basic_stringstream<>*)
  ...
  and 45 from 44 other places


total runtime: 0.66s.
calls to allocation functions: 203769 (307343/s)
temporary memory allocations: 24 (36/s)
peak heap memory consumption: 152.99K
peak RSS (including heaptrack overhead): 5.55M
total memory leaked: 6.97K
suppressed leaks: 1.63K
```

Each entry is a place that calls an allocation function, with the number of
calls it made. The first is `joinV0` itself: 203,490 calls in 102 calls of
`joinV0`, which is 1,995 per call. The next two are GoogleTest's own
bookkeeping, 55 and 45 calls, three orders of magnitude down; the process
total at the bottom, 203,769, is those three and the rest of the program
together.

"From 6 other places" means heaptrack merged six call stacks into the entry.
They are the two allocations inside `joinV0` -- the temporary built for
`out + part` and the bigger buffer `+ sep` needs -- reached from the three
places the test calls it: the check, the warmup and the measured loop. With
this rig's compiler both allocation calls are made from `joinV0`'s own code,
so both are reported under its name; see
[What Should Reproduce](#what-should-reproduce) for how another build splits
them.

What to conclude: `joinV0` asks the allocator for memory about twice for
every part it appends, where `joinV1` asks once in all, and allocating less
often is the change that separates them. What not to conclude: that
allocation is all of `joinV0`'s time. Each new buffer is also filled by
copying everything joined so far, and heaptrack does not measure that
copying; a CPU profiler does.

## Step 4: Confirm the Fix

Record `joinV1` the same way:

```bash
bench run ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler --profile heaptrack \
  --cycles 100 --repeats 1 --profile-output-dir heaptrack-v1 \
  -- --gtest_filter='Heaptrack.JoinV1'
heaptrack_print --print-peaks 0 --print-temporary 0 --peak-limit 3 --sub-peak-limit 0 \
  heaptrack-v1/BenchDemo_15_HeaptrackProfiler.heaptrack/run.*
```

Captured output of the run, trimmed to the command, the result and
heaptrack's totals:

```
Running: heaptrack -o heaptrack-v1/BenchDemo_15_HeaptrackProfiler.heaptrack/run ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler --cycles 100 --repeats 1 --profile heaptrack --profile-output-dir heaptrack-v1 --gtest_filter=Heaptrack.JoinV1
...
[Heaptrack.JoinV1]  29.310 us/call  CV=0.0%  ~34.1K calls/s  (p10=29.310 p90=29.310 sd=0.000)
...
heaptrack stats:
        allocations:            381
        leaked allocations:     37
        temporary allocations:  125
...
```

and of the report, with the binary's path cut:

```
reading file "heaptrack-v1/BenchDemo_15_HeaptrackProfiler.heaptrack/run.zst" - please wait, this might take some time...
Debuggee command was: ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler --cycles 100 --repeats 1 --profile heaptrack --profile-output-dir heaptrack-v1 --gtest_filter=Heaptrack.JoinV1
finished reading file, now analyzing data:

MOST CALLS TO ALLOCATION FUNCTIONS
102 calls to allocation functions with 7.49K peak consumption from
vernier::bench::demo::joinV1(std::vector<> const&, char)
  ...
  and 102 from 3 other places

55 calls to allocation functions with 0B peak consumption from
testing::Message::Message()
  ...
  and 55 from 54 other places

45 calls to allocation functions with 162B peak consumption from
testing::internal::StringStreamToString(std::__cxx11::basic_stringstream<>*)
  ...
  and 45 from 44 other places


total runtime: 0.05s.
calls to allocation functions: 381 (8106/s)
temporary memory allocations: 126 (2680/s)
peak heap memory consumption: 123.04K
peak RSS (including heaptrack overhead): 5.47M
total memory leaked: 6.97K
suppressed leaks: 1.63K
```

`joinV1` made 102 calls to allocation functions in 102 calls: one each, the
`reserve`. It still ranks first, but now it sits next to GoogleTest's 55 and
45, and the whole process made 381 allocations against `joinV0`'s 203,769. The
reading changed the way the timing did, and for the reason the report named.

## What Should Reproduce

| Reading                        | On this rig                                                           | Elsewhere                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| allocations per call, `joinV0` | 1,995, all reported under `joinV0`                                    | 1,995 with GNU libstdc++, in optimized and unoptimized builds; on x86-64 with GCC 11, heaptrack reports them as 997 under `joinV0` and 998 under `_M_mutate` |
| allocations per call, `joinV1` | 1                                                                     | 1                                                                                                                                                            |
| `joinV0` / `joinV1` time       | 51x here; 44x to 51x over seven runs                                  | tens of times in an optimized build                                                                                                                          |
| time under heaptrack           | `joinV0` 4.2x, `joinV1` 1.5x its time without heaptrack               | grows with the allocation rate; depends on the machine                                                                                                       |
| absolute times                 | 966.1 and 19.0 us/call (935 to 1036 and 19.0 to 21.3 over seven runs) | will differ                                                                                                                                                  |

The allocation counts are the readings to carry elsewhere: they depend on the
code and the standard library, not on the machine. The same counts, 1,995 and
1 per call at 1,000 parts, were measured with GCC 11.4 and Clang 21.1.8 on an
x86-64 laptop, at `-O0` and `-O3`, and with GCC 14.2 on this rig at both. The
names the report files them under depend on what the compiler inlined: the
laptop's GCC 11 build kept the string's growth path out of line, so the report
there has two entries, 997 per call under `joinV0` and 998 under
`std::__cxx11::basic_string<>::_M_mutate`, adding to the same 1,995.

## If It Does Not Match

- **Both versions report a handful of allocations.** tcmalloc is loaded:
  its own `operator new` serves the C++ allocations without calling `malloc`,
  so heaptrack never sees them. On this rig, a build configured with
  `-DVERNIER_LINK_TCMALLOC=ON` records 19 allocations for either version's
  whole run, and the benchmark prints a warning before it measures. The
  doctor names it. `bench doctor ./build/bin/ptests/BenchDemo_15_HeaptrackProfiler`
  prints `[OK]   heaptrack  heaptrack available` for the default build, which
  does not link tcmalloc, and this for the build configured with the option:

  ```
    [WARN] heaptrack  heaptrack available, but libtcmalloc is loaded: C++ allocations will be missing
               Use a build without tcmalloc (-DVERNIER_LINK_TCMALLOC=OFF, the default) and do not preload it.
  ```

  Reconfigure without the option, and do not preload tcmalloc.

- **`bench run` stops before the test starts:**

  ```
  Error: tool not found: 'heaptrack' is not on PATH; --profile heaptrack runs the benchmark under it. Install heaptrack, or run `bench doctor` to see which profilers this machine can use
  ```

  heaptrack is not installed, or not on `PATH`; the doctor reports
  `[FAIL] heaptrack  heaptrack binary not found on PATH`. Install the package
  from the rig document's one-time setup.

- **The counts do not divide into a whole number per call.** The run made a
  different number of calls: one for the check, the warmup (one call by
  default), then `--cycles` times `--repeats`. `--target-time` chooses
  `--cycles` by timing the call, and under heaptrack that timing is
  heaptrack's; profile with a fixed `--cycles`.

## Check Against the Reference

A capture from this rig is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/21_heaptrack_profiler.csv run1.csv
```

Output for the step-1 run above, from the `bench` CLI built from this tree:

```
Test                  Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
----------------  ------------  ------------  ----------  --------  --------  --------  ------------
Heaptrack.JoinV0     947.47100     966.06700   +18.59600     +2.0%      0.1%      0.1%  neutral
Heaptrack.JoinV1      21.01100      19.04840    -1.96260     -9.3%      2.1%      0.5%  IMPROVEMENT

  1 improvement(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by the same binary on this board 1 hour 41
minutes before that run. As the note under the table says, a label only
compares the median change with the threshold, so `joinV1`'s row says
`IMPROVEMENT` at -9.3% although nothing changed. Against the reference, the
session's six other runs put `joinV1` between 9.3% below it and 1.2% above,
and `joinV0` between 1.4% below and 9.3% above: read the medians and the CV
columns yourself. The reference holds times, not allocation counts; the
counts are checked by the example's unit tests, below.

## What Keeps This Page True

Two things check what this page shows, and both fail loudly:

- The example's unit tests count the allocations each version makes per call,
  by replacing `operator new` in their test binary with one that counts:
  `JoinAllocationSizesTest.V1AllocatesOnce` fails unless `joinV1` makes
  exactly one at 10, 100, 1,000 and 10,000 parts, and
  `JoinAllocationTest.V0AllocatesFarMoreOftenThanV1` fails unless `joinV0`, at
  1,000 parts, makes at least 500 times as many as `joinV1`. They are
  registered with `ctest`, so ordinary CI runs them, with the other examples'
  tests under the same label:

  ```bash
  ctest --test-dir build -L demo
  ```

  ```
  100% tests passed, 0 tests failed out of 15
  ```

  The check lives in the example's tests, not in the demo, because a counting
  `operator new` and heaptrack each disturb the other. The counter hands
  every C++ allocation to `malloc`, so in the demo heaptrack would see them
  even in a build with tcmalloc, and the demo would stop behaving like a
  program of your own; and under heaptrack, the counter also counts
  heaptrack's own bookkeeping.

- `bench doctor` reports heaptrack as a warning whenever tcmalloc is loaded
  into the benchmark, and `bench doctor --require heaptrack` then fails. The
  bench library's unit tests check that in a build configured with tcmalloc,
  and check that a default build reports heaptrack as ready.

This repository has no continuous-integration lane on the reference board, so
nothing runs the heaptrack steps automatically. Before a release they are run
on the rig by hand, with the commands above, and the reference CSV is
re-captured when the numbers move.

## See Also

- [Demo 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example, timed
- [Demo 14: Massif](14_MASSIF_PROFILER.md) -- heap size over time, where
  heaptrack counts calls
- [Demo 22: jemalloc](22_JEMALLOC_PROFILER.md) -- sampled allocation profiling
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
