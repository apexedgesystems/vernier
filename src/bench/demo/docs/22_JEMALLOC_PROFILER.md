# Demo 22: jemalloc's Heap Profiler -- Sampled Allocations, by Function

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md), with a jemalloc
built with profiling (the rig document's
[one-time setup](../../docs/rigs/RIG_PI4.md#2-one-time-setup) builds it)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-10-17 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`; jemalloc 5.3.0 built from the project's release with
`--enable-prof`; `jeprof` from Debian's `libjemalloc-dev` 5.3.0-3. The
reference CSV in [Check Against the Reference](#check-against-the-reference)
is the first step-1 run of the same session.

## Overview

A timer says how long a function takes. jemalloc's heap profiler says which
functions the allocated bytes come from, by sampling allocations inside the
allocator while the program runs. This walkthrough profiles the `join` example
from [Demo 01](01_BASIC_WORKFLOW.md): on this rig `joinV0` takes about 46 times
as long as `joinV1`, and jemalloc puts all of the bytes `joinV0`'s run
allocates under `joinV0`, about 11.8 MB per call, and none under `joinV1`.

## What is jemalloc's heap profiler?

jemalloc is a `malloc` implementation. Built with profiling (configure
`--enable-prof`), it can record a sample of the allocations it serves: on
average one per 512 KiB allocated, with the call stack that asked. With
`MALLOC_CONF=prof:true` it samples; with `prof_accum:true` it keeps, for each
stack, the bytes allocated over the whole run, not only those still in use;
with `prof_final:true` it writes all of it to a file when the process exits.
`jeprof`, a script that ships with jemalloc, reads that file and estimates, for
each function, the bytes allocated under it.

It samples, so its numbers are estimates: a function that allocates little
may take no sample at all, and the totals move a few percent between runs.
Its cost is paid per sample, on the allocations that take one, so it grows
with the bytes a program allocates per second; [What Sampling
Costs](#what-sampling-costs) measures it on this example. For exact counts of
every allocation, use [heaptrack](21_HEAPTRACK_PROFILER.md).

**In Vernier:** `bench run <binary> --profile jemalloc` starts the benchmark
with jemalloc preloaded, `LD_PRELOAD=libjemalloc.so.2`, and
`MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:<dir>/<binary>.jemalloc/jeprof`,
where `<dir>` is `--profile-output-dir` (`bench-out` when you give none).
jemalloc writes one dump for the whole process when it exits,
`jeprof.<pid>.<n>.f.heap`; `bench run` then checks that the dump is the
benchmark's own and prints the `jeprof` command that reads it. Because the dump
covers the whole process, profile one test per run with `--gtest_filter`.

**Needs:** a jemalloc built with profiling, which neither Debian's nor
Ubuntu's arm64 `libjemalloc2` is (Debian's packaging enables profiling on
amd64 and i386 only): this rig's one-time setup builds jemalloc 5.3.0 into a
folder of your own, and `LD_LIBRARY_PATH` selects it, since the loader looks
there first. Reading the dump needs `jeprof` (Debian's and Ubuntu's
`libjemalloc-dev`) and binutils, which `jeprof` names functions with; no
privileges. The doctor checks the library the loader would preload:

```bash
export JEMALLOC_PROF=<the folder the one-time setup installed into>
LD_LIBRARY_PATH=$JEMALLOC_PROF/lib bench doctor ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  --profile jemalloc
```

prints, in its selected row, with the folder shortened (`...`):

```
  [OK]   jemalloc   .../jemalloc-5.3.0-prof/lib/libjemalloc.so.2 profiles: preloaded with bench run's options, it wrote its final heap dump at exit (probe: /bin/cat /proc/self/maps); analyzer /usr/bin/jeprof
```

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
`out + part` builds a new string holding everything joined so far plus the
part, and adding `sep` needs a bigger buffer again: two allocations for every
part, each the size of the text joined so far, so the bytes `joinV0` allocates
grow with the square of the number of parts. `joinV1` allocates once, for the
final size.

The demo measures each version in its own test, over the same 1,000 words as
Demo 01:

```cpp
PERF_THROUGHPUT(Jemalloc, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}
```

Each test calls only its own version, so a dump of `Jemalloc.JoinV0` holds none
of `joinV1`'s allocations, and it calls it a known number of times: once for
the check, once to warm up, then `--cycles` times per repeat. With
`--cycles 100 --repeats 1` that is 102 calls.

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from Jemalloc
[ RUN      ] Jemalloc.JoinV0
[target-time] 50.000 ms -> cycles=54 (calibrated 922.0000 us/call, batch of 2)
[Jemalloc.JoinV0]  925.407 us/call  CV=0.1%  ~1.1K calls/s  (p10=925.331 p90=926.435 sd=0.583)
[       OK ] Jemalloc.JoinV0 (517 ms)
[ RUN      ] Jemalloc.JoinV1
[target-time] 50.000 ms -> cycles=2519 (calibrated 19.8438 us/call, batch of 64)
[Jemalloc.JoinV1]  19.745 us/call  CV=0.3%  ~50.6K calls/s  (p10=19.722 p90=19.861 sd=0.057)
[       OK ] Jemalloc.JoinV1 (500 ms)
[----------] 2 tests from Jemalloc (1018 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (1018 ms total)
[  PASSED  ] 2 tests.

=================================================================
Test              Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------
Jemalloc.JoinV0       925.407    0.1%        1.1K  OK
Jemalloc.JoinV1        19.745    0.3%       50.6K  OK
-----------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

This is Demo 01's measurement under this demo's test names, with the C
library's allocator: `joinV0` takes 925.4 us per call and `joinV1` 19.7 us, 47
times less. Over the session's seven runs `joinV0` took 925 to 965 us and
`joinV1` 19.7 to 21.2 us, 45 to 47 times less. Nothing in these lines says
where the difference goes. The next step asks the allocator.

## Step 2: Profile

Profile `joinV0` alone, with a fixed number of calls, the profiling build
selected:

```bash
LD_LIBRARY_PATH=$JEMALLOC_PROF/lib bench run ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  --profile jemalloc --cycles 100 --repeats 1 --profile-output-dir jemalloc-v0 \
  -- --gtest_filter='Jemalloc.JoinV0'
```

Captured output:

```
Running: LD_PRELOAD=libjemalloc.so.2 MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof ./build/bin/ptests/BenchDemo_16_JemallocProfiler --cycles 100 --repeats 1 --profile jemalloc --profile-output-dir jemalloc-v0 --gtest_filter=Jemalloc.JoinV0
Note: Google Test filter = Jemalloc.JoinV0
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Jemalloc
[ RUN      ] Jemalloc.JoinV0
[jemalloc] jemalloc profiles this process and writes its heap dump when the
[jemalloc] process exits, as jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35135.<n>.f.heap.
[jemalloc] bench run checks that dump after the exit.
[Jemalloc.JoinV0]  1468.610 us/call  CV=0.0%  ~681 calls/s  (p10=1468.610 p90=1468.610 sd=0.000)
[       OK ] Jemalloc.JoinV0 (164 ms)
[----------] 1 test from Jemalloc (164 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (165 ms total)
[  PASSED  ] 1 test.
[bench] jemalloc wrote jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35135.0.f.heap (7328 bytes) when the benchmark exited: 2394 allocations sampled over the run, 1 of them in use at exit
[bench] Read it: jeprof --text --alloc_space ./build/bin/ptests/BenchDemo_16_JemallocProfiler jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35135.0.f.heap
```

The `Running:` line is the command `bench run` built: the preload and
jemalloc's options in front of the benchmark. The benchmark's jemalloc backend
asked the allocator it runs on whether it profiles, and says where jemalloc
will write its dump, when the process exits; it never says a dump was
written, because jemalloc writes it after everything else in the process has
ended. The last two lines are `bench run`'s, after the exit: it found the dump
named after the benchmark's own process (35135), checked that it is a heap
profile of the benchmark, and printed what it holds, 2,394 sampled
allocations, and the command that reads it. The dump is the only file the run
writes:

```
jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35135.0.f.heap
```

The time on the result line is one repeat under the profiler, 1.6 times
step 1's; it is not a measurement of `joinV0`.

## Step 3: Read the Report

`jeprof --text` lists functions with five columns: the bytes attributed to
the function itself (flat), its share, the running total of that share, the
bytes under the function and everything it called (cumulative), and that
share. `--alloc_space` reads the bytes allocated over the run, which
`prof_accum` keeps; `--cum` sorts by the cumulative column:

```bash
jeprof --text --alloc_space --cum ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.*.f.heap
```

Captured output, the program's own rows kept and the rest cut (`...`):

```
Using local file ./build/bin/ptests/BenchDemo_16_JemallocProfiler.
Using local file jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35135.0.f.heap.
Total: 1203.5 MB
     0.0   0.0%   0.0%   1203.5 100.0% _start
...
  1203.5 100.0% 100.0%   1203.5 100.0% prof_backtrace_impl
     0.0   0.0% 100.0%   1202.9 100.0% Jemalloc_JoinV0_Test::TestBody
...
     0.0   0.0% 100.0%   1202.9 100.0% std::operator+ (inline)
...
     0.0   0.0% 100.0%   1202.9 100.0% vernier::bench::demo::joinV0
     0.0   0.0% 100.0%   1202.9 100.0% void* fallback_impl
     0.0   0.0% 100.0%   1187.3  98.7% std::_Function_handler::_M_invoke
     0.0   0.0% 100.0%   1173.2  97.5% vernier::bench::PerfCase::measured
     0.0   0.0% 100.0%    787.8  65.5% std::__cxx11::basic_string::_M_mutate (inline)
     0.0   0.0% 100.0%    787.8  65.5% std::__cxx11::basic_string::_M_replace_aux (inline)
     0.0   0.0% 100.0%    787.8  65.5% std::__cxx11::basic_string::append (inline)
     0.0   0.0% 100.0%    415.1  34.5% std::__cxx11::basic_string::reserve (inline)
     0.0   0.0% 100.0%    415.1  34.5% std::__str_concat (inline)
     0.0   0.0% 100.0%      0.5   0.0% _dl_init
...
```

Read the cumulative column. `joinV0` holds 1,202.9 MB of the 1,203.5 MB
sampled in the process: all of it but one sample of the start-up's memory
(`_dl_init`), and 11.8 MB per call over the 102 calls. Below it, the two
allocations per part show as the two ways the standard library grows a
string: 787.8 MB (65.5%) under `append`, where `+ sep` needs a bigger buffer,
and 415.1 MB (34.5%) under `__str_concat`, the new string `out + part`
reserves. `PerfCase::measured`, the 100 measured calls, holds 1,173.2 MB; the
check and the warmup hold the rest.

The flat column does not point at `joinV0`: all of it lands on
`prof_backtrace_impl`, jemalloc's own function that takes the sample. The
profiling build keeps its debug symbols, so jemalloc's own frames are named
and counted; with a stripped library the flat column lands on whatever name
its symbols give that spot instead. Either way the program's functions appear
only in the cumulative column, which is the one to read.

The numbers are estimates from samples: five runs of these 102 calls on this
rig put between 1,164.5 and 1,218.8 MB under `joinV0`. They are also above
what `joinV0` asks for. heaptrack, which counts every allocation, counted
1,145,083,603 bytes asked for by the whole process in the same 102 calls on
this rig (1,092 of `jeprof`'s MB, which are MiB); jemalloc counts each
sampled allocation by the size class it served, which rounds the size asked
up.

Without `--alloc_space`, `jeprof` reads the bytes still in use when the dump
was written, and `joinV0`'s strings are all freed by then:

```bash
jeprof --text ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.*.f.heap
```

```
Using local file ./build/bin/ptests/BenchDemo_16_JemallocProfiler.
Using local file jemalloc-v0/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35135.0.f.heap.
Total: 0.5 MB
     0.5 100.0% 100.0%      0.5 100.0% prof_backtrace_impl
     0.0   0.0% 100.0%      0.5 100.0% _dl_init
...
```

What to conclude: the bytes `joinV0` allocates are all its own, split
between the two string operations of the one-liner, and allocating less is
what separates it from `joinV1`. What not to conclude: that the time goes to
the allocator. Each new buffer is also filled by copying everything joined so
far; a CPU profiler measures that.

## Step 4: Confirm the Fix

Profile `joinV1` the same way:

```bash
LD_LIBRARY_PATH=$JEMALLOC_PROF/lib bench run ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  --profile jemalloc --cycles 100 --repeats 1 --profile-output-dir jemalloc-v1 \
  -- --gtest_filter='Jemalloc.JoinV1'
jeprof --text --alloc_space --cum ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  jemalloc-v1/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.*.f.heap
```

Captured output of the run, trimmed to its last lines:

```
...
[Jemalloc.JoinV1]  22.900 us/call  CV=0.0%  ~43.7K calls/s  (p10=22.900 p90=22.900 sd=0.000)
...
[bench] jemalloc wrote jemalloc-v1/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35414.0.f.heap (5580 bytes) when the benchmark exited: 1 allocation sampled over the run, 1 of them in use at exit
[bench] Read it: jeprof --text --alloc_space ./build/bin/ptests/BenchDemo_16_JemallocProfiler jemalloc-v1/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35414.0.f.heap
```

and of the report:

```
Using local file ./build/bin/ptests/BenchDemo_16_JemallocProfiler.
Using local file jemalloc-v1/BenchDemo_16_JemallocProfiler.jemalloc/jeprof.35414.0.f.heap.
Total: 0.5 MB
     0.0   0.0%   0.0%      0.5 100.0% _dl_init
...
     0.5 100.0% 100.0%      0.5 100.0% prof_backtrace_impl
     0.0   0.0% 100.0%      0.5 100.0% std::__to_chars_bfloat16_t@@GLIBCXX_3.4.31
```

The run took one sample, the start-up's, and none of `joinV1`'s: its 102
calls allocate about 7.5 KB each, under 1 MB in all, where jemalloc samples
once per 512 KiB on average, so no sample, or one or two, is the ordinary
outcome (one, about 0.5 MB, in one of the check's runs on this rig). From
1,202.9 MB under `joinV0` to nothing under `joinV1`: the reading changed the
way the timing did, and for the reason the report named. (The last row names
the start-up sample's frame after the nearest symbol the C++ library exports;
it is that library's own start-up allocation, the same in every run.)

## What Sampling Costs

The same jemalloc build preloaded into the same binary, with profiling off
(`MALLOC_CONF` unset) and on (`bench run`'s options), timed as step 1 does,
five times each, alternating:

```bash
LD_LIBRARY_PATH=$JEMALLOC_PROF/lib LD_PRELOAD=libjemalloc.so.2 \
  taskset -c 3 ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  --target-time 50ms --repeats 10 --csv cost_off_1.csv
LD_LIBRARY_PATH=$JEMALLOC_PROF/lib LD_PRELOAD=libjemalloc.so.2 \
  MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:cost_on_1/jeprof \
  taskset -c 3 ./build/bin/ptests/BenchDemo_16_JemallocProfiler \
  --target-time 50ms --repeats 10 --csv cost_on_1.csv
```

| Median of five runs, us per call | Profiling off | Profiling on | On / off            |
| -------------------------------- | ------------- | ------------ | ------------------- |
| `joinV0`, 11.8 MB per call       | 841.9         | 1,458.3      | 1.73 (1.70 to 1.73) |
| `joinV1`, 7.5 KB per call        | 22.3          | 22.6         | 1.01 (0.99 to 1.08) |

The ratios in brackets are the five pairs' range. The cost follows the bytes
allocated: `joinV0` takes about 23 samples per call, which comes to about
26 us per sample on this core, while `joinV1` takes one sample every 70 calls
or so and runs within its own spread. Sampling is cheap for code that
allocates at a modest rate and costly for code that allocates megabytes per
millisecond; measure it on your own workload before leaving it on for a long
run. This is the profiling cost alone, both columns on the same jemalloc; the
C library's allocator of step 1 is another allocator and is not part of the
comparison.

## What Should Reproduce

| Reading                         | On this rig                                                        | Elsewhere                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| bytes under `joinV0`            | 100% of the run's; 1,164.5 to 1,218.8 MB in five runs of 102 calls | 100%; the bytes follow the standard library's string growth, not the machine (x86-64: 1,146.7 to 1,246.9 MB) |
| bytes under `joinV1`            | nothing, or one sample of about 0.5 MB                             | the same                                                                                                     |
| `joinV0` / `joinV1` time        | 47x here; 45x to 47x over seven runs                               | tens of times in an optimized build                                                                          |
| time with profiling, same build | `joinV0` 1.73x, `joinV1` 1.01x                                     | grows with the bytes allocated per second; depends on the core and the unwinder                              |
| absolute times                  | 925.4 and 19.7 us/call (925 to 965 and 19.7 to 21.2, seven runs)   | will differ                                                                                                  |

The share under `joinV0` and the collapse to nothing under `joinV1` are the
readings to carry elsewhere. The x86-64 figures are from a Debug build in
the project's development container, whose Ubuntu jemalloc 5.3.0 is built
with profiling.

## If It Does Not Match

- **The run fails at its start, "built without profiling".** The loader
  preloaded a jemalloc built without profiling, here this rig's own:

  ```
  <jemalloc>: Invalid conf pair: prof:true
  <jemalloc>: Invalid conf pair: prof_final:true
  <jemalloc>: Invalid conf pair: prof_accum:true
  <jemalloc>: Invalid conf pair: prof_prefix:jemalloc-stock/BenchDemo_16_JemallocProfiler.jemalloc/jeprof
  [FAIL] Profiler 'jemalloc': missing: the jemalloc this process runs on, /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 (5.3.0-0-g54eaed1d8b56b1aa528be3bdd1877e59c56fa90c), is built without profiling
     Use a jemalloc built with profiling (configure --enable-prof; Debian's and Ubuntu's libjemalloc2 have it on amd64 and i386, not on arm64), with its library folder first on LD_LIBRARY_PATH when it is not the system's; or profile the heap with --profile heaptrack.
     Nothing is collected for this request; the run will fail (exit status 4 if the tests pass).
  ...
  Error: the requested profile failed (the benchmark's report above says why); the benchmark exited with status 4
  ```

  The doctor says the same before any run
  (`bench doctor <binary> --profile jemalloc`):

  ```
    [FAIL] jemalloc   missing: /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 is built without profiling: preloaded with bench run's options, it rejects them ("<jemalloc>: Invalid conf pair: prof:true")
  ```

  Build jemalloc as the rig document's one-time setup does, and put its
  `lib` folder first on `LD_LIBRARY_PATH`. Installing `jeprof` does not help:
  it only reads dumps.

- **`bench run` stops before the benchmark starts:**

  ```
  Error: tool not found: the loader cannot preload libjemalloc.so.2 ("ERROR: ld.so: object 'libjemalloc.so.2' from LD_PRELOAD cannot be preloaded (file too short): ignored."), and --profile jemalloc runs the benchmark with it preloaded. Use a jemalloc built with profiling (configure --enable-prof; Debian's and Ubuntu's libjemalloc2 have it on amd64 and i386, not on arm64), with its library folder first on LD_LIBRARY_PATH when it is not the system's; bench doctor <binary> --profile jemalloc checks it
  ```

  The loader cannot preload what `LD_LIBRARY_PATH` and the system's folders
  give it for `libjemalloc.so.2`; the part in brackets is the loader's own
  reason. This capture put an empty file of that name first on
  `LD_LIBRARY_PATH`; with no jemalloc at all the reason reads
  `cannot open shared object file`. Nothing is created and the benchmark does
  not run. Point `LD_LIBRARY_PATH` at the profiling build's `lib` folder.

- **`jeprof` is not installed.** The run collects as usual and its last line
  says so:

  ```
  [bench] jeprof is not on PATH: reading the dump needs it (Debian and Ubuntu ship it in libjemalloc-dev); collecting it did not
  ```

  and the doctor's row stays `[OK]`, ending
  `jeprof is not on PATH (only reading the dumps needs it)`. Install
  `libjemalloc-dev` to read the dump.

- **The report's `Total` is a fraction of a MB.** `jeprof` read the bytes in
  use at exit (no `--alloc_space`), as in step 3's last block; or the run did
  not keep the bytes allocated over it. A wrap made by hand without
  `prof_accum:true` collects with a warning:

  ```
  [WARN] Profiler 'jemalloc': jemalloc profiles this process (.../jemalloc-5.3.0-prof/lib/libjemalloc.so.2 (5.3.0-0-g54eaed1d8b56b1aa528be3bdd1877e59c56fa90c)), with limits: its dumps hold the bytes in use when they are written, not those allocated over the run (prof_accum is off), which jeprof --alloc_space reads
     Set MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:./jeprof, as bench run --profile jemalloc does.
  ```

  and `jeprof --text --alloc_space` on its dump prints its two
  `Using local file` lines and nothing else. `bench run --profile jemalloc`
  sets the options for you.

## Check Against the Reference

The first step-1 run of the session is committed with the demo:

```bash
bench compare src/bench/demo/reference/pi4/22_jemalloc_profiler.csv run2.csv
```

Output for the session's second step-1 run, from the `bench` CLI built from
this tree:

```
Test                 Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
---------------  ------------  ------------  ----------  --------  --------  --------  ------------
Jemalloc.JoinV0     925.40700     960.99000   +35.58300     +3.8%      0.1%      0.0%  neutral
Jemalloc.JoinV1      19.74490      21.23500    +1.49010     +7.5%      0.3%      3.0%  REGRESSION

  1 regression(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

Nothing changed between the two runs: they are the same binary a second
apart. The reference is the session's first run, chosen so before it ran, and
it was the session's fastest for `joinV1`: the other six put `joinV1` between
5.0% and 7.5% above it, past the 5% threshold, and `joinV0` between 1.3% and
4.3% above. As the note under the table says, a label only compares the
median change with the threshold: read the medians and the CV columns
yourself. The reference holds times, not the profile; the profile is checked
by the test below.

## What Keeps This Page True

- `TestDemoJemalloc` runs `Jemalloc.JoinV0` and `Jemalloc.JoinV1` as
  `bench run --profile jemalloc` wraps them and reads the folders they leave:
  each run must leave one dump, the demo's own, and `jeprof --alloc_space`
  must put at least 90% of `JoinV0`'s sampled bytes, and at least 400 MB for
  its 102 calls, under `joinV0`, while `JoinV1`'s run puts under `joinV1` less
  than 2% of that. It is registered with `ctest` under the `demo` and
  `jemalloc` labels. On this rig, with the profiling build selected:

  ```bash
  LD_LIBRARY_PATH=$JEMALLOC_PROF/lib ctest --test-dir build -L jemalloc
  ```

  ```
  100% tests passed, 0 tests failed out of 2
  ```

  With the distribution's jemalloc it skips, quoting jemalloc's
  `Invalid conf pair: prof:true`; it also skips, saying why, without `jeprof`
  or binutils, and in a build with the address, thread or (clang's)
  undefined-behaviour sanitizer.

- The bench library's tests check the backend's decisions: a jemalloc built
  without profiling, one the loader cannot preload, a run that does not run on
  jemalloc, options that write nothing, and the doctor and the run agreeing
  for the same request.

This repository has no continuous-integration lane on the reference board.
The hosted CI's C++ job runs `TestDemoJemalloc` in the project's development
container, whose jemalloc profiles; on the rig the steps above, and the test
with the profiling build selected, are run by hand before a release, and the
reference CSV is re-captured when the numbers move.

## See Also

- [Demo 21: heaptrack](21_HEAPTRACK_PROFILER.md) -- every allocation counted,
  on the same example
- [Demo 14: Massif](14_MASSIF_PROFILER.md) -- heap size over time
- [Demo 01: basic workflow](01_BASIC_WORKFLOW.md) -- the same example, timed
- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
