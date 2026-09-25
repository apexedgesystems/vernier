# Demo 01: Basic Workflow

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-20, written for the Vernier 1.0.4 release; captured from
the development tree at project version 1.0.3, whose CLI reported `bench 1.0.2`.
The two `bench compare` outputs were produced from the CSVs that session saved,
by the CLI of a later development tree, which reports `bench 1.0.3`.

## Overview

This is the whole Vernier cycle, once, on one small function: write a
performance test, run it, read the result line, write a CSV, and compare two
CSVs. The example is `join`, which glues a list of words together with a
separator. It has two versions that return the same string, and on this rig
one of them takes 44 times longer than the other.

## What a Perf Test Is

A perf test is a GoogleTest test that measures instead of asserting facts
about values. `PERF_GUARD(perf)` gives the test a measuring harness named
after it; `perf.throughputLoop(op, label)` calls `op` a fixed number of times
per repeat, times each repeat, and prints the per-call median with its spread.
Nothing about it is special to this demo: this is how every measurement in
this repository is written.

**In Vernier:** perf tests build as `ptest` binaries under
`build/bin/ptests/` and are deliberately not registered with `ctest`, so a
measurement never runs beside a parallel test suite. You run them yourself,
with `--repeats N` for how many samples to take, `--target-time DUR` to size
each sample, and `--csv FILE` to write the results. The `bench` CLI reads
those CSVs.

**Needs:** a Release build and a machine held still. The
[rig document](../../docs/rigs/RIG_PI4.md) has the build command, the governor
and the core to pin; sections 3 and 4 are all this page assumes.

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

The difference is not a coding-style preference. `joinV0` copies everything
joined so far once per part, so its work grows with the square of the number
of parts, while `joinV1`'s grows in step with them. Both return the same
string, and the example's unit tests
([`Join_uTest.cpp`](../examples/join/utst/Join_uTest.cpp)) hold them to that
at six input sizes, so a timing difference between them can only be about how
they build the answer.

The demo measures each version in its own test, over 1,000 words:

```cpp
PERF_THROUGHPUT(BasicWorkflow, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}
```

One measurement per test, because the CSV keeps one row per test: a test that
measures twice writes only its last measurement. The `volatile` sink is there
so the compiler cannot decide the result is unused and delete the call. A
third test, `BasicWorkflow.JoinSpeedup`, times both versions and fails if V0
is not at least three times slower than V1 -- see
[What Keeps This Page True](#what-keeps-this-page-true).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_01_BasicWorkflow \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from BasicWorkflow
[ RUN      ] BasicWorkflow.JoinV0
[target-time] 50.000 ms -> cycles=54 (calibrated 922.0000 us/call, batch of 2)
[BasicWorkflow.JoinV0]  927.611 us/call  CV=0.1%  ~1.1K calls/s  (p10=927.056 p90=928.317 sd=0.544)
[       OK ] BasicWorkflow.JoinV0 (509 ms)
[ RUN      ] BasicWorkflow.JoinV1
[target-time] 50.000 ms -> cycles=2492 (calibrated 20.0625 us/call, batch of 64)
[BasicWorkflow.JoinV1]  21.044 us/call  CV=0.7%  ~47.5K calls/s  (p10=20.752 p90=21.123 sd=0.149)
[       OK ] BasicWorkflow.JoinV1 (526 ms)
[ RUN      ] BasicWorkflow.JoinSpeedup
[BasicWorkflow.JoinSpeedup]  V0 927.564 us/call  V1 20.511 us/call  45.2x
[       OK ] BasicWorkflow.JoinSpeedup (48 ms)
[----------] 3 tests from BasicWorkflow (1084 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (1084 ms total)
[  PASSED  ] 3 tests.

======================================================================
Test                   Median (us)     CV%     Calls/s  Status
----------------------------------------------------------------------
BasicWorkflow.JoinV0       927.611    0.1%        1.1K  OK
BasicWorkflow.JoinV1        21.044    0.7%       47.5K  OK
----------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

Read the spread before the median. `CV` is the coefficient of variation
across the ten repeats: 0.1% for V0 and 0.7% for V1 here, and `p10`, `p90` and
`sd` say the same thing in microseconds. A difference is worth talking about
only when it is much larger than that spread, which this one is: 927.6 us
against 21.0 us, a factor of 44, against repeat-to-repeat noise below one
percent.

The `[target-time]` line is the harness sizing the work: asked for 50 ms
samples, it timed a small batch of each version and chose 54 calls per repeat
for V0 and 2,492 for V1. Without `--target-time` both tests run the default
10,000 calls per repeat, which for V0 on this rig is 9.3 seconds per repeat.

`--csv run1.csv` wrote one row per measuring test, next to the columns that
say how the measurement was taken (`cycles`, `repeats`, `warmup`, `threads`)
and where it came from (`timestamp`, `gitHash`, `hostname`, `platform`).

## Step 2: Read the CSV

```bash
bench summary run1.csv
```

Captured output:

```
Test                   Median (us)       P10       P90        CV       Calls/sec  Stable
--------------------  ------------  --------  --------  --------  --------------  ------
BasicWorkflow.JoinV0     927.61100  927.05600  928.31700      0.1%            1078  yes
BasicWorkflow.JoinV1      21.04410  20.75180  21.12340      0.7%           47519  yes

  2 tests, sorted by name
```

Same numbers as the console, sorted and without the run's noise. `Stable` is
the harness's own flag: it compares the CV against a threshold that depends on
how much work each call does, and the CSV carries both the flag and the
threshold it used, so a reader can disagree with it.

## Step 3: Compare Two Runs

Comparison is how a measurement earns its keep: you keep the CSV from before
a change and compare it with the CSV from after. Here, to see what the tool
does, run the same binary a second time without changing anything.

```bash
taskset -c 3 ./build/bin/ptests/BenchDemo_01_BasicWorkflow \
  --target-time 50ms --repeats 10 --csv run2.csv
bench compare run1.csv run2.csv
```

Output, for the captured session's `run1.csv` and `run2.csv`:

```
Test                      Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
--------------------  ------------  ------------  ----------  --------  --------  --------  ------------
BasicWorkflow.JoinV0     927.61100     940.56600   +12.95500     +1.4%      0.1%      0.1%  neutral
BasicWorkflow.JoinV1      21.04410      19.45060    -1.59350     -7.6%      0.7%      2.1%  IMPROVEMENT

  1 improvement(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

Nothing changed between those two runs: same binary, same input, seconds
apart. `bench compare` joins the two files on test name, subtracts the
medians, and labels a row when its median moved by more than the threshold, as
a percentage of the baseline's -- 5% by default, `--threshold` to change it.
That is what the labels say, and all they say: `IMPROVEMENT` on the V1 row
means "the median moved 7.6% in the faster direction", not "this code got
faster".

The movement is real and it is not this demo's doing: a fresh process
measures a little differently. Across nine runs of this binary in one session
on this rig, V0's median ranged from 927.6 to 1014.5 us/call, a spread of
9.4%, while the CV inside each of those runs stayed between 0.0% and 0.3%:
the gap between runs is about thirty times the jitter within one. V1's median
ranged from 19.5 to 21.0 us/call, a spread of 8.2% against within-run CVs of
0.2% to 2.2% -- about four to one rather than thirty to one, and still a
spread its own CV does not predict. The CV in a CSV describes the repeats
inside one run; it says nothing about the next run.

So treat the labels as a filter that says which rows to look at, and read the
medians and the CV yourself. Three more things to know before leaning on them:

- The threshold is a percentage, not a statement about the noise. A test that
  moves inside its own run-to-run spread is labelled if that spread is wider
  than the threshold.
- The labels are not a significance test: the CSVs carry each run's summary
  statistics, not the individual repeats such a test would need. The `Base CV`
  and `Cand CV` columns are each run's own spread, shown as context; neither
  measures the run-to-run spread described above.
- A test that appears in only one of the two files is named under the table,
  as missing from the candidate or new in the candidate; a renamed test shows
  up as both. Two CSVs with no test name in common are an error: the command
  exits 1 and names the tests each side had.

## What Should Reproduce

| Reading                | On this rig                                                   | Elsewhere                                |
| ---------------------- | ------------------------------------------------------------- | ---------------------------------------- |
| V0 is slower than V1   | yes, by 44x here; 44x to 50x over nine runs                   | yes; tens of times in an optimized build |
| V0 median              | 927.6 us/call (928 to 1015 over nine runs)                    | will differ                              |
| V1 median              | 21.0 us/call (19.5 to 21.0 over nine runs)                    | will differ                              |
| CV within one run      | 0.1% (V0) and 0.7% (V1); 0.0-0.3% and 0.2-2.2% over nine runs | depends on how still the machine is      |
| the ratio without `-O` | about 7x                                                      | about 6x on an x86 laptop                |

The ratio is the reading to carry elsewhere. Measured on an x86 laptop
(Release, one pinned core) the same demo reported 113.9 us against 3.4 us,
which is 33x; built without optimization it reported 302.7 us against 47.6 us,
6.4x, and on this rig 2381.6 us against 328.1 us, 7.3x. The absolute times
belong to the machine; the direction and the order of magnitude do not.

## If It Does Not Match

- **The ratio is about 7x, not 44x.** That is the unoptimized build. The
  walkthroughs assume Release; see the rig document's build section.
- **`bench: command not found`.** `source build/.env` in the build you just
  made, or call `./build/bin/tools/rust/bench` by path.
- **Each test takes tens of seconds.** `--target-time` is missing, so each
  repeat runs the default 10,000 calls.
- **CV above a few percent.** The governor is back to `ondemand`, the test is
  not pinned, or something else is running. The rig document's measurement
  section has the governor and `taskset` recipe, and `vcgencmd get_throttled`
  should read `0x0` before and after.

## Check Against the Reference

A capture from this rig is committed with the demo, so you can compare a run
of your own against it:

```bash
bench compare src/bench/demo/reference/pi4/01_basic_workflow.csv run1.csv
```

Output, for the captured session's `run1.csv`:

```
Test                      Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
--------------------  ------------  ------------  ----------  --------  --------  --------  ------------
BasicWorkflow.JoinV0    1014.54000     927.61100   -86.92900     -8.6%      0.2%      0.1%  IMPROVEMENT
BasicWorkflow.JoinV1      20.94160      21.04410    +0.10250     +0.5%      2.2%      0.7%  neutral

  1 improvement(s)  1 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by this same binary on this same board, minutes
before the run above, so the -8.6% on the V0 row is the run-to-run spread from
the previous section and nothing else; the reference holds the slowest V0 of
the session. Expect labels here, and read the medians yourself. Measured
against the reference, the session's eight other runs put V0 between 0.6% and
8.6% below it and V1 between 7.1% below and 0.5% above it, while the ratio
between the two rows stayed between 44x and 50x. The ratio is the reading
that holds still; the medians are not.

On any other machine the absolute numbers will differ and the labels say
nothing at all. What should survive is that V0's row is tens of times larger
than V1's.

## What Keeps This Page True

Two things run against this example, and both fail loudly:

- `BasicWorkflow.JoinSpeedup`, in the demo binary, times both versions and
  fails unless V0 is at least three times slower than V1. If an optimizer, a
  library change or an edit to the example ever erases the effect, the demo
  fails instead of quietly demonstrating nothing.
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
rig by hand, with the command in step 1, and the reference CSV is re-captured
when the numbers move.

## See Also

- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [CPU benchmarking guide](../../docs/CPU_GUIDE.md) -- the harness API behind
  `PERF_GUARD` and `throughputLoop`
- [Demo 02: perf profiler](02_PERF_PROFILER.md) -- reaching for a profiler
  when a timing difference needs an explanation
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
