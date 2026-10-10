# Demo 01: Basic Workflow

**Reference rig:** [Raspberry Pi 4](../../docs/rigs/RIG_PI4.md)
**Build:** Release
**Example:** [`join`](../examples/join/inc/Join.hpp) (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** 2026-09-28 (UTC), written for the Vernier 1.0.4 release; captured
from the development tree at project version 1.0.3, whose CLI reported
`bench 1.0.3`

## Overview

This is the whole Vernier cycle, once, on one small function: write a
performance test, run it, read the result line, write a CSV, and compare two
CSVs. The example is `join`, which glues a list of words together with a
separator, the way code builds a CSV row, a log line or a query a piece at a
time. It has two versions that return the same string, and on this rig one of
them takes 50 times longer than the other.

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

The demo, [`cpu/01_BasicWorkflow_Demo.cpp`](../cpu/01_BasicWorkflow_Demo.cpp),
measures each version in its own test, over 1,000 words:

```cpp
PERF_THROUGHPUT(BasicWorkflow, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), demo::joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}
```

One measurement per test, because the CSV keeps one row per test: a test that
measures twice writes only its last measurement. The size check uses the
example's `joinedSize()`, which computes the length without allocating. The
`volatile` sink is there so the compiler cannot decide the result is unused
and delete the call. That is the whole demo: two tests that measure. The check
that V0 stays at least three times slower than V1 is a performance test of its
own beside the example ([What Keeps This Page True](#what-keeps-this-page-true)).

## Step 1: Measure

```bash
source build/.env     # puts the bench CLI on PATH
taskset -c 3 ./build/bin/ptests/BenchDemo_01_BasicWorkflow \
  --target-time 50ms --repeats 10 --csv run1.csv
```

Captured output:

```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from BasicWorkflow
[ RUN      ] BasicWorkflow.JoinV0
[target-time] 50.000 ms -> cycles=49 (calibrated 1006.0000 us/call, batch of 1)
[BasicWorkflow.JoinV0]  1000.531 us/call  CV=0.1%  ~999 calls/s  (p10=999.845 p90=1001.631 sd=0.879)
[       OK ] BasicWorkflow.JoinV0 (497 ms)
[ RUN      ] BasicWorkflow.JoinV1
[target-time] 50.000 ms -> cycles=2507 (calibrated 19.9375 us/call, batch of 64)
[BasicWorkflow.JoinV1]  20.003 us/call  CV=0.1%  ~50.0K calls/s  (p10=19.975 p90=20.022 sd=0.023)
[       OK ] BasicWorkflow.JoinV1 (504 ms)
[----------] 2 tests from BasicWorkflow (1001 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (1002 ms total)
[  PASSED  ] 2 tests.

======================================================================
Test                   Median (us)     CV%     Calls/s  Status
----------------------------------------------------------------------
BasicWorkflow.JoinV0      1000.531    0.1%         999  OK
BasicWorkflow.JoinV1        20.003    0.1%       50.0K  OK
----------------------------------------------------------------------
2 tests | 2 stable | 0 unstable
```

Read the spread before the median. `CV` is the coefficient of variation
across the ten repeats: 0.1% for both versions here, and `p10`, `p90` and `sd`
say the same thing in microseconds. A difference is worth talking about only
when it is much larger than that spread, which this one is: 1000.5 us against
20.0 us, a factor of 50, against repeat-to-repeat noise of a tenth of a
percent.

The `[target-time]` line is the harness sizing the work: asked for 50 ms
samples, it timed a small batch of each version and chose 49 calls per repeat
for V0 and 2,507 for V1. Without `--target-time` both tests run the default
10,000 calls per repeat, which for V0 on this rig is 10 seconds per repeat.

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
BasicWorkflow.JoinV0    1000.53000  999.84500  1001.63000      0.1%             999  yes
BasicWorkflow.JoinV1      20.00320  19.97520  20.02200      0.1%           49992  yes

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
BasicWorkflow.JoinV0    1000.53000     968.88800   -31.64200     -3.2%      0.1%      0.2%  neutral
BasicWorkflow.JoinV1      20.00320      20.02720    +0.02400     +0.1%      0.1%      2.8%  neutral

  2 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

Nothing changed between those two runs: same binary, same input, a second
apart. `bench compare` joins the two files on test name, subtracts the
medians, and labels a row when its median moved by more than the threshold,
as a percentage of the baseline's -- 5% by default, `--threshold` to change
it. Here V0's median moved 3.2% and V1's 0.1%, both inside the threshold, so
both rows say `neutral`. That is all the label says, whichever way it reads:
a label on unchanged code, which the reference comparison below shows, means
"the median moved by more than 5%", not "this code got slower or faster".

The movement is real and it is not this demo's doing: a fresh process
measures a little differently. Across thirteen runs of this command in one
session on this rig, V0's median ranged from 920.9 to 1033.5 us/call, a spread
of 12.2%, while the CV inside each of those runs stayed between 0.0% and 0.3%:
the gap between runs is about forty times the largest jitter within one. V1's
median ranged from 19.7 to 21.2 us/call, a spread of 7.5%, against within-run
CVs of 0.1% to 2.8%: nearly three times the largest, and still a spread its
own CV does not predict. V0's median is the noisiest reading on this page. The
CV in a CSV describes the repeats inside one run; it says nothing about the
next run.

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

## Measuring Your Own Code

The demo is the pattern to copy:

1. Give each version of your code a test of its own, shaped like
   `BasicWorkflow.JoinV0` above: `PERF_THROUGHPUT` and `PERF_GUARD`, the input
   built once outside the timed calls, the answer checked, a warmup, then
   `perf.throughputLoop` around one call whose result goes to a `volatile`
   sink. The [CPU guide](../../docs/CPU_GUIDE.md#build-and-run) shows how to
   build such a file into your own project.
2. Run it the way step 1 runs the demo: on one pinned core, with
   `--target-time` sizing each repeat, `--repeats 10`, and `--csv` keeping the
   result.
3. Keep that CSV from before a change, run again after it, and
   `bench compare before.csv after.csv`. Read the medians and the CVs as step
   3 does.
4. Judge a change by more than one run of each side: with nothing changed,
   V0's median on this page moved 12% across one session's runs.

## What Should Reproduce

| Reading                | On this rig                                                       | Elsewhere                                                             |
| ---------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| V0 is slower than V1   | yes, by 50x here; 44.5x to 51.5x over thirteen runs               | yes; tens of times in an optimized build; 32x to 37x on an x86 laptop |
| V0 median              | 1000.5 us/call (920.9 to 1033.5 over thirteen runs)               | will differ                                                           |
| V1 median              | 20.0 us/call (19.7 to 21.2 over thirteen runs)                    | will differ                                                           |
| CV within one run      | 0.1% (V0) and 0.1% (V1); 0.0-0.3% and 0.1-2.8% over thirteen runs | depends on how still the machine is                                   |
| the ratio without `-O` | 6.6x to 6.7x over three runs                                      | 7.5x to 7.6x on an x86 laptop                                         |

The thirteen runs are step 1's command run ten times, then the reference
capture, then the page's two runs, in one session on this rig. Those ranges
describe those runs; they are not a bound a run has to meet, and another run
of unchanged code can land outside them. The ratio is the reading to carry
elsewhere. Measured on an x86 laptop (Release, one pinned core, with other
work running on it) the same demo reported 102.6 to 119.4 us against 3.0 to
3.2 us over five runs, 32x to 37x. Built without optimization (the Debug build
type) it reported 314.1 to 316.0 us against 41.8 to 42.0 us, 7.5x to 7.6x, and
on this rig 2330.4 to 2361.2 us against 351.1 to 355.9 us, 6.6x to 6.7x. The
absolute times belong to the machine; the direction and the order of
magnitude do not.

## If It Does Not Match

- **The ratio is about 7x, not 50x.** That is the unoptimized build. The
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
BasicWorkflow.JoinV0     937.41300    1000.53000   +63.11700     +6.7%      0.1%      0.1%  REGRESSION
BasicWorkflow.JoinV1      21.06390      20.00320    -1.06070     -5.0%      0.5%      0.1%  IMPROVEMENT

  1 regression(s)  1 improvement(s)  0 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

The reference was captured by this same binary on this same board seconds
before the run above, and nothing changed between them, yet both rows carry a
label: V0's median moved 6.7% in the slower direction and V1's 5.0% in the
faster one. That is the run-to-run spread from step 3 crossing the 5%
threshold, and nothing else; the ratio between the two rows moved from 44.5x
to 50.0x. Measured against the reference, the session's twelve other runs put
V0 between 1.8% below and 10.3% above it and V1 between 6.3% below and 0.8%
above it, and three of the twelve would have drawn a label on each row. Expect
labels here, and read the medians yourself. The CSVs' `hostname` column
records the name visible to the process that captured each: this reference was
captured in a UTS namespace named `pi4` on the same rig, so it records `pi4`,
while the rig's ordinary captures, step 1's `run1.csv` among them, record
`raspberrypi`.

On any other machine the absolute numbers will differ and the labels say
nothing at all. What should survive is that V0's row is tens of times larger
than V1's.

## What Keeps This Page True

Two things run against this example, and both fail loudly:

- `BasicWorkflow.JoinSpeedup` times both versions in one test, with a timer
  of its own, and fails unless V0 is at least three times slower than V1. If
  an optimizer, a library change or an edit to the example ever erases the
  effect, it fails instead of the demo quietly demonstrating nothing. It is a
  performance test beside the join example,
  [`JoinSpeedup_pTest.cpp`](../examples/join/ptst/JoinSpeedup_pTest.cpp),
  built as `JoinSpeedup`: comparing the two versions needs both timings in one
  test, which the demo's one row per test does not give. A timing belongs to
  the machine that takes it, so it is not registered with `ctest`; it is run
  on the rig by hand before a release:

  ```bash
  taskset -c 3 ./build/bin/ptests/JoinSpeedup
  ```

  Captured on this rig:

  ```
  [==========] Running 1 test from 1 test suite.
  [----------] Global test environment set-up.
  [----------] 1 test from BasicWorkflow
  [ RUN      ] BasicWorkflow.JoinSpeedup
  [BasicWorkflow.JoinSpeedup]  V0 971.548 us/call  V1 20.142 us/call  48.2x
  [       OK ] BasicWorkflow.JoinSpeedup (50 ms)
  [----------] 1 test from BasicWorkflow (50 ms total)

  [----------] Global test environment tear-down
  [==========] 1 test from 1 test suite ran. (51 ms total)
  [  PASSED  ] 1 test.
  ```

  Over ten runs in this session it read 45.3x to 48.2x. With `joinV0` given
  `joinV1`'s body it read 1.0x and failed: "the demo has stopped
  demonstrating".

- The example's unit tests hold every version to the same answers and are
  registered with `ctest`, so an ordinary test run includes them; every test
  the label selects should pass:

  ```bash
  ctest --test-dir build -L demo
  ```

This repository has no continuous-integration lane on the reference board, so
nothing runs the demo or its timing check automatically. Before a release they
are run on the rig by hand, with the commands above, and the reference CSV is
re-captured when the numbers move.

## See Also

- [Reference rigs](../../docs/rigs/README.md) -- what a rig is, and what
  reproduces on one
- [CPU benchmarking guide](../../docs/CPU_GUIDE.md) -- the harness API behind
  `PERF_GUARD` and `throughputLoop`
- [Demo 02: perf profiler](02_PERF_PROFILER.md) -- reaching for a profiler
  when a timing difference needs an explanation
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
