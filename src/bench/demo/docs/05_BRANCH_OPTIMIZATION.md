# Demo 05: Branch Prediction and Branchless Programming

The branch example lives in [walkthrough 02](02_PERF_PROFILER.md) now, as the
second example of the perf page:
[The Second Example: a Filter](02_PERF_PROFILER.md#the-second-example-a-filter).
There is no `BenchDemo_05_BranchOptimization` binary; the filter is measured
by `BenchDemo_02_PerfProfiler`, and its counters were captured on the
[Raspberry Pi 4 rig](../../docs/rigs/RIG_PI4.md).

## Why One Example, and Why a Filter

A branch example has to keep its branch through an optimizing build. A sum
of the values above a threshold did not: a compiler may turn a conditional
addition into a conditional select, and the branchy and branchless sums are
then the same machine code, timing the same on every input. A store made
only when the value passes is harder to remove, because a compiler may not
add a store the source does not make; so the example is a filter, and its
branchless twin stores every value and advances the output cursor by the
comparison instead. The filter kept its branch in every build walkthrough 02
reports on, and its unit tests fail in a build where it does not.
Walkthrough 02 measures and profiles it with the hardware counters that show
the difference, and one demo owns it: a second demo on the same example
would be a copy of that one.

## What This Page Owed, and Where It Is Now

| Obligation                                                                                 | Where it is met                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| One filter implementation, with and without a branch                                       | [`examples/filter`](../examples/filter/inc/Filter.hpp): `filterBranchy` and `filterBranchless`, and the inputs `makeValues` and `makeSortedValues`                                                                 |
| Correctness: every version keeps the same values, in the same order                        | the example's unit tests, [`Filter_uTest.cpp`](../examples/filter/utst/Filter_uTest.cpp), registered with `ctest` under the `demo` label                                                                           |
| The optimized-code check: a build that loses the branch fails                              | `FilterBranchTest.ConditionalStoreKeepsItsBranch` in the same tests: on random input the branchy filter must mispredict at least a tenth of a branch per value, and at least ten times as often as on sorted input |
| The demonstration: random input mispredicts, sorted input and the branchless filter do not | `FilterBranchTest.BranchlessFormRemovesTheMisses` beside it in the same tests, and [steps 5 to 7](02_PERF_PROFILER.md#step-5-count-the-branchy-filter-on-random-input) of walkthrough 02                           |
| Hardware-counter evidence from the reference rig                                           | walkthrough 02's captured `stat.txt` reports and its [What Should Reproduce](02_PERF_PROFILER.md#what-should-reproduce) table                                                                                      |
| The lesson: when a branch costs, and what sorting or a branchless form does about it       | walkthrough 02, from [The Second Example: a Filter](02_PERF_PROFILER.md#the-second-example-a-filter) on                                                                                                            |

## See Also

- [Walkthrough 02: perf](02_PERF_PROFILER.md) -- hardware counters on the
  join and the filter examples
- [CPU guide](../../docs/CPU_GUIDE.md) -- the framework's CPU benchmarking
  reference
