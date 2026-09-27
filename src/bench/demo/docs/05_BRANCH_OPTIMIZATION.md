# Demo 05: Branch Prediction and Branchless Programming

This walkthrough has moved. The branch example is now the second example of
walkthrough 02:
[The Second Example: a Filter](02_PERF_PROFILER.md#the-second-example-a-filter).

`BenchDemo_05_BranchOptimization` is removed. Its replacement is
`BenchDemo_02_PerfProfiler`, whose filter cases
(`PerfProfiler.FilterBranchyRandom`, `PerfProfiler.FilterBranchySorted` and
`PerfProfiler.FilterBranchless`) take the place of demo 05's branchy random,
branchy sorted and branchless cases. From the tree root, with the Release
build walkthrough 02 uses, run them with the timing flags of its step 1:

```bash
./build/bin/ptests/BenchDemo_02_PerfProfiler --gtest_filter='PerfProfiler.Filter*' \
  --target-time 50ms --repeats 10
```
