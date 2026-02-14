# Demo 05: Branch Prediction and Branchless Programming

## Overview

Demonstrates how unpredictable branches cause pipeline stalls and how
branchless coding eliminates the penalty. A three-way comparison shows
the branch predictor in action.

## Prerequisites

```bash
make compose-debug
make tools-rust
```

## Step 1: Run All Three Variants

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_05_BranchOptimization --quick --csv /tmp/branch_demo.csv
  ./bin/tools/rust/bench summary /tmp/branch_demo.csv --sort median
'
```

Expected output (sorted fastest to slowest):

```
BranchOptimization.BranchlessRandomData   ~45 us/call   ~22K calls/s
BranchOptimization.BranchySortedData      ~50 us/call   ~20K calls/s
BranchOptimization.BranchyRandomData      ~90 us/call   ~11K calls/s
```

## Step 2: Profile Branch Misses

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug

  # Profile branchy + random (worst case)
  ./bin/ptests/BenchDemo_05_BranchOptimization --profile perf \
    --gtest_filter="*BranchyRandomData*" --cycles 1000

  # Profile branchless + random (best case)
  ./bin/ptests/BenchDemo_05_BranchOptimization --profile perf \
    --gtest_filter="*BranchlessRandomData*" --cycles 1000
'
```

Expected perf counters:

```
Branchy + Random:
  branch-misses:  ~50,000,000  (50% mispredict rate)

Branchless + Random:
  branch-misses:     ~100,000  (near zero)
```

## Step 3: Diagnose

### Why random data is slow with branches

The conditional `if (val > 0.5)` is taken for ~50% of random values in
unpredictable order. The CPU's branch predictor cannot learn the pattern,
resulting in ~50% misprediction rate. Each mispredict costs 15-20 cycles
of pipeline flush and refill.

### Why sorted data helps branchy code

With sorted data, all values below 0.5 come first (branch not taken),
then all values above 0.5 (branch taken). The predictor sees one
transition point and achieves near-100% accuracy after warming up.

### Why branchless always wins

The branchless version `val * (val > threshold)` converts the branch
into a multiply-by-zero-or-one. No branch instruction means no
misprediction, regardless of data order.

## Step 4: Compare

| Variant             | Median | Branch Misses | Notes                      |
| ------------------- | ------ | ------------- | -------------------------- |
| Branchy + Random    | ~90 us | ~50M          | Worst case: 50% mispredict |
| Branchy + Sorted    | ~50 us | ~1K           | Predictor learns pattern   |
| Branchless + Random | ~45 us | ~100K         | No branches to mispredict  |

## Key Takeaways

- Unpredictable branches in hot loops cause 2-3x slowdowns
- Sorting data helps if the branch depends on sorted values
- Branchless coding (`val * predicate`) eliminates the penalty entirely
- Use `perf` with `branch-misses` counter to identify the problem
- Branchless matters most in: hot inner loops, random data, 50/50 conditions
- For rare branches (<5% or >95% taken), the predictor handles it well

## Further Reading

- `docs/CPU_GUIDE.md` -- Branch prediction section
- Demo 02 (perf) -- Hardware counter profiling
