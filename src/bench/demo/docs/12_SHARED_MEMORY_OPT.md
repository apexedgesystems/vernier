# Demo 12: Shared Memory Bank Conflicts

**Reference rig:** [NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
**Build:** Release
**Example:** a 1,024 x 1,024 matrix transpose private to this demo
([`03_SharedMemoryOpt_Transpose.cu`](../gpu/03_SharedMemoryOpt_Transpose.cu)):
no shared example has a bank conflict to show
**Captured:** 2026-09-30 (UTC). Written for the Vernier 1.0.4 release;
captured from the development tree at project version 1.0.3, whose CLI
reported `bench 1.0.3`; Nsight Compute 2025.3.1. The RTX 5000 Ada figures in
[What Should Reproduce](#what-should-reproduce) come from one run of the same
tree in the project's `dev-cuda` container on the same day.

## Overview

A matrix transpose three ways: through global memory alone, through a tile
in shared memory, and through the same tile with each row padded by one
float. On this rig the tile takes the kernel from 102.9 to 76.5 us per
launch, and the padding takes it to 40.6 us: 2.5x end to end, 1.9x from one
float per row. The timer says the padding mattered; it cannot say why. Nsight
Compute can: the tiled kernel's reads of one tile column land in one
shared-memory bank and are served 32 times over, and the padded tile's land
in 32 banks and are served once. This page reads that from ncu's counters,
and a check under `ctest` holds the demo to it.

This is the last of the GPU walkthroughs. It assumes walkthrough
[10](10_GPU_BASIC_WORKFLOW.md) (the GPU harness and its CSV) and
[11](11_NSIGHT_PROFILER.md) (Nsight Compute, `bench run --profile ncu`, and
what a replay does to a kernel's time).

## What Is a Bank Conflict, and What Nsight Compute Counts

Shared memory is split into 32 banks, each four bytes wide, so consecutive
floats sit in consecutive banks and the pattern repeats every 32 floats. A
warp's load from shared memory is served in one pass, one _wavefront_, when
its 32 lanes touch 32 different banks (or the same word). Lanes that touch
different words in the same bank are served in turn, one wavefront each: a
warp whose 32 lanes all read one bank takes 32. Nothing in the source says
which happens; the addresses decide, and the counters show it.

Nsight Compute counts it, per launch:

- `smsp__inst_executed_op_shared_ld.sum`: the shared-memory load
  instructions the kernel executed, one per warp per load in the source
  (ncu's rules call each one a request).
- `smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`: the
  wavefronts those instructions took, as the instruction stream sees them.
  One per instruction is the best case.
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`: ncu's
  bank-conflict counter for shared loads, the figure its rule quotes. On this
  page's launches it reads as the wavefronts the L1TEX unit served beyond one
  per request (Step 4 shows the arithmetic).

The first two make the mechanism exact (32 against 1 per request, below); the
third is the headline. ncu's `MemoryWorkloadAnalysis_Tables` section turns
them into a sentence naming the conflict's degree.

**In Vernier:** `--profile ncu` selects the Nsight Compute backend, and
`bench run <binary> --profile ncu` starts the binary under
`ncu -o bench-out/<binary>.ncu/kernel_profile -f --target-processes all`. That
wrap collects ncu's default sections (launch statistics, occupancy and
throughput, as walkthrough 11 reads them) into
`bench-out/<binary>.ncu/kernel_profile.ncu-rep`, and none of them carries a
bank-conflict figure; `bench run` passes ncu no option of its own
(`--profile-args` goes to the binary). So this page types the ncu command
itself, with the sections and metrics it reads, as walkthrough 14 types
massif's for `--time-unit`. The CSV's `occupancy` column is the harness's
estimate from the launch shape, the device's limits and the static shared
memory the demo hands `withLaunchConfig()`; its `cuptiStaticSmemBytes` column
is CUPTI's reading of the kernel's static shared memory.

**Needs:** the CUDA toolkit on `PATH` with `ncu`, and a GPU build: the rig's
[setup](../../docs/rigs/RIG_THOR_AGX.md#2-one-time-setup) and
[build](../../docs/rigs/RIG_THOR_AGX.md#3-build). On this rig the GPU's
performance counters are for administrators only, so `ncu` runs under `sudo`.

## The Example

Every kernel maps one thread to one element of a 1,024 x 1,024 matrix of
floats (4 MiB in, 4 MiB out), in blocks of 32 x 32 threads, and reads its
input along a row, so its reads are coalesced. They differ in how the
transposed element reaches the output. The naive kernel writes it straight to
its column:

```cpp
__global__ void transposeNaive(const float* input, float* output, int dim) {
  const int X = blockIdx.x * TILE_DIM + threadIdx.x;
  const int Y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (X < dim && Y < dim) {
    // A warp reads 32 consecutive floats of one row and writes them dim
    // floats apart, down one column: one transaction in, 32 out.
    output[X * dim + Y] = input[Y * dim + X];
  }
}
```

The tiled kernel stages a 32 x 32 tile in shared memory, so both the global
reads and the global writes run along rows; the transpose happens in the
tile, where each warp reads one column:

```cpp
__global__ void transposeSharedConflict(const float* input, float* output, int dim) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  const int X = blockIdx.x * TILE_DIM + threadIdx.x;
  const int Y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (X < dim && Y < dim) {
    // Each warp fills one row of the tile from one row of the input.
    tile[threadIdx.y][threadIdx.x] = input[Y * dim + X];
  }
  __syncthreads();

  const int OUT_X = blockIdx.y * TILE_DIM + threadIdx.x;
  const int OUT_Y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (OUT_X < dim && OUT_Y < dim) {
    // Each warp reads one column of the tile and writes it as one row of
    // the output. The 32 elements of a column are TILE_DIM floats apart,
    // and TILE_DIM is a multiple of the bank count, so all 32 lanes read the
    // same bank: the read is served 32 times over.
    output[OUT_Y * dim + OUT_X] = tile[threadIdx.x][threadIdx.y];
  }
}
```

The padded kernel is the same code with the tile one float wider per row:

```cpp
__global__ void transposeSharedPadded(const float* input, float* output, int dim) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // rows TILE_DIM + 1 floats apart

  const int X = blockIdx.x * TILE_DIM + threadIdx.x;
  const int Y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (X < dim && Y < dim) {
    tile[threadIdx.y][threadIdx.x] = input[Y * dim + X];
  }
  __syncthreads();

  const int OUT_X = blockIdx.y * TILE_DIM + threadIdx.x;
  const int OUT_Y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (OUT_X < dim && OUT_Y < dim) {
    // The same column read, but the elements are TILE_DIM + 1 floats apart:
    // each lands one bank further than the last, so the 32 lanes read 32
    // banks and the read is served once.
    output[OUT_Y * dim + OUT_X] = tile[threadIdx.x][threadIdx.y];
  }
}
```

The arithmetic of the column read: element `[k][c]` of a 32-wide tile sits
`k * 32 + c` floats from the tile's start, and `(k * 32 + c) mod 32` is `c`
whatever `k` is, so a warp reading column `c` reads 32 words of bank `c`: 32
wavefronts. In the 33-wide tile the same element sits `k * 33 + c` floats in,
and `(k * 33 + c) mod 32` is `(k + c) mod 32`: 32 different banks for the 32
rows, one wavefront. The pad costs 128 bytes per tile. It fixes this column
read of a row-major tile; it is not a general cure for strided access along a
row, and this page measures no other stride.

The demo ([`03_SharedMemoryOpt_Demo.cu`](../gpu/03_SharedMemoryOpt_Demo.cu))
measures each kernel in a test of its own, one CSV row each. The tiled one:

```cpp
/**
 * @test The transpose through a shared-memory tile. The global reads and
 *       writes are both coalesced; every warp's reads of one tile column
 *       land in one bank, so each is served 32 times over.
 */
PERF_GPU_TEST(SharedMemoryOpt, SharedWithBankConflicts) {
  PERF_GPU_GUARD(perf);

  DeviceMatrices device;
  ASSERT_TRUE(device.ok()) << "device allocation failed";
  const std::vector<float> INPUT = rampMatrix();
  ASSERT_TRUE(device.upload(INPUT)) << "device upload failed";

  const dim3 GRID = sm::transposeGrid(sm::MATRIX_DIM);
  const dim3 BLOCK = sm::transposeBlock();
  const auto LAUNCH = [&](cudaStream_t s) {
    sm::transposeSharedConflict<<<GRID, BLOCK, 0, s>>>(device.input(), device.output(),
                                                       sm::MATRIX_DIM);
  };
  perf.cudaWarmup(LAUNCH);
  // The tile is static shared memory, declared in the kernel; the harness is
  // told its size so its occupancy estimate can account for it.
  perf.cudaKernel(LAUNCH, "transpose_shared_conflict")
      .withLaunchConfig(GRID, BLOCK, sm::TILE_BYTES)
      .measure();

  std::vector<float> output(N);
  ASSERT_TRUE(device.download(output)) << "device download failed";
  EXPECT_EQ(transposeMismatches(INPUT, output, sm::MATRIX_DIM), 0U)
      << "the kernel did not transpose the input";
}
```

`withLaunchConfig()` is handed the tile's bytes (`TILE_BYTES`, 4,096; the
padded tile's `TILE_PADDED_BYTES`, 4,224), which the harness's occupancy
estimate accounts for. After the measurement the test reads the output back
and fails unless it is the transpose of the input, so a kernel that stops
transposing fails its own test. The three kernels are also held to a CPU
transpose, and their declared shared memory to those constants, by
`TestDemoTranspose` ([What Keeps This Page True](#what-keeps-this-page-true)).

## Step 1: Measure

Inside the rig document's
[measurement procedure](../../docs/rigs/RIG_THOR_AGX.md#4-running-a-measurement)
(clocks locked with `jetson_clocks` and restored afterwards, the host thread
on core 13), from the source tree, after the rig's build:

```bash
taskset -c 13 ./build/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt --repeats 10 --csv shared_memory_opt.csv
```

The CSV lands in the working directory under the name given; the demo writes
nothing else. Captured output (the reference run):

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from SharedMemoryOpt
[ RUN      ] SharedMemoryOpt.NaiveGlobalMemory
[SharedMemoryOpt.NaiveGlobalMemory]  102.912 us/call  CV=0.1%  ~9.7K calls/s  (p10=102.882 p90=102.926 sd=0.064)
[       OK ] SharedMemoryOpt.NaiveGlobalMemory (10474 ms)
[ RUN      ] SharedMemoryOpt.SharedWithBankConflicts
[SharedMemoryOpt.SharedWithBankConflicts]  76.548 us/call  CV=0.0%  ~13.1K calls/s  (p10=76.531 p90=76.551 sd=0.010)
[       OK ] SharedMemoryOpt.SharedWithBankConflicts (7683 ms)
[ RUN      ] SharedMemoryOpt.SharedPadded
[SharedMemoryOpt.SharedPadded]  40.576 us/call  CV=0.0%  ~24.6K calls/s  (p10=40.557 p90=40.591 sd=0.016)
[       OK ] SharedMemoryOpt.SharedPadded (4082 ms)
[----------] 3 tests from SharedMemoryOpt (22240 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (22240 ms total)
[  PASSED  ] 3 tests.

=========================================================================================
Test                                      Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------------------------------
SharedMemoryOpt.NaiveGlobalMemory             102.912    0.1%        9.7K  OK
SharedMemoryOpt.SharedWithBankConflicts        76.548    0.0%       13.1K  OK
SharedMemoryOpt.SharedPadded                   40.576    0.0%       24.6K  OK
-----------------------------------------------------------------------------------------
3 tests | 3 stable | 0 unstable
```

What to read: the three medians and their CVs. The naive transpose takes
102.9 us per launch, the tiled one 76.5 us and the padded one 40.6 us, each
steady within its run (CV 0.1% or less): the tile is worth 1.34x, the
padding another 1.89x, 2.54x in all.

## Step 2: Read the GPU Columns

```bash
source build/.env
bench summary shared_memory_opt.csv
```

```
Test                                      Median (us)       P10       P90        CV       Calls/sec  Stable
---------------------------------------  ------------  --------  --------  --------  --------------  ------
SharedMemoryOpt.NaiveGlobalMemory           102.91200  102.88200  102.92600      0.1%            9717  yes
SharedMemoryOpt.SharedPadded                 40.57560  40.55690  40.59090      0.0%           24645  yes
SharedMemoryOpt.SharedWithBankConflicts      76.54810  76.53070  76.55120      0.0%           13064  yes

  3 tests, sorted by name
```

The GPU columns are in the CSV, by name:

```bash
awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; print "test kernelTimeUs occupancy cuptiStaticSmemBytes cuptiKernelLaunches"; next }
         { print $col["test"], $col["kernelTimeUs"], $col["occupancy"], $col["cuptiStaticSmemBytes"], $col["cuptiKernelLaunches"] }' \
  shared_memory_opt.csv | column -t
```

```
test                                     kernelTimeUs  occupancy  cuptiStaticSmemBytes  cuptiKernelLaunches
SharedMemoryOpt.NaiveGlobalMemory        102.912201    0.666667   0                     100000
SharedMemoryOpt.SharedWithBankConflicts  76.548117     0.666667   4096                  100000
SharedMemoryOpt.SharedPadded             40.575644     0.666667   4224                  100000
```

`kernelTimeUs` is the wall median: these tests declare no transfers, so a
call is one launch. `cuptiStaticSmemBytes` is what CUPTI read: no shared
memory in the naive kernel, one 32 x 32 tile of floats (4,096 bytes) in the
tiled one, 32 rows of 33 (4,224 bytes) in the padded one; the demo hands the
harness the same numbers. `cuptiKernelLaunches` counts the 100,000 measured
launches (10,000 cycles by 10 repeats).

`occupancy` reads 0.667 for all three, and that is right. A block of 1,024
threads is 32 warps; an SM on this GPU holds 48 warps but only one such
block, so the estimate is 32 of 48 whatever the tile weighs. Telling the
harness the tile's bytes changes nothing here, because the block, not the
tile, is the limit. Step 3 shows ncu reaching the same figure the same way.

## Step 3: Count the Bank Conflicts

With the clocks still locked, profile the three kernels with the three
metrics and the sections this page reads. ncu replays every launch, so keep
the launches few (`--cycles 3 --repeats 1`: with the harness's warmup, 14
launches per kernel). As root on this rig, then give the report back to your
user:

```bash
sudo env PATH="$PATH" ncu --target-processes all \
  --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis_Tables \
  --metrics smsp__inst_executed_op_shared_ld.sum,smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
  -o shared_memory_opt -f ./build/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt --cycles 3 --repeats 1
sudo chown "$(id -u):$(id -g)" shared_memory_opt.ncu-rep
```

Captured output, with all but the first `==PROF== Profiling` line of each
kernel cut, and the checkout's path shortened to `...` here and below:

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from SharedMemoryOpt
[ RUN      ] SharedMemoryOpt.NaiveGlobalMemory
==PROF== Connected to process 85848 (.../build/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt)
==PROF== Profiling "transposeNaive" - 0: 0%....50%....100% - 31 passes
...
[SharedMemoryOpt.NaiveGlobalMemory]  1089568.359 us/call  CV=0.0%  ~1 calls/s  (p10=1089568.359 p90=1089568.359 sd=0.000)
[       OK ] SharedMemoryOpt.NaiveGlobalMemory (20342 ms)
[ RUN      ] SharedMemoryOpt.SharedWithBankConflicts
==PROF== Profiling "transposeSharedConflict" - 14: 0%....50%....100% - 31 passes
...
[SharedMemoryOpt.SharedWithBankConflicts]  1083525.065 us/call  CV=0.0%  ~1 calls/s  (p10=1083525.065 p90=1083525.065 sd=0.000)
[       OK ] SharedMemoryOpt.SharedWithBankConflicts (15265 ms)
[ RUN      ] SharedMemoryOpt.SharedPadded
==PROF== Profiling "transposeSharedPadded" - 28: 0%....50%....100% - 31 passes
...
[SharedMemoryOpt.SharedPadded]  1093090.007 us/call  CV=0.0%  ~1 calls/s  (p10=1093090.007 p90=1093090.007 sd=0.000)
[       OK ] SharedMemoryOpt.SharedPadded (15304 ms)
[----------] 3 tests from SharedMemoryOpt (50912 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (50913 ms total)
[  PASSED  ] 3 tests.

=========================================================================================
Test                                      Median (us)     CV%     Calls/s  Status
-----------------------------------------------------------------------------------------
SharedMemoryOpt.NaiveGlobalMemory         1089568.359    0.0%           1  OK
SharedMemoryOpt.SharedWithBankConflicts   1083525.065    0.0%           1  OK
SharedMemoryOpt.SharedPadded              1093090.007    0.0%           1  OK
-----------------------------------------------------------------------------------------
3 tests | 3 stable | 0 unstable
==PROF== Disconnected from process 85848
==PROF== Report: .../shared_memory_opt.ncu-rep
```

The run took 51 seconds for 42 launches at 31 passes each. The times the
harness prints are ncu's replays, about 1.09 s per launch, not the kernel's:
the timings are Step 1's. The report is `shared_memory_opt.ncu-rep` in the
working directory. Read it with the same `ncu`:

```bash
ncu --import shared_memory_opt.ncu-rep --print-summary per-kernel
```

The metrics and four rows of the Occupancy section, per kernel, minimum,
maximum and average over its 14 launches (the Launch Statistics section and
the other Occupancy rows are cut):

```
  transposeNaive(const float *, float *, int) (32, 32, 1)x(32, 32, 1), Device 0, CC 11.0, Invocations 14
    Section: Command line profiler metrics
    -------------------------------------------------------------- ----------- ------- ------- -------
    Metric Name                                                    Metric Unit Minimum Maximum Average
    -------------------------------------------------------------- ----------- ------- ------- -------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                      0.00    0.00    0.00
    smsp__inst_executed_op_shared_ld.sum                                  inst    0.00    0.00    0.00
    smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                0.00    0.00    0.00
    -------------------------------------------------------------- ----------- ------- ------- -------

    Section: Occupancy
    ...
    Block Limit Shared Mem                block    8.00    8.00    8.00
    Block Limit Warps                     block    1.00    1.00    1.00
    Theoretical Occupancy                     %   66.67   66.67   66.67
    Achieved Occupancy                        %   34.68   35.78   35.26
    ------------------------------- ----------- ------- ------- -------

  transposeSharedConflict(const float *, float *, int) (32, 32, 1)x(32, 32, 1), Device 0, CC 11.0, Invocations 14
    Section: Command line profiler metrics
    -------------------------------------------------------------- ----------- ------------ ------------ ------------
    Metric Name                                                    Metric Unit      Minimum      Maximum      Average
    -------------------------------------------------------------- ----------- ------------ ------------ ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                   1,016,796.00 1,016,893.00 1,016,842.21
    smsp__inst_executed_op_shared_ld.sum                                  inst    32,768.00    32,768.00    32,768.00
    smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum             1,048,576.00 1,048,576.00 1,048,576.00
    -------------------------------------------------------------- ----------- ------------ ------------ ------------

    Section: Occupancy
    ...
    Block Limit Shared Mem                block    1.00    1.00    1.00
    Block Limit Warps                     block    1.00    1.00    1.00
    Theoretical Occupancy                     %   66.67   66.67   66.67
    Achieved Occupancy                        %   65.08   65.36   65.27
    ------------------------------- ----------- ------- ------- -------

  transposeSharedPadded(const float *, float *, int) (32, 32, 1)x(32, 32, 1), Device 0, CC 11.0, Invocations 14
    Section: Command line profiler metrics
    -------------------------------------------------------------- ----------- --------- --------- ---------
    Metric Name                                                    Metric Unit   Minimum   Maximum   Average
    -------------------------------------------------------------- ----------- --------- --------- ---------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                      314.00    375.00    343.21
    smsp__inst_executed_op_shared_ld.sum                                  inst 32,768.00 32,768.00 32,768.00
    smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum             32,768.00 32,768.00 32,768.00
    -------------------------------------------------------------- ----------- --------- --------- ---------

    Section: Occupancy
    ...
    Block Limit Shared Mem                block    1.00    1.00    1.00
    Block Limit Warps                     block    1.00    1.00    1.00
    Theoretical Occupancy                     %   66.67   66.67   66.67
    Achieved Occupancy                        %   63.75   63.92   63.80
    ------------------------------- ----------- ------- ------- -------
```

## Step 4: Read the Report

`--page details` prints every launch with the rules ncu applies to it:

```bash
ncu --import shared_memory_opt.ncu-rep --page details
```

For the first launch of the tiled kernel, under its Memory Workload Analysis
Tables section:

```
    OPT   Est. Local Speedup: 96.88%
          The memory access pattern for shared loads might not be optimal and causes on average a 32.0 - way bank
          conflict across all 32768 shared load requests.This results in 1016797 bank conflicts,  which represent
          96.88% of the overall 1049565 wavefronts for shared loads. Check the Source Counters section for uncoalesced
          shared loads.
```

- **The tiled kernel's column reads conflict 32 ways.** 32,768 requests per
  launch: 1,024 blocks of 32 warps, each warp reading one column of its tile
  (`smsp__inst_executed_op_shared_ld.sum`). They took 1,048,576 wavefronts
  (`smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`): 32
  per request, the same on all 14 launches. That is the "32.0 - way bank
  conflict" of the rule: every lane of the warp reading the same bank, served
  one after another. ncu's counter reads 1,016,796 to 1,016,893 per launch,
  31.03 per request: the wavefronts its L1TEX unit served beyond one per
  request (on this launch, 1,049,565 - 32,768 = 1,016,797, the rule's own
  figures), which is 96.88% of them, 31 of every 32.
- **The padded kernel's do not.** 32,768 requests, 32,768 wavefronts: one
  each, on every launch. The counter reads 314 to 375, one for about every
  hundred requests, and ncu prints no rule for its shared loads. The L1TEX
  unit's wavefront counts run a little above the instruction stream's (989
  of 1,049,565 on the tiled kernel's launch above, and those few hundred
  here), an extra the tile's access pattern does not predict and this page
  does not explain; the check bounds it at 2% of the tiled kernel's count.
- **The naive kernel loads nothing from shared memory.** 0 requests, 0
  wavefronts, 0 conflicts. Its rule is about the writes the tile was
  introduced to fix, under the same section of its first launch:

  ```
      OPT   Est. Speedup: 49.27%
            The memory access pattern for global stores to L2 might not be optimal. On average, only 4.0 of the 32 bytes
            transmitted per sector are utilized by each thread. This applies to the 97.0% of sectors missed in L1TEX.
            This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced
            global stores.
  ```

  Each warp writes 32 floats a whole row apart, so every 32-byte sector it
  touches carries four useful bytes. That is what the tile fixed (1.34x in
  Step 1), and the padding then fixed the tile (1.89x).

- **Occupancy is the same for all three, and ncu says why.** Theoretical
  Occupancy 66.67%: `Block Limit Warps 1`, a 1,024-thread block being 32 of
  the SM's 48 warps and only one fitting; the tiled kernels' rule:

  ```
      OPT   Est. Local Speedup: 33.33%
            The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
            hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the required amount of
            shared memory, and the number of warps within each block.
  ```

  `Block Limit Shared Mem` is 8 for the naive kernel and 1 for the tiled
  ones, because the driver gave the tiled kernels an 8.19 KB shared-memory
  configuration that holds one tile plus its own 1.02 KB per block; the
  limit that binds is still the warps. The harness's `occupancy` column in
  Step 2 is this theoretical figure; the achieved figures (35.26%, 65.27%
  and 63.80% on average) are ncu's measurement.

What not to conclude:

- **That the padded kernel is 32 times faster.** The rule's "Est. Local
  Speedup: 96.88%" is ncu's estimate for the shared loads alone; the kernel
  also reads and writes global memory, and the padding took 1.89x off its
  time, not 32x.
- **That "conflict free" is what the counter reads.** The padded kernel
  still counts a few hundred conflicts per launch, and the counter moves by
  about a hundred between launches of the same kernel; the exact figures are
  the wavefronts per request, 32 and 1.
- **That every rule is about the loads.** In this session ncu also printed,
  for the padded kernel's stores, a rule of the same shape:

  ```
      OPT   Est. Local Speedup: 11.32%
            The memory access pattern for shared stores might not be optimal and causes on average a 1.1 - way bank
            conflict across all 32768 shared store requests.This results in 4183 bank conflicts,  which represent 11.32%
            of the overall 36951 wavefronts for shared stores. Check the Source Counters section for uncoalesced shared
            stores.
  ```

  A 1.1-way conflict on the stores is the same few-percent extra as above, on
  the store side; the tile's store pattern predicts none, and the rule did
  not appear on the tiled kernel's first launch in this session.

- **That the times under ncu mean anything.** They are replays (Step 3).
  ncu also prints, on every launch of the details page, that "Data
  collection happened without fixed GPU frequencies": it did not fix the
  clocks itself. This session had locked them with `jetson_clocks`, and the
  counts do not depend on them: with the clocks as found the counter read
  1,016,826 to 1,016,973 on this rig.

## What Should Reproduce

| Reading                                            | On this rig                                                                                                | Elsewhere                                                                                                                          |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| wavefronts per shared-load request, tiled / padded | 32 / 1, on every one of 14 launches each                                                                   | should match on a GPU with 32 four-byte banks: 32 / 1 on an RTX 5000 Ada laptop GPU                                                |
| ncu's bank-conflict counter per launch             | 1,016,796 to 1,016,893 / 314 to 375 over 14 launches each; 1,016,826 to 1,016,973 with the clocks as found | the same shape, with a larger extra on the padded kernel: 1,017,170 to 1,017,223 / 3,375 to 3,626 on the RTX 5000 Ada (ncu 2026.2) |
| padded against tiled, time                         | 1.89x (76.5 against 40.6 us); 1.88x to 1.89x over five locked runs                                         | the padded kernel should be faster; 1.6x on the RTX 5000 Ada                                                                       |
| tiled against naive                                | 1.34x; 1.34x to 1.35x                                                                                      | depends on the memory system: 2.4x on the RTX 5000 Ada                                                                             |
| padded against naive                               | 2.54x; 2.53x to 2.54x                                                                                      | 3.9x on the RTX 5000 Ada                                                                                                           |
| `occupancy` column                                 | 0.667 for all three                                                                                        | depends on the GPU's block and warp limits; 0.667 for all three on the RTX 5000 Ada                                                |
| absolute times                                     | 102.9 / 76.5 / 40.6 us                                                                                     | will differ                                                                                                                        |

The wavefronts per request are the reading to carry elsewhere; they depend
on the code and on the bank width, not on the machine. The five locked runs
are Step 1's command run five times in one session, the reference among
them; a range describes those runs, not a bound, and another run of the same
command can land outside it. The naive kernel is the noisiest measurement on
this page: CV 0.1% with the clocks locked, 1.0% with them as found. The RTX
5000 Ada figures are what one run of Step 1's command measured in the
project's `dev-cuda` container, with the clocks as found and other work
running on the laptop, and what the check read there (six launches per
kernel); they are stated for their direction, not their size.

## If It Does Not Match

- **`ncu` has no permission for the counters.** Run without `sudo` on this
  rig, the tests pass, ncu writes no metrics and exits 1:

  ```
  ...
  [ RUN      ] SharedMemoryOpt.NaiveGlobalMemory
  ==PROF== Connected to process 87374 (.../build/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt)
  ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
  [SharedMemoryOpt.NaiveGlobalMemory]  104.992 us/call  CV=0.0%  ~9.5K calls/s  (p10=104.992 p90=104.992 sd=0.000)
  [       OK ] SharedMemoryOpt.NaiveGlobalMemory (355 ms)
  ...
  [  PASSED  ] 1 test.
  ==PROF== Disconnected from process 87374
  ```

  Run it as Step 3 does, with `sudo env PATH="$PATH"`.

- **`bench run --profile ncu` shows no bank conflicts.** Its wrap collects
  ncu's default sections. Run on the padded kernel alone, as root:

  ```
  Running: ncu -o bench-out/BenchDemo_Gpu_03_SharedMemoryOpt.ncu/kernel_profile -f --target-processes all ./build/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt --cycles 1 --repeats 1 --profile ncu --gpu-warmup 1 --gtest_filter=SharedMemoryOpt.SharedPadded
  ...
  [ RUN      ] SharedMemoryOpt.SharedPadded
  ==PROF== Connected to process 87417 (.../build/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt)
  [WARN] Profiler 'ncu': unverified: collection is owned by the ncu wrap; completion is checked at exit
  ==WARNING== Unable to access the following 8 metrics: mcc__cycles_active.avg, mcc__cycles_active.max, mcc__cycles_active.min, mcc__cycles_active.sum, mcc__cycles_elapsed.avg, mcc__cycles_elapsed.max, mcc__cycles_elapsed.min, mcc__cycles_elapsed.sum.
  ...
  [nsight] external ncu wrap active; skipping in-process attach.
  [gpu] in-process CUPTI collection disabled for this run (external Nsight session or VERNIER_DISABLE_CUPTI); CUPTI CSV columns will be empty.
  ...
  [SharedMemoryOpt.SharedPadded]  888230.835 us/call  CV=0.0%  ~1 calls/s  (p10=888230.835 p90=888230.835 sd=0.000)
  [       OK ] SharedMemoryOpt.SharedPadded (7296 ms)
  ...
  [  PASSED  ] 1 test.
  ...
  ==PROF== Report: .../bench-out/BenchDemo_Gpu_03_SharedMemoryOpt.ncu/kernel_profile.ncu-rep
  ```

  The `[WARN]` line is the harness's readiness report for the profiler
  request, printed once per run: under `bench run`'s wrap the capture belongs
  to `ncu`, so the harness runs no check of its own and reports the request
  unverified (walkthrough 11 explains it). The eight `mcc__` metrics are not
  readable on this rig. The report holds four sections and no counter:

  ```
    transposeSharedPadded(const float *, float *, int) (32, 32, 1)x(32, 32, 1), Device 0, CC 11.0, Invocations 3
      Section: GPU Speed Of Light Throughput
      Section: GPU and Memory Workload Distribution
      Section: Launch Statistics
      Section: Occupancy
  ```

  Use Step 3's command for the counter.

- **`ncu` runs for minutes.** It replays every launch, 31 passes each with
  Step 3's sections and metrics, about 1.1 s per launch here. Keep
  `--cycles 3 --repeats 1`; the default 10,000 cycles per repeat would take
  hours per kernel.

- **The clocks were not locked.** The launches read a few percent slower and
  the naive kernel's CV rises to 1.0%; the ratios hold (1.85x to 1.86x and
  2.55x to 2.56x over three runs as found). One of them against the
  reference:

  ```
  Test                                         Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
  ---------------------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
  SharedMemoryOpt.NaiveGlobalMemory           102.91200     106.76900    +3.85700     +3.7%      0.1%      1.0%  neutral
  SharedMemoryOpt.SharedPadded                 40.57560      41.88320    +1.30760     +3.2%      0.0%      0.0%  neutral
  SharedMemoryOpt.SharedWithBankConflicts      76.54810      77.66450    +1.11640     +1.5%      0.0%      0.0%  neutral

    3 neutral

    Labels compare the median change against the 5.0% threshold.
    They describe the difference between two runs, not a significance test;
    the CV of each run is its own spread, not the spread between the runs.
  ```

- **The report names no kernel of the demo.** A `--gtest_filter` that
  matches none of the three tests profiles nothing; the kernels appear in
  the report as `transposeNaive`, `transposeSharedConflict` and
  `transposeSharedPadded`.

## Check Against the Reference

```bash
bench compare src/bench/demo/reference/thor/12_shared_memory_opt.csv shared_memory_opt.csv
```

A later run in the same session, clocks locked:

```
Test                                         Baseline     Candidate       Delta         %   Base CV   Cand CV        Result
---------------------------------------  ------------  ------------  ----------  --------  --------  --------  ------------
SharedMemoryOpt.NaiveGlobalMemory           102.91200     103.00900    +0.09700     +0.1%      0.1%      0.1%  neutral
SharedMemoryOpt.SharedPadded                 40.57560      40.60370    +0.02810     +0.1%      0.0%      0.0%  neutral
SharedMemoryOpt.SharedWithBankConflicts      76.54810      76.59630    +0.04820     +0.1%      0.0%      0.0%  neutral

  3 neutral

  Labels compare the median change against the 5.0% threshold.
  They describe the difference between two runs, not a significance test;
  the CV of each run is its own spread, not the spread between the runs.
```

`bench compare` labels a test `REGRESSION` when its median is more than the
threshold (5% unless `--threshold` says otherwise) above the reference's,
`IMPROVEMENT` when it is more than that below it, and `neutral` otherwise. The
labels describe the difference between two runs and are not a significance
test; the two CV columns are each run's own spread, not the spread between
the runs. The kernel rows on this page hold still: the widest spread across
the five locked runs was the naive kernel's, 102.91 to 103.05 us.

The reference CSV is one of five runs of Step 1's command captured with the
clocks locked: the one whose three medians sit closest to the median of each
test across the five. It was captured from a copy of the development tree
without its git metadata, so its `gitHash` column reads `unknown`; its
`hostname` column reads `kalex`, the name this board reports.

## What Keeps This Page True

Three things check what this page shows, and all fail loudly:

- `BankConflicts.CountedByNsightCompute`, a test program of its own beside
  the demo
  ([`03_SharedMemoryOpt_BankConflicts_uTest.cpp`](../gpu/utst/03_SharedMemoryOpt_BankConflicts_uTest.cpp),
  built as `TestDemoBankConflicts`), runs the demo binary under `ncu` with
  Step 3's three metrics on three launches per kernel and reads ncu's CSV.
  It fails unless every launch reads what this page says: no shared load in
  the naive kernel; one request per warp in the tiled kernels; 32 wavefronts
  per request for the tiled kernel and its counter at least 30 per request;
  one wavefront per request for the padded kernel and its counter below 2%
  of the tiled kernel's. A kernel the report names no launch of fails it, and
  so does a padded launch whose counter the report leaves out or gives as
  other than a whole number. It skips where `ncu` is not on `PATH` or no CUDA
  device is present, and where ncu refuses the counters or has no value for a
  metric, quoting ncu's own line. It is registered with `ctest` under the
  `demo` and `ncu` labels, so an ordinary test run includes it. On this rig
  the counters are for administrators, so as your user
  `ctest --test-dir build -L demo` reports it skipped, quoting the refusal;
  to run it:

  ```bash
  sudo env PATH="$PATH" ctest --test-dir build -L ncu
  sudo chown -R "$(id -u):$(id -g)" build
  ```

  A run that fails keeps the check's temporary directory,
  `vernier-demo-gpu03-` followed by six random characters, with ncu's CSV and
  the demo's output in it. Run as root, the check leaves it under `/tmp`,
  owned by root, and the `chown` of `build` above does not reach it. Read it
  as root, then remove it:

  ```bash
  sudo rm -rf /tmp/vernier-demo-gpu03-*
  ```

- `TestDemoTranspose`
  ([`03_SharedMemoryOpt_uTest.cu`](../gpu/utst/03_SharedMemoryOpt_uTest.cu)),
  under the `demo` label, holds the three kernels to a CPU transpose at seven
  matrix sizes (1, 7, 31, 32, 33, 100 and 1,024) and each kernel's declared
  shared memory to the bytes the demo hands the harness.
- The demo's own tests fail unless the kernel they measured transposed the
  input.

An ordinary test run includes the first two, and every test it runs should
pass:

```bash
ctest --test-dir build -L demo
```

The demo's timing tests are not registered: what they measure belongs to the
machine they run on. This repository has no continuous-integration lane on
the reference board, so before a release the page's commands are run on the
rig by hand, and the page and its reference CSV are re-captured when what
they show changes.

## See Also

- [Demo 10: GPU Basic Workflow](10_GPU_BASIC_WORKFLOW.md) -- the GPU harness
  and its CSV columns
- [Demo 11: Nsight Systems and Nsight Compute](11_NSIGHT_PROFILER.md) -- `bench run --profile ncu`,
  what a replay does to a kernel's time, and the occupancy rows read in full
- [GPU Guide: understanding occupancy](../../docs/GPU_GUIDE.md#understanding-occupancy)
- [Reference Rig: NVIDIA Jetson AGX Thor](../../docs/rigs/RIG_THOR_AGX.md)
- [Demos README](../README.md) -- every demo, and the contract each
  walkthrough meets
