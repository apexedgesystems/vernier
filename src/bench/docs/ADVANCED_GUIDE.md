# Advanced Benchmarking Guide

**Namespace:** `vernier::bench`
**Platform:** Linux-only
**C++ Standard:** C++20 (C++23 used when available)

Complete guide to advanced features, patterns, and best practices for the benchmarking framework.

---

## Table of Contents

- [Memory Profiling Deep Dive](#memory-profiling-deep-dive)
- [Parameterized Tests](#parameterized-tests)
- [CSV Schema Reference](#csv-schema-reference)
- [Profiler Integration](#profiler-integration)
- [Quick Iteration Mode](#quick-iteration-mode)
- [Custom Configuration](#custom-configuration)
- [GPU Advanced Topics](#gpu-advanced-topics)

---

## Memory Profiling Deep Dive

### Understanding MemoryProfile

`MemoryProfile` tracks memory operations to calculate bandwidth and identify CPU-bound vs memory-bound code.

```cpp
struct MemoryProfile {
  size_t bytesRead;       // Input data read from memory
  size_t bytesWritten;    // Output data written to memory
  size_t bytesAllocated;  // Heap allocations during operation

  // Calculate total bandwidth (MB/s) -- read + write only (allocations excluded)
  double bandwidthMBs(double durationUs) const {
    const double totalBytes = bytesRead + bytesWritten;
    const double durationS = durationUs / 1e6;
    return (totalBytes / durationS) / 1e6;  // MB/s
  }

  // Calculate efficiency vs theoretical peak
  double efficiency(double durationUs, double peakMBs) const {
    return (bandwidthMBs(durationUs) / peakMBs) * 100.0;
  }
};
```

### Field Explanations

**`bytesRead`** - Total bytes loaded from memory into CPU/cache:

```cpp
// Reading array
bytesRead = sizeof(data);

// Reading struct members
bytesRead = sizeof(input.field1) + sizeof(input.field2);

// Reading via pointer dereference
bytesRead = input_size_in_bytes;
```

**`bytesWritten`** - Total bytes stored from CPU/cache to memory:

```cpp
// Writing array
bytesWritten = sizeof(output);

// Writing struct members
bytesWritten = sizeof(result.x) + sizeof(result.y);

// Modifying in-place
bytesWritten = bytesRead;  // Read-modify-write
```

**`bytesAllocated`** - Heap allocations during operation:

```cpp
// No allocations (typical for performance tests)
bytesAllocated = 0;

// Testing allocation overhead
bytesAllocated = malloc_size * allocation_count;

// Vector resize
bytesAllocated = vec.capacity() * sizeof(T);
```

### When to Use Each Field

| Scenario                | bytesRead     | bytesWritten   | bytesAllocated |
| ----------------------- | ------------- | -------------- | -------------- |
| Read-only scan          | Size of input | 0              | 0              |
| Write-only fill         | 0             | Size of output | 0              |
| Transform (read->write) | Input size    | Output size    | 0              |
| In-place modify         | Data size     | Data size      | 0              |
| With allocation         | Input size    | Output size    | Alloc size     |

### Bandwidth vs Throughput

**Throughput:** How much data your algorithm processes

```cpp
// Process 1MB of payload data
throughput = 1 MB / time = 1000 MB/s
```

**Bandwidth:** Total memory system utilization

```cpp
// But you read 1MB AND write 1MB
bandwidth = (1 MB read + 1 MB write) / time = 2000 MB/s
```

**Key insight:** Bandwidth can be much higher than throughput!

### Complete Example

```cpp
PERF_TEST(Codec, Encode) {
  PERF_GUARD(perf);  // Also attaches the profiler selected by --profile

  const size_t INPUT_SIZE = 1024;   // 1KB input
  const size_t OUTPUT_SIZE = 2048;  // 2KB output (with framing overhead)

  std::vector<uint8_t> input(INPUT_SIZE);
  std::vector<uint8_t> output(OUTPUT_SIZE);

  // Define memory profile
  ub::MemoryProfile memProfile{
    .bytesRead = INPUT_SIZE,      // Read entire input
    .bytesWritten = OUTPUT_SIZE,  // Write entire output
    .bytesAllocated = 0           // No heap allocations
  };

  perf.warmup([&] {
    for (int i = 0; i < perf.cycles(); ++i) {
      volatile int result = encode(input.data(), INPUT_SIZE,
                                   output.data(), OUTPUT_SIZE);
      (void)result;
    }
  });

  volatile int sink = 0;
  auto result = perf.throughputLoop([&] {
    sink += encode(input.data(), INPUT_SIZE, output.data(), OUTPUT_SIZE);
  }, "encode", memProfile);

  // Output shows:
  // - Throughput: 1000 MB/s of input processed
  // - Bandwidth: 3000 MB/s total memory traffic (1MB read + 2MB write)
  // - Efficiency: 25% of DDR4 peak (~12 GB/s)

  (void)sink;
}
```

### When to Omit MemoryProfile

**Omit when:**

- Quick development iteration
- Testing pure compute (no memory I/O)
- Bandwidth isn't relevant to your optimization

```cpp
// Simple version - just get throughput
auto result = perf.throughputLoop([&] {
  work();
}, "simple");
// CSV: wallMedian, callsPerSecond (same schema either way)
```

**Include when:**

- Analyzing memory-bound code
- Need bandwidth in reports
- Identifying cache effects
- Production benchmarks

```cpp
// Detailed version - with bandwidth analysis
ub::MemoryProfile mp{INPUT_SIZE, OUTPUT_SIZE, 0};
auto result = perf.throughputLoop([&] {
  work();
}, "detailed", mp);
// CSV schema is unchanged; bandwidth from the MemoryProfile is printed to the
// console, not written as extra CSV columns.
```

**Performance impact:** None - it's just metadata for the console bandwidth report.

---

## Parameterized Tests

### PERF_TEST_P

`PERF_TEST_P(Fixture, Name)` is an alias of GoogleTest's `TEST_P`, as
`PERF_TEST_F` is of `TEST_F` (`PerfTestMacros.hpp`), so either spelling works.
A parameterized test needs what `TEST_P` needs: a fixture derived from
`::testing::TestWithParam<T>` and an `INSTANTIATE_TEST_SUITE_P` for its values.
`PERF_GUARD` works inside it when every value runs with the global
configuration. To give each value a configuration of its own, build the
`PerfCase` yourself and attach the profiler hooks to it:

```cpp
class MyTest : public ::testing::TestWithParam<int> {
protected:
  const ub::PerfConfig& getCfg() {
    return ub::detail::getPerfConfig();
  }
};

PERF_TEST_P(MyTest, Case) {  // Same as TEST_P(MyTest, Case)
  const int size = GetParam();

  ub::PerfConfig cfg = getCfg();
  cfg.msgBytes = size;

  std::string testName = ::testing::UnitTest::GetInstance()
                           ->current_test_info()->test_suite_name() +
                         std::string(".") +
                         ::testing::UnitTest::GetInstance()
                           ->current_test_info()->name();

  ub::PerfCase perf{testName, cfg};
  ub::attachProfilerHooks(perf, cfg);  // A PerfCase you build needs this

  // Test implementation...
}

INSTANTIATE_TEST_SUITE_P(
  Params,
  MyTest,
  ::testing::Values(64, 256, 1024)
);
```

GoogleTest names these cases `Params/MyTest.Case/0`, `Params/MyTest.Case/1`
and `Params/MyTest.Case/2`, and `testName` above has the same form.

### Standard Pattern with PERF_GUARD

For simple tests without parameters:

```cpp
PERF_TEST(MyComponent, Basic) {
  PERF_GUARD(perf);  // Names the case "MyComponent.Basic" and attaches the profiler

  // Test code...
}
```

### Custom Configuration Pattern

For parameter sweeps, manually construct `PerfCase` with custom config:

```cpp
PERF_TEST(MyComponent, PayloadSweep) {
  for (int payloadSize : {64, 256, 1024, 4096}) {
    // Get default config
    ub::PerfConfig cfg = ub::detail::getPerfConfig();

    // Override specific field
    cfg.msgBytes = payloadSize;

    // Create custom PerfCase (can't use PERF_GUARD here!)
    std::string testName = "MyComponent.PayloadSweep/" + std::to_string(payloadSize);
    ub::PerfCase perf{testName, cfg};
    ub::attachProfilerHooks(perf, cfg);

    // Setup test data
    std::vector<uint8_t> data(payloadSize);

    // Run test
    perf.warmup([&] {
      for (int i = 0; i < perf.cycles(); ++i) {
        volatile auto result = process(data.data(), payloadSize);
        (void)result;
      }
    });

    volatile uint64_t sink = 0;
    auto result = perf.throughputLoop([&] {
      sink += process(data.data(), payloadSize);
    }, "throughput");

    // Each iteration gets separate CSV row with different msgBytes
    (void)sink;
  }
}
```

`attachProfilerHooks()` gives the case the profiler `--profile` names;
`ub::makePerfCaseWithProfiler(testName, cfg)` constructs the case and
attaches it in one call. A `PerfCase` constructed without either gets no
profiler (see [attachProfilerHooks()](#attachprofilerhooks-function)).

### Why Custom Config?

**`PERF_GUARD(perf)` limitations:**

- Gets test name from GoogleTest automatically
- Uses global config (can't override per-parameter)
- All parameter values would have same `msgBytes` in CSV

**Manual `PerfCase` constructor:**

- Set custom test name with parameter value
- Override config fields per parameter
- Each parameter gets unique CSV row

### GoogleTest Parameterized Tests

For more complex parameter combinations, use standard GoogleTest fixtures:

```cpp
class PayloadTest : public ::testing::TestWithParam<int> {
protected:
  const ub::PerfConfig& getCfg() {
    return ub::detail::getPerfConfig();
  }
};

TEST_P(PayloadTest, Encode) {  // PERF_TEST_P(PayloadTest, Encode) is the same
  const int payloadSize = GetParam();

  // Custom config with parameter
  ub::PerfConfig cfg = ub::detail::getPerfConfig();
  cfg.msgBytes = payloadSize;

  // Custom name with parameter
  std::string testName = ::testing::UnitTest::GetInstance()
                           ->current_test_info()->test_suite_name() +
                         std::string(".") +
                         ::testing::UnitTest::GetInstance()
                           ->current_test_info()->name();

  ub::PerfCase perf{testName, cfg};
  ub::attachProfilerHooks(perf, cfg);

  // Test implementation...
}

INSTANTIATE_TEST_SUITE_P(
  PayloadSizes,
  PayloadTest,
  ::testing::Values(64, 256, 1024, 4096)
);
```

**CSV output** (`test` and `msgBytes` columns, when the implementation measures
once per case):

```
test,msgBytes
PayloadSizes/PayloadTest.Encode/0,64
PayloadSizes/PayloadTest.Encode/1,256
PayloadSizes/PayloadTest.Encode/2,1024
PayloadSizes/PayloadTest.Encode/3,4096
```

GoogleTest puts the `INSTANTIATE_TEST_SUITE_P` prefix first and numbers the
values in order, so the value itself is in the `msgBytes` column, not the name.

### Key Points

1. **Use `PERF_GUARD`** for simple tests with no parameters
2. **Use manual constructor** for parameter sweeps
3. **Always set `cfg.msgBytes`** to actual payload size
4. **Include parameter in test name** for CSV clarity
5. **Call `attachProfilerHooks`** with your custom config on a `PerfCase` you
   construct; `PERF_GUARD` attaches the hooks itself

---

## CSV Schema Reference

`--csv` writes one row per measurement. The API reference's
[CSV Schema](API_REFERENCE.md#csv-schema) gives the columns, their order,
what each holds and when a cell is empty. `MemoryProfile`'s bandwidth is
printed to the console, not written to the file.

### Using CSV Data

**Plot scaling curves:**

```bash
bench-plot plot results.csv --output plots/
```

**Compare implementations:**

```bash
bench compare baseline.csv optimized.csv
```

Each test both CSVs report is labelled by how far its median moved against
the threshold (5% by default), and tests only one of the two ran are listed
separately. The labels compare two runs; they are not a significance test.

**Choose the threshold:**

```bash
bench compare baseline.csv candidate.csv --threshold 5
```

**Filter by test**: the header line, which `bench` and `bench-plot` read the
columns from, and the rows whose test starts with a name. A CSV reader keeps
a quoted test name (a measurement's label may hold a comma) whole, where a
line-based filter would drop it or split it:

```bash
python3 -c '
import csv, sys
path, prefix = sys.argv[1:]
rows = csv.DictReader(open(path, newline=""))
out = csv.DictWriter(sys.stdout, rows.fieldnames, lineterminator="\n")
out.writeheader()
for row in rows:
    if None in row or None in row.values():
        sys.exit(f"{path}: a row whose cells do not match the header")
    if row["test"].startswith(prefix):
        out.writerow(row)
' results.csv MyComponent.Encode > encode_only.csv
```

---

## Profiler Integration

### attachProfilerHooks() Function

```cpp
void attachProfilerHooks(PerfCase& perf, const PerfConfig& cfg);
```

**Purpose:**
Attaches the profiler that `cfg.profileTool` names (the `--profile` flag) to a
`PerfCase` you construct yourself, so that `--profile` covers its measured
phase. `PERF_GUARD` calls it for you, so a test that uses `PERF_GUARD` needs no
call of its own.

**When to call:**

- Right after constructing a `PerfCase` with a configuration of your own, as in
  the parameterized and sweep patterns above

**What it does:**

1. Creates the backend that `cfg.profileTool` names (a no-op when no
   `--profile` was given), after checking that the request can run here; a
   request that cannot gets no backend, and the run exits with status 4
   after a report of the failure (see
   [Troubleshooting](TROUBLESHOOTING.md#--profile--failed-and-exit-status-4))
2. Starts it before the measured phase and stops it after, through the
   PerfCase's before/after measure hooks
3. Records the tool and its artifact folder for the CSV's `profileTool` and
   `profileDir` columns

**Which cases get a profiler:** only the cases that create one, through
`PERF_GUARD` (`UB_PERF_GUARD`), `makePerfCaseWithProfiler()` or
`attachProfilerHooks()`, and on the GPU `PERF_GPU_GUARD` or
`attachGpuProfilerHooks()`. A case built with `PERF_GUARD_NOPROFILE`, or a
bare `PerfCase`, gets none.

**What happens if omitted** (on a `PerfCase` you construct): the test runs, but
`--profile` attaches no profiler to it: the backend does nothing around its
measured phase (no perf counting, no callgrind instrumentation window, no NVTX
range) and no artifact folder is created for it. A tool that runs the whole
process, such as valgrind, heaptrack, nsys, ncu or compute-sanitizer started
by `bench run` or by hand, still records the case with the rest of the
process. When no case that ran created a profiler, the run ends with a notice
and its exit status is unchanged:

```
[profile] --profile perf: no case that ran was built with the profiler guard, so nothing was profiled.
```

and, under `bench run`'s wrap of a tool:

```
[profile] --profile massif: no case that ran was built with the profiler guard; the massif wrap still recorded the whole process.
```

**Example:**

```cpp
PERF_TEST(MyComponent, Throughput) {
  ub::PerfConfig cfg = ub::detail::getPerfConfig();
  cfg.msgBytes = 256;

  ub::PerfCase perf{"MyComponent.Throughput", cfg};
  ub::attachProfilerHooks(perf, cfg);  // Without it, --profile skips this case

  // If user runs: ./test --profile perf
  // this case's measured phase is profiled
}
```

### Available Profilers

Every backend below self-registers via the profiler registry; `bench doctor`
(or `--profile-check`) walks the list and reports each one's environment
readiness with the exact remediation hint. "Under the tool" means the process
must be started by that tool: `bench run --profile <name>` starts it (for
every name marked so but `rocprof`), or you start it yourself; a run started
without it fails with the command that would start it.

In **Writes**, `<case>` is the test's name, `Suite.Test`: a per-case folder
is created in the working directory, and the folder `bench run` creates,
`<binary>.<profiler>/`, in `bench-out/`; `--profile-output-dir` moves either
root. **Cost** is what a walkthrough measured, on the rig it names; it depends
on the code and the machine, so time one profiled call before sizing a run.

| Profiler             | Layer | Purpose                                                             | Requirements                                                                          | Writes                                                                                                                  | Cost                                                                                                                     |
| -------------------- | ----- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `perf`               | CPU   | Hardware counters (`stat`/`record`/`mem`/`c2c`)                     | perf for the running kernel; `perf_event_paranoid` 2 or lower, or root                | `<case>.perf/stat.txt`; `perf.data` with `record`                                                                       | none seen in `stat` mode: 919.7 against 926.9 to 1,002.6 us a call, Pi 4 ([02](../demo/docs/02_PERF_PROFILER.md))        |
| `gperf`              | CPU   | gperftools sampling (CPU; heap with `--profile-args heap`)          | libgperftools; heap mode also a build with `-DVERNIER_LINK_TCMALLOC=ON`               | `<case>.gperf/cpu.prof`; `heap.<N>.heap` in heap mode                                                                   | about 1%, within the runs' own spread, at 100 samples a second, Pi 4 ([03](../demo/docs/03_GPERF_PROFILER.md))           |
| `callgrind`          | CPU   | Deterministic instruction counts                                    | valgrind, under the tool                                                              | `<binary>.callgrind/callgrind.out`                                                                                      | 50x and 94x, Pi 4 ([07](../demo/docs/07_CALLGRIND_PROFILER.md))                                                          |
| `bpftrace`           | CPU   | Kernel tracing (fsync / write latency, etc.)                        | bpftrace as root (`BENCH_SUDO=1` with a sudoers grant)                                | `<case>.bpf/<script>.out.text`                                                                                          | per traced event: about 1.5 us a `write()` under `write_latency.bt`, Pi 4 ([09](../demo/docs/09_BPFTRACE_PROFILER.md))   |
| `rapl`               | CPU   | Package energy per call                                             | Intel CPU, msr driver, read access to `/dev/cpu/0/msr` (see below)                    | `<case>.rapl/energy.txt`                                                                                                | two register reads a window, outside the timed calls ([08](../demo/docs/08_RAPL_PROFILER.md))                            |
| `massif`             | CPU   | Heap usage timeline                                                 | valgrind, under the tool                                                              | `<binary>.massif/massif.out`                                                                                            | 2.3x and 6.2x, Pi 4 ([14](../demo/docs/14_MASSIF_PROFILER.md))                                                           |
| `memcheck`           | CPU   | Memory errors and leaks                                             | valgrind, under the tool                                                              | `<binary>.memcheck/memcheck.log`                                                                                        | about 115x, Pi 4 ([15](../demo/docs/15_MEMCHECK_PROFILER.md))                                                            |
| `helgrind`           | CPU   | Data races and lock-order violations; DRD with `--profile-args drd` | valgrind, under the tool                                                              | `<binary>.helgrind/helgrind.log`                                                                                        | thousands of times, five times apart between runs, Pi 4 ([20](../demo/docs/20_HELGRIND_PROFILER.md))                     |
| `offcpu`             | CPU   | Off-CPU stack profiling                                             | bpftrace as root (`BENCH_SUDO=1` with a sudoers grant), tracefs                       | `<case>.offcpu/offcpu.txt`                                                                                              | not measured: timed medians with and without it overlapped, Pi 4 ([16](../demo/docs/16_OFFCPU_PROFILER.md))              |
| `heaptrack`          | CPU   | Heap allocations and their call stacks                              | heaptrack, under the tool                                                             | `<binary>.heaptrack/run.zst` (or `run.gz`)                                                                              | 1.4x to 1.5x and 4.2x to 4.3x, growing with the allocation rate, Pi 4 ([21](../demo/docs/21_HEAPTRACK_PROFILER.md))      |
| `jemalloc`           | CPU   | jemalloc's sampling heap profiler                                   | jemalloc built with `--enable-prof`, preloaded                                        | `<binary>.jemalloc/jeprof.<pid>.<n>.f.heap`                                                                             | 1.01x and 1.73x, growing with the bytes allocated, Pi 4 ([22](../demo/docs/22_JEMALLOC_PROFILER.md#what-sampling-costs)) |
| `nsight` (or `nsys`) | GPU   | Nsight Systems / Compute (auto stats)                               | nsys (Systems) or ncu (the compute modes), under the tool                             | `<binary>.nsight/profile.nsys-rep` and the nsys summaries; `kernel_profile.ncu-rep` there with `--profile-args compute` | none seen: 3,367 against 3,439 us a call, Thor ([11](../demo/docs/11_NSIGHT_PROFILER.md))                                |
| `ncu`                | GPU   | Nsight Compute per-kernel counters                                  | ncu, under the tool; the GPU's performance counters, which a driver can keep for root | `<binary>.ncu/kernel_profile.ncu-rep`                                                                                   | about a second a profiled launch, Thor ([11](../demo/docs/11_NSIGHT_PROFILER.md))                                        |
| `compute-sanitizer`  | GPU   | GPU memcheck / racecheck / synccheck / initcheck                    | compute-sanitizer (CUDA toolkit), under the tool                                      | `<binary>.compute-sanitizer/sanitizer.log`                                                                              | 54x to 57x, Thor; about 116x, RTX 5000 Ada laptop ([17](../demo/docs/17_COMPUTE_SANITIZER.md))                           |
| `rocprof`            | GPU   | AMD ROCm GPU profiler, not validated on AMD hardware                | rocprof (ROCm), under the tool; never reported ready                                  | rocprof's files, where its `-o` says                                                                                    | not measured ([18](../demo/docs/18_ROCPROF_PROFILER.md))                                                                 |

CUPTI activity counters (per-launch register count, static + dynamic shared
memory, kernel count) populate the GPU CSV section on every `PERF_GPU_*`
test without requiring `--profile` -- spawning external `ncu` is unnecessary
for those metrics.

### Window sizing without the guesswork

Microsecond-scale operations cannot hold tight CV bars in tiny
measurement windows -- and the right `--cycles` differs per test size.
`--target-time 100ms` sizes the window instead of the iteration count:
the harness times a batch of calls before the measured phase (doubled
until the sample spans about 1 ms) and picks cycles so a repeat targets
the requested wall time (per test, so a parameterized sweep gets a
per-size cycle count automatically). Applies where the harness owns
the loop (`throughputLoop`, `contentionRun`); bodies driven by raw
`measured()` keep explicit `--cycles`.
The floor is one cycle: a call that already outlasts the target runs
once per repeat.
The sizing is best effort. Taking the least per-call time among the
calibration samples reduces isolated scheduling interference, but a
first call stretched past the target, or consecutive stretched samples,
can still make the repeats shorter than the target. An operation slower
than about 1 ms but shorter than the target is called three times to
calibrate.

Run `bench doctor <ptest-binary>` (or `<ptest-binary> --profile-check`) to
see each backend's status on the current host. For CI, the same data is
machine-readable and enforceable:

```bash
# Record the host's capability manifest as a build artifact (gate lanes):
bench doctor MyComponent_PTEST --json > doctor.json

# Fail a profile lane fast unless its tools are actually ready
# ("warn" is not ready: --require accepts only [OK]):
bench doctor MyComponent_PTEST --require offcpu,heaptrack

# Check the request the lane runs, mode included, as bench run passes it:
bench doctor MyComponent_PTEST --profile massif --profile-args pages --require massif
```

`bench doctor` takes the binary's full path. With `--json` its stdout is one
JSON document and the `--require` lines go to stderr.

Assess-and-record on gate lanes, assess-and-enforce on profile lanes;
the doctor never installs anything.

### Using perf

```bash
# Enable perf (requires root, one-time setup)
sudo sysctl -w kernel.perf_event_paranoid=-1

# Run test with perf profiling, recording call stacks for a report
./MyComponent_PTEST --profile perf --profile-args "record -g" --target-time 250ms \
    --csv results.csv

# Analyze results
perf report -i MyComponent.Throughput.perf/perf.data
perf annotate -i MyComponent.Throughput.perf/perf.data
```

**Artifacts:** each profiled test gets a `<Test>.perf/` folder: `stat.txt` from
the default `perf stat`, or `perf.data` from `--profile-args "record ..."`. The
`profileTool` and `profileDir` CSV columns record the tool and that folder.
Counter detail lives in those files, not in extra CSV columns.

### Using RAPL

```bash
# Needs read access to /dev/cpu/0/msr: root, or CAP_SYS_RAWIO together with
# read permission on the device
sudo ./MyComponent_PTEST --profile rapl

# Each case's report
cat MyComponent.Throughput.rapl/energy.txt
```

**Access:** the msr driver checks CAP_SYS_RAWIO, and the device's file mode
(`0600 root` by default) is a separate requirement that the capability does
not lift. Vernier does not raise its privileges for RAPL; a run without
access fails with exit status 4, naming the cause (`denied`, `missing`,
`unsupported`), and creates no folder.

**Energy report (`<Test>.rapl/energy.txt`):** a header (the CPU, the scope:
package 0 and nothing off it, the counter's 32-bit wrap, the per-call
definition), then one numbered section per measured window, in the order of
the case's CSV rows:

- `Energy consumed` - energy over the window (J): the counter's delta in units
- `Duration` - the window's wall-clock duration (s)
- `Average power` - energy over duration (W)
- `Energy per operation` - average power x median time per call (mJ/call)
- `Median time per call`, `Energy unit`, `Start counter`, `End counter`,
  `Delta counter`

The counter is the whole package's, whatever runs on it. Energy stays in this
file, not in CSV columns: the row's `profileDir` names the folder.

### Custom Profilers

See API_REFERENCE.md for `Profiler` base class interface to implement custom profilers.

---

## Quick Iteration Mode

### What --quick Does

The `--quick` flag reduces test duration for fast feedback during development:

```bash
./MyComponent_PTEST --quick --csv results.csv
```

**Default values:**

```cpp
// Normal mode (production)
cycles = 10000;
repeats = 10;
warmup = 1;

// Quick mode (if not explicitly overridden)
cycles = 5000;   // 2x fewer operations
repeats = 5;     // 2x fewer samples
warmup = 2;      // Extra warmup to stabilize
```

**Performance impact:**

- With the default counts, 25,000 measured calls instead of 100,000
  (5,000 x 5 against 10,000 x 10, per thread under `contentionRun()`). The
  time this saves depends on the workload: work that grows with the calls
  takes a quarter as long, while work done once per repeat inside the
  measured window, such as a GPU test's copies or the start and join of
  `contentionRun()`'s threads, only halves
- Less statistical confidence (higher CV%)
- May miss outliers/edge cases
- Same CSV format (compatible with analysis tools)

### When to Use Quick Mode

**GOOD: Use --quick for:**

- Active development (tight iteration loop)
- Debugging test failures
- Sanity checking builds
- Rapid experimentation
- Local development

**BAD: Don't use --quick for:**

- Production benchmarks
- CI/CD regression testing
- Publishing results
- Comparing implementations
- Performance validation

### Development Workflow

```bash
# Development (fast feedback)
make
./test --quick --csv dev.csv
# Make changes...
make
./test --quick --csv dev2.csv
bench compare dev.csv dev2.csv

# Validation (more samples)
./test --cycles 50000 --repeats 20 --csv validate.csv

# Production (full characterization)
./test --cycles 100000 --repeats 30 --csv production.csv
```

### Quick Mode and CV%

Quick mode may show higher coefficient of variation:

| Mode       | Typical CV% | Acceptable?        |
| ---------- | ----------- | ------------------ |
| Production | <5%         | Good               |
| Normal     | <10%        | Good               |
| Quick      | <15%        | Acceptable for dev |

Use `EXPECT_STABLE_CV_CPU` - it automatically relaxes thresholds in quick mode.

---

## Custom Configuration

### PERF_MAIN() Macro

```cpp
PERF_MAIN()  // Replaces int main()
```

**Purpose:**
Provides complete main() function with:

- GoogleTest initialization
- Command-line parsing (`--csv`, `--profile`, `--quick`, etc.)
- PerfConfig singleton setup
- CSV export after all tests
- Profiler lifecycle management
- An exit status for CI/CD: the tests' status, or 4 when the tests passed and
  the requested `--profile` failed

**Usage:**
Place at end of test file:

```cpp
#include <gtest/gtest.h>
#include "Perf.hpp"

namespace ub = vernier::bench;

PERF_TEST(MyComponent, Test1) {
  // ...
}

PERF_TEST(MyComponent, Test2) {
  // ...
}

PERF_MAIN()  // That's it - no custom main() needed!
```

**What it expands to:**

```cpp
int main(int argc, char** argv) {
  // 1. Check that this benchmark and the libbench it loaded share one layout
  vernier::bench::ensureBenchAbi();

  // 2. Parse performance flags
  auto& cfg = vernier::bench::detail::perfConfigSingleton();
  vernier::bench::parsePerfFlags(cfg, &argc, argv);

  // 3. Register global config for CSV export
  vernier::bench::setGlobalPerfConfig(&cfg);

  // 4. Install CSV listener
  vernier::bench::installPerfEventListener(cfg);

  // 5. Initialize GoogleTest
  ::testing::InitGoogleTest(&argc, argv);

  // 6. Run all tests
  const int rc = RUN_ALL_TESTS();

  // 7. Warn when --profile was given and the filter matched no test
  VERNIER_WARN_IF_NO_TESTS_RAN_UNDER_PROFILE(cfg);

  // 8. Report each failure of the --profile request and return the run's
  //    status: the tests' status, or 4 when they passed and the profile failed
  return vernier::bench::ProfilerRegistry::instance().finishRun(
      cfg, rc, ::testing::UnitTest::GetInstance()->test_to_run_count());
}
```

Step 8 comes after `RUN_ALL_TESTS()` returns, in its own statement: GoogleTest
counts the selected tests during the run.

**When NOT to use:**

- Custom main() logic needed
- Embedding tests in larger application
- Multiple test binaries with shared setup

In those cases, manually implement the steps above.

### Accessing Global Config

```cpp
// Get read-only config in tests
const ub::PerfConfig& cfg = ub::detail::getPerfConfig();

// Check flags
if (cfg.quickMode) {
  // Adjust test behavior
}

// Use values
std::printf("Running with %d cycles\n", cfg.cycles);
```

### Command-Line Flag Reference

**Core flags:**

```bash
--cycles N        # Operations per repeat (default: 10000)
--repeats N       # Samples collected (default: 10)
--warmup N        # Warmup iterations (default: 1, 0=auto)
--threads N       # Worker threads (default: 1)
--msg-bytes N     # Payload size (default: 64)
--quick           # Fast mode (reduced cycles/repeats)
```

**Output flags:**

```bash
--csv PATH        # CSV output file
--console         # Echo to console (default: false)
--nonblocking     # Non-blocking mode
--min-level STR   # Minimum log level (default: INFO)
```

**Profiling flags:**

```bash
--profile TOOL         # Profiler: perf|gperf|bpftrace|rapl|callgrind|massif|memcheck|helgrind|offcpu|heaptrack|jemalloc|nsight (or nsys)|ncu|compute-sanitizer|rocprof
--profile-args ARGS    # Profiler-specific arguments
--artifact-root DIR    # Output directory (default: .)
--profile-frequency N  # gperf CPU sampling rate, 1 to 4000 Hz (default: gperftools' 100); bench run applies it, see API_REFERENCE.md
--profile-analyze      # Run the profile's analysis (callgrind: bench run, after valgrind exits)
--bpf LIST             # BPF script names/paths (comma-separated): fsync_latency,write_latency
--bpf-scripts DIR      # Directory --bpf names are looked up in (sets PERF_BPF_SCRIPTS)
```

**GPU flags:**

```bash
--gpu-device N      # CUDA device ID (default: 0)
--gpu-warmup N      # GPU warmup iterations (default: 10)
--gpu-memory MODE   # Strategy for the tests that read it: explicit|unified|pinned|mapped; others exit 2
--min-speedup F     # Minimum expected speedup vs CPU
--capture-um        # Unified-memory counters from CUPTI (where it records them)
```

**Example combinations:**

```bash
# Development
./test --quick --csv dev.csv

# Production CI
./test --cycles 100000 --repeats 30 --csv ci_results.csv

# With profiling
./test --profile perf --gtest_filter="*Encode*"

# Multi-threaded
./test --threads 8 --csv mt_results.csv

# GPU test
./test --gpu-device 0 --gpu-memory unified --csv gpu_results.csv
```

---

## GPU Advanced Topics

### GPU Parameterized Tests

GPU tests follow the same pattern as CPU but use `PERF_GPU_GUARD`:

```cpp
class GpuPayloadTest : public ::testing::TestWithParam<int> {
protected:
  const ub::PerfConfig& getCfg() {
    return ub::detail::getPerfConfig();
  }
};

TEST_P(GpuPayloadTest, Kernel) {
  const int arraySize = GetParam();

  ub::PerfConfig cfg = getCfg();
  cfg.msgBytes = arraySize * sizeof(float);

  std::string testName = ::testing::UnitTest::GetInstance()
                           ->current_test_info()->test_suite_name() +
                         std::string(".") +
                         ::testing::UnitTest::GetInstance()
                           ->current_test_info()->name();

  // Note: PerfGpuCase uses PerfConfig, not custom config override
  ub::PerfGpuCase perf{testName, ub::detail::getPerfConfig()};

  // Allocate arrays
  std::vector<float> h_data(arraySize);
  float* d_data;
  cudaMalloc(&d_data, arraySize * sizeof(float));

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, arraySize);
  });

  // Measure
  auto result = perf.cudaKernel([&](cudaStream_t s) {
    myKernel<<<grid, block, 0, s>>>(d_data, arraySize);
  }, "kernel")
    .withLaunchConfig(grid, block)
    .withHostToDevice(h_data.data(), d_data, arraySize * sizeof(float))
    .withDeviceToHost(d_data, h_data.data(), arraySize * sizeof(float))
    .measure();

  cudaFree(d_data);
}

INSTANTIATE_TEST_SUITE_P(
  ArraySizes,
  GpuPayloadTest,
  ::testing::Values(1024, 4096, 16384, 65536)
);
```

### GPU Memory Strategy Selection

`--gpu-memory explicit|unified|pinned|mapped` selects a memory strategy for
the tests that allocate by one; any other value stops the run with status 2.
The harness does not read it: it allocates no test's memory and copies only
what a test declares (`withHostToDevice()`, `withDeviceToHost()`). Nor does
the flag ask for unified-memory counters; `--capture-um` does (see the GPU
guide's [Unified-Memory Counters](GPU_GUIDE.md#unified-memory-counters)).

The test reads the strategy, not the harness.
`GpuMemoryStrategies.SelectedStrategy`
([`GpuMemoryStrategies_pTest.cu`](../ptst/gpu/GpuMemoryStrategies_pTest.cu))
reads `perf.gpuConfig().memStrategy` (`explicit` without the flag) and
allocates its input and output, 64K floats each, with a helper of its own:

| `--gpu-memory`       | The host's buffers  | The kernel's buffers     | Copies declared | Each repeat times                                        |
| -------------------- | ------------------- | ------------------------ | --------------- | -------------------------------------------------------- |
| `explicit` (default) | pageable            | device memory            | in and out      | the copy in, the launches, the copy out                  |
| `pinned`             | page-locked         | device memory            | in and out      | the same, from and to page-locked memory                 |
| `unified`            | managed             | the same managed buffers | none            | the launches; the first moves the input to the GPU       |
| `mapped`             | page-locked, mapped | the same host buffers    | none            | the launches, each reading and writing the host's memory |

```cpp
const auto STRATEGY = perf.gpuConfig().memStrategy; // the test reads it
StrategyBuffer in(STRATEGY, N), out(STRATEGY, N);   // and allocates by it

ub::CudaKernelBuilder kernel = perf.cudaKernel(LAUNCH);
kernel.withLaunchConfig(GRID, BLOCK);
if (in.needsCopy()) { // explicit and pinned: the kernel's buffers are device memory
  kernel.withHostToDevice(in.host(), in.kernel(), BYTES)
      .withDeviceToHost(out.kernel(), out.host(), BYTES);
}
const ub::PerfGpuResult RESULT = kernel.measure();
```

Run it once per strategy, each into a CSV of its own; the test prints the
strategy it allocated by:

```bash
./BenchmarkGPU_PTEST --gtest_filter=GpuMemoryStrategies.SelectedStrategy --csv explicit.csv
./BenchmarkGPU_PTEST --gtest_filter=GpuMemoryStrategies.SelectedStrategy --gpu-memory pinned --csv pinned.csv
./BenchmarkGPU_PTEST --gtest_filter=GpuMemoryStrategies.SelectedStrategy --gpu-memory unified --csv unified.csv
./BenchmarkGPU_PTEST --gtest_filter=GpuMemoryStrategies.SelectedStrategy --gpu-memory mapped --csv mapped.csv
```

```text
[GpuMemoryStrategies.SelectedStrategy] memory strategy pinned: page-locked host buffers, copied to and from device buffers
```

On an RTX 5000 Ada laptop GPU, at the default cycles and repeats, the
pinned row's copies took 94 to 103 us against 146 to 178 us for explicit,
at the same 4.1 to 4.3 us per launch; the unified row had no copies and the
same launch time, the input's one move to the GPU diluted into the median;
and the mapped row's launches took 41 to 42 us, about ten times the others',
each reaching the host's memory over PCIe. The sample's four other cases
each measure one strategy, whatever the flag says.

### Multi-GPU Parameterized Tests

Test scaling across multiple GPUs:

```cpp
class MultiGpuTest : public ::testing::TestWithParam<int> {};

TEST_P(MultiGpuTest, Scaling) {
  const int deviceCount = GetParam();

  PERF_GPU_GUARD(perf);

  // Allocate per-device data
  std::vector<float*> d_data(deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&d_data[i], SIZE * sizeof(float));
  }

  // Measure multi-GPU performance
  auto result = perf.cudaKernelMultiGpu(deviceCount,
    [&](int dev, cudaStream_t stream) {
      myKernel<<<grid, block, 0, stream>>>(d_data[dev], SIZE);
    }, "multi-gpu")
    .withLaunchConfig(grid, block)
    .withP2PAccess()
    .measure();

  // Check scaling efficiency
  EXPECT_GT(result.aggregatedStats.multiGpu->scalingEfficiency, 0.8)
    << "Poor scaling with " << deviceCount << " GPUs";

  // Cleanup
  for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaFree(d_data[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
  DeviceCounts,
  MultiGpuTest,
  ::testing::Values(1, 2, 4, 8)
);
```

### GPU CSV Schema

A GPU benchmark's file has 29 more columns: the device, the kernel and copy
times, NVML's power and thermal readings, CUPTI's kernel metrics, the
multi-GPU columns and the unified-memory counters. The API reference's
[CSV Schema](API_REFERENCE.md#csv-schema) describes each, and when it is
empty.

### GPU Profiler Integration

The GPU profilers, `nsight`, `ncu`, `compute-sanitizer` and `rocprof`, are
rows of [Available Profilers](#available-profilers), with what each needs,
writes and costs.

**Using Nsight:**

```bash
# Profile specific kernel (default Systems mode): bench run starts it under nsys
bench run ./test --profile nsight -- --gtest_filter="*MyKernel"
# Generates: bench-out/test.nsight/profile.nsys-rep and the nsys stats summaries

# Kernel deep-dive (Compute replay): bench run does not wrap a replay. Run the
# benchmark directly: it fails, printing the ncu --metrics command to run
./test --profile nsight --profile-args replay --gtest_filter="*MyKernel"
# That command writes ./kernel_replay.ncu-rep

# Analyze with Nsight UI
ncu-ui kernel_replay.ncu-rep
```

### GPU Best Practices

1. **Always warmup GPU kernels** - First launch includes JIT compilation

   ```cpp
   perf.cudaWarmup([&](cudaStream_t s) {
     myKernel<<<grid, block, 0, s>>>(d_data);
   });
   ```

2. **Declare the launch configuration** - compare launch shapes by their
   times

   ```cpp
   auto result = perf.cudaKernel([&](cudaStream_t s) {
       myKernel<<<grid, block, sharedMemBytes, s>>>(d_data);
     }, "kernel")
     .withLaunchConfig(grid, block, sharedMemBytes)
     .measure();

   // The occupancy figure is the launch shape's estimate, not how busy the
   // GPU was, so it grades neither the kernel nor the GPU's use. A test
   // tuned for this shape can require the estimate it gave, so that a
   // changed shape shows
   EXPECT_GE(result.stats.occupancy.achievedOccupancy, TUNED_SHAPE_ESTIMATE);
   ```

3. **Profile with multiple block sizes** - Find optimal configuration

   ```cpp
   for (int blockSize : {128, 256, 512, 1024}) {
     dim3 block(blockSize);
     dim3 grid((N + blockSize - 1) / blockSize);

     auto result = perf.cudaKernel([&](cudaStream_t s) {
         myKernel<<<grid, block, 0, s>>>(d_data, N);
       }, "kernel")
       .withLaunchConfig(grid, block)
       .measure();
   }
   ```

4. **Read the copies and the launch apart** - one measurement times both

   ```cpp
   auto full = perf.cudaKernel([&](cudaStream_t s) {
       myKernel<<<grid, block, 0, s>>>(d_in, d_out, N);
     }, "full")
     .withHostToDevice(h_in, d_in, SIZE)
     .withDeviceToHost(d_out, h_out, SIZE)
     .withLaunchConfig(grid, block)
     .measure();

   // One launch, the two copy medians and one round trip: medians taken
   // separately, so the first two need not add up to the third, and their
   // ratio to it is not a share of one round trip
   std::printf("launch %.1f us, copies %.1f us, round trip %.1f us\n",
               full.kernelTimeUs, full.transferTimeUs, full.totalTimeUs);
   ```

   An Nsight Systems timeline gives the copies' share of each round trip
   (the GPU guide's [Transfer Overhead](GPU_GUIDE.md#transfer-overhead)).

---

## Best Practices Summary

### CPU Benchmarking

1. **Use `PERF_GUARD`**, or call `attachProfilerHooks()` on a `PerfCase` you
   construct, so that `--profile` reaches the test
2. **Use `MemoryProfile`** for memory-bound code analysis
3. **Set `cfg.msgBytes`** to actual payload size in parameterized tests
4. **Use `--quick`** during development, full config for production
5. **Check CV%** with `EXPECT_STABLE_CV_CPU` for result stability

### GPU Benchmarking

6. **Always warmup GPU kernels** - First launch includes JIT compilation
7. **Read occupancy as an estimate** - the launch shape's, not the GPU's use;
   compare launch shapes by their times
8. **Profile multiple block sizes** - Find optimal configuration
9. **Read the copies and the launch apart** - an Nsight Systems timeline for
   the copies' share of a round trip
10. **Use appropriate memory strategy** - Test explicit vs unified vs pinned

### General

11. **Call `PERF_MAIN()`** to handle all boilerplate
12. **Check CSV schema** to understand available metrics
13. **Use `PERF_TEST_P` or `TEST_P` for parameterized tests** - they are the
    same macro

---

## See Also

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking guide
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking guide
- **[Demos](../demo/docs/)** - Interactive demos with step-by-step walkthroughs
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
