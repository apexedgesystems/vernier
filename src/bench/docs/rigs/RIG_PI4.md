# Reference Rig: Raspberry Pi 4 Model B

**Used by:** CPU walkthroughs
**Captured:** 2026-09-19
**See also:** [Reference Rigs](README.md)

---

## Table of Contents

1. [Hardware and Software](#1-hardware-and-software)
2. [One-Time Setup](#2-one-time-setup)
3. [Build](#3-build)
4. [Running a Measurement](#4-running-a-measurement)
5. [Rig-Specific Behavior](#5-rig-specific-behavior)
6. [Verify Your Rig](#6-verify-your-rig)

---

## 1. Hardware and Software

| Item      | Value                                                                             |
| --------- | --------------------------------------------------------------------------------- |
| Board     | Raspberry Pi 4 Model B Rev 1.5, 8 GB                                              |
| CPU       | 4x Cortex-A72, aarch64, 1800 MHz max; 32 KiB L1d per core, 1 MiB shared L2, no L3 |
| OS        | Debian GNU/Linux 13 (trixie), Linux 6.18 (64-bit)                                 |
| Compiler  | g++ 14.2.0, cmake 3.31.6                                                          |
| Profilers | valgrind 3.24.0, heaptrack 1.5.0, bpftrace 0.23.2, perf 6.18, gperftools 2.0      |

## 2. One-Time Setup

**Packages.**

```bash
sudo apt install build-essential cmake ninja-build git \
  valgrind heaptrack bpftrace linux-perf google-perftools libgoogle-perftools-dev
```

**Rust toolchain.** Needed once to build the `bench` CLI. Install with
`rustup` and make sure `cargo` is on PATH.

**perf.** With `kernel.perf_event_paranoid=2`, the default, the user-space
counters Vernier collects work without any sysctl change.

**Kernel-probe backends** (`offcpu`, `bpftrace`). These attach scheduler
tracepoints and need root. Either run as root, or grant your user
passwordless sudo for `bpftrace` and `kill` and set `BENCH_SUDO=1`; only
the tracer is elevated, and the test and its output files stay yours.

## 3. Build

Release, native on the board:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVERNIER_BUILD_GPU=OFF -DVERNIER_BUILD_TOOLS=ON -DPROJECT_BUILD_DOCS=OFF
cmake --build build -j4
source build/.env          # puts the bench CLI on PATH
```

On this rig the build takes about eight and a half minutes, most of it the
Rust CLI, and the unit tests pass (`ctest --test-dir build`).

## 4. Running a Measurement

Pin the CPU frequency governor for the measurement and restore it
afterwards, in the same script. The governor is changed only after the
current one has been read, and a failed restore is reported and fails the
script:

```bash
set -euo pipefail
GOV=/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
SAVED=$(head -1 /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
[ -n "$SAVED" ]                           # no saved governor, no change
restore_governor() {
  echo "$SAVED" | sudo tee $GOV > /dev/null ||
    { echo "governor restore FAILED: set it back to '$SAVED' by hand" >&2; exit 1; }
}
trap restore_governor EXIT
echo performance | sudo tee $GOV > /dev/null

vcgencmd get_throttled                    # expect throttled=0x0
taskset -c 3 ./build/bin/ptests/<Test> --repeats 10 --csv out.csv
vcgencmd get_throttled                    # unchanged, or distrust the run
```

- **Governor.** The default is `ondemand`. On a test with 34-millisecond
  rounds, pinning the governor and the core took the CV from 2.1-2.9% to
  1.3-2.3% over three runs each; the median did not move.
- **Core.** Pin the test to the last core.
- **Throttling.** The throttle word has sticky bits. If it changes across
  a run, an under-voltage or thermal event happened during the
  measurement. Use a supply that holds 5 V under load and a heat sink or
  fan; this rig idles at 36 C.
- **Valgrind tools.** Callgrind, Massif, Memcheck and Helgrind run 20 to
  100 times slower, on an already slow core. The CPU walkthroughs size
  their profiled runs for this board; keep `--cycles` small under
  `--profile`.

## 5. Rig-Specific Behavior

- **Small caches.** 1 MiB of shared L2 and no L3: cache effects appear at
  much smaller working sets than on a desktop CPU.
- **jemalloc.** The distribution's jemalloc is built without profiling
  (`prof:true` is rejected); the doctor reports it, and the jemalloc
  walkthrough does not use this rig.
- **Energy.** RAPL is Intel-only and is not available here.

## 6. Verify Your Rig

```bash
BENCH_SUDO=1 bench doctor build/bin/ptests/BenchmarkCPU_PTEST
```

Expected on this rig:

```
  [OK]   bpftrace   bpftrace available via BENCH_SUDO (tracepoint attach verified)
  [OK]   callgrind  valgrind available
  [OK]   gperf      gperftools linked: cpu
  [OK]   heaptrack  heaptrack available
  [OK]   helgrind   valgrind available (helgrind + drd thread-error detectors ship with it)
  [WARN] jemalloc   libjemalloc present but built without profiling (prof:true rejected)
  [OK]   massif     valgrind available (massif tool ships with it)
  [OK]   memcheck   valgrind available (memcheck is the default tool)
  [OK]   offcpu     bpftrace available via BENCH_SUDO (tracepoint attach verified)
  [WARN] perf       perf_event_paranoid=2 (kernel profiling blocked; userspace counters still work)
  [FAIL] rapl       RAPL not available (Intel CPU + MSR access required)
  [FAIL] rocprof    ROCm not detected (no rocprof on PATH, no /opt/rocm)
  12 backend(s), 2 fail.
```

The two failures are an Intel-only and an AMD-only backend. Without
`BENCH_SUDO=1`, `bpftrace` and `offcpu` report `[WARN] ... not running as
root`.
