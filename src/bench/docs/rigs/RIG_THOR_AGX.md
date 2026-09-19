# Reference Rig: NVIDIA Jetson AGX Thor Developer Kit

**Used by:** GPU walkthroughs
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

| Item      | Value                                                                          |
| --------- | ------------------------------------------------------------------------------ |
| Board     | NVIDIA Jetson AGX Thor Developer Kit                                           |
| CPU       | 14 cores, aarch64, 2601 MHz max; 64 KiB L1d and 1 MiB L2 per core              |
| GPU       | "NVIDIA Thor", compute capability 11.0, driver 580.00                          |
| Memory    | 122 GiB, shared by the CPU and the GPU                                         |
| OS        | Ubuntu 24.04.4 LTS, Linux 6.8.12-tegra, L4T R38 revision 2.2                   |
| CUDA      | 13.0 (V13.0.88) in `/usr/local/cuda`                                           |
| Nsight    | Nsight Systems 2025.3.2, Nsight Compute CLI 2025.3.1                           |
| Compiler  | g++ 13.3.0, cmake 3.31.5                                                       |
| Profilers | valgrind 3.22.0, heaptrack 1.5.0, bpftrace 0.20.2, perf 6.8.12, gperftools 2.0 |

## 2. One-Time Setup

**CUDA on PATH.** The toolkit is installed but not on the default PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

**perf.** Vendor kernels ship no matching `linux-tools` package, and the
distribution's `/usr/bin/perf` is a shim that refuses to run. Install a
generic perf build and put it first on PATH (`/usr/local/bin/perf` on this
rig). With `kernel.perf_event_paranoid=2`, the default, the user-space
counters Vernier collects work without any sysctl change.

**Kernel-probe backends** (`offcpu`, `bpftrace`). These attach scheduler
tracepoints and need root. Either run as root, or grant your user
passwordless sudo for `bpftrace` and `kill` and set `BENCH_SUDO=1`; only
the tracer is elevated, and the test and its output files stay yours.

**Nsight Compute.** GPU performance counters are restricted to
administrators on Jetson, so `--profile ncu` runs under `sudo`.

**Rust toolchain.** Needed once to build the `bench` CLI (`cargo` on PATH).

## 3. Build

Release, native on the board, with the GPU architecture stated. Vernier
reads the architecture from its own `CUDA_ARCHS` option (default `89`) and
assigns it over `CMAKE_CUDA_ARCHITECTURES`, so pass `CUDA_ARCHS`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVERNIER_BUILD_GPU=ON -DCUDA_ARCHS=110 \
  -DVERNIER_BUILD_TOOLS=ON -DPROJECT_BUILD_DOCS=OFF
cmake --build build -j$(nproc)
source build/.env          # puts the bench CLI on PATH
```

Check that the GPU code was compiled for this board:

```bash
grep -rho "generate-code=[^ ]*" build/src/bench/CMakeFiles/*/flags.make | sort -u
# --generate-code=arch=compute_110,code=[compute_110,sm_110]
```

On this rig the build takes about 35 seconds and the unit tests pass
(`ctest --test-dir build`: 117 of 117).

## 4. Running a Measurement

Lock the clocks for the measurement and restore them afterwards, in the
same script. The clocks are locked only after their state has been saved,
and a failed restore is reported and fails the script:

```bash
set -euo pipefail
STATE=/tmp/jetson_clocks.state
sudo rm -f "$STATE"                       # --store prompts if the file exists
sudo jetson_clocks --store "$STATE"       # no snapshot, no lock: set -e stops here
restore_clocks() {
  sudo jetson_clocks --restore "$STATE" ||
    { echo "clock restore FAILED: run 'sudo jetson_clocks --restore $STATE'" >&2; exit 1; }
}
trap restore_clocks EXIT
sudo jetson_clocks

taskset -c 13 ./build/bin/ptests/<Test> --repeats 10 --csv out.csv
```

- **Clocks.** Left to the governor, short tests run partly at a low clock.
  Locking took one 6-microsecond case from 7.7% to 0.7% CV.
- **Core.** Pin the host thread to the last core. Unpinned, small
  operations jitter at 18-21% CV; pinned, 1-3%.
- **Settle.** After a multi-threaded run, wait a few seconds before a
  single-threaded one.
- **One at a time.** Do not measure while anything else is running on the
  board.

## 5. Rig-Specific Behavior

- **Shared memory.** The CPU and GPU share physical memory, so a
  host-to-device copy is a memory-to-memory copy (about 75 GB/s here), not
  a PCIe transfer. Walkthrough numbers that involve transfers are smaller
  on this rig than on a discrete GPU, and each walkthrough says which ones.
- **heaptrack and tcmalloc.** With `libgoogle-perftools-dev` installed (the
  gperf backend needs it), the build links tcmalloc; the doctor's `gperf`
  line shows it as `cpu heap`. tcmalloc provides its own `operator new`,
  which heaptrack does not intercept, so heaptrack misses C++ allocations
  in this build even though the doctor reports heaptrack `[OK]`.
- **jemalloc.** The distribution's jemalloc is built without profiling;
  the doctor reports it, and the jemalloc walkthrough does not use this rig.
- **Energy.** RAPL is Intel-only and is not available here.

## 6. Verify Your Rig

```bash
BENCH_SUDO=1 bench doctor build/bin/ptests/BenchmarkGPU_PTEST
```

Expected on this rig:

```
  [OK]   bpftrace   bpftrace available via BENCH_SUDO (tracepoint attach verified)
  [OK]   callgrind  valgrind available
  [OK]   compute-sanitizer compute-sanitizer available
  [OK]   gperf      gperftools linked: cpu heap
  [OK]   heaptrack  heaptrack available
  [OK]   helgrind   valgrind available (helgrind + drd thread-error detectors ship with it)
  [WARN] jemalloc   libjemalloc present but built without profiling (prof:true rejected)
  [OK]   massif     valgrind available (massif tool ships with it)
  [OK]   memcheck   valgrind available (memcheck is the default tool)
  [OK]   ncu        ncu available
  [OK]   nsight     nsys + ncu available
  [OK]   offcpu     bpftrace available via BENCH_SUDO (tracepoint attach verified)
  [WARN] perf       perf_event_paranoid=2 (kernel profiling blocked; userspace counters still work)
  [FAIL] rapl       RAPL not available (Intel CPU + MSR access required)
  [FAIL] rocprof    ROCm not detected (no rocprof on PATH, no /opt/rocm)
  15 backend(s), 2 fail.
```

The two failures are an Intel-only and an AMD-only backend. Without
`BENCH_SUDO=1`, `bpftrace` and `offcpu` report `[WARN] ... not running as
root`.
