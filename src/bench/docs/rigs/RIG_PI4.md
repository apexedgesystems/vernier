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
| Profilers | valgrind 3.24.0, heaptrack 1.5.0, bpftrace 0.23.2, perf 6.18, gperftools 2.16     |

The gperftools version is the installed packages' (`google-perftools` and
`libgoogle-perftools-dev`, 2.16-1). `google-pprof --version` prints
`pprof (part of gperftools 2.0)`: the script carries that number as a
constant of its own (`$PPROF_VERSION`), so it is not the package's version.

## 2. One-Time Setup

**Packages.**

```bash
sudo apt install build-essential cmake ninja-build git \
  valgrind heaptrack bpftrace linux-perf google-perftools libgoogle-perftools-dev \
  libc6-dbg
```

**C library symbols.** `libc6-dbg` holds the names of the C library's
internal functions, which `google-pprof` reads to name the samples taken
inside the library. Without it, a sample there takes the nearest name the
library exports: the gperftools walkthrough's report then shows its top row,
`memcpy`, as
[`__xpg_strerror_r`](../../demo/docs/03_GPERF_PROFILER.md#if-it-does-not-match).

**Rust toolchain.** Needed once to build the `bench` CLI. Install with
`rustup` and make sure `cargo` is on PATH.

**perf.** With `kernel.perf_event_paranoid=2`, the default, the user-space
counters Vernier collects work without any sysctl change.

**jemalloc with profiling** (the
[jemalloc walkthrough](../../demo/docs/22_JEMALLOC_PROFILER.md) only).
Debian's packaging enables jemalloc's heap profiling on amd64 and i386 only;
this rig's `libjemalloc2`, for arm64, rejects `prof:true`. The walkthrough
uses jemalloc 5.3.0, the version Debian packages, built from the project's
release with profiling, into a folder of your own; nothing is installed
system-wide, and the system's package stays as it is:

```bash
export JEMALLOC_PROF=<a folder of yours>/jemalloc-5.3.0-prof
curl -LO https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
echo "2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa  jemalloc-5.3.0.tar.bz2" |
  sha256sum -c
tar -xjf jemalloc-5.3.0.tar.bz2
cd jemalloc-5.3.0
./configure --prefix="$JEMALLOC_PROF" --enable-prof
make -j4
make install_bin install_include install_lib
```

The checksum is the one Debian's source package (`jemalloc_5.3.0-3.dsc`)
records for that release archive. `--enable-prof` is upstream's switch for
heap profiling: jemalloc 5.3.0's `INSTALL.md` says it will "Enable heap
profiling and leak detection functionality", backtracing with the first
approach that works on the machine; here configure chose libgcc's unwinder.
On this rig the build takes about three minutes. Select it by putting
its library folder first on `LD_LIBRARY_PATH`: the loader then resolves
`libjemalloc.so.2` there, for the doctor's probe and for `bench run` alike.
Reading the dumps needs `jeprof`, from Debian's `libjemalloc-dev`
(`sudo apt install libjemalloc-dev`).

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
- **jemalloc.** Debian's `libjemalloc2` (5.3.0 for arm64) is built without
  profiling: it rejects `prof:true`, and the doctor fails `jemalloc`, naming
  the library. The
  [jemalloc walkthrough](../../demo/docs/22_JEMALLOC_PROFILER.md) runs with
  the build from the [one-time setup](#2-one-time-setup), selected with
  `LD_LIBRARY_PATH`.
- **Energy.** RAPL is Intel-only and is not available here.

## 6. Verify Your Rig

```bash
BENCH_SUDO=1 bench doctor build/bin/ptests/BenchmarkCPU_PTEST
```

Expected on this rig, captured 2026-10-17 (UTC) from the development tree at
project version 1.0.3 (`bench 1.0.3`), the binary's readiness section above
it left out:

```
=== Profiler Backend Doctor (default mode of each backend) ===

  [OK]   bpftrace   write_latency, fsync_latency: a probe copy of each with a 5 s self-exit stayed running for 1000 ms through sudo -n (BENCH_SUDO=1) and stopped on SIGINT through sudo -n kill (probe with /usr/bin/bpftrace); not checked: the grant for the run's own command (/usr/bin/bpftrace -q -B none <capture folder>/write_latency.tmp.bt, /usr/bin/bpftrace -q -B none <capture folder>/fsync_latency.tmp.bt), SIGTERM and SIGKILL through sudo, and the run's capture
  [OK]   callgrind  valgrind starts callgrind (probe: /usr/bin/valgrind --tool=callgrind --callgrind-out-file=/dev/null /bin/true)
  [FAIL] compute-sanitizer missing: compute-sanitizer not found on PATH
             Install the CUDA toolkit, which ships compute-sanitizer, or add its bin folder (for example /usr/local/cuda/bin) to PATH.
  [OK]   gperf      gperftools profiles cpu (built: cpu); analyzer /usr/bin/google-pprof
  [OK]   heaptrack  heaptrack records /bin/true (probe: /usr/bin/heaptrack -o <private directory>/probe /bin/true, which wrote probe.zst)
  [OK]   helgrind   valgrind starts helgrind (probe: /usr/bin/valgrind --tool=helgrind --log-file=/dev/null /bin/true)
  [FAIL] jemalloc   missing: /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 is built without profiling: preloaded with bench run's options, it rejects them ("<jemalloc>: Invalid conf pair: prof:true")
             Use a jemalloc built with profiling (configure --enable-prof; Debian's and Ubuntu's libjemalloc2 have it on amd64 and i386, not on arm64), with its library folder first on LD_LIBRARY_PATH when it is not the system's; or profile the heap with --profile heaptrack.
  [OK]   massif     valgrind starts massif (probe: /usr/bin/valgrind --tool=massif --massif-out-file=/dev/null /bin/true)
  [OK]   memcheck   valgrind starts memcheck (probe: /usr/bin/valgrind --tool=memcheck --leak-check=full --error-exitcode=0 --log-file=/dev/null /bin/true)
  [FAIL] ncu        missing: ncu not found on PATH
             Install the CUDA toolkit's Nsight Compute (ncu), or add its bin folder (for example /usr/local/cuda/bin) to PATH.
  [FAIL] nsight     missing: nsys not found on PATH
             Install the CUDA toolkit's Nsight Systems (nsys), or add its bin folder (for example /usr/local/cuda/bin) to PATH.
  [OK]   offcpu     the off-CPU script, with a 5 s self-exit added, stayed running for the 1500 ms start grace through sudo -n (BENCH_SUDO=1) and stopped on SIGINT through sudo -n kill (probe with /usr/bin/bpftrace); not checked: the grant for the run's own command (/usr/bin/bpftrace -e <the off-CPU script> <benchmark pid>), SIGTERM and SIGKILL through sudo, and the run's capture
  [WARN] perf       perf stat counts this process, but branches:u <not supported> here; those columns stay empty; kernel.perf_event_paranoid=2 limits this user to user-space events
  [FAIL] rapl       RAPL not available (Intel CPU + MSR access required)
             sudo modprobe msr (then re-run with sudo or CAP_SYS_RAWIO).
  [FAIL] rocprof    missing: rocprof not found on PATH
             Install ROCm's rocprofiler (apt install rocprofiler on Debian and Ubuntu).

  15 backend(s), 6 fail.

  Each row checks one backend's default mode for this user and environment.
  A run checks its own --profile request when its profiler is created, and
  only cases built with the profiler guard create one. Add --profile <name>
  [--profile-args <args>] to --profile-check to check one request.
  --require accepts only [OK]; a run may proceed with a [WARN] caveat.
```

Six backends fail here: the three NVIDIA tools, which this rig does not
have, `rapl`, which is Intel-only, `rocprof`, which is AMD's, and `jemalloc`,
because the distribution's jemalloc is built without profiling. With the
profiling build from the [one-time setup](#2-one-time-setup) first on
`LD_LIBRARY_PATH`, as the jemalloc walkthrough selects it, the `jemalloc` row
is `[OK]`, naming that library. `perf` warns because this core has no
`branches:u` event, so the branch columns stay empty. Without `BENCH_SUDO=1`,
`bpftrace` and `offcpu` fail as `denied:`, quoting bpftrace: `ERROR: bpftrace
currently only supports running as the root user.`
