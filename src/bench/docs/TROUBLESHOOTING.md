# Troubleshooting Guide

**Namespace:** `vernier::bench`
**Platform:** Linux-only
**C++ Standard:** C++20 (C++23 used when available)

Solutions to common issues when using the benchmarking framework. This guide covers measurement problems, build issues, profiling, containers, and more.

---

## Table of Contents

- [Measurement Issues](#measurement-issues)
- [Build Issues](#build-issues)
- [Runtime Issues](#runtime-issues)
- [GPU Issues](#gpu-issues)
- [Profiler Issues](#profiler-issues)
- [Container Issues](#container-issues)
- [Analysis Tool Issues](#analysis-tool-issues)
- [Performance Debugging](#performance-debugging)

---

## Measurement Issues

### High Jitter (CV% > 10%)

**Symptoms:**

- Large spread between p10/median/p90
- High standard deviation in results
- `wallCV` column shows values >0.10

**Root causes:**

- Process migrating between CPU cores
- Background processes interfering
- Thermal throttling
- Measurement overhead dominating (workload too small)

**Solutions:**

**1. Enable CPU pinning** (most effective):

```bash
taskset -c 2-9 ./MyComponent_PTEST --csv results.csv

# Or use the bench run command
bench run ./MyComponent_PTEST --taskset 2-9
```

**2. Increase warmup iterations**:

```bash
./MyComponent_PTEST --warmup 10 --csv results.csv
```

**3. Close background processes**:

```bash
# Check CPU usage
top

# Stop heavy services temporarily
sudo systemctl stop docker mysqld
```

**4. Check for thermal throttling**:

```bash
# Monitor temperatures
sensors
watch -n 1 sensors

# For GPU
watch -n 1 nvidia-smi
```

**5. Increase cycles to amortize overhead**:

```bash
./MyComponent_PTEST --cycles 20000 --csv results.csv
```

**When high CV% is acceptable:**

- Small payloads (<64B): CV up to 20% is normal
- Quick mode: Higher variance expected
- I/O operations: Inherently variable

---

### Inconsistent Results Between Runs

**Symptoms:**

- Different median latencies across executions
- Results vary by >5% run-to-run

**Solutions:**

**1. Use explicit warmup**:

```bash
# Instead of auto-scaling
./MyComponent_PTEST --warmup 3 --csv run1.csv
./MyComponent_PTEST --warmup 3 --csv run2.csv
```

**2. Increase repeats for confidence**:

```bash
./MyComponent_PTEST --repeats 30 --csv baseline.csv
```

**3. Don't use --quick for baselines**:

```bash
# BAD: for baselines
./MyComponent_PTEST --quick --csv baseline.csv

# GOOD: for baselines
./MyComponent_PTEST --cycles 20000 --repeats 30 --csv baseline.csv
```

**4. Use fixed seeds for random data**:

```cpp
// GOOD: Deterministic
std::mt19937 rng(42);

// BAD: Non-deterministic
std::random_device rd;
std::mt19937 rng(rd());
```

**5. Check system load**:

```bash
uptime  # Check load average
top     # Check for CPU hogs
```

---

### Results Don't Make Sense

**Symptoms:**

- Optimization made things slower
- Performance inconsistent with expectations

**Debugging steps:**

**1. Verify correctness first**:

```cpp
// Always validate results before trusting performance
ASSERT_EQ(result, expected) << "Incorrect computation!";
```

**2. Check compiler didn't optimize away code**:

```cpp
// BAD: Compiler might remove this
int sum = 0;
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
// sum never used!

// GOOD: Force compiler to compute
int sum = 0;
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
volatile int sink = sum;  // Tell compiler: sum is used
```

**3. Profile to find hotspots**:

```bash
./MyComponent_PTEST --profile perf --cycles 100000
bench flamegraph MyComponent.Test.perf/perf.data \
    --output hotspots.svg
```

**4. Compare before/after**:

```bash
./MyComponent_PTEST --csv before.csv
# Make changes
./MyComponent_PTEST --csv after.csv
bench compare before.csv after.csv
```

**5. Check warmup is sufficient**:

```bash
# Look at warmup column in CSV
grep "warmup" results.csv
```

---

## Build Issues

### CMake Can't Find GoogleTest

**Symptoms:**

```
Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR)
```

**Solutions:**

**Ubuntu/Debian:**

```bash
sudo apt-get install libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

**macOS:**

```bash
brew install googletest
```

**From source:**

```bash
git clone https://github.com/google/googletest.git
cd googletest
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
sudo cmake --install build
```

---

### CUDA Not Found During Build

**Symptoms:**

```
Could not find CUDA toolkit
```

**Solutions:**

**1. Install CUDA Toolkit**:

```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-0
```

**2. Set CUDA path**:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**3. Specify CUDA architecture**:

```bash
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="75;80;89"
```

**4. Skip GPU tests** (CPU-only build):

```bash
# GPU tests automatically skipped if CUDA not found
cmake -B build -S .
```

---

### Compilation Errors in GPU Code

**Common issues:**

**1. Architecture mismatch**:

```
warning: arch compute_XX not supported
```

**Solution:**

```bash
# Check your GPU capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for your GPU
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="89"
```

**2. NVML symbols undefined**:

```
undefined reference to `nvmlInit_v2'
```

**Solution:**

```bash
# Ensure NVML symlink exists
ls -la /usr/local/cuda/targets/x86_64-linux/lib/libnvidia-ml.so

# If missing, NVML features will be disabled (non-fatal)
```

---

## Runtime Issues

### "ABI mismatch" and Exit Status 3

**Symptoms:** the benchmark prints one line and exits with status 3 before
any profiler is constructed:

```
[bench] ABI mismatch: this benchmark and the libbench it loaded were built from different vernier headers (sizeof(PerfConfig): benchmark 272, library 240). Rebuild the benchmark against this libbench, or load the libbench that matches the benchmark's headers. Exiting.
```

The parentheses list each value that differs: `ABI version`,
`sizeof(PerfConfig)`, `sizeof(Stats)`, and for `libbench_cuda` also
`sizeof(PerfGpuConfig)`.

**Cause:** the benchmark was compiled from the vernier headers of another
build than the `libbench` or `libbench_cuda` it loaded, for example an
installed package next to newer headers, or a stale copy of the library found
first on the library path. The two sides exchange `PerfConfig`, `Stats` and
`PerfGpuConfig` objects, so the run stops instead of reading them with the
wrong layout.

**Fix:** rebuild the benchmark against the headers that ship with the library
it loads, or have it load the library it was built with. `ldd` shows which
file the loader picked:

```bash
ldd ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST | grep libbench
```

---

### "libbench.so.1: cannot open shared object file"

**Symptoms:** a benchmark built against an earlier release does not start;
nothing from vernier is printed and the exit status is 127:

```
./MyComponent_PTEST: error while loading shared libraries: libbench.so.1: cannot open shared object file: No such file or directory
```

**Cause:** the benchmark was linked against the `.so.1` bench libraries. The
current ones are `libbench.so.2` and `libbench_cuda.so.2`: the layout they
share with the benchmark changed, and there is no `libbench.so.1` link to
them, so an old binary stops at the loader instead of running against an
incompatible layout.

**Fix:** rebuild the benchmark against the current headers and libraries. To
keep running the old binary as it is, keep the earlier release's
`libbench.so.1` where its loader looks; both versions can be installed in one
directory. See [Library versions](../../../README.md#library-versions).

---

### CSV File Not Generated

**Symptoms:** No CSV after test completes.

**Solutions:**

**1. Ensure --csv flag is passed**:

```bash
./MyComponent_PTEST --csv results.csv
```

**2. Check directory permissions**:

```bash
ls -ld .
# Should show write permissions

mkdir -p results/
./MyComponent_PTEST --csv results/output.csv
```

**3. Verify PERF_MAIN() is used**:

```cpp
// GOOD: Automatic CSV listener
PERF_MAIN()

// BAD: Manual main() needs explicit listener setup
int main(int argc, char** argv) {
  // Missing CSV listener installation!
}
```

**4. Check for early exit**:

```bash
# Run with verbose output
./MyComponent_PTEST --csv results.csv --gtest_print_time=1
```

---

### Segmentation Fault

**Common causes:**

**1. Buffer overflow**:

```cpp
// Check bounds
std::vector<int> data(100);
for (int i = 0; i <= 100; ++i) {  // BAD: Off-by-one!
  data[i] = i;
}
```

**2. Invalid pointer in lambda**:

```cpp
// BAD: data goes out of scope
{
  auto data = prepareData();
} // data destroyed here
perf.throughputLoop([&]{
  process(data.data());  // BAD: data is invalid!
}, "op");

// GOOD: data outlives lambda
auto data = prepareData();
perf.throughputLoop([&]{
  process(data.data());  // GOOD: data still valid
}, "op");
```

**3. Race condition in multi-threaded test**:

```cpp
// BAD: No synchronization
int counter = 0;
perf.contentionRun([&]{
  counter++;  // BAD: Data race!
});

// GOOD: Proper synchronization
std::atomic<int> counter{0};
perf.contentionRun([&]{
  counter.fetch_add(1);  // GOOD: Thread-safe
});
```

---

### Test Hangs / Never Completes

**Common causes:**

**1. Deadlock in contention test**:

```cpp
// Check lock ordering
// Avoid nested locks without careful design
std::mutex m1, m2;
perf.contentionRun([&]{
  std::lock_guard<std::mutex> lock1(m1);
  std::lock_guard<std::mutex> lock2(m2);  // Potential deadlock!
});
```

**2. Infinite loop**:

```cpp
// Verify loop terminates
perf.throughputLoop([&]{
  while (condition) {  // Does this always terminate?
    work();
  }
}, "op");
```

**3. Profiler hanging**:

```bash
# Try without profiler
./MyComponent_PTEST --csv results.csv

# If that works, profiler is the issue
```

**4. Drain loops in concurrency tests**:

A `while (counter < N) yield()` pattern hangs under profiler overhead when
task rejections prevent the counter from ever reaching N. Read the test
source before running with `--profile` and exclude these via
`--gtest_filter=-*DrainLoop*`.

**5. Blocking `recv` / batch-receive tests**:

Tests that call a blocking `recv()`, `recvfrom()`, or batch-receive waiting
for N frames hang when fewer frames arrive than expected. These pass
interactively (the operator can Ctrl-C) but deadlock under `--repeats N`
because the framework waits indefinitely. Exclude with `--gtest_filter`
before any profiler or `--repeats` run.

**Built-in watchdog:**

Under `--profile X`, vernier installs a SIGALRM-based per-test watchdog
that auto-aborts after 300 s with a precise diagnostic naming the test
and the profile tool. Override with `--profile-test-timeout <seconds>`;
set to `0` to disable. The watchdog only fires while a profile is active.

**Debug:**

```bash
# External timeout (always available)
timeout 60s ./MyComponent_PTEST --csv results.csv

# Attach debugger to find what is blocking
gdb -p $(pgrep MyComponent_PTEST)
(gdb) bt
```

---

## GPU Issues

### GPU Not Detected

**Symptoms:**

```
CUDA error: no CUDA-capable device is detected
```

**Solutions:**

**1. Check GPU is present**:

```bash
nvidia-smi
lspci | grep -i nvidia
```

**2. Verify CUDA driver**:

```bash
cat /proc/driver/nvidia/version
nvidia-smi
```

**3. Set correct device**:

```bash
# If multiple GPUs
./MyKernel_GPU_PTEST --gpu-device 1
```

**4. Check permissions**:

```bash
# Ensure user has access to GPU
ls -l /dev/nvidia*
# Should be accessible by your user/group
```

---

### Low GPU Speedup (<2x vs CPU)

**Possible causes:**

**1. Workload too small**:

```cpp
// BAD: Too small for GPU
const int N = 64;

// GOOD: Large enough for GPU benefit
const int N = 1024 * 1024;
```

**2. Transfer overhead dominates**:

```bash
# Check CSV: compare transferTimeUs against kernelTimeUs
bench summary results.csv --json | grep -E "transferTimeUs|kernelTimeUs"
# If transfer time rivals kernel time, transfers dominate

# Keep data on GPU longer or use Unified Memory
```

**3. Poor occupancy**:

```bash
grep "occupancy" results.csv
# If <0.5, GPU underutilized

# Try different block sizes
./test --csv results_128.csv  # Try 128 threads/block
./test --csv results_256.csv  # Try 256 threads/block
```

**4. Thermal throttling**:

```bash
# Monitor during test
watch -n 1 nvidia-smi

# Look for GPU clocks dropping
```

**Debug steps:**

```bash
# Profile to identify bottleneck
./MyKernel_GPU_PTEST --profile nsight --gtest_filter="*CpuVsGpu*"

# Check all GPU metrics in CSV
cat results.csv | grep -E "(transfer|occupancy|speedup)"
```

---

### NVML Warnings

**Symptoms:**

```
Warning: NVML initialization failed
```

**Impact:**

- Non-fatal - tests still run
- GPU clock monitoring disabled
- No throttling detection

**Solutions:**

**1. Install NVIDIA utils**:

```bash
sudo apt-get install nvidia-utils
```

**2. Check NVML symlink**:

```bash
ls -la /usr/local/cuda/targets/x86_64-linux/lib/libnvidia-ml.so
```

**3. Ignore if not needed** - NVML is optional

---

### CUDA Out of Memory

**Symptoms:**

```
CUDA error: out of memory
```

**Solutions:**

**1. Check available memory**:

```cpp
size_t free, total;
cudaMemGetInfo(&free, &total);
std::printf("GPU: %.1f / %.1f MB\n", free/1e6, total/1e6);
```

**2. Reduce problem size**:

```cpp
const int N = 512 * 1024;  // Instead of 4M
```

**3. Free memory between tests**:

```cpp
cudaFree(d_data);
cudaDeviceReset();  // Nuclear option
```

**4. Use Unified Memory** (trades speed for capacity):

```bash
./test --gpu-memory unified
```

---

## Profiler Issues

### `bench run`: "tool not found: ... is not on PATH"

**Symptoms:** `bench run` exits with status 1 before it starts the benchmark
or creates an output folder:

```
Error: tool not found: 'nsys' is not on PATH; --profile nsight runs the benchmark under it. Install nsys, or run `bench doctor` to see which profilers this machine can use
```

**Cause:** these profiles run the benchmark under an external program that
must be on `PATH`: `callgrind`, `massif`, `memcheck` and `helgrind`
(`valgrind`), `heaptrack`, `compute-sanitizer`, `nsight` (`nsys`) and `ncu`.
`--taskset` needs `taskset` the same way. `bench profile-all` reports the same
line for that profiler and moves on to the next one.

**Fix:** install the named program, or add the directory that holds it to
`PATH`. `bench doctor <binary>` lists each profiler backend the binary has and
whether its tool is available.

---

### perf Not Working

**Symptoms:**

```
perf: command not found
```

or Ubuntu's `perf` launcher finds no build for the running kernel:

```
WARNING: perf not found for kernel 6.8.0-138
```

**Solutions:**

**1. Install perf** for the running kernel:

```bash
sudo apt-get install linux-tools-generic linux-tools-$(uname -r)
```

A package for the running kernel may not exist (a vendor kernel, for example);
then install a generic perf build and put it first on `PATH`, as the
[Thor rig](rigs/RIG_THOR_AGX.md) does. In a container, see
[Perf Doesn't Work in Container](#perf-doesnt-work-in-container).

**2. Check permissions** (a `perf` that runs can still be refused):

```bash
# Option 1: Run as root
sudo ./MyComponent_PTEST --profile perf

# Option 2: Adjust paranoid level
sudo sysctl -w kernel.perf_event_paranoid=-1

# Option 3: Add capability
sudo setcap cap_perfmon=ep ./MyComponent_PTEST
```

**3. Check perf data is valid**:

```bash
perf report -i test.perf/perf.data --stdio
```

---

### bpftrace Requires Sudo

**Symptoms:**

```
bpftrace: insufficient privileges
```

**Solutions:**

**1. Run with sudo**:

```bash
sudo ./MyComponent_PTEST --profile bpftrace
```

**2. Add CAP_BPF** (Linux 5.8+):

```bash
sudo setcap cap_bpf,cap_perfmon=ep ./MyComponent_PTEST
```

**3. Skip if unavailable**:

```bash
# bpftrace is optional
./MyComponent_PTEST --csv results.csv
```

---

### Nsight Not Generating Reports

**Symptoms:** No `.nsys-rep` or `.ncu-rep` files.

**Solutions:**

**1. Check Nsight is installed**:

```bash
which nsys
which ncu
```

**2. Check the artifact directory**. A direct run writes one folder per
profiled test, named after the test; under `bench run` the whole run's data is
in one folder named after the binary, and no per-test folder is created:

```bash
# ./test --profile nsight
ls -la Suite.Case.nsight/

# bench run ./test --profile nsight
ls -la bench-out/test.nsight/
```

`--profile ncu` writes `.ncu` folders in place of `.nsight` in both cases.

**3. Use manual profiling**:

```bash
# Nsight Systems
nsys profile --trace=cuda ./test --gtest_filter="*Specific*"

# Nsight Compute
ncu --set full ./test --gtest_filter="*Specific*"
```

**4. `nsys` runs but the report has no GPU activity**: nsys completes without
error and writes a `.nsys-rep`, yet `nsys stats --report cuda_gpu_kern_sum`
reports "does not contain CUDA kernel data" and `nsight-parse` writes 0 rows.
Two distinct, fixable causes:

- **Build toolkit ahead of the driver.** A binary built with a CUDA toolkit
  whose minor version is higher than the CUDA the driver provides (the "CUDA
  Version" `nvidia-smi` reports) still runs via minor-version forward
  compatibility, but nsys cannot trace its CUDA activity. Build with a toolkit
  whose CUDA is `<=` the driver's, or upgrade the driver. `ncu` and the
  in-process CUPTI columns use other interfaces and keep working, which is the
  tell-tale: GPU benchmarks and CSV metrics succeed while the nsys timeline is
  empty.

- **The in-process CUPTI collector holds the single client slot.** CUPTI allows
  one client per process. When the binary uses the in-process CUPTI collector
  (the automatic GPU CSV columns), it claims that slot and starves an external
  nsys / ncu session. Set `VERNIER_DISABLE_CUPTI=1` when wrapping the binary so
  the in-process collector stands down and the external tool can attach:

  ```bash
  VERNIER_DISABLE_CUPTI=1 nsys profile -o out --trace=cuda,nvtx \
      ./test --profile nsight --gtest_filter="*Specific*"
  ```

If you only need per-kernel metrics, the in-process CUPTI columns (kernel time,
registers, shared memory, launch count) populate automatically with no nsys
required; for occupancy / warp stalls / cache / roofline, use `ncu` (see
`GPU_GUIDE.md`).

---

### RAPL Energy Measurement Fails

**Symptoms:**

```
RAPL: Could not read MSR
```

**Solutions:**

**1. Load MSR module**:

```bash
sudo modprobe msr
lsmod | grep msr
```

**2. Run as root or with capability**:

```bash
# Option 1: Root
sudo ./MyComponent_PTEST --profile rapl

# Option 2: Capability
sudo setcap cap_sys_rawio=ep ./MyComponent_PTEST
./MyComponent_PTEST --profile rapl
```

**3. Check CPU support**:

```bash
# RAPL requires Intel Haswell+ (~2013+)
lscpu | grep "Model name"
```

---

## Container Issues

### Container Validation Fails

**Symptoms:** `validate_container.sh` reports errors.

**Solutions:**

**Check specific failures:**

```bash
docker run --rm mybench:latest ./tst/validate_container.sh
```

**Common fixes:**

**1. Python packages missing**:

```dockerfile
RUN pip3 install pandas matplotlib seaborn scipy
```

**2. FlameGraph tools not found**:

```dockerfile
RUN git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph
ENV PATH="/opt/FlameGraph:${PATH}"
ENV FLAMEGRAPH_DIR="/opt/FlameGraph"
```

**3. Perf not available**: see
[Perf Doesn't Work in Container](#perf-doesnt-work-in-container).

---

### Perf Doesn't Work in Container

**Problem:** perf fails or produces empty data.

**Solutions:**

**1. Give the image a perf for the host's kernel.** On Ubuntu, `perf` runs
the build for the running kernel, and a container runs on the host's kernel.
The dev images install that build when they are built; rebuild them on the
host that runs the container after a kernel change, and in place of an image
pulled from the registry:

```bash
make docker-dev        # or: make docker-dev-cuda
```

`docker compose run` and the `compose-*` targets do not rebuild the image. A
package for the host's kernel may not exist: the build then prints
`WARN: linux-tools-<release> unavailable; perf may not match the host kernel`.
Details, and images of your own:
[Docker Setup Guide](DOCKER_SETUP.md#perf-profiling).

**2. Access is a separate question**, and a rebuild does not change it: the
host's `kernel.perf_event_paranoid`, `CAP_PERFMON` and the container's policy
decide. At `perf_event_paranoid=4` the privileged `dev` service counts as root
and refuses the default user. For user profiling, lower the level on the host:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```

---

### GPU Not Visible in Container

**Problem:** nvidia-smi fails.

**Solutions:**

**1. Install NVIDIA Container Toolkit** (on host):

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**2. Use --gpus flag**:

```bash
docker run --rm --gpus all mybench-gpu:latest nvidia-smi
```

**3. Test with CUDA base image**:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

### High Variance in Container

**Problem:** Results inconsistent.

**Solutions:**

**1. Pin CPUs**:

```bash
docker run --rm --cpuset-cpus="2-9" mybench:latest
```

**2. Set resource limits**:

```bash
docker run --rm \
    --memory=4g \
    --cpu-shares=2048 \
    mybench:latest
```

**3. Use dedicated runners** (not shared CI):

- Avoid shared runners for benchmarking
- Use self-hosted runners with dedicated hardware

---

### `timeout` Drops Env Vars

**Problem:** Wrapping a profile command with `timeout` silently swallows your
env vars.

```bash
# BAD: timeout treats CPUPROFILE_FREQUENCY as the command name
timeout 120 CPUPROFILE_FREQUENCY=1000 ./MyTest --profile gperf
# -> "timeout: cannot run CPUPROFILE_FREQUENCY=1000: No such file or directory"
```

**Fix:** insert `env` so `timeout` sees a real command:

```bash
timeout 120 env CPUPROFILE_FREQUENCY=1000 ./MyTest --profile gperf
```

The same trap applies to `nice`, `taskset`, and any other wrapper that takes a
command as its first positional argument.

---

### CSV / Profile Artifacts Vanish on Container Exit

**Problem:** `--csv results.csv` or `--profile gperf` writes inside the
container; when the container exits, the files are gone.

**Built-in detection:** when vernier detects it is running in a container and
`--csv` points outside the workspace mount, it prints a warning naming the
offending path. Heed the warning.

**Fix:** write under a host-mounted path:

```bash
# BAD (path inside container fs):
docker compose run --rm -T dev ./MyTest --csv /tmp/results.csv

# GOOD (under the workspace mount):
docker compose run --rm -T dev ./MyTest \
    --csv /home/$USER/workspace/results.csv

# Same for --profile-output-dir (or --artifact-root):
docker compose run --rm -T dev ./MyTest --profile gperf \
    --profile-output-dir /home/$USER/workspace/profile-out/
```

---

### Profile Artifacts Land in CWD; Move Immediately

**Problem:** without `--profile-output-dir`, each `--profile X` run drops a
`<TestName>.<tool>/` directory in the current working directory. After a few
runs the project root is polluted with stale artifacts that git complains
about.

**Built-in fix:** pass `--profile-output-dir <path>` (or its alias
`--artifact-root <path>`):

```bash
./MyTest --profile gperf --profile-output-dir bench-out/2026-05-24/
```

A direct run writes each profiled test's artifacts to a `<Test>.<tool>/`
subdirectory of that root (`.bpf` for bpftrace). A `/` in a parameterized
test's name is written `+2F` and a `+` as `+2B`, so `Parts/Join.V0/n1000` gets
`Parts+2FJoin.V0+2Fn1000.gperf/`. Under `bench run`, a profiler that
`bench run` wraps around the whole process (the valgrind tools, heaptrack,
compute-sanitizer, nsight, ncu, and jemalloc when its library can be
preloaded) writes one `<binary>.<tool>/` folder for the run instead, and no
per-test folders; its root is `--profile-output-dir`, else `bench-out/`. For
multi-tool runs (`--profile gperf` then `--profile callgrind`), reuse the same
root: the tool is part of every folder name, so their artifacts stay apart.

---

## Analysis Tool Issues

### Python Import Errors

**Symptoms:**

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**

```bash
pip install pandas matplotlib seaborn scipy
```

---

### Flamegraph Generation Fails

**Symptoms:**

```
flamegraph.pl not found
```

**Solutions:**

**1. Install FlameGraph tools**:

```bash
git clone https://github.com/brendangregg/FlameGraph.git
export PATH=$PWD/FlameGraph:$PATH
export FLAMEGRAPH_DIR=$PWD/FlameGraph
```

**2. Check perf.data is valid**:

```bash
perf report -i test.perf/perf.data --stdio
```

**3. Verify perf was profiling**:

```bash
./test --profile perf --artifact-root artifacts/
ls -lh artifacts/test.perf/perf.data
# Should be >1KB
```

---

### CSV Parsing Errors

**Symptoms:**

```
Error: Could not parse CSV
```

**Solutions:**

**1. Validate CSV format**:

```bash
bench validate results.csv
```

**2. Check file isn't empty**:

```bash
wc -l results.csv
# Should be >1 line
```

**3. Inspect CSV**:

```bash
head results.csv
# Should have headers and data
```

---

## Performance Debugging

### "Why is my code slow?"

**Systematic debugging approach:**

**1. Measure first**:

```bash
./test --csv baseline.csv
```

**2. Profile to find hotspots**:

```bash
./test --profile perf --cycles 100000
bench flamegraph test.perf/perf.data
```

**3. Check memory bandwidth**:

```cpp
// Add memory profile to see if memory-bound
auto result = perf.throughputLoop([&]{ work(); }, "op",
    MemoryProfile{read_bytes, write_bytes, 0});
// Framework prints bandwidth utilization
```

**4. Try optimizations**:

- Reduce cache misses (improve data layout)
- Vectorization (use aligned data, enable AVX)
- Parallelize (use contentionRun for multi-threaded)

**5. Measure again**:

```bash
./test --csv optimized.csv
bench compare baseline.csv optimized.csv
```

---

### "My optimization made things worse!"

**Check:**

**1. Correctness**:

```cpp
ASSERT_EQ(result, expected) << "Wrong answer!";
```

**2. Compiler didn't break something**:

```bash
# Try different optimization levels
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
```

**3. Warmup is sufficient**:

```bash
./test --warmup 10 --csv results.csv
```

**4. Compare with differential flamegraph**:

```bash
bench flamegraph optimized.perf/perf.data \
    --baseline baseline.perf/perf.data
```

---

## Getting More Help

### Where to Look

**1. Check documentation first:**

- [CPU Guide](CPU_GUIDE.md) - CPU benchmarking
- [GPU Guide](GPU_GUIDE.md) - GPU benchmarking
- [API Reference](API_REFERENCE.md) - Complete API
- **CLI Tools:** `tools/README.md` - CLI tools reference

**2. Search issues:** Check if someone else had the same problem

**3. Create detailed issue:**
Include:

- Full error message
- Minimal reproducible example
- System info (OS, compiler, GPU)
- What you've already tried

---

## Quick Diagnosis Checklist

**Measurement unstable?**

- [ ] Try `taskset -c 2-9 ./test`
- [ ] Increase `--warmup`
- [ ] Check `wallCV` in CSV

**Results don't make sense?**

- [ ] Verify correctness first
- [ ] Check warmup is sufficient
- [ ] Profile with `--profile perf`
- [ ] Use `volatile` sink to prevent dead code elimination

**GPU not faster?**

- [ ] Check transfer overhead %
- [ ] Verify occupancy > 50%
- [ ] Ensure workload is large enough

**CSV not generated?**

- [ ] Confirm `--csv` flag
- [ ] Check `PERF_MAIN()` is used
- [ ] Verify directory permissions

**Build failing?**

- [ ] Check GoogleTest installed
- [ ] For GPU: Verify CUDA toolkit
- [ ] Check CMake version >= 3.24

**Profiler not working?**

- [ ] Check profiler is installed
- [ ] Verify permissions (sudo/capabilities)
- [ ] Try without profiler first

---

**Still stuck?** See the guides linked above or create an issue with details!
