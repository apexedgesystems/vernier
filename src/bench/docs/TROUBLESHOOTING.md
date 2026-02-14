# Troubleshooting Guide

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
bench run ./MyComponent_PTEST \
    --taskset --cpuset 2-9
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
bench flamegraph \
    --perf-data MyComponent.Test.perf/perf.data \
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

**Debug:**

```bash
# Run with timeout
timeout 60s ./MyComponent_PTEST --csv results.csv

# Attach debugger
gdb -p $(pgrep MyComponent_PTEST)
(gdb) bt  # Print backtrace
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
# Check CSV
grep "transferOverheadPct" results.csv
# If >50%, transfers dominate

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

### perf Not Working

**Symptoms:**

```
perf: command not found
```

**Solutions:**

**1. Install perf**:

```bash
sudo apt-get install linux-tools-generic linux-tools-$(uname -r)
```

**2. Check permissions**:

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

**Symptoms:** No `.qdrep` or `.ncu-rep` files.

**Solutions:**

**1. Check Nsight is installed**:

```bash
which nsys
which ncu
```

**2. Check artifact directory**:

```bash
ls -la test.nsight/
```

**3. Use manual profiling**:

```bash
# Nsight Systems
nsys profile --trace=cuda ./test --gtest_filter="*Specific*"

# Nsight Compute
ncu --set full ./test --gtest_filter="*Specific*"
```

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

**3. Perf not available** (must mount from host):

```bash
docker run --rm \
    -v /usr/bin/perf:/usr/bin/perf:ro \
    -v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
    mybench:latest
```

---

### Perf Doesn't Work in Container

**Problem:** perf fails or produces empty data.

**Solutions:**

**1. Must mount from host** (kernel version must match):

```bash
docker run --rm --privileged \
    -v /usr/bin/perf:/usr/bin/perf:ro \
    -v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
    -v /dev/cpu:/dev/cpu \
    mybench:latest
```

**2. Need --privileged or adjust paranoid**:

```bash
# On host
sudo sysctl -w kernel.perf_event_paranoid=-1

# Or use --privileged
docker run --rm --privileged ...
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
bench flamegraph \
    --perf-data test.perf/perf.data
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
bench flamegraph \
    --perf-data optimized.perf/perf.data \
    --baseline-perf baseline.perf/perf.data
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
- [ ] Check CMake version >= 3.15

**Profiler not working?**

- [ ] Check profiler is installed
- [ ] Verify permissions (sudo/capabilities)
- [ ] Try without profiler first

---

**Still stuck?** See the guides linked above or create an issue with details!
