# Docker Setup Guide

Build and run benchmarks in containers: in the project's own development images,
or in an image of your own project. A container shares the host's kernel, so a
profiler that reads kernel counters needs the setup described under
[Profiling in Containers](#profiling-in-containers).

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [CPU Benchmarks](#cpu-benchmarks)
- [GPU Benchmarks](#gpu-benchmarks)
- [Container Validation](#container-validation)
- [Profiling in Containers](#profiling-in-containers)
- [CI Integration](#ci-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Two Ways to Use Containers

- **The project's development images** build and run this repository's own tree:
  `make docker-dev` and `make docker-dev-cuda` build them, and the README's Quick
  Start uses them through `docker compose` (`make compose-release`).
- **An image of your own project** builds your benchmarks together with
  vernier, as in the recipes below. They assume a project whose
  `CMakeLists.txt` fetches vernier's sources (for example with `FetchContent`)
  and defines `MyComponent_PTEST` at its top level, linking `vernier::bench`, so
  the executable lands in `build/`. The README's
  [Install as Library](../../../README.md#install-as-library) describes the
  targets.

### What You'll Need

**Host system:**

```bash
docker --version

# GPU images need the NVIDIA Container Toolkit; this lists the GPUs a container sees
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
```

**Image contents:**

- Base: Ubuntu 24.04. Its CMake (3.28) meets vernier's minimum of 3.24; Ubuntu
  22.04's (3.22) does not
- A C++20 compiler (`build-essential`), `cmake`, and `git` with
  `ca-certificates`, for the sources CMake fetches (vernier, GoogleTest)
- GPU: a CUDA 12 or newer development image

---

## Quick Start

### CPU Benchmarks

```bash
# Build the image (Dockerfile below)
docker build -t mybench:latest .

# Run, keeping the CSV on the host
mkdir -p results
docker run --rm -v "$PWD/results:/results" mybench:latest \
  ./build/MyComponent_PTEST --csv /results/results.csv
```

### GPU Benchmarks

```bash
# Build the GPU image (Dockerfile.gpu below)
docker build -t mybench-gpu:latest -f Dockerfile.gpu .

# Run with GPU access
docker run --rm --gpus all mybench-gpu:latest
```

---

## CPU Benchmarks

### Minimal Dockerfile

**Dockerfile:**

```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Compiler, CMake, and git for the sources CMake fetches
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /workspace
COPY . .

# Build benchmarks
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel "$(nproc)"

# Default command
CMD ["./build/MyComponent_PTEST", "--csv", "results.csv"]
```

### Multi-Stage Build (Smaller Images)

The runtime stage keeps only the executable and the shared libraries it loads
from `build/lib`, at the same paths, and leaves the compiler and sources behind.

**Dockerfile.multistage:**

```dockerfile
# ============ Stage 1: Build ============
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel "$(nproc)"

# ============ Stage 2: Runtime ============
FROM ubuntu:24.04

# The executable finds its libraries by the build tree's paths
WORKDIR /workspace
COPY --from=builder /workspace/build/MyComponent_PTEST ./build/
COPY --from=builder /workspace/build/lib ./build/lib

CMD ["./build/MyComponent_PTEST", "--csv", "results.csv"]
```

**Build and use:**

```bash
docker build -t mybench:slim -f Dockerfile.multistage .
docker run --rm mybench:slim
```

---

## GPU Benchmarks

### GPU Dockerfile

**Dockerfile.gpu** (a project that also builds `MyKernel_GPU_PTEST`, linking
`vernier::bench_cuda` and `vernier::bench`):

```dockerfile
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

# Build for the compute capabilities you run on (7.5, 8.0, 8.9 here)
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;89" && \
    cmake --build build --parallel "$(nproc)"

CMD ["./build/MyKernel_GPU_PTEST", "--csv", "results.csv"]
```

### Running GPU Containers

```bash
# Run with GPU access
docker run --rm --gpus all mybench-gpu:latest

# Run on one GPU
docker run --rm --gpus '"device=0"' mybench-gpu:latest

# Run with results saved to host
mkdir -p results
docker run --rm --gpus all -v "$PWD/results:/results" mybench-gpu:latest \
  ./build/MyKernel_GPU_PTEST --csv /results/gpu_results.csv

# Check GPU access
docker run --rm --gpus all mybench-gpu:latest nvidia-smi
```

---

## Container Validation

### Built-in Environment Checks

`bench validate` reports the profiling tools and settings the container has.
In the project's dev containers, `bench` comes from the build's `.env`; in the
profiling image built below it is on `PATH` already:

**Running validation:**

```bash
# In a project dev container, after make release
source build/native-linux-release/.env
bench validate

# From the host, in the profiling image (see Complete Profiling Example)
docker run --rm --privileged mybench-prof:latest bench validate
```

For GPU containers, add `bench gpu-env` to check GPU environment readiness
(CUDA, Nsight Systems, Nsight Compute). For a deeper per-backend check tied
to a specific binary, run `bench doctor <ptest-binary>`.

**What it checks:**

1. **Python Dependencies** - pandas, matplotlib, seaborn, scipy, plotly
2. **FlameGraph Tools** - PATH, $FLAMEGRAPH_DIR, scripts available
3. **Perf Profiling** - whether `perf` runs (on Ubuntu it needs the build for
   the running kernel) and the `perf_event_paranoid` level, as separate checks;
   see [Perf Profiling](#perf-profiling)
4. **GPU Tools** (optional) - CUDA, Nsight Systems, Nsight Compute
5. **Framework Smoke Test** - Actual benchmark execution

**Example output:**

```
Container Integrity Validation
==================================
1. Python Dependencies
[OK] All required Python packages available
pandas: 2.1.1, scipy: 1.15.3, plotly: 5.18.0

2. FlameGraph Tools
[OK] FLAMEGRAPH_DIR set: /opt/FlameGraph
[OK] flamegraph.pl found in PATH
[OK] stackcollapse-perf.pl available

3. Perf Profiling Tools
[OK] perf available: perf version 6.8.12
[OK] Kernel-specific perf tools available
[OK] perf has sufficient permissions

4. GPU Tools (Optional)
[OK] CUDA available: 12.0
[OK] Nsight Systems: 2025.5.1
[OK] Nsight Compute available

5. Framework Smoke Test
[OK] Test binary exists
[OK] Benchmark execution successful
[OK] CSV output valid
[OK] Python analysis tools working

==================================
Validation Summary
All checks passed!
Container is ready for benchmarking.
```

### Adding Validation to Dockerfile

```dockerfile
# Add validation as health check
FROM ubuntu:22.04

# ... install dependencies, copy source ...

# Build the tools and source the build .env so bench is on PATH
RUN make tools-rust

# Run validation during build (fails build if issues)
RUN . build/native-linux-debug/.env && bench validate || \
(echo "ERROR: Container validation failed" && exit 1)

# Also use as runtime health check
HEALTHCHECK --interval=60s --timeout=10s \
CMD . build/native-linux-debug/.env && bench validate || exit 1
```

---

## Profiling in Containers

### Perf Profiling

On Ubuntu, `perf` is a launcher script: it runs the perf build for the running
kernel from `/usr/lib/linux-tools/<release>/` and refuses to run without one. A
container shares the host's kernel, so an image needs the perf build for the
kernel of the host that runs it.

**The project's dev images** install it when they are built:
`docker/base.Dockerfile` installs `linux-tools-<release>` for the kernel named
by the `HOST_KERNEL` build argument, and `make docker-dev` and
`make docker-dev-cuda` pass the running kernel (`uname -r`). Rebuild on the
host that runs the container after it boots a different kernel, and in place
of an image pulled from the registry, which is built without `HOST_KERNEL`:

```bash
make docker-dev        # CPU dev image (the dev service)
make docker-dev-cuda   # CUDA dev image (dev-cuda, where make compose-debug runs)
```

`docker compose run` and the `compose-*` targets start the image that is
tagged and do not rebuild it. A stale image prints:

```
WARNING: perf not found for kernel 6.8.0-138
  You may need to install the following packages for this specific kernel:
    linux-tools-6.8.0-138-generic
  ...
```

What a rebuild does not do:

- **Find a package for every kernel.** A vendor kernel, for example, may have
  none. The build then prints
  `WARN: linux-tools-<release> unavailable; perf may not match the host kernel`
  and the image keeps only the generic build; install a generic perf build and
  put it first on `PATH`, as the [Thor rig](rigs/RIG_THOR_AGX.md) does.
- **Grant access.** Whether counters can be read is decided by the host's
  `kernel.perf_event_paranoid`, the capabilities of the process (`CAP_PERFMON`,
  `CAP_SYS_ADMIN`) and the container's policy. On the tested host, at
  `perf_event_paranoid=4`, `perf stat` in the privileged `dev` service works
  when run as root (uid 0) and is refused for the image's non-root user
  (uid 1001), in the same image.
  `bench validate` reports the `perf` executable and `perf_event_paranoid` on
  separate lines.

**An image of your own** can do the same: take the host's `uname -r` as a
build argument and install `linux-tools-<release>`, as `docker/base.Dockerfile`
does. Mounting the host's `/usr/bin/perf` and `/usr/lib/linux-tools` into the
container is not enough on every host: on an Ubuntu 22.04 host with an HWE
kernel, `/usr/lib/linux-tools/<release>/perf` is a link into
`/usr/lib/linux-hwe-6.8-tools-<version>/`, which those mounts leave out, and
the launcher still reports no perf for the kernel.

### FlameGraph Tools

**Install in Dockerfile:**

```dockerfile
# Install FlameGraph tools
RUN git clone --depth 1 https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph

# Tell bench flamegraph where they are
ENV FLAMEGRAPH_DIR="/opt/FlameGraph"
```

**Use in container** (the profiling image below has perf, FlameGraph and
`bench`; perf's counters need a privileged container running as root, or the
access described under [Perf Profiling](#perf-profiling)):

```bash
mkdir -p results
docker run --rm --privileged -v "$PWD/results:/results" mybench-prof:latest bash -c '
  ./build/MyComponent_PTEST --profile perf --profile-args "record -g" \
      --target-time 250ms --artifact-root /results &&
  bench flamegraph /results/MyComponent.Throughput.perf/perf.data \
      --output /results/flamegraph.svg'
```

### RAPL Energy Profiling

**Requirements:**

- An Intel CPU, with the `msr` module loaded on the host (`sudo modprobe msr`)
- `--privileged` with the `/dev/cpu` mount, or `CAP_SYS_RAWIO` with the MSR
  device the backend reads, `/dev/cpu/0/msr`

```bash
docker run --rm --privileged \
  -v /dev/cpu:/dev/cpu \
  mybench:latest \
  ./build/MyComponent_PTEST --profile rapl
```

**Alternative (more secure):** a mount alone is not enough without
`--privileged`, because the container may not open the device; pass it with
`--device` instead:

```bash
# Only SYS_RAWIO capability and CPU 0's MSR device
docker run --rm \
  --cap-add=SYS_RAWIO \
  --device=/dev/cpu/0/msr \
  mybench:latest \
  ./build/MyComponent_PTEST --profile rapl
```

### Complete Profiling Example

**Dockerfile.prof** builds on the benchmark image (`mybench:latest`): perf for
the host's kernel, FlameGraph, and vernier's `bench` CLI, which needs a Rust
toolchain newer than Ubuntu 24.04's `cargo` (1.75 cannot read the CLI's
`Cargo.lock`), so it comes from rustup:

```dockerfile
FROM mybench:latest

ARG HOST_KERNEL

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    linux-tools-common \
    "linux-tools-${HOST_KERNEL}" \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone --depth 1 https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph
ENV FLAMEGRAPH_DIR="/opt/FlameGraph"

# Reconfigure with the CLI tools; bench lands in build/bin/tools/rust.
# Dropping the Rust build tree afterwards keeps the image smaller.
RUN cmake -S . -B build -DVERNIER_BUILD_TOOLS=ON && \
    cmake --build build --parallel "$(nproc)" && \
    rm -rf build/vernier-rust-target
ENV PATH="/workspace/build/bin/tools/rust:${PATH}"
```

**Build and run:**

```bash
docker build -t mybench-prof:latest --build-arg HOST_KERNEL="$(uname -r)" \
  -f Dockerfile.prof .

docker run --rm --privileged mybench-prof:latest bash -c '
  perf --version && bench --version &&
  ./build/MyComponent_PTEST --profile perf --gtest_filter="*Throughput" &&
  cat MyComponent.Throughput.perf/stat.txt'
```

---

## CI Integration

### GitHub Actions with Docker

```yaml
name: Docker Benchmarks

on: [pull_request]

jobs:
benchmark:
runs-on: ubuntu-latest

steps:
- name: Checkout
uses: actions/checkout@v4

- name: Build Docker image
run: docker build -t mybench:latest .

- name: Validate container
run: |
docker run --rm --privileged \
-v /usr/bin/perf:/usr/bin/perf:ro \
mybench:latest \
bash -c "source build/native-linux-debug/.env && bench validate"

- name: Run benchmarks
run: |
docker run --rm \
-v $(pwd)/results:/results \
mybench:latest \
./ptests/MyComponent_PTEST --csv /results/results.csv

- name: Upload results
uses: actions/upload-artifact@v4
with:
name: benchmark-results
path: results/
```

### GitLab CI with Docker

```yaml
benchmark:
image: docker:latest
services:
- docker:dind

script:
# Build image
- docker build -t mybench:$CI_COMMIT_SHA .

# Validate
- docker run --rm mybench:$CI_COMMIT_SHA
bash -c "source build/native-linux-debug/.env && bench validate"

# Run benchmarks
- docker run --rm
-v $(pwd)/results:/results
mybench:$CI_COMMIT_SHA
./ptests/MyComponent_PTEST --csv /results/results.csv

artifacts:
paths:
- results/
```

---

## Best Practices

### 1. Use Multi-Stage Builds

Keeps runtime images small:

```dockerfile
# Builder stage: compiler, sources, build tree
FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y build-essential cmake ...
RUN cmake --build build

# Runtime stage: the executable and the libraries it loads
FROM ubuntu:24.04
COPY --from=builder /workspace/build/MyComponent_PTEST /workspace/build/
COPY --from=builder /workspace/build/lib /workspace/build/lib
```

### 2. Cache Dependencies

Speed up builds with layer caching:

```dockerfile
# Install dependencies first (cached layer)
RUN apt-get update && apt-get install -y \
    build-essential cmake ...

# Then copy source (changes frequently)
COPY . .
RUN cmake --build build
```

### 3. Validate on Build

Fail fast if container is broken:

```dockerfile
# Run validation during build
RUN make tools-rust
RUN . build/native-linux-debug/.env && bench validate || exit 1
```

### 4. Use Specific Base Images

```dockerfile
# BAD: latest tag changes
FROM ubuntu:latest

# Good: specific version
FROM ubuntu:24.04

# Good: specific CUDA version
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04
```

### 5. Non-Root User

Run benchmarks as non-root (`ubuntu:24.04` already has a user `ubuntu` with uid
1000, so let `useradd` pick the uid):

```dockerfile
# Create user
RUN useradd -m benchmark && \
    chown -R benchmark:benchmark /workspace

USER benchmark

# Benchmarks run as 'benchmark' user
CMD ["./build/MyComponent_PTEST"]
```

### 6. Volume Mounts for Results

```bash
# Mount results directory
mkdir -p results
docker run --rm \
  -v "$PWD/results:/results" \
  mybench:latest \
  ./build/MyComponent_PTEST --csv /results/results.csv

# Results persist on host after container exits
ls results/results.csv
```

The container runs as root unless its image sets a user, so what it writes into a
mount belongs to root on the host; add `--user "$(id -u):$(id -g)"` to write as
yourself.

---

## Troubleshooting

### Container Validation Fails

**Problem:** `bench validate` reports errors.

**Solutions:**

```bash
# Check which checks failed
docker run --rm mybench:latest \
bash -c "source build/native-linux-debug/.env && bench validate"

# Common issues:

# 1. Python packages missing
RUN pip3 install pandas matplotlib seaborn scipy

# 2. FlameGraph tools not found
RUN git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph
ENV PATH="/opt/FlameGraph:${PATH}"

# 3. Perf not available
# Install linux-tools for the host's kernel when the image is built
# (see Perf Profiling)
```

### Perf Doesn't Work in Container

**Problem:** `perf` command fails or produces empty data.

**Solutions:**

```bash
# 1. The image needs perf for the host's kernel (see Perf Profiling).
#    Rebuild the dev images on the host that runs them:
make docker-dev        # or: make docker-dev-cuda

# 2. Access is separate from the executable: on the tested host, at
#    perf_event_paranoid=4, perf stat in the privileged dev service works
#    when run as root (uid 0) and is refused for the image's non-root user
#    (uid 1001).
#    For user profiling, lower the level on the host:
sudo sysctl -w kernel.perf_event_paranoid=-1

# 3. Mount /dev/cpu for RAPL
docker run --rm --privileged \
-v /dev/cpu:/dev/cpu \
...
```

### GPU Not Visible

**Problem:** `nvidia-smi` fails in container.

**Solutions:**

```bash
# 1. Install the NVIDIA Container Toolkit on the host, configure Docker for it
#    and restart Docker, as NVIDIA's installation guide shows

# 2. Use --gpus flag
docker run --rm --gpus all mybench-gpu:latest nvidia-smi

# 3. Check GPU access: on the host first, then in a plain image
nvidia-smi
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

The installation guide is
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>.

### Container Builds Slowly

**Problem:** Docker builds take too long.

**Solutions:**

```dockerfile
# 1. Use multi-stage build
FROM ubuntu:24.04 AS builder
# ... build ...
FROM ubuntu:24.04
COPY --from=builder /workspace/build/MyComponent_PTEST /workspace/build/
COPY --from=builder /workspace/build/lib /workspace/build/lib

# 2. Order layers by change frequency: packages change rarely, sources often
RUN apt-get update && apt-get install -y --no-install-recommends ...
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel "$(nproc)"

# 3. Use BuildKit
# In docker build command:
DOCKER_BUILDKIT=1 docker build ...
```

### High Variance in Container Benchmarks

**Problem:** Results inconsistent between runs.

**Solutions:**

```bash
# 1. Pin CPUs
docker run --rm \
  --cpuset-cpus="2-9" \
  mybench:latest

# 2. Weight the container above others (default 1024) when CPUs are contended
docker run --rm \
  --cpu-shares=2048 \
  mybench:latest

# 3. Set memory limits
docker run --rm \
  --memory=4g \
  --memory-swap=4g \
  mybench:latest

# 4. Use dedicated/self-hosted runners in CI
# Avoid shared runners for benchmarking
```

---

## See Also

- **[CI/CD Integration](CI_CD_INTEGRATION.md)** - Using containers in CI pipelines
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking best practices
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking guide
- **CLI Tools:** `tools/README.md` - CLI tools reference
- **[Main README](../../../README.md)** - Framework overview
