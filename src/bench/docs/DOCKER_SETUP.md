# Docker Setup Guide

Run benchmarks in containers with reproducible, validated environments. Perfect for CI/CD and ensuring consistent results across different machines.

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

### Why Use Docker?

**Reproducible** - Same environment everywhere (local, CI, production)
**Isolated** - No dependency conflicts with host system
**Portable** - Works on any machine with Docker installed
**CI-ready** - Easy integration with GitHub Actions, GitLab CI, Jenkins
**Validated** - Built-in validation script ensures everything works

### What You'll Need

**Host system:**

```bash
# Docker Engine 20.10+
docker --version

# For GPU: NVIDIA Container Toolkit
nvidia-docker --version # or 'docker run --gpus all'
```

**Container requirements:**

- Base: Ubuntu 22.04 or later
- Build tools: cmake, g++, python3
- Framework deps: libgtest-dev
- GPU optional: CUDA Toolkit 11.0+

---

## Quick Start

### CPU Benchmarks

```bash
# Build image
docker build -t mybench:latest -f Dockerfile .

# Run benchmarks
docker run --rm mybench:latest \
./build/bin/ptests/MyComponent_PTEST --csv results.csv

# Save results to host
docker run --rm -v $(pwd)/results:/results mybench:latest \
./build/bin/ptests/MyComponent_PTEST --csv /results/results.csv
```

### GPU Benchmarks

```bash
# Build GPU image
docker build -t mybench-gpu:latest -f Dockerfile.gpu .

# Run with GPU access
docker run --rm --gpus all mybench-gpu:latest \
./build/bin/ptests/MyKernel_GPU_PTEST --csv results.csv
```

---

## CPU Benchmarks

### Minimal Dockerfile

**Dockerfile:**

```dockerfile
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
build-essential \
cmake \
git \
python3 \
python3-pip \
libgtest-dev \
&& rm -rf /var/lib/apt/lists/*

# Install Python packages for analysis
RUN pip3 install --no-cache-dir pandas matplotlib seaborn scipy

# Copy source code
WORKDIR /workspace
COPY . .

# Build benchmarks
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
cmake --build build --parallel $(nproc)

# Default command
CMD ["./build/bin/ptests/MyComponent_PTEST", "--csv", "results.csv"]
```

### Multi-Stage Build (Smaller Images)

**Dockerfile.multistage:**

```dockerfile
# ============ Stage 1: Build ============
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
build-essential \
cmake \
git \
libgtest-dev \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Build benchmarks
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
cmake --build build --parallel $(nproc) && \
strip build/bin/ptests/* # Reduce binary size

# ============ Stage 2: Runtime ============
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
python3 \
python3-pip \
libstdc++6 \
&& rm -rf /var/lib/apt/lists/*

# Install Python analysis tools
RUN pip3 install --no-cache-dir pandas matplotlib seaborn scipy

# Copy only built binaries and tools
WORKDIR /benchmarks
COPY --from=builder /src/build/bin/ptests ./ptests
COPY --from=builder /src/tools ./tools

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
CMD test -f ./ptests/MyComponent_PTEST || exit 1

CMD ["./ptests/MyComponent_PTEST", "--csv", "results.csv"]
```

**Build and use:**

```bash
# Build (~2GB vs ~4GB for single-stage)
docker build -t mybench:slim -f Dockerfile.multistage .

# Run
docker run --rm mybench:slim
```

---

## GPU Benchmarks

### GPU Dockerfile

**Dockerfile.gpu:**

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
build-essential \
cmake \
git \
python3 \
python3-pip \
libgtest-dev \
&& rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir pandas matplotlib seaborn scipy

# Copy source
WORKDIR /workspace
COPY . .

# Build with CUDA support
RUN cmake -B build -S . \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_ARCHITECTURES="75;80;89" && \
cmake --build build --parallel $(nproc)

CMD ["./build/bin/ptests/MyKernel_GPU_PTEST", "--csv", "results.csv"]
```

### Running GPU Containers

```bash
# Build
docker build -t mybench-gpu:latest -f Dockerfile.gpu .

# Run with GPU access
docker run --rm --gpus all mybench-gpu:latest

# Run specific GPU
docker run --rm --gpus '"device=1"' mybench-gpu:latest \
./ptests/MyKernel_GPU_PTEST --gpu-device 1

# Run with results saved to host
docker run --rm --gpus all \
-v $(pwd)/results:/results \
mybench-gpu:latest \
./ptests/MyKernel_GPU_PTEST --csv /results/gpu_results.csv
```

### Multi-GPU Support

```bash
# All GPUs
docker run --rm --gpus all mybench-gpu:latest

# Specific GPUs
docker run --rm --gpus '"device=0,1"' mybench-gpu:latest \
./ptests/MyKernel_GPU_PTEST

# Check GPU access
docker run --rm --gpus all mybench-gpu:latest nvidia-smi
```

---

## Container Validation

### Built-in Validation Script

The framework includes a comprehensive validation script to ensure your container has everything needed:

**Running validation:**

```bash
# Inside container
./vernier/tst/validate_container.sh

# From host
docker run --rm --privileged \
-v /usr/bin/perf:/usr/bin/perf:ro \
-v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
mybench:latest \
./vernier/tst/validate_container.sh
```

**What it checks:**

1. **Python Dependencies** - pandas, matplotlib, seaborn, scipy, plotly
2. **FlameGraph Tools** - PATH, $FLAMEGRAPH_DIR, scripts available
3. **Perf Profiling** - perf command, kernel match, permissions
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

# ... install dependencies ...

# Copy validation script
COPY tst/validate_container.sh /usr/local/bin/

# Run validation during build (fails build if issues)
RUN /usr/local/bin/validate_container.sh || \
(echo "ERROR: Container validation failed" && exit 1)

# Also use as runtime health check
HEALTHCHECK --interval=60s --timeout=10s \
CMD /usr/local/bin/validate_container.sh || exit 1
```

---

## Profiling in Containers

### Perf Profiling

**Challenge:** perf must match the host kernel version.

**Solution:** Mount perf from host:

```bash
docker run --rm --privileged \
-v /usr/bin/perf:/usr/bin/perf:ro \
-v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
-v /dev/cpu:/dev/cpu \
mybench:latest \
./ptests/MyComponent_PTEST --profile perf
```

**Why each mount:**

- `/usr/bin/perf` - perf command
- `/usr/lib/linux-tools` - Kernel-specific tools
- `/dev/cpu` - MSR access for RAPL energy profiling

### FlameGraph Tools

**Install in Dockerfile:**

```dockerfile
# Install FlameGraph tools
RUN git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph && \
chmod +x /opt/FlameGraph/*.pl

# Add to PATH
ENV PATH="/opt/FlameGraph:${PATH}"
ENV FLAMEGRAPH_DIR="/opt/FlameGraph"
```

**Use in container:**

```bash
docker run --rm --privileged \
-v /usr/bin/perf:/usr/bin/perf:ro \
-v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
mybench:latest \
bash -c "
./ptests/MyComponent_PTEST --profile perf --artifact-root artifacts/
bench flamegraph \
--perf-data artifacts/MyComponent.Test.perf/perf.data \
--output flamegraph.svg
"
```

### RAPL Energy Profiling

**Requirements:**

- `--privileged` flag or `--cap-add=SYS_RAWIO`
- `/dev/cpu` mount

```bash
docker run --rm --privileged \
-v /dev/cpu:/dev/cpu \
mybench:latest \
./ptests/MyComponent_PTEST --profile rapl
```

**Alternative (more secure):**

```bash
# Only SYS_RAWIO capability
docker run --rm \
--cap-add=SYS_RAWIO \
-v /dev/cpu:/dev/cpu \
mybench:latest \
./ptests/MyComponent_PTEST --profile rapl
```

### Complete Profiling Example

**Dockerfile with all profiling tools:**

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install profiling tools
RUN apt-get update && apt-get install -y \
build-essential cmake git \
python3 python3-pip \
libgtest-dev \
linux-tools-generic \
&& rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --no-cache-dir pandas matplotlib seaborn scipy

# FlameGraph tools
RUN git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph && \
chmod +x /opt/FlameGraph/*.pl

ENV PATH="/opt/FlameGraph:${PATH}"
ENV FLAMEGRAPH_DIR="/opt/FlameGraph"

# Copy source and build
WORKDIR /workspace
COPY . .
RUN cmake -B build -S . && cmake --build build -j$(nproc)

# Welcome message
RUN echo 'echo " Benchmarking Container"' >> /etc/bash.bashrc && \
echo 'echo " Tools: perf, FlameGraph, Python analysis"' >> /etc/bash.bashrc && \
echo 'echo " Run: validate_container.sh to verify setup"' >> /etc/bash.bashrc
```

**Run with full profiling:**

```bash
docker run --rm -it --privileged \
-v /usr/bin/perf:/usr/bin/perf:ro \
-v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
-v /dev/cpu:/dev/cpu \
-v $(pwd)/results:/results \
mybench:latest bash

# Inside container:
./validate_container.sh # Verify everything works
./ptests/MyComponent_PTEST --profile perf --artifact-root /results/
bench flamegraph \
--perf-data /results/MyComponent.Test.perf/perf.data \
--output /results/flamegraph.svg
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
./tst/validate_container.sh

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
./tst/validate_container.sh

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
# Builder stage: ~4GB
FROM ubuntu:22.04 AS builder
RUN apt-get install build-essential cmake ...
RUN cmake --build build

# Runtime stage: ~1GB
FROM ubuntu:22.04
COPY --from=builder /src/build/bin ./bin
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
COPY tst/validate_container.sh /usr/local/bin/
RUN validate_container.sh || exit 1
```

### 4. Use Specific Base Images

```dockerfile
# BAD: latest tag changes
FROM ubuntu:latest

# Good: specific version
FROM ubuntu:22.04

# Good: specific CUDA version
FROM nvidia/cuda:12.0-devel-ubuntu22.04
```

### 5. Non-Root User

Run benchmarks as non-root:

```dockerfile
# Create user
RUN useradd -m -u 1000 benchmark && \
chown -R benchmark:benchmark /workspace

USER benchmark

# Benchmarks run as 'benchmark' user
CMD ["./ptests/MyComponent_PTEST"]
```

### 6. Volume Mounts for Results

```bash
# Mount results directory
docker run --rm \
-v $(pwd)/results:/results \
mybench:latest \
./ptests/MyComponent_PTEST --csv /results/results.csv

# Results persist on host after container exits
ls results/results.csv
```

---

## Troubleshooting

### Container Validation Fails

**Problem:** `validate_container.sh` reports errors.

**Solutions:**

```bash
# Check which checks failed
docker run --rm mybench:latest ./tst/validate_container.sh

# Common issues:

# 1. Python packages missing
RUN pip3 install pandas matplotlib seaborn scipy

# 2. FlameGraph tools not found
RUN git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph
ENV PATH="/opt/FlameGraph:${PATH}"

# 3. Perf not available
# Mount from host (can't install in container)
docker run -v /usr/bin/perf:/usr/bin/perf:ro ...
```

### Perf Doesn't Work in Container

**Problem:** `perf` command fails or produces empty data.

**Solutions:**

```bash
# 1. Must mount from host (kernel version match)
docker run --rm \
-v /usr/bin/perf:/usr/bin/perf:ro \
-v /usr/lib/linux-tools:/usr/lib/linux-tools:ro \
mybench:latest

# 2. Need --privileged or perf_event_paranoid
docker run --rm --privileged ...

# Or on host:
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
# 1. Install NVIDIA Container Toolkit on host
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 2. Use --gpus flag
docker run --rm --gpus all mybench-gpu:latest nvidia-smi

# 3. Check GPU access
nvidia-smi # On host first
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi # Test image
```

### Container Builds Slowly

**Problem:** Docker builds take too long.

**Solutions:**

```dockerfile
# 1. Use multi-stage build
FROM ubuntu:22.04 AS builder
# ... build ...
FROM ubuntu:22.04
COPY --from=builder /src/build ./build

# 2. Order layers by change frequency
RUN apt-get install ... # Changes rarely
COPY requirements.txt . # Changes occasionally
RUN pip install -r ... # Changes occasionally
COPY . . # Changes frequently
RUN cmake --build build # Changes frequently

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

# 2. Limit CPU shares (prevent throttling)
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

## Example: Production Docker Setup

**Complete production-ready setup:**

**Dockerfile.production:**

```dockerfile
# ============ Stage 1: Build ============
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
build-essential \
cmake \
git \
libgtest-dev \
&& rm -rf /var/lib/apt/lists/*

# Build benchmarks
WORKDIR /src
COPY . .
RUN cmake -B build -S . \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_TESTING=OFF && \
cmake --build build --parallel $(nproc) && \
strip build/bin/ptests/*

# ============ Stage 2: Runtime ============
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
python3 \
python3-pip \
libstdc++6 \
&& rm -rf /var/lib/apt/lists/*

# Python analysis tools
RUN pip3 install --no-cache-dir pandas matplotlib seaborn scipy plotly

# FlameGraph tools
RUN git clone --depth 1 \
https://github.com/brendangregg/FlameGraph.git \
/opt/FlameGraph && \
chmod +x /opt/FlameGraph/*.pl

ENV PATH="/opt/FlameGraph:${PATH}"
ENV FLAMEGRAPH_DIR="/opt/FlameGraph"

# Create non-root user
RUN useradd -m -u 1000 benchmark && \
mkdir -p /results && \
chown -R benchmark:benchmark /results

# Copy built binaries
WORKDIR /benchmarks
COPY --from=builder --chown=benchmark:benchmark \
/src/build/bin/ptests ./ptests
COPY --from=builder --chown=benchmark:benchmark \
/src/tools ./tools
COPY --from=builder --chown=benchmark:benchmark \
/src/tst/validate_container.sh ./

# Validate during build
RUN ./validate_container.sh || \
(echo "ERROR: Validation failed" && exit 1)

# Switch to non-root
USER benchmark

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
CMD test -f ./ptests/MyComponent_PTEST || exit 1

# Welcome message
ENV PS1=' benchmark@docker:\w\$ '

# Default: run validation
CMD ["./validate_container.sh"]
```

**Build and run:**

```bash
# Build production image
docker build -t mybench:prod -f Dockerfile.production .

# Run validation
docker run --rm mybench:prod

# Run benchmarks with results
docker run --rm \
--cpuset-cpus="2-9" \
--memory=4g \
-v $(pwd)/results:/results \
mybench:prod \
./ptests/MyComponent_PTEST --csv /results/results.csv

# Interactive shell
docker run --rm -it mybench:prod bash
```

---

## See Also

- **[CI/CD Integration](CI_CD_INTEGRATION.md)** - Using containers in CI pipelines
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking best practices
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking guide
- **CLI Tools:** `tools/README.md` - CLI tools reference
- **[Main README](../../../README.md)** - Framework overview
