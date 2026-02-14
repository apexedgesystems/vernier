# ==============================================================================
# base.Dockerfile - Shared build environment and system dependencies
#
# Foundation layer for all Vernier development images. Provides compilers,
# build tools, profiling tools, and formatters. Not run directly.
# ==============================================================================
FROM ubuntu:24.04

# Build-time arguments
ARG USER
ARG UID
ARG GID
ARG CMAKE_VERSION=4.0.2
ARG UPX_VERSION=5.0.0
ARG HADOLINT_VERSION=v2.12.0
ARG SHFMT_VERSION=v3.11.0

LABEL org.opencontainers.image.title="vernier.base" \
      org.opencontainers.image.description="Base tooling layer for Vernier development" \
      org.opencontainers.image.vendor="Vernier"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Environment Configuration
# ==============================================================================
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ENV CONTAINER=yes

# Thread safety: prevent thread explosion during parallel builds/tests.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_MAX_ACTIVE_LEVELS=1

# ccache: mount volume at /ccache to persist across runs
ENV CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=5G \
    CCACHE_COMPRESS=1

RUN mkdir -p /ccache && chmod 1777 /ccache

# ==============================================================================
# System Packages
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      wget curl lsb-release gnupg ca-certificates \
      git vim \
      make ninja-build \
      ccache mold \
      python3 python3-pip python3-venv \
      doxygen graphviz \
      sudo xz-utils file

# ==============================================================================
# LLVM/Clang 21
# ==============================================================================
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | \
      gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-21 main" \
      >> /etc/apt/sources.list.d/llvm.list

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      clang-21 llvm-21 clang-tidy-21 clang-format-21 libclang-rt-21-dev \
      lld libc++-dev libc++abi-dev lcov gdb

RUN ln -sf /usr/bin/clang-format-21 /usr/local/bin/clang-format && \
    ln -sf /usr/bin/clang-tidy-21   /usr/local/bin/clang-tidy && \
    ln -sf /usr/bin/clang-21        /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-21      /usr/local/bin/clang++

# ==============================================================================
# CMake
# ==============================================================================
RUN wget --progress=dot:giga \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh" && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# ==============================================================================
# UPX - Executable Packer
# ==============================================================================
RUN wget --progress=dot:giga -O /tmp/upx.tar.xz \
      "https://github.com/upx/upx/releases/download/v${UPX_VERSION}/upx-${UPX_VERSION}-amd64_linux.tar.xz" && \
    tar -C /tmp -xJf /tmp/upx.tar.xz && \
    mv "/tmp/upx-${UPX_VERSION}-amd64_linux/upx" /usr/local/bin/upx && \
    chmod +x /usr/local/bin/upx && \
    rm -rf /tmp/upx.tar.xz "/tmp/upx-${UPX_VERSION}-amd64_linux"

# ==============================================================================
# Linters and Formatters
# ==============================================================================
RUN wget --progress=dot:giga -O /usr/local/bin/hadolint \
      "https://github.com/hadolint/hadolint/releases/download/${HADOLINT_VERSION}/hadolint-Linux-x86_64" && \
    chmod +x /usr/local/bin/hadolint

RUN wget --progress=dot:giga -O /usr/local/bin/shfmt \
      "https://github.com/mvdan/sh/releases/download/${SHFMT_VERSION}/shfmt_${SHFMT_VERSION}_linux_amd64" && \
    chmod +x /usr/local/bin/shfmt

# ==============================================================================
# Node.js (required by pre-commit hooks: prettier, markdownlint-cli)
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      nodejs npm

# ==============================================================================
# Python Tools
# ==============================================================================
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-cache-dir \
      cmakelang poetry pre-commit

# ==============================================================================
# Rust Toolchain
# ==============================================================================
ARG RUST_VERSION=stable
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal && \
    /opt/rust/cargo/bin/rustup component add clippy rustfmt && \
    chmod -R a+rx /opt/rust
ENV PATH="/opt/rust/cargo/bin:$PATH"

# ==============================================================================
# FlameGraph
# ==============================================================================
RUN git clone --depth 1 https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph && \
    ln -s /opt/FlameGraph/flamegraph.pl /usr/local/bin/flamegraph.pl && \
    ln -s /opt/FlameGraph/stackcollapse-perf.pl /usr/local/bin/stackcollapse-perf.pl && \
    ln -s /opt/FlameGraph/difffolded.pl /usr/local/bin/difffolded.pl && \
    chmod +x /opt/FlameGraph/*.pl

ENV FLAMEGRAPH_DIR=/opt/FlameGraph

# ==============================================================================
# Profiling Tools
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      linux-tools-common \
      linux-tools-generic \
      google-perftools \
      libgoogle-perftools-dev \
      libunwind-dev \
      bpftrace \
      valgrind

# ==============================================================================
# User Setup Scripts
# ==============================================================================
RUN printf '%s\n' '#!/bin/bash' \
      'set -e' \
      '_USER="$1"; _UID="$2"; _GID="$3"' \
      'groupadd --gid "$_GID" "$_USER" 2>/dev/null || true' \
      'useradd --uid "$_UID" --gid "$_GID" --create-home --shell /bin/bash -p "*" "$_USER" 2>/dev/null || true' \
      'echo "$_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers' \
      'chown -R "$_UID:$_GID" "/home/$_USER"' \
      > /usr/local/bin/setup-user.sh && \
    chmod +x /usr/local/bin/setup-user.sh

RUN printf '%s\n' '#!/bin/bash' \
      'set -e' \
      '_USER="$1"; _UID="$2"; _GID="$3"; COLOR="$4"; LABEL="$5"' \
      'echo "if [ -n \"\$PS1\" ]; then export PS1=\"\\[\\e[1;${COLOR}m\\][${LABEL}] \\u@\\h:\\w \\$\\[\\e[0m\\] \"; fi" >> "/home/$_USER/.bashrc"' \
      'chown "$_UID:$_GID" "/home/$_USER/.bashrc"' \
      > /usr/local/bin/setup-prompt.sh && \
    chmod +x /usr/local/bin/setup-prompt.sh

# ==============================================================================
# Cleanup and Validation
# ==============================================================================
RUN rm -rf /usr/local/man /tmp/*

RUN echo "Validating base image..." && \
    cmake --version && \
    clang --version && \
    ccache --version && \
    mold --version && \
    upx --version | head -1 && \
    rustc --version && \
    echo "Base image validation: OK"

WORKDIR /home/${USER}
