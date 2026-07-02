# ==============================================================================
# base.Dockerfile - Tiered base images
#
# build-base : compile + link + test the release artifacts. The release builders
#              and the build-test gate inherit only this.
# dev-base   : build-base + tooling (scanners, formatters, profilers, docs).
#              The dev shells (`vernier.base` consumers) inherit this.
#
# The `base` compose service builds the dev-base target, so `vernier.base`
# carries the full toolset and every current consumer is unaffected.
#
# Usage:
#   docker compose build base                    # -> vernier.base (dev-base)
#   docker build --target build-base -f this .   # -> the lean compile/test tier
# ==============================================================================

# ==============================================================================
# Stage: build-base - compile, link, and test the release artifacts
# ==============================================================================
FROM ubuntu:24.04 AS build-base

# Build-time arguments
ARG USER
ARG HOST_UID
ARG HOST_GID
ARG CMAKE_VERSION=4.0.2
ARG UPX_VERSION=5.0.0

LABEL org.opencontainers.image.title="vernier.build-base" \
      org.opencontainers.image.description="Compile/link/test tier for Vernier builders" \
      org.opencontainers.image.vendor="Vernier"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Environment Configuration
# ==============================================================================
# Build-time dependency fetches (pip/poetry from PyPI, cargo from crates.io) are
# the build's only compile-time network dependency. Transient transport flakes
# there have failed builds, so make the fetches resilient: bounded retries, a
# longer timeout, and plain HTTP/1.1 for cargo (sidesteps HTTP/2 framing resets
# against crates.io).
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    CARGO_NET_RETRY=5 \
    CARGO_HTTP_MULTIPLEXING=false

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
# Compile toolchain and the utilities the build/test gate needs. The dev-only
# editor, docs, and analysis tooling live in dev-base.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      wget curl lsb-release gnupg ca-certificates \
      git \
      make ninja-build \
      ccache mold \
      python3 python3-pip python3-venv \
      sudo \
      xz-utils file

# ==============================================================================
# LLVM/Clang 21 apt repository + compilers
# ==============================================================================
# The analysis/format tooling (clang-tidy, clang-format, lcov, gdb) is added in
# dev-base; the repo and key persist in the image so dev-base reuses them.
# Download the key to a file with retries (apt.llvm.org throttles/5xx-flakes on CI
# runners) then dearmor -- a piped one-shot fails the whole layer on any transient
# blip, and --retry-on-http-error avoids dearmoring an error page.
RUN wget --tries=5 --retry-connrefused --retry-on-http-error=429,500,502,503,504 \
      --waitretry=15 --timeout=30 -qO /tmp/llvm-snapshot.gpg.key \
      https://apt.llvm.org/llvm-snapshot.gpg.key && \
    gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg /tmp/llvm-snapshot.gpg.key && \
    rm -f /tmp/llvm-snapshot.gpg.key && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-21 main" \
      >> /etc/apt/sources.list.d/llvm.list

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      clang-21 llvm-21 libclang-rt-21-dev \
      lld libc++-dev libc++abi-dev

# Unversioned compiler symlinks default to Clang 21. cc/c++ also point at clang
# so cargo and other tools that default to `cc` have a C/C++ driver.
RUN ln -sf /usr/bin/clang-21  /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-21 /usr/local/bin/clang++ && \
    ln -sf /usr/bin/clang-21  /usr/local/bin/cc && \
    ln -sf /usr/bin/clang++-21 /usr/local/bin/c++

# ==============================================================================
# CMake
# ==============================================================================
RUN wget --progress=dot:giga --tries=5 --retry-connrefused \
      --retry-on-http-error=429,500,502,503,504 --waitretry=15 --timeout=30 \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh" && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# ==============================================================================
# UPX - Executable Packer
# ==============================================================================
RUN wget --progress=dot:giga --tries=5 --retry-connrefused \
      --retry-on-http-error=429,500,502,503,504 --waitretry=15 --timeout=30 \
      -O /tmp/upx.tar.xz \
      "https://github.com/upx/upx/releases/download/v${UPX_VERSION}/upx-${UPX_VERSION}-amd64_linux.tar.xz" && \
    tar -C /tmp -xJf /tmp/upx.tar.xz && \
    mv "/tmp/upx-${UPX_VERSION}-amd64_linux/upx" /usr/local/bin/upx && \
    chmod +x /usr/local/bin/upx && \
    rm -rf /tmp/upx.tar.xz "/tmp/upx-${UPX_VERSION}-amd64_linux"

# ==============================================================================
# Rust Toolchain
# ==============================================================================
# Installed to /opt for system-wide access. Includes clippy + rustfmt;
# llvm-tools-preview supplies the profdata tooling the coverage driver
# (cargo-llvm-cov, added in dev-base) needs.
ARG RUST_VERSION=stable
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal && \
    /opt/rust/cargo/bin/rustup component add clippy rustfmt llvm-tools-preview && \
    chmod -R a+rwX /opt/rust
ENV PATH="/opt/rust/cargo/bin:$PATH"

# ==============================================================================
# Rust dependency cache (hermetic, offline release builds)
# ==============================================================================
# Pre-fetch the rust tools' pinned dependencies into the shared cargo registry
# so the release build -- and clean rebuilds -- never reach crates.io. A
# crates.io outage, throttle, or yanked version can no longer fail a build, and
# the fetch is paid once per dependency change instead of every build. Keyed on
# Cargo.lock (the CI image graph watches it too, so the cache never goes stale).
# --target restricts the fetch to the host triple, dropping Windows/wasm crates.
COPY tools/rust/Cargo.toml tools/rust/Cargo.lock /tmp/rust-fetch/
RUN cd /tmp/rust-fetch && \
    cargo fetch --locked --target x86_64-unknown-linux-gnu && \
    rm -rf /tmp/rust-fetch && \
    chmod -R a+rwX /opt/rust/cargo

# Opt the rust-tools build into offline mode against the cache above via the
# VERNIER_RUST_OFFLINE flag (read by tools/CMakeLists.txt). Scoped to this build
# so it never forces an unrelated build offline; dev-base unsets it so
# interactive dependency work still resolves online.
ENV VERNIER_RUST_OFFLINE=1

# ==============================================================================
# FetchContent source cache (hermetic, offline configure)
# ==============================================================================
# Clone the build's FetchContent dependencies (googletest) at their pinned tags
# so `cmake` configures offline -- no GitHub at build time.
# ExternalDependencies.cmake redirects FetchContent to these via VERNIER_DEPS_DIR.
# Keyed on ExternalDependencies.cmake (the pinned-version source of truth).
COPY ExternalDependencies.cmake /tmp/deps/ExternalDependencies.cmake
COPY docker/scripts/bake-external-deps.sh /tmp/deps/bake-external-deps.sh
RUN bash /tmp/deps/bake-external-deps.sh /tmp/deps/ExternalDependencies.cmake /opt/vernier-deps && \
    rm -rf /tmp/deps && \
    chmod -R a+rwX /opt/vernier-deps
ENV VERNIER_DEPS_DIR=/opt/vernier-deps

# ==============================================================================
# Poetry - Python package manager for the python tools
# ==============================================================================
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir --break-system-packages poetry

# ==============================================================================
# Python dependency wheelhouse (hermetic, offline python-tools build)
# ==============================================================================
# Download the python tools' locked dependencies as wheels so the build installs
# them offline -- a PyPI outage/yank can no longer fail the build. The tools
# build adds --no-index --find-links when VERNIER_PIP_WHEELHOUSE is set. poetry
# build itself is already offline. Keyed on poetry.lock.
COPY tools/py/pyproject.toml tools/py/poetry.lock /tmp/py-fetch/
RUN pip3 install --no-cache-dir --break-system-packages poetry-plugin-export && \
    cd /tmp/py-fetch && \
    poetry export --format requirements.txt --output req.txt --without-hashes && \
    pip3 download --no-cache-dir --requirement req.txt --dest /opt/vernier-pip-wheels && \
    rm -rf /tmp/py-fetch && \
    chmod -R a+rwX /opt/vernier-pip-wheels
ENV VERNIER_PIP_WHEELHOUSE=/opt/vernier-pip-wheels

# ==============================================================================
# Profiler link libraries
# ==============================================================================
# The perf tests link tcmalloc/profiler when present, so the link-time libs ship
# in build-base for parity with the dev build. The profiler binaries/backends
# live in dev-base.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      libgoogle-perftools-dev \
      libunwind-dev

# ==============================================================================
# User Setup Scripts
# ==============================================================================
# Defined here so every downstream image (builders off build-base and dev shells
# off dev-base) can create the host-matched user.
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

RUN echo "Validating build-base image..." && \
    cmake --version && \
    clang --version && \
    clang++ --version && \
    ccache --version && \
    mold --version && \
    upx --version | { head -n1; cat >/dev/null; } && \
    rustc --version && \
    cargo --version && \
    echo "build-base image validation: OK"

WORKDIR /home/${USER}

# ==============================================================================
# Stage: dev-base - build-base + tooling (scanners, formatters, profilers, docs)
# ==============================================================================
FROM build-base AS dev-base

# Interactive dev resolves dependencies online; the offline guarantee is for the
# release builders (build-base tier), not the dev shell. The baked caches are
# still inherited, so a clean dev build reuses them and only fetches genuinely
# new crates/wheels.
ENV VERNIER_RUST_OFFLINE= \
    VERNIER_PIP_WHEELHOUSE=

ARG USER
ARG HADOLINT_VERSION=v2.12.0
ARG SHFMT_VERSION=v3.11.0

LABEL org.opencontainers.image.title="vernier.base" \
      org.opencontainers.image.description="Base tooling layer for Vernier development" \
      org.opencontainers.image.vendor="Vernier"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Dev System Packages
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      vim \
      doxygen graphviz

# ==============================================================================
# Node.js (required by pre-commit hooks: prettier, markdownlint-cli)
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      nodejs npm

# ==============================================================================
# LLVM/Clang analysis and format tooling
# ==============================================================================
# The compilers themselves live in build-base; this adds tidy/format/debug.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      clang-tidy-21 clang-format-21 \
      lcov gdb

RUN ln -sf /usr/bin/clang-format-21 /usr/local/bin/clang-format && \
    ln -sf /usr/bin/clang-tidy-21   /usr/local/bin/clang-tidy

# ==============================================================================
# Linters and Formatters
# ==============================================================================
RUN wget --progress=dot:giga --tries=5 --retry-connrefused \
      --retry-on-http-error=429,500,502,503,504 --waitretry=15 --timeout=30 \
      -O /usr/local/bin/hadolint \
      "https://github.com/hadolint/hadolint/releases/download/${HADOLINT_VERSION}/hadolint-Linux-x86_64" && \
    chmod +x /usr/local/bin/hadolint

RUN wget --progress=dot:giga --tries=5 --retry-connrefused \
      --retry-on-http-error=429,500,502,503,504 --waitretry=15 --timeout=30 \
      -O /usr/local/bin/shfmt \
      "https://github.com/mvdan/sh/releases/download/${SHFMT_VERSION}/shfmt_${SHFMT_VERSION}_linux_amd64" && \
    chmod +x /usr/local/bin/shfmt

# ==============================================================================
# Python Formatters and Hooks
# ==============================================================================
# cmakelang: CMake formatter (cmake-format, cmake-lint); pre-commit: hook runner.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-cache-dir \
      cmakelang pre-commit

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
# linux-tools:      perf (common + generic; host-matched added below)
# google-perftools: pprof CPU/heap profiler driver (tcmalloc lib is in build-base)
# valgrind:         callgrind, massif, memcheck (vernier backends)
# bpftrace:         dynamic tracing + off-CPU (vernier offcpu backend)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      linux-tools-common \
      linux-tools-generic \
      google-perftools \
      valgrind \
      bpftrace

# Host-kernel-matched perf. linux-tools-generic tracks the latest kernel, which
# drifts ahead of the host's RUNNING kernel; perf needs the exact match. Install
# linux-tools-${HOST_KERNEL} (uname -r from the makefile). Tolerant: warns
# rather than fails if that version isn't in the archive.
ARG HOST_KERNEL
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ -n "${HOST_KERNEL}" ]; then \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        "linux-tools-${HOST_KERNEL}" \
      || echo "WARN: linux-tools-${HOST_KERNEL} unavailable; perf may not match the host kernel"; \
    else \
      echo "WARN: HOST_KERNEL build-arg empty; perf may not match the host kernel"; \
    fi

# ==============================================================================
# Cleanup and Validation
# ==============================================================================
RUN rm -rf /usr/local/man /tmp/*

RUN echo "Validating dev-base image..." && \
    clang-tidy --version && \
    clang-format --version && \
    hadolint --version && \
    shfmt --version && \
    echo "dev-base image validation: OK"

WORKDIR /home/${USER}
