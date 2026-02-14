# ==============================================================================
# toolchain/aarch64.Dockerfile - AArch64 cross-compilation toolchain
#
# Provides GCC cross-compiler, binutils, and ARM64 sysroot libraries for
# building Jetson/ARM64 targets from an x86_64 host. This is a toolchain-only
# image used as a COPY source; use dev/jetson for interactive development.
# ==============================================================================
FROM vernier.base:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.toolchain.aarch64" \
      org.opencontainers.image.description="AArch64 cross-compilation toolchain"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Cross-compilation Toolchain
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      crossbuild-essential-arm64 \
      binutils-aarch64-linux-gnu \
      pkg-config \
      qemu-user-static \
      file

# pkg-config needs to know where ARM64 libraries live
ENV AARCH64_SYSROOT=/usr/aarch64-linux-gnu
ENV PKG_CONFIG_LIBDIR=${AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${AARCH64_SYSROOT}/usr/lib/pkgconfig:${AARCH64_SYSROOT}/usr/share/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=${AARCH64_SYSROOT}

# ==============================================================================
# Multi-arch Apt Sources
# ==============================================================================
# Ubuntu 24.04 uses DEB822 format (ubuntu.sources) instead of sources.list.
# Remove it and write explicit arch-filtered sources so arm64 goes to
# ports.ubuntu.com and amd64 stays on archive.ubuntu.com.
RUN rm -f /etc/apt/sources.list.d/ubuntu.sources && \
    printf '%s\n' \
      'deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse' \
      'deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse' \
      'deb [arch=amd64] http://security.ubuntu.com/ubuntu noble-security main restricted universe multiverse' \
      'deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble-backports main restricted universe multiverse' \
      > /etc/apt/sources.list && \
    printf '%s\n' \
      'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports noble main restricted universe multiverse' \
      'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports noble-updates main restricted universe multiverse' \
      'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports noble-security main restricted universe multiverse' \
      'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports noble-backports main restricted universe multiverse' \
      > /etc/apt/sources.list.d/ubuntu-arm64-ports.list

# ==============================================================================
# ARM64 Sysroot Libraries
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    dpkg --add-architecture arm64 && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      libgoogle-perftools-dev:arm64 \
      libtcmalloc-minimal4t64:arm64 \
      libunwind-dev:arm64 \
      zlib1g-dev:arm64

# ==============================================================================
# CI Profiling Support
# ==============================================================================
RUN if [ -n "${USER}" ]; then \
      echo "${USER} ALL=(ALL) NOPASSWD: /usr/bin/perf, /usr/bin/bpftrace" > /etc/sudoers.d/profilers && \
      chmod 0440 /etc/sudoers.d/profilers; \
    fi

# ==============================================================================
# Validation
# ==============================================================================
RUN aarch64-linux-gnu-gcc --version && \
    aarch64-linux-gnu-g++ --version && \
    file /usr/bin/qemu-aarch64-static && \
    echo "AArch64 toolchain validation: OK"

WORKDIR /home/${USER}
