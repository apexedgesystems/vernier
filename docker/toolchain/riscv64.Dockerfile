# ==============================================================================
# toolchain/riscv64.Dockerfile - RISC-V 64-bit cross-compilation toolchain
#
# Provides cross-compilation for RISC-V 64-bit Linux targets.
# Supports SiFive, StarFive VisionFive, Milk-V, and other RV64GC boards.
# ==============================================================================
FROM vernier.base:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.toolchain.riscv64" \
      org.opencontainers.image.description="RISC-V 64-bit Linux cross-compilation toolchain"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# RISC-V Cross Toolchain
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      # Cross compiler and binutils
      gcc-riscv64-linux-gnu \
      g++-riscv64-linux-gnu \
      binutils-riscv64-linux-gnu \
      # QEMU for running RV64 binaries
      qemu-user-static \
      # Useful utilities
      file

# ==============================================================================
# Multi-arch Apt Sources
# ==============================================================================
# Ubuntu 24.04 uses DEB822 format (ubuntu.sources) instead of sources.list.
# Remove it and write explicit arch-filtered sources so riscv64 goes to
# ports.ubuntu.com and amd64 stays on archive.ubuntu.com.
RUN rm -f /etc/apt/sources.list.d/ubuntu.sources && \
    printf '%s\n' \
      'deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse' \
      'deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse' \
      'deb [arch=amd64] http://security.ubuntu.com/ubuntu noble-security main restricted universe multiverse' \
      'deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble-backports main restricted universe multiverse' \
      > /etc/apt/sources.list && \
    printf '%s\n' \
      'deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble main restricted universe multiverse' \
      'deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble-updates main restricted universe multiverse' \
      'deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble-security main restricted universe multiverse' \
      'deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports noble-backports main restricted universe multiverse' \
      > /etc/apt/sources.list.d/ubuntu-riscv64-ports.list

# ==============================================================================
# RISC-V Sysroot Libraries
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    dpkg --add-architecture riscv64 && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      libgoogle-perftools-dev:riscv64 \
      libtcmalloc-minimal4t64:riscv64 \
      libunwind-dev:riscv64 \
      zlib1g-dev:riscv64

# ==============================================================================
# Cross Environment
# ==============================================================================
ENV CROSS_COMPILE=riscv64-linux-gnu-

# ==============================================================================
# Validation
# ==============================================================================
RUN riscv64-linux-gnu-gcc --version && \
    qemu-riscv64-static --version && \
    echo "RISC-V toolchain validation: OK"

WORKDIR /home/${USER}
