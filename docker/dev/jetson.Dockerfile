# ==============================================================================
# dev/jetson.Dockerfile - CUDA + AArch64 cross-compilation environment
#
# Combines the CUDA development environment with AArch64 cross-toolchain for
# building Jetson binaries. Includes CUDA cross-compilation support (sbsa).
#
# Usage:
#   make shell-dev-jetson
#   docker compose run dev-jetson
# ==============================================================================
FROM vernier.dev.cuda:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.dev.jetson" \
      org.opencontainers.image.description="CUDA + AArch64 cross-compilation environment for Jetson"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Toolchain Overlay
# ==============================================================================
COPY --from=vernier.toolchain.aarch64:latest / /

# ==============================================================================
# Environment Re-export
# ==============================================================================
ENV FLAMEGRAPH_DIR=/opt/FlameGraph
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100
ENV CONTAINER=yes
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    OMP_MAX_ACTIVE_LEVELS=1 \
    OMP_NESTED=false
ENV CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=5G \
    CCACHE_COMPRESS=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ==============================================================================
# FlameGraph Symlinks
# ==============================================================================
RUN ln -sf /opt/FlameGraph/flamegraph.pl /usr/local/bin/flamegraph.pl && \
    ln -sf /opt/FlameGraph/stackcollapse-perf.pl /usr/local/bin/stackcollapse-perf.pl && \
    ln -sf /opt/FlameGraph/difffolded.pl /usr/local/bin/difffolded.pl

# ==============================================================================
# CUDA Cross-compilation Support
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -euo pipefail && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources && \
    wget -qO /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    rm -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/ /" \
      > /etc/apt/sources.list.d/cuda-cross.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y cuda-cross-sbsa || true && \
    CUDA_MM="$(nvcc --version | sed -n 's/^.*release \([0-9]\+\)\.\([0-9]\+\).*/\1-\2/p')" && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y "cuda-cross-sbsa-${CUDA_MM}" || true && \
    PKG="$(apt-cache search '^cuda-cross-sbsa-[0-9]\+-[0-9]\+$' | awk '{print $1}' | sort -V | tail -1 || true)" && \
    { [ -z "${PKG}" ] || DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y "${PKG}"; }

# ==============================================================================
# CUDA AArch64 Target Symlink
# ==============================================================================
RUN set -euo pipefail && \
    if [ ! -f /usr/local/cuda/targets/aarch64-linux/include/cuda_runtime.h ]; then \
      shopt -s nullglob; cand=( \
        /usr/local/cuda/targets/aarch64-linux \
        /usr/local/cuda/targets/sbsa-linux \
        /usr/local/cuda-*/targets/aarch64-linux \
        /usr/local/cuda-*/targets/sbsa-linux ); \
      if [ "${#cand[@]}" -gt 0 ]; then \
        target="${cand[-1]}"; mkdir -p /usr/local/cuda/targets; ln -sfn "${target}" /usr/local/cuda/targets/aarch64-linux; \
      fi; \
    fi && \
    test -f /usr/local/cuda/targets/aarch64-linux/include/cuda_runtime.h \
    || { find /usr/local -maxdepth 3 -type d -path '*/targets/*' -print || true; exit 1; } && \
    (command -v ldconfig >/dev/null 2>&1 && ldconfig || true)

# Cross-compilation pkg-config paths
ENV AARCH64_SYSROOT=/usr/aarch64-linux-gnu
ENV PKG_CONFIG_LIBDIR=${AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${AARCH64_SYSROOT}/usr/lib/pkgconfig:${AARCH64_SYSROOT}/usr/share/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=${AARCH64_SYSROOT}

# ==============================================================================
# User Setup (recreate after COPY overwrites /etc/passwd)
# ==============================================================================
RUN setup-user.sh "${USER}" "${UID}" "${GID}" && \
    setup-prompt.sh "${USER}" "${UID}" "${GID}" "36" "JETSON"

# ==============================================================================
# Validation
# ==============================================================================
RUN aarch64-linux-gnu-gcc --version && \
    nvcc --version && \
    echo "Jetson cross-compile dev image validation: OK"

USER ${USER}
WORKDIR /home/${USER}
