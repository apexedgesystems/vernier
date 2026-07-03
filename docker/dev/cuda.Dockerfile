# ==============================================================================
# dev/cuda.Dockerfile - Interactive CUDA development shell
#
# CUDA-enabled development environment with Nsight profiling tools. Use this
# for GPU kernel development, CUDA debugging, and performance analysis.
#
# Usage:
#   make shell-dev-cuda          # Interactive shell
#   docker compose run dev-cuda  # Via compose
# ==============================================================================
# OVERLAY selects which tier's filesystem is overlaid onto the NVIDIA image:
# vernier.base (dev shell, full tooling) or vernier.build-base (the lean
# release-builder variant, vernier.build.cuda). COPY --from cannot expand a
# variable directly, so the tier is aliased through a named stage.
ARG OVERLAY=vernier.base:latest
FROM ${OVERLAY} AS overlay

FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ARG USER
ARG HOST_UID
ARG HOST_GID
# WITH_NSIGHT=0 drops the Nsight profilers from the build variant (dev-only).
ARG WITH_NSIGHT=1

LABEL org.opencontainers.image.title="vernier.dev.cuda" \
      org.opencontainers.image.description="CUDA development environment for Vernier"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Base Tooling Overlay
# ==============================================================================
# Overlay our base tools (Clang, CMake, formatters, etc.) onto the NVIDIA image.
COPY --from=overlay / /

# CUDA paths (ensure nvcc and libs are found)
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ==============================================================================
# Environment Re-export
# ==============================================================================
# COPY --from overlays files but does NOT preserve ENV declarations from the
# source image. We must re-declare all environment variables from vernier.base.
#
# Hermetic switches: the overlay carries the baked caches on disk, but the env
# that activates them is lost with the rest. The build variant (vernier.build.cuda)
# passes RUST_OFFLINE/PIP_WHEELHOUSE so its tools build offline; the dev shell
# leaves them empty (resolve online), matching dev-base. The FetchContent
# redirect dir is present in both overlays, so it is re-exported unconditionally.
ARG RUST_OFFLINE=
ARG PIP_WHEELHOUSE=
ENV VERNIER_RUST_OFFLINE=${RUST_OFFLINE} \
    VERNIER_PIP_WHEELHOUSE=${PIP_WHEELHOUSE} \
    VERNIER_DEPS_DIR=/opt/vernier-deps
ENV FLAMEGRAPH_DIR=/opt/FlameGraph
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100
ENV CONTAINER=yes
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_MAX_ACTIVE_LEVELS=1
ENV CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=5G \
    CCACHE_COMPRESS=1
# Rust toolchain (installed to /opt/rust in base image)
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH="/opt/rust/cargo/bin:${PATH}"

# ==============================================================================
# FlameGraph Symlinks
# ==============================================================================
# COPY --from doesn't reliably preserve symlinks. Recreate them. FlameGraph is
# dev-tier tooling, so the lean build overlay does not carry it -- skip there.
RUN if [ -d /opt/FlameGraph ]; then \
      ln -sf /opt/FlameGraph/flamegraph.pl /usr/local/bin/flamegraph.pl && \
      ln -sf /opt/FlameGraph/stackcollapse-perf.pl /usr/local/bin/stackcollapse-perf.pl && \
      ln -sf /opt/FlameGraph/difffolded.pl /usr/local/bin/difffolded.pl && \
      chmod +x /opt/FlameGraph/*.pl; \
    else \
      echo "skip FlameGraph symlinks (lean build tier)"; \
    fi

# ==============================================================================
# Nsight Profiling Tools
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "${WITH_NSIGHT}" = "1" ]; then \
      apt-get update && \
      wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/nvidia.pub | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg && \
      echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/ /" \
        > /etc/apt/sources.list.d/nvidia-devtools.list && \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        nsight-systems-cli \
        nsight-compute; \
    else \
      echo "skip Nsight profilers (lean build tier)"; \
    fi

RUN if [ "${WITH_NSIGHT}" = "1" ]; then \
      { nsys --version 2>/dev/null || echo "nsys installed (requires GPU at runtime)"; } && \
      { ncu --version 2>/dev/null || echo "ncu installed (requires GPU at runtime)"; } ; \
    fi

# ==============================================================================
# NVML Stub Linking
# ==============================================================================
# Link NVML stub so builds succeed without a GPU present.
RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so \
           /usr/local/cuda/targets/x86_64-linux/lib/libnvidia-ml.so && \
    ldconfig

# ==============================================================================
# User Setup
# ==============================================================================
RUN setup-user.sh "${USER}" "${HOST_UID}" "${HOST_GID}" && \
    setup-prompt.sh "${USER}" "${HOST_UID}" "${HOST_GID}" "32" "CUDA"

# ==============================================================================
# Validation
# ==============================================================================
RUN nvcc --version && \
    echo "CUDA dev image validation: OK"

USER ${USER}
WORKDIR /home/${USER}
