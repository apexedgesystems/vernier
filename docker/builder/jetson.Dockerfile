# ==============================================================================
# builder/jetson.Dockerfile - Jetson (AArch64) release artifact builder
#
# Cross-compiles release artifacts for NVIDIA Jetson (aarch64) with CUDA.
# Used in CI pipelines for artifact generation.
#
# Output: build/jetson-aarch64-release/
#
# Usage:
#   docker compose build builder-jetson
#   make docker-builder-jetson
# ==============================================================================
FROM vernier.dev.jetson:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.builder.jetson" \
      org.opencontainers.image.description="Jetson (AArch64) cross-compile release artifact builder"

ENV CONTAINER=yes

USER ${USER}
WORKDIR /home/${USER}/workspace

# ==============================================================================
# Source Code
# ==============================================================================
COPY --chown=${UID}:${GID} . .

# ==============================================================================
# Build Release Artifacts
# ==============================================================================
RUN make distclean 2>/dev/null || true && \
    make install-jetson
