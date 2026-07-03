# ==============================================================================
# builder/cuda.Dockerfile - CUDA release artifact builder
#
# Builds release artifacts for x86_64 Linux with CUDA support.
# Used in CI pipelines for artifact generation.
#
# Usage:
#   docker compose build builder-cuda
#   make docker-builder-cuda
# ==============================================================================
FROM vernier.build.cuda:latest

ARG USER
ARG HOST_UID
ARG HOST_GID

LABEL org.opencontainers.image.title="vernier.builder.cuda" \
      org.opencontainers.image.description="CUDA release artifact builder"

ENV CONTAINER=yes

USER ${USER}
WORKDIR /home/${USER}/workspace

# ==============================================================================
# Source Code
# ==============================================================================
COPY --chown=${HOST_UID}:${HOST_GID} . .

# ==============================================================================
# Build Release Artifacts
# ==============================================================================
RUN make distclean 2>/dev/null || true && \
    make install && \
    make install-tools
