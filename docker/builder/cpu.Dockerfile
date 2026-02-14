# ==============================================================================
# builder/cpu.Dockerfile - CPU (non-CUDA) release artifact builder
#
# Builds release artifacts for x86_64 Linux without CUDA support.
# Used in CI pipelines for artifact generation.
#
# Usage:
#   docker compose build builder-cpu
#   make docker-builder-cpu
# ==============================================================================
FROM vernier.dev.cpu:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.builder.cpu" \
      org.opencontainers.image.description="CPU release artifact builder"

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
    make install && \
    make install-tools
