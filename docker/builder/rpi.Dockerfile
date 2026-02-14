# ==============================================================================
# builder/rpi.Dockerfile - Raspberry Pi release artifact builder
#
# Cross-compiles release artifacts for Raspberry Pi (aarch64 Linux).
# Used in CI pipelines for artifact generation.
#
# Output: build/rpi-aarch64-release/
#
# Usage:
#   docker compose build builder-rpi
#   make docker-builder-rpi
# ==============================================================================
FROM vernier.dev.rpi:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.builder.rpi" \
      org.opencontainers.image.description="Raspberry Pi cross-compile release artifact builder"

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
    make install-rpi
