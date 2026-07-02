# ==============================================================================
# builder/riscv64.Dockerfile - RISC-V 64-bit release artifact builder
#
# Cross-compiles release artifacts for RISC-V 64-bit Linux.
# Used in CI pipelines for artifact generation.
#
# Output: build/riscv64-linux-release/
#
# Usage:
#   docker compose build builder-riscv64
#   make docker-builder-riscv64
# ==============================================================================
FROM vernier.build.riscv64:latest

ARG USER
ARG HOST_UID
ARG HOST_GID

LABEL org.opencontainers.image.title="vernier.builder.riscv64" \
      org.opencontainers.image.description="RISC-V 64-bit cross-compile release artifact builder"

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
    make install-riscv
