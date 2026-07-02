# ==============================================================================
# dev/riscv64.Dockerfile - RISC-V 64-bit development shell
#
# Interactive development environment for RISC-V cross-compilation.
# Layers RISC-V toolchain on top of CPU dev image.
#
# Usage:
#   make shell-dev-riscv64
#   docker compose run --rm dev-riscv64
# ==============================================================================
# BASE selects the tier: vernier.dev.cpu (dev shell) or vernier.build.cpu (the
# lean release-builder variant, vernier.build.riscv64).
ARG BASE=vernier.dev.cpu:latest
FROM ${BASE}

ARG USER
ARG HOST_UID
ARG HOST_GID

LABEL org.opencontainers.image.title="vernier.dev.riscv64" \
      org.opencontainers.image.description="RISC-V 64-bit development shell"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Layer in RISC-V toolchain
# ==============================================================================
COPY --from=vernier.toolchain.riscv64:latest / /

# ==============================================================================
# User Setup (recreate after COPY overwrites /etc/passwd)
# ==============================================================================
RUN setup-user.sh "${USER}" "${HOST_UID}" "${HOST_GID}" && \
    setup-prompt.sh "${USER}" "${HOST_UID}" "${HOST_GID}" "37" "RISCV"

# ==============================================================================
# Environment
# ==============================================================================
ENV CROSS_COMPILE=riscv64-linux-gnu-
ENV RISCV_SYSROOT=/opt/sysroots/riscv64

USER ${USER}
WORKDIR /home/${USER}
