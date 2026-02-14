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
FROM vernier.dev.cpu:latest

ARG USER
ARG UID
ARG GID

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
RUN setup-user.sh "${USER}" "${UID}" "${GID}" && \
    setup-prompt.sh "${USER}" "${UID}" "${GID}" "37" "RISCV"

# ==============================================================================
# Environment
# ==============================================================================
ENV CROSS_COMPILE=riscv64-linux-gnu-
ENV RISCV_SYSROOT=/opt/sysroots/riscv64

USER ${USER}
WORKDIR /home/${USER}
