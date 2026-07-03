# ==============================================================================
# dev/rpi.Dockerfile - Raspberry Pi development shell
#
# Interactive development environment for Raspberry Pi cross-compilation.
# Layers Pi toolchain on top of CPU dev image.
#
# Usage:
#   make shell-dev-rpi
#   docker compose run --rm dev-rpi
# ==============================================================================
# BASE selects the tier: vernier.dev.cpu (dev shell) or vernier.build.cpu (the
# lean release-builder variant, vernier.build.rpi).
ARG BASE=vernier.dev.cpu:latest
FROM ${BASE}

ARG USER
ARG HOST_UID
ARG HOST_GID

LABEL org.opencontainers.image.title="vernier.dev.rpi" \
      org.opencontainers.image.description="Raspberry Pi development shell"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Layer in Raspberry Pi toolchain
# ==============================================================================
COPY --from=vernier.toolchain.rpi:latest / /

# ==============================================================================
# User Setup (recreate after COPY overwrites /etc/passwd)
# ==============================================================================
RUN setup-user.sh "${USER}" "${HOST_UID}" "${HOST_GID}" && \
    setup-prompt.sh "${USER}" "${HOST_UID}" "${HOST_GID}" "35" "RPI"

# ==============================================================================
# Environment
# ==============================================================================
ENV RPI_SYSROOT=/opt/sysroots/rpi

USER ${USER}
WORKDIR /home/${USER}
