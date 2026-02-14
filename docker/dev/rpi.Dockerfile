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
FROM vernier.dev.cpu:latest

ARG USER
ARG UID
ARG GID

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
RUN setup-user.sh "${USER}" "${UID}" "${GID}" && \
    setup-prompt.sh "${USER}" "${UID}" "${GID}" "35" "RPI"

# ==============================================================================
# Environment
# ==============================================================================
ENV RPI_SYSROOT=/opt/sysroots/rpi

USER ${USER}
WORKDIR /home/${USER}
