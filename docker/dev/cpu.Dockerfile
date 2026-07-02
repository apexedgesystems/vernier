# ==============================================================================
# dev/cpu.Dockerfile - Interactive CPU (x86_64) development shell
#
# Primary development environment for native x86_64 builds. Use this for
# day-to-day development, debugging, and testing when CUDA is not needed.
#
# Usage:
#   make shell-dev              # Interactive shell
#   docker compose run dev      # Via compose
# ==============================================================================
# BASE selects the tier: vernier.base (dev shell, full tooling) or
# vernier.build-base (the lean release-builder variant, vernier.build.cpu).
ARG BASE=vernier.base:latest
FROM ${BASE}

ARG USER
ARG HOST_UID
ARG HOST_GID

LABEL org.opencontainers.image.title="vernier.dev.cpu" \
      org.opencontainers.image.description="Native x86_64 development environment"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# User Setup
# ==============================================================================
RUN setup-user.sh "${USER}" "${HOST_UID}" "${HOST_GID}" && \
    setup-prompt.sh "${USER}" "${HOST_UID}" "${HOST_GID}" "34" "CPU"

# ==============================================================================
# Validation
# ==============================================================================
RUN echo "CPU dev image validation: OK"

USER ${USER}
WORKDIR /home/${USER}
