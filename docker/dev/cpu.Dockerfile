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
FROM vernier.base:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.dev.cpu" \
      org.opencontainers.image.description="Native x86_64 development environment"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# User Setup
# ==============================================================================
RUN setup-user.sh "${USER}" "${UID}" "${GID}" && \
    setup-prompt.sh "${USER}" "${UID}" "${GID}" "34" "CPU"

# ==============================================================================
# Validation
# ==============================================================================
RUN echo "CPU dev image validation: OK"

USER ${USER}
WORKDIR /home/${USER}
