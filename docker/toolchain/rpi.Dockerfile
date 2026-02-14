# ==============================================================================
# toolchain/rpi.Dockerfile - Raspberry Pi cross-compilation toolchain
#
# Extends the aarch64 toolchain with Raspberry Pi-specific libraries and tools.
# Supports Pi 3/4/5 (64-bit). Use dev/rpi for interactive development.
# ==============================================================================
FROM vernier.toolchain.aarch64:latest

ARG USER
ARG UID
ARG GID

LABEL org.opencontainers.image.title="vernier.toolchain.rpi" \
      org.opencontainers.image.description="Raspberry Pi cross-compilation toolchain"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ==============================================================================
# Raspberry Pi Sysroot (Optional)
# ==============================================================================
# The base aarch64 toolchain provides cross-compilation capability.
# For Pi-specific libraries (pigpio, wiringpi, etc.), mount or copy a sysroot
# created from a running Pi at /opt/sysroots/rpi.
#
# Create sysroot from Pi:
#   rsync -avz pi@raspberrypi:/lib /opt/sysroots/rpi/
#   rsync -avz pi@raspberrypi:/usr/lib /opt/sysroots/rpi/usr/
#   rsync -avz pi@raspberrypi:/usr/include /opt/sysroots/rpi/usr/

RUN mkdir -p /opt/sysroots/rpi

ENV RPI_SYSROOT=/opt/sysroots/rpi

# ==============================================================================
# Pi Development Tools
# ==============================================================================
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
      # Disk image manipulation
      kpartx \
      parted \
      dosfstools \
      # Remote deployment
      rsync \
      sshpass

# ==============================================================================
# Validation
# ==============================================================================
RUN aarch64-linux-gnu-gcc --version && \
    echo "Raspberry Pi toolchain validation: OK"

WORKDIR /home/${USER}
