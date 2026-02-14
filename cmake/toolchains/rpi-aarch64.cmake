# ==============================================================================
# Toolchain: rpi-aarch64 (Raspberry Pi 4/5)
# ==============================================================================
#
# Cross-compilation for Raspberry Pi (64-bit ARM).
#
# Features:
#   - aarch64 target with cross compilers
#   - Optional sysroot with Pi-specific libraries (pigpio, camera, etc.)
#   - No CUDA (unlike Jetson toolchain)
#
# Usage:
#   cmake --preset rpi-aarch64-release
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/rpi-aarch64.cmake ..
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Base aarch64 configuration
# ------------------------------------------------------------------------------
include("${CMAKE_CURRENT_LIST_DIR}/aarch64-linux-gnu-base.cmake")

# ------------------------------------------------------------------------------
# Sysroot configuration
# ------------------------------------------------------------------------------
# Set RPI_SYSROOT to a Raspberry Pi OS sysroot for access to Pi-specific
# libraries (pigpio, wiringpi, etc.)
#
# Create sysroot from a running Pi:
#   rsync -avz pi@raspberrypi:/lib /opt/sysroots/rpi/
#   rsync -avz pi@raspberrypi:/usr/lib /opt/sysroots/rpi/usr/
#   rsync -avz pi@raspberrypi:/usr/include /opt/sysroots/rpi/usr/
#
set(RPI_SYSROOT
    ""
    CACHE PATH "Path to Raspberry Pi sysroot"
)

if (RPI_SYSROOT AND EXISTS "${RPI_SYSROOT}")
  set(CMAKE_SYSROOT "${RPI_SYSROOT}")
  set(CMAKE_FIND_ROOT_PATH "${RPI_SYSROOT}")
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

  # Pi-specific library paths
  list(APPEND CMAKE_LIBRARY_PATH "${RPI_SYSROOT}/opt/vc/lib"
       "${RPI_SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
  list(APPEND CMAKE_INCLUDE_PATH "${RPI_SYSROOT}/opt/vc/include")
endif ()

# ------------------------------------------------------------------------------
# Debug output
# ------------------------------------------------------------------------------
if (VERNIER_TOOLCHAIN_VERBOSE)
  message(STATUS "[rpi] CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}")
  message(STATUS "[rpi] SYSROOT=${CMAKE_SYSROOT}")
endif ()
