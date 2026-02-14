# ==============================================================================
# Toolchain: aarch64-linux-gnu-base
# ==============================================================================
#
# Common configuration for aarch64 Linux cross-compilation.
# Include this from platform-specific toolchains (Jetson, Pi, etc.)
#
# Provides:
#   - Target platform settings (CMAKE_SYSTEM_NAME, CMAKE_SYSTEM_PROCESSOR)
#   - Cross compiler defaults (aarch64-linux-gnu-gcc/g++)
#   - Multiarch hints for find_library()
#   - Try-compile configuration
#   - QEMU emulator setup
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Target platform
# ------------------------------------------------------------------------------
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ------------------------------------------------------------------------------
# Cross compilers (do not override user-specified)
# ------------------------------------------------------------------------------
if (NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
endif ()
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif ()
if (NOT DEFINED CMAKE_C_COMPILER_TARGET)
  set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
endif ()
if (NOT DEFINED CMAKE_CXX_COMPILER_TARGET)
  set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
endif ()

# ------------------------------------------------------------------------------
# Multiarch hints (let find_library see aarch64 dirs)
# ------------------------------------------------------------------------------
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)
list(APPEND CMAKE_LIBRARY_PATH "/usr/aarch64-linux-gnu/lib" "/usr/lib/aarch64-linux-gnu"
     "/lib/aarch64-linux-gnu"
)

# ------------------------------------------------------------------------------
# Try-compile: never run target binaries on host
# ------------------------------------------------------------------------------
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ------------------------------------------------------------------------------
# QEMU emulator for test discovery
# ------------------------------------------------------------------------------
find_program(QEMU_AARCH64 qemu-aarch64-static)
if (QEMU_AARCH64)
  set(CMAKE_CROSSCOMPILING_EMULATOR
      "${QEMU_AARCH64}"
      CACHE FILEPATH "" FORCE
  )
endif ()

# ------------------------------------------------------------------------------
# Debug output
# ------------------------------------------------------------------------------
option(VERNIER_TOOLCHAIN_VERBOSE "Print toolchain debug lines" OFF)
