# ==============================================================================
# Toolchain: riscv64-linux-gnu (RISC-V 64-bit Linux)
# ==============================================================================
#
# Purpose:
#   - Cross-compile for RISC-V 64-bit Linux targets
#   - Supports SiFive boards, StarFive VisionFive, Milk-V, etc.
#
# Usage:
#   cmake --preset riscv64-linux-release
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake ..
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Target platform
# ------------------------------------------------------------------------------
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# ------------------------------------------------------------------------------
# Cross compilers
# ------------------------------------------------------------------------------
# Try riscv64-linux-gnu (Debian/Ubuntu) first, fall back to riscv64-unknown-linux-gnu
if (NOT DEFINED CMAKE_C_COMPILER)
  find_program(_RISCV_GCC NAMES riscv64-linux-gnu-gcc riscv64-unknown-linux-gnu-gcc)
  if (_RISCV_GCC)
    set(CMAKE_C_COMPILER "${_RISCV_GCC}")
  else ()
    set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
  endif ()
endif ()

if (NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(_RISCV_GXX NAMES riscv64-linux-gnu-g++ riscv64-unknown-linux-gnu-g++)
  if (_RISCV_GXX)
    set(CMAKE_CXX_COMPILER "${_RISCV_GXX}")
  else ()
    set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
  endif ()
endif ()

# Compiler target triple
get_filename_component(_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
string(REGEX REPLACE "-gcc$" "" _TARGET_TRIPLE "${_COMPILER_NAME}")
set(CMAKE_C_COMPILER_TARGET "${_TARGET_TRIPLE}")
set(CMAKE_CXX_COMPILER_TARGET "${_TARGET_TRIPLE}")

# ------------------------------------------------------------------------------
# RISC-V architecture flags
# ------------------------------------------------------------------------------
# rv64gc = rv64imafdc (general-purpose 64-bit with compressed instructions)
# lp64d = 64-bit long/pointer, double-precision float ABI
set(RISCV_ARCH
    "rv64gc"
    CACHE STRING "RISC-V architecture string"
)
set(RISCV_ABI
    "lp64d"
    CACHE STRING "RISC-V ABI"
)

add_compile_options(-march=${RISCV_ARCH} -mabi=${RISCV_ABI})
add_link_options(-march=${RISCV_ARCH} -mabi=${RISCV_ABI})

# ------------------------------------------------------------------------------
# Multiarch hints
# ------------------------------------------------------------------------------
set(CMAKE_LIBRARY_ARCHITECTURE riscv64-linux-gnu)
list(APPEND CMAKE_LIBRARY_PATH "/usr/riscv64-linux-gnu/lib" "/usr/lib/riscv64-linux-gnu"
     "/lib/riscv64-linux-gnu"
)

# ------------------------------------------------------------------------------
# Sysroot configuration
# ------------------------------------------------------------------------------
set(RISCV_SYSROOT
    ""
    CACHE PATH "Path to RISC-V sysroot"
)

if (RISCV_SYSROOT AND EXISTS "${RISCV_SYSROOT}")
  set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
  set(CMAKE_FIND_ROOT_PATH "${RISCV_SYSROOT}")
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif ()

# ------------------------------------------------------------------------------
# Try-compile: never run target binaries on host
# ------------------------------------------------------------------------------
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ------------------------------------------------------------------------------
# QEMU emulator for test discovery
# ------------------------------------------------------------------------------
find_program(QEMU_RISCV64 NAMES qemu-riscv64-static qemu-riscv64)
if (QEMU_RISCV64)
  set(CMAKE_CROSSCOMPILING_EMULATOR
      "${QEMU_RISCV64}"
      CACHE FILEPATH "" FORCE
  )
endif ()

# ------------------------------------------------------------------------------
# Debug output
# ------------------------------------------------------------------------------
set(VERNIER_TOOLCHAIN_VERBOSE
    OFF
    CACHE BOOL "Print toolchain debug lines"
)
if (VERNIER_TOOLCHAIN_VERBOSE)
  message(STATUS "[riscv64] CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}")
  message(STATUS "[riscv64] ARCH=${RISCV_ARCH} ABI=${RISCV_ABI}")
  message(STATUS "[riscv64] SYSROOT=${CMAKE_SYSROOT}")
endif ()
