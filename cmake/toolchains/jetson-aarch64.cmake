# ==============================================================================
# Toolchain: jetson-aarch64 (Jetson / Thor AGX)
# ==============================================================================
#
# Cross-compilation for NVIDIA Jetson platforms with CUDA support.
#
# Features:
#   - aarch64 target with cross compilers
#   - CUDA toolkit detection for aarch64/sbsa targets
#   - Linker flags for CUDA cross-library resolution
#
# Usage:
#   cmake --preset jetson-aarch64-release
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/jetson-aarch64.cmake ..
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Base aarch64 configuration
# ------------------------------------------------------------------------------
include("${CMAKE_CURRENT_LIST_DIR}/aarch64-linux-gnu-base.cmake")

# ------------------------------------------------------------------------------
# Optional sysroot
# ------------------------------------------------------------------------------
if (DEFINED CMAKE_SYSROOT
    AND CMAKE_SYSROOT
    AND NOT CMAKE_SYSROOT STREQUAL "/usr/aarch64-linux-gnu"
)
  set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif ()

# ------------------------------------------------------------------------------
# CUDA configuration
# ------------------------------------------------------------------------------
option(VERNIER_REQUIRE_CUDA "Fail configure if aarch64 CUDA target is missing" OFF)

if (NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()
if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-g++)
endif ()
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED CUDA_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif ()

# Prefer shared CUDA runtime (static not shipped in cross tree)
set(CMAKE_CUDA_RUNTIME_LIBRARY
    Shared
    CACHE STRING "" FORCE
)
set(CUDA_USE_STATIC_CUDA_RUNTIME
    OFF
    CACHE BOOL "" FORCE
)

# Locate CUDA target root (aarch64-linux or sbsa-linux)
set(_CUDA_AARCH64_ROOT "")
if (DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
  set(_CUDA_AARCH64_ROOT "${CUDAToolkit_ROOT}")
else ()
  set(_try_a "/usr/local/cuda/targets/aarch64-linux")
  set(_try_b "/usr/local/cuda/targets/sbsa-linux")
  if (EXISTS "${_try_a}/include/cuda_runtime.h")
    set(_CUDA_AARCH64_ROOT "${_try_a}")
  elseif (EXISTS "${_try_b}/include/cuda_runtime.h")
    set(_CUDA_AARCH64_ROOT "${_try_b}")
  endif ()
  if (_CUDA_AARCH64_ROOT)
    set(CUDAToolkit_ROOT "${_CUDA_AARCH64_ROOT}")
  endif ()
endif ()

# Inject linker flags for cross CUDA libdir
if (_CUDA_AARCH64_ROOT)
  set(_CUDA_AARCH64_LIB "${_CUDA_AARCH64_ROOT}/lib")
  if (EXISTS "${_CUDA_AARCH64_LIB}")
    set(CMAKE_EXE_LINKER_FLAGS_INIT
        "${CMAKE_EXE_LINKER_FLAGS_INIT} -L${_CUDA_AARCH64_LIB} -Wl,-rpath-link,${_CUDA_AARCH64_LIB}"
    )
    set(CMAKE_SHARED_LINKER_FLAGS_INIT
        "${CMAKE_SHARED_LINKER_FLAGS_INIT} -L${_CUDA_AARCH64_LIB} -Wl,-rpath-link,${_CUDA_AARCH64_LIB}"
    )
    set(CMAKE_MODULE_LINKER_FLAGS_INIT
        "${CMAKE_MODULE_LINKER_FLAGS_INIT} -L${_CUDA_AARCH64_LIB} -Wl,-rpath-link,${_CUDA_AARCH64_LIB}"
    )
  endif ()
else ()
  if (VERNIER_REQUIRE_CUDA)
    message(FATAL_ERROR "Required aarch64 CUDA target not found under /usr/local/cuda/targets "
                        "(looked in 'aarch64-linux' and 'sbsa-linux'). Install cuda-cross-aarch64."
    )
  endif ()
  set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit
      ON
      CACHE BOOL "" FORCE
  )
endif ()

# ------------------------------------------------------------------------------
# Debug output
# ------------------------------------------------------------------------------
if (VERNIER_TOOLCHAIN_VERBOSE)
  message(STATUS "[jetson] CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}")
  message(STATUS "[jetson] CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "[jetson] CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
endif ()
