# ==============================================================================
# vernier/Cuda.cmake - CUDA source integration and NVML support
# ==============================================================================

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# vernier_cuda_sources(<target> FILES <cu...>
#                      [NO_CUDART] [SEPARABLE] [RESOLVE_DEVICE_SYMBOLS]
#                      [DEFINE <n>])
#
# Add CUDA source files to an existing target (no-op when CUDA inactive).
# Links CUDA::cudart by default. Defines VERNIER_HAS_CUDA=1 unless overridden.
# ------------------------------------------------------------------------------
function (vernier_cuda_sources _target)
  vernier_guard(${_target})

  # Require both toolkit and language
  if (NOT CUDAToolkit_FOUND OR NOT CMAKE_CUDA_COMPILER)
    return()
  endif ()

  cmake_parse_arguments(CU "NO_CUDART;SEPARABLE;RESOLVE_DEVICE_SYMBOLS" "DEFINE" "FILES" ${ARGN})

  if (NOT CU_FILES)
    message(FATAL_ERROR "vernier_cuda_sources: no FILES specified for '${_target}'")
  endif ()

  # Determine target type for PRIVATE vs INTERFACE
  get_target_property(_tgt_type "${_target}" TYPE)
  if (_tgt_type STREQUAL "INTERFACE_LIBRARY")
    set(_scope INTERFACE)
  else ()
    set(_scope PRIVATE)
  endif ()

  target_sources(${_target} ${_scope} ${CU_FILES})

  # Device-link properties (default OFF to avoid pulling libcudadevrt)
  if (NOT _tgt_type STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(
      ${_target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    if (CU_SEPARABLE)
      set_target_properties(${_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif ()
    if (CU_RESOLVE_DEVICE_SYMBOLS)
      set_target_properties(${_target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif ()
  endif ()

  # Compile definition for header guards
  set(_cuda_define "${CU_DEFINE}")
  if (NOT _cuda_define)
    set(_cuda_define "VERNIER_HAS_CUDA")
  endif ()
  target_compile_definitions(${_target} ${_scope} ${_cuda_define}=1)

  # Link CUDA runtime
  if (NOT CU_NO_CUDART)
    if (TARGET CUDA::cudart)
      target_link_libraries(${_target} ${_scope} CUDA::cudart)
    else ()
      message(WARNING "vernier_cuda_sources: CUDA::cudart not found")
    endif ()
  endif ()
endfunction ()

# ==============================================================================
# NVML and CUPTI Integration
# ==============================================================================
#
# These functions handle cross-compilation correctly by:
# 1. Using CUDAToolkit_LIBRARY_DIR (already target-arch-specific) as primary source
# 2. Only adding native system paths when NOT cross-compiling
# 3. Supporting environment variables for custom installations
#
# Users can override search paths via:
#   - NVML_ROOT / NVIDIA_ML_ROOT environment variables
#   - CUDAToolkit_ROOT cmake variable
# ==============================================================================

# Helper: Build library search paths appropriate for current build type
function (_vernier_cuda_lib_paths _out_paths _out_stub_paths)
  set(_paths)
  set(_stubs)

  # CUDA toolkit library dir is always the primary source (already arch-specific)
  if (CUDAToolkit_LIBRARY_DIR)
    list(APPEND _paths "${CUDAToolkit_LIBRARY_DIR}")
    list(APPEND _stubs "${CUDAToolkit_LIBRARY_DIR}/stubs")
  endif ()

  # Native builds: add system paths
  if (NOT CMAKE_CROSSCOMPILING)
    list(
      APPEND
      _paths
      /usr/lib/x86_64-linux-gnu
      /usr/lib64
      /usr/lib
      /usr/local/lib
      /usr/lib/wsl/lib
      /usr/local/cuda/lib64
    )
    list(APPEND _stubs /usr/local/cuda/lib64/stubs /usr/local/cuda/targets/x86_64-linux/lib/stubs)
  endif ()

  set(${_out_paths}
      "${_paths}"
      PARENT_SCOPE
  )
  set(${_out_stub_paths}
      "${_stubs}"
      PARENT_SCOPE
  )
endfunction ()

# Helper: Build include search paths appropriate for current build type
function (_vernier_cuda_inc_paths _out_paths)
  set(_paths)

  if (CUDAToolkit_INCLUDE_DIRS)
    list(APPEND _paths "${CUDAToolkit_INCLUDE_DIRS}")
  endif ()

  if (NOT CMAKE_CROSSCOMPILING)
    list(APPEND _paths /usr/local/cuda/include /usr/include /usr/local/include /usr/include/nvidia)
  endif ()

  set(${_out_paths}
      "${_paths}"
      PARENT_SCOPE
  )
endfunction ()

# Helper: Determine visibility scope based on target type
function (_vernier_cuda_target_scope _target _out_scope)
  get_target_property(_tgt_type "${_target}" TYPE)
  if (_tgt_type STREQUAL "INTERFACE_LIBRARY")
    set(${_out_scope}
        INTERFACE
        PARENT_SCOPE
    )
  else ()
    set(${_out_scope}
        PUBLIC
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# ------------------------------------------------------------------------------
# vernier_nvml_enable(<target>)
#
# Enable NVML usage when libnvidia-ml and nvml.h are available.
# Defines COMPAT_NVML_AVAILABLE=1 (found) or COMPAT_NVML_AVAILABLE=0 (missing).
#
# Search priority:
#   1. NVML_ROOT / NVIDIA_ML_ROOT environment variables
#   2. CUDAToolkit_LIBRARY_DIR (target-arch-specific)
#   3. System paths (native builds only)
#   4. Stubs (link-time only, runtime requires driver)
# ------------------------------------------------------------------------------
function (vernier_nvml_enable _target)
  # Silently return if target doesn't exist (skipped by BAREMETAL guard)
  if (NOT TARGET "${_target}")
    return()
  endif ()

  # Determine visibility scope for this target type
  _vernier_cuda_target_scope(${_target} _scope)

  if (CMAKE_SYSTEM_NAME STREQUAL "Generic")
    target_compile_definitions(${_target} ${_scope} COMPAT_NVML_AVAILABLE=0)
    return()
  endif ()

  # Use cached result if already searched
  if (DEFINED NVML_LIBRARY AND DEFINED NVML_INCLUDE_DIR)
    if (NVML_INCLUDE_DIR AND NVML_LIBRARY)
      target_compile_definitions(${_target} ${_scope} COMPAT_NVML_AVAILABLE=1)
      target_include_directories(${_target} ${_scope} "${NVML_INCLUDE_DIR}")
      target_link_libraries(${_target} ${_scope} "${NVML_LIBRARY}")
    else ()
      target_compile_definitions(${_target} ${_scope} COMPAT_NVML_AVAILABLE=0)
    endif ()
    return()
  endif ()

  _vernier_cuda_inc_paths(_inc_paths)
  _vernier_cuda_lib_paths(_lib_paths _stub_paths)

  find_path(
    NVML_INCLUDE_DIR
    NAMES nvml.h
    HINTS ENV NVML_ROOT ENV NVIDIA_ML_ROOT
    PATH_SUFFIXES include
    PATHS ${_inc_paths}
  )

  find_library(
    NVML_LIBRARY
    NAMES nvidia-ml nvml
    HINTS ENV NVML_ROOT ENV NVIDIA_ML_ROOT
    PATH_SUFFIXES lib lib64 tegra
    PATHS ${_lib_paths}
  )

  # Fallback to stubs (link-time only)
  set(_using_stubs FALSE)
  if (NOT NVML_LIBRARY AND _stub_paths)
    find_library(
      NVML_LIBRARY
      NAMES nvidia-ml nvml
      NO_DEFAULT_PATH
      PATHS ${_stub_paths}
    )
    if (NVML_LIBRARY)
      set(_using_stubs TRUE)
    endif ()
  endif ()

  if (NVML_INCLUDE_DIR AND NVML_LIBRARY)
    if (_using_stubs)
      message(STATUS "NVML: using stubs (runtime requires nvidia driver)")
    else ()
      message(STATUS "NVML found: ${NVML_LIBRARY}")
    endif ()
    target_compile_definitions(${_target} ${_scope} COMPAT_NVML_AVAILABLE=1)
    target_include_directories(${_target} ${_scope} "${NVML_INCLUDE_DIR}")
    target_link_libraries(${_target} ${_scope} "${NVML_LIBRARY}")

    # RPATH handling only for non-INTERFACE targets
    get_target_property(_tgt_type "${_target}" TYPE)
    if (NOT _tgt_type STREQUAL "INTERFACE_LIBRARY")
      get_filename_component(_nvml_dir "${NVML_LIBRARY}" DIRECTORY)
      if (UNIX AND NOT APPLE)
        set_property(
          TARGET ${_target}
          APPEND
          PROPERTY BUILD_RPATH "${_nvml_dir}"
        )
        set_property(
          TARGET ${_target}
          APPEND
          PROPERTY INSTALL_RPATH "${_nvml_dir}"
        )
      endif ()
    endif ()
  else ()
    message(STATUS "NVML not found - building without NVML (set NVML_ROOT if installed)")
    target_compile_definitions(${_target} ${_scope} COMPAT_NVML_AVAILABLE=0)
  endif ()
endfunction ()

# ------------------------------------------------------------------------------
# vernier_cupti_enable(<target>)
#
# Enable CUPTI usage when libcupti is available.
# Defines COMPAT_CUPTI_AVAILABLE=1 (found) or COMPAT_CUPTI_AVAILABLE=0 (missing).
#
# Search priority:
#   1. CUDAToolkit paths (extras/CUPTI or main lib dir)
#   2. System CUDA paths (native builds only)
# ------------------------------------------------------------------------------
function (vernier_cupti_enable _target)
  vernier_guard(${_target})

  # Determine visibility scope for this target type
  _vernier_cuda_target_scope(${_target} _scope)

  if (NOT CUDAToolkit_FOUND)
    target_compile_definitions(${_target} ${_scope} COMPAT_CUPTI_AVAILABLE=0)
    return()
  endif ()

  # Use cached result if already searched
  if (DEFINED CUPTI_LIBRARY)
    if (CUPTI_LIBRARY)
      target_compile_definitions(${_target} ${_scope} COMPAT_CUPTI_AVAILABLE=1)
      target_link_libraries(${_target} ${_scope} ${CUPTI_LIBRARY})
      if (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        target_include_directories(
          ${_target} ${_scope} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../extras/CUPTI/include
        )
      endif ()
    else ()
      target_compile_definitions(${_target} ${_scope} COMPAT_CUPTI_AVAILABLE=0)
    endif ()
    return()
  endif ()

  # Build search paths - CUPTI is in extras/CUPTI or main lib dir
  set(_lib_paths)

  if (CUDAToolkit_LIBRARY_DIR)
    get_filename_component(_cuda_root "${CUDAToolkit_LIBRARY_DIR}/.." ABSOLUTE)
    list(APPEND _lib_paths "${_cuda_root}/extras/CUPTI/lib64" "${_cuda_root}/extras/CUPTI/lib"
         "${CUDAToolkit_LIBRARY_DIR}"
    )
  endif ()

  if (NOT CMAKE_CROSSCOMPILING)
    if (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
      list(APPEND _lib_paths "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../extras/CUPTI/lib64"
           "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64"
      )
    endif ()
    list(APPEND _lib_paths /usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/lib64)
  endif ()

  find_library(
    CUPTI_LIBRARY
    NAMES cupti
    PATHS ${_lib_paths}
    NO_DEFAULT_PATH
  )

  if (CUPTI_LIBRARY)
    message(STATUS "CUPTI found: ${CUPTI_LIBRARY}")
    target_compile_definitions(${_target} ${_scope} COMPAT_CUPTI_AVAILABLE=1)
    target_link_libraries(${_target} ${_scope} ${CUPTI_LIBRARY})
    if (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
      target_include_directories(
        ${_target} ${_scope} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../extras/CUPTI/include
      )
    endif ()
  else ()
    message(STATUS "CUPTI not found - building without CUPTI")
    target_compile_definitions(${_target} ${_scope} COMPAT_CUPTI_AVAILABLE=0)
  endif ()
endfunction ()
