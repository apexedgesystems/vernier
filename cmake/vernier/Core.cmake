# ==============================================================================
# vernier/Core.cmake - Foundation utilities for Vernier CMake infrastructure
# ==============================================================================

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# vernier_require(<VAR>...)
#
# Validate that all listed variables are defined and non-empty.
# Emits FATAL_ERROR with clear message if any are missing.
# ------------------------------------------------------------------------------
function (vernier_require)
  foreach (_var IN LISTS ARGN)
    if (NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
      message(FATAL_ERROR "vernier_require: '${_var}' is required but not set")
    endif ()
  endforeach ()
endfunction ()

# ------------------------------------------------------------------------------
# vernier_guard(<target>)
#
# Validate target exists. Emits FATAL_ERROR if not.
# ------------------------------------------------------------------------------
function (vernier_guard _target)
  if (NOT TARGET "${_target}")
    message(FATAL_ERROR "vernier_guard: target '${_target}' does not exist")
  endif ()
endfunction ()

# ------------------------------------------------------------------------------
# vernier_module(<name>)
#
# Set standard module variables from naming convention:
#   LIB_NAME    = <name>
#   SRC_DIR     = ${CMAKE_CURRENT_SOURCE_DIR}/src
#   INC_DIR     = ${CMAKE_CURRENT_SOURCE_DIR}/inc
#   TST_DIR     = ${CMAKE_CURRENT_SOURCE_DIR}/tst
#   UTST_DIR    = ${CMAKE_CURRENT_SOURCE_DIR}/utst
#   PTST_DIR    = ${CMAKE_CURRENT_SOURCE_DIR}/ptst
#   DTST_DIR    = ${CMAKE_CURRENT_SOURCE_DIR}/dtst
#   README_FILE = ${CMAKE_CURRENT_SOURCE_DIR}/README.md
#
# Test directory conventions:
#   utst/ - Unit tests (run with make test)
#   ptst/ - Performance tests (run with make test, longer duration)
#   dtst/ - Development tests (manual execution only, external dependencies)
#   tst/  - General test bucket (legacy)
#
# All directories are optional. CMakeLists should check EXISTS before adding.
# ------------------------------------------------------------------------------
function (vernier_module _name)
  set(LIB_NAME
      "${_name}"
      PARENT_SCOPE
  )
  set(SRC_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      PARENT_SCOPE
  )
  set(INC_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/inc"
      PARENT_SCOPE
  )
  set(TST_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/tst"
      PARENT_SCOPE
  )
  set(UTST_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/utst"
      PARENT_SCOPE
  )
  set(PTST_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/ptst"
      PARENT_SCOPE
  )
  set(DTST_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/dtst"
      PARENT_SCOPE
  )
  set(README_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
      PARENT_SCOPE
  )
endfunction ()

# ------------------------------------------------------------------------------
# vernier_is_cuda_target(<target> <out_var>)
#
# Set <out_var> to TRUE if target has CUDA sources, FALSE otherwise.
# ------------------------------------------------------------------------------
function (vernier_is_cuda_target _target _out_var)
  set(_has_cuda FALSE)

  get_target_property(_sources ${_target} SOURCES)
  if (_sources)
    foreach (_src IN LISTS _sources)
      if (_src MATCHES "\\.(cu|cuh)$")
        set(_has_cuda TRUE)
        break()
      endif ()
    endforeach ()
  endif ()

  set(${_out_var}
      ${_has_cuda}
      PARENT_SCOPE
  )
endfunction ()

# ------------------------------------------------------------------------------
# vernier_standard_optins(<target>)
#
# Apply standard opt-in features to a target:
#   - vernier_enable_coverage (Debug only, respects ENABLE_COVERAGE)
#   - vernier_add_doxygen (respects PROJECT_BUILD_DOCS)
#   - vernier_add_upx_copy (respects ENABLE_UPX)
#   - vernier_clang_tidy_cuda (if target has CUDA sources, respects ENABLE_CLANG_TIDY)
#
# Safe to call on any target type. No-ops gracefully when features disabled
# or when target was skipped (e.g., due to BAREMETAL guard).
# ------------------------------------------------------------------------------
function (vernier_standard_optins _target)
  # Silently return if target doesn't exist (skipped by BAREMETAL guard)
  if (NOT TARGET "${_target}")
    return()
  endif ()

  # Doxygen documentation
  vernier_add_doxygen(${_target})

  # UPX compression
  vernier_add_upx_copy(${_target})

  # CUDA-specific: clang-tidy on .cu files
  vernier_is_cuda_target(${_target} _has_cuda)
  if (_has_cuda)
    get_target_property(_sources ${_target} SOURCES)
    set(_cu_files)
    foreach (_src IN LISTS _sources)
      if (_src MATCHES "\\.(cu)$")
        list(APPEND _cu_files "${_src}")
      endif ()
    endforeach ()
    if (_cu_files)
      vernier_clang_tidy_cuda(${_target} FILES ${_cu_files} ALLOW_FAILURE)
    endif ()
  endif ()
endfunction ()
