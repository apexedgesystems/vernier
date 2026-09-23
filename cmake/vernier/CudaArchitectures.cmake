# ==============================================================================
# vernier/CudaArchitectures.cmake - Which GPU architectures CUDA code targets
# ==============================================================================

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# vernier_resolve_cuda_architectures(<out_archs> <out_source>)
#
# Decide the value of CMAKE_CUDA_ARCHITECTURES. Call before
# enable_language(CUDA): once the language is enabled CMake has cached the
# compiler's default, which can no longer be told apart from a user's choice.
#
# Inputs, strongest first:
#   CMAKE_CUDA_ARCHITECTURES  the standard variable (command line, preset,
#                             toolchain file, or a parent project)
#   CUDAARCHS                 the standard environment variable, which CMake
#                             reads only when the variable above is not set
#   CUDA_ARCHS                this project's shorthand for the same setting
#   VERNIER_CUDA_ARCHITECTURES_DEFAULT  used when none of the above is set
#
# The standard setting and CUDA_ARCHS may both be given when they name the same
# architectures (order ignored). When they differ, configuration stops: device
# code built for the wrong architecture still runs, through driver JIT
# compilation or not at all on older drivers, so a silent pick would go
# unnoticed.
#
# <out_source> receives the name of the input that decided the value.
# ------------------------------------------------------------------------------
function (vernier_resolve_cuda_architectures _out_archs _out_source)
  set(_standard "")
  set(_standard_name "")
  if (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(_standard "${CMAKE_CUDA_ARCHITECTURES}")
    set(_standard_name "CMAKE_CUDA_ARCHITECTURES")
  elseif (NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(_standard "$ENV{CUDAARCHS}")
    set(_standard_name "CUDAARCHS (environment)")
  endif ()

  set(_shorthand "")
  if (DEFINED CUDA_ARCHS)
    set(_shorthand "${CUDA_ARCHS}")
  endif ()

  if (NOT _standard STREQUAL "" AND NOT _shorthand STREQUAL "")
    set(_lhs ${_standard})
    set(_rhs ${_shorthand})
    list(SORT _lhs)
    list(SORT _rhs)
    if (NOT "${_lhs}" STREQUAL "${_rhs}")
      message(
        FATAL_ERROR
          "Conflicting CUDA architectures: ${_standard_name}='${_standard}' but "
          "CUDA_ARCHS='${_shorthand}'. Set one of them, or give both the same value. "
          "In an existing build directory either may be a cached value from an earlier "
          "configure: remove the stale one with -UCUDA_ARCHS or -UCMAKE_CUDA_ARCHITECTURES, "
          "or start from an empty build directory."
      )
    endif ()
  endif ()

  if (NOT _standard STREQUAL "")
    set(${_out_archs}
        "${_standard}"
        PARENT_SCOPE
    )
    set(${_out_source}
        "${_standard_name}"
        PARENT_SCOPE
    )
  elseif (NOT _shorthand STREQUAL "")
    set(${_out_archs}
        "${_shorthand}"
        PARENT_SCOPE
    )
    set(${_out_source}
        "CUDA_ARCHS"
        PARENT_SCOPE
    )
  else ()
    set(${_out_archs}
        "${VERNIER_CUDA_ARCHITECTURES_DEFAULT}"
        PARENT_SCOPE
    )
    set(${_out_source}
        "default"
        PARENT_SCOPE
    )
  endif ()
endfunction ()
