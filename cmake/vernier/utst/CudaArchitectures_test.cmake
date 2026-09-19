# ==============================================================================
# CudaArchitectures_test.cmake - Script-mode test of the architecture resolver
#
# Run with: cmake -P CudaArchitectures_test.cmake
#
# Needs no CUDA toolkit and no GPU. Each case runs the resolver in a child
# cmake process so that a FATAL_ERROR is an observable result, and so that the
# CUDAARCHS environment variable is controlled per case instead of inherited.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

# ------------------------------------------------------------------------------
# Child Mode
# ------------------------------------------------------------------------------
# Inputs arrive as CASE_* so that "not set" stays distinguishable from "empty".
# Lists are written with commas: a semicolon does not survive the argument
# parsing and process boundary between the case table and this child.
# ------------------------------------------------------------------------------

if (DEFINED CASE_RUN)
  if (DEFINED CASE_STANDARD)
    string(REPLACE "," ";" CMAKE_CUDA_ARCHITECTURES "${CASE_STANDARD}")
  endif ()
  if (DEFINED CASE_SHORTHAND)
    string(REPLACE "," ";" CUDA_ARCHS "${CASE_SHORTHAND}")
  endif ()
  set(VERNIER_CUDA_ARCHITECTURES_DEFAULT "89")
  include("${CMAKE_CURRENT_LIST_DIR}/../CudaArchitectures.cmake")
  vernier_resolve_cuda_architectures(_archs _source)
  string(REPLACE ";" "," _archs "${_archs}")
  message("RESOLVED=[${_archs}] SOURCE=[${_source}]")
  return()
endif ()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

set(_failures 0)

# run_case(<name> [STANDARD <v>] [SHORTHAND <v>] [ENV <v>]
#          (EXPECT <archs> FROM <source> | EXPECT_CONFLICT <text>...))
function (run_case _name)
  cmake_parse_arguments(C "" "STANDARD;SHORTHAND;ENV;EXPECT;FROM" "EXPECT_CONFLICT" ${ARGN})

  set(_env --unset=CUDAARCHS)
  if (DEFINED C_ENV)
    set(_env "CUDAARCHS=${C_ENV}")
  endif ()
  set(_defs -DCASE_RUN=1)
  if (DEFINED C_STANDARD)
    list(APPEND _defs "-DCASE_STANDARD=${C_STANDARD}")
  endif ()
  if (DEFINED C_SHORTHAND)
    list(APPEND _defs "-DCASE_SHORTHAND=${C_SHORTHAND}")
  endif ()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${_env} "${CMAKE_COMMAND}" ${_defs} -P
            "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )

  set(_why "")
  if (DEFINED C_EXPECT_CONFLICT)
    if (_rc EQUAL 0)
      set(_why "expected configuration to stop, but it resolved: ${_out}")
    endif ()
    foreach (_text IN LISTS C_EXPECT_CONFLICT)
      string(FIND "${_out}" "${_text}" _pos)
      if (_pos EQUAL -1)
        string(APPEND _why " message lacks '${_text}';")
      endif ()
    endforeach ()
  else ()
    set(_want "RESOLVED=[${C_EXPECT}] SOURCE=[${C_FROM}]")
    string(FIND "${_out}" "${_want}" _pos)
    if (NOT _rc EQUAL 0 OR _pos EQUAL -1)
      set(_why "wanted ${_want}, got rc=${_rc}: ${_out}")
    endif ()
  endif ()

  if (_why STREQUAL "")
    message(STATUS "PASS ${_name}")
  else ()
    message(STATUS "FAIL ${_name}: ${_why}")
    math(EXPR _failures "${_failures} + 1")
    set(_failures
        "${_failures}"
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# ------------------------------------------------------------------------------
# Cases (12)
# ------------------------------------------------------------------------------

run_case(NothingSetUsesDefault EXPECT "89" FROM "default")
run_case(
  EmptyShorthandUsesDefault
  SHORTHAND
  ""
  EXPECT
  "89"
  FROM
  "default"
)
run_case(
  ShorthandAlone
  SHORTHAND
  "110"
  EXPECT
  "110"
  FROM
  "CUDA_ARCHS"
)
run_case(
  StandardAloneIsHonored
  STANDARD
  "110"
  EXPECT
  "110"
  FROM
  "CMAKE_CUDA_ARCHITECTURES"
)
run_case(
  StandardWithEmptyShorthand
  STANDARD
  "110"
  SHORTHAND
  ""
  EXPECT
  "110"
  FROM
  "CMAKE_CUDA_ARCHITECTURES"
)
run_case(
  StandardKeywordPassesThrough
  STANDARD
  "native"
  EXPECT
  "native"
  FROM
  "CMAKE_CUDA_ARCHITECTURES"
)
run_case(
  EnvironmentAloneIsHonored
  ENV
  "87"
  EXPECT
  "87"
  FROM
  "CUDAARCHS (environment)"
)
run_case(
  StandardOutranksEnvironment
  STANDARD
  "110"
  ENV
  "87"
  EXPECT
  "110"
  FROM
  "CMAKE_CUDA_ARCHITECTURES"
)
run_case(
  BothAgree
  STANDARD
  "110"
  SHORTHAND
  "110"
  EXPECT
  "110"
  FROM
  "CMAKE_CUDA_ARCHITECTURES"
)
run_case(
  BothAgreeInAnyOrder
  STANDARD
  "80,86"
  SHORTHAND
  "86,80"
  EXPECT
  "80,86"
  FROM
  "CMAKE_CUDA_ARCHITECTURES"
)
run_case(
  StandardAndShorthandDiffer
  STANDARD
  "110"
  SHORTHAND
  "89"
  EXPECT_CONFLICT
  "CMAKE_CUDA_ARCHITECTURES='110'"
  "CUDA_ARCHS='89'"
)
run_case(
  EnvironmentAndShorthandDiffer
  ENV
  "87"
  SHORTHAND
  "89"
  EXPECT_CONFLICT
  "CUDAARCHS (environment)='87'"
  "CUDA_ARCHS='89'"
)

if (_failures GREATER 0)
  message(FATAL_ERROR "${_failures} case(s) failed")
endif ()
message(STATUS "All cases passed")
