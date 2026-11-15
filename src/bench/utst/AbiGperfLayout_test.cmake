# ==============================================================================
# AbiGperfLayout_test.cmake - Runs the two AbiGperfConsumer programs
#
# Run with: cmake -DDEFAULT=<exe> -DHIDDEN=<exe> -P AbiGperfLayout_test.cmake
#
# DEFAULT was compiled with the default include path, HIDDEN with gperftools'
# headers removed from it (AbiGperfLayout.cmake). Each prints whether it saw
# gperftools' headers and its sizeof(GperfProfiler) beside libbench's, and
# exits 0 only when the two agree. Passes when both exit 0 and, where the
# default program saw the headers, the other did not.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

foreach (_var DEFAULT HIDDEN)
  if (NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "AbiGperfLayout_test.cmake: -D${_var}=... is required")
  endif ()
endforeach ()

set(_problems "")
foreach (_variant DEFAULT HIDDEN)
  execute_process(
    COMMAND "${${_variant}}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  message(STATUS "${_variant}: exit status ${_rc}: ${_out}")
  string(REGEX MATCH "gperftools headers ([a-z]+);" _match "${_out}")
  set(_headers_${_variant} "${CMAKE_MATCH_1}")
  if (_headers_${_variant} STREQUAL "")
    string(APPEND _problems " ${_variant}: no report;")
  elseif (NOT _rc EQUAL 0)
    string(APPEND _problems " ${_variant}: its sizeof(GperfProfiler) is not libbench's;")
  endif ()
endforeach ()
if (_headers_DEFAULT STREQUAL "visible" AND NOT _headers_HIDDEN STREQUAL "hidden")
  string(APPEND _problems " HIDDEN saw gperftools' headers, so it tests nothing;")
endif ()

if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "AbiGperfLayout:${_problems}")
endif ()
message(STATUS "Both consumers lay GperfProfiler out as libbench does")
