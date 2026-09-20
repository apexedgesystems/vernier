# ==============================================================================
# AbiConsumer_test.cmake - Runs one AbiConsumer build and checks how it ends
#
# Run with: cmake -DCONSUMER=<exe> -DEXPECT=match|mismatch -DWORK_DIR=<dir>
#                 -P AbiConsumer_test.cmake
#
# match:    exit 0 and the profiler was constructed.
# mismatch: the layout message naming both PerfConfig sizes, the mismatch exit
#           status, and no profiler constructed. Sanitizer reports change the
#           exit status or the output, so a sanitizer build fails this too.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
execute_process(
  COMMAND "${CONSUMER}" "${WORK_DIR}"
  WORKING_DIRECTORY "${WORK_DIR}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
file(GLOB _left_behind "${WORK_DIR}/*")
file(REMOVE_RECURSE "${WORK_DIR}")
message(STATUS "exit status: ${_rc}\n${_out}")

set(_problems "")
string(FIND "${_out}" "constructed profiler" _constructed)
if (_out MATCHES "Sanitizer")
  string(APPEND _problems " sanitizer report in the output;")
endif ()

if (EXPECT STREQUAL "match")
  if (NOT _rc EQUAL 0)
    string(APPEND _problems " expected exit 0;")
  endif ()
  if (_constructed EQUAL -1)
    string(APPEND _problems " the profiler was not constructed;")
  endif ()
elseif (EXPECT STREQUAL "mismatch")
  if (NOT _rc EQUAL 3)
    string(APPEND _problems " expected the mismatch exit status 3;")
  endif ()
  if (NOT _out MATCHES
      "\\[bench\\] ABI mismatch: .*sizeof\\(PerfConfig\\): benchmark [0-9]+, library [0-9]+"
  )
    string(APPEND _problems " no ABI mismatch message naming both PerfConfig sizes;")
  endif ()
  if (NOT _constructed EQUAL -1)
    string(APPEND _problems " a profiler was constructed from a mismatched PerfConfig;")
  endif ()
  if (_left_behind)
    string(APPEND _problems " an artifact folder was created: ${_left_behind};")
  endif ()
else ()
  message(FATAL_ERROR "EXPECT must be match or mismatch")
endif ()

if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "AbiConsumer (${EXPECT}):${_problems}")
endif ()
