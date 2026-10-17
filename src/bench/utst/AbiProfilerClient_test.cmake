# ==============================================================================
# AbiProfilerClient_test.cmake - Runs one AbiProfilerClient build and checks
# how it ends
#
# Run with: cmake -DCLIENT=<exe> -DEXPECT=accepted|rejected -P AbiProfilerClient_test.cmake
#
# accepted: exit 0, the profiler constructed and its abortMeasure() returned.
# rejected: the mismatch exit status 3 with the start-up check's message
#           naming the ABI versions, before any profiler is constructed.
# A sanitizer report fails either expectation.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

execute_process(
  COMMAND "${CLIENT}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
message(STATUS "exit status: ${_rc}\n${_out}")

set(_problems "")
if (_out MATCHES "Sanitizer" OR _out MATCHES "runtime error:")
  string(APPEND _problems " sanitizer report in the output;")
endif ()
if (EXPECT STREQUAL "accepted")
  if (NOT _rc EQUAL 0)
    string(APPEND _problems " expected exit 0;")
  endif ()
  if (NOT _out MATCHES "client: abortMeasure\\(\\) returned")
    string(APPEND _problems " abortMeasure() did not return;")
  endif ()
elseif (EXPECT STREQUAL "rejected")
  if (NOT _rc EQUAL 3)
    string(APPEND _problems " expected the mismatch exit status 3;")
  endif ()
  if (NOT _out MATCHES "\\[bench\\] ABI mismatch: .*ABI version: benchmark [0-9]+, library 2")
    string(APPEND _problems " no ABI mismatch message naming the library's version 2;")
  endif ()
  if (_out MATCHES "client: constructed")
    string(APPEND _problems " a profiler was constructed past the check;")
  endif ()
else ()
  message(FATAL_ERROR "EXPECT must be accepted or rejected")
endif ()

if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "AbiProfilerClient (${EXPECT}):${_problems}")
endif ()
