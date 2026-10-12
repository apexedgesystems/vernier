# ==============================================================================
# CallgrindWindowReaderAssertion_test.cmake - Holds CallgrindWindowProbe_test.cmake
# to the evidence for valgrind's reader assertion
#
# Run with: cmake -DDRIVER=<CallgrindWindowProbe_test.cmake>
#                 -DFAKE=<CallgrindWindowReaderAssertion_fake_valgrind.sh>
#                 -DREADER=<ValgrindReaderAssertionCli> -DWORK_DIR=<dir>
#                 -P CallgrindWindowReaderAssertion_test.cmake
#
# Runs the window test's driver three times, in its runner case, against a
# stand-in valgrind that prints its reader's assertion after callgrind's
# opening lines: right after them and killed by SIGSEGV, as valgrind 3.18.1
# stops before the program starts, where the driver must skip quoting the
# line; after a line of the program's; and with exit status 1. In the last two
# the driver must fail, saying the probe's tests did not start. Each run's exit
# status is checked, and its skip or failure; nothing here needs valgrind.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/bin")
get_filename_component(_fake_name "${FAKE}" NAME)
file(
  COPY "${FAKE}"
  DESTINATION "${WORK_DIR}/bin"
  FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE
)
file(RENAME "${WORK_DIR}/bin/${_fake_name}" "${WORK_DIR}/bin/valgrind")

set(_failures "")
foreach (_case stopped_before_start program_wrote_first exit_1)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env "PATH=${WORK_DIR}/bin:$ENV{PATH}" "FAKE_READER_ASSERTION=${_case}"
      "${CMAKE_COMMAND}" -DPROBE=/stand-in/CallgrindWindowProbe "-DREADER=${READER}" -DCASE=runner
      "-DWORK_DIR=${WORK_DIR}/${_case}" -P "${DRIVER}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  message(STATUS "${_case}: exit status ${_rc}\n${_out}")
  if (_case STREQUAL "stopped_before_start")
    if (NOT _rc EQUAL 0)
      string(APPEND _failures " the driver failed where valgrind stopped before the probe"
             " started (exit status ${_rc});"
      )
    elseif (NOT _out MATCHES "-- SKIPPED: [^\n]*It printed:\nvalgrind: m_debuginfo/readelf\\.c:")
      string(APPEND _failures " the driver did not skip quoting valgrind's line;")
    endif ()
  elseif (_rc EQUAL 0)
    string(APPEND _failures " the driver accepted the ${_case} run;")
  elseif (NOT _out MATCHES "did[ \n]+not[ \n]+start[ \n]+under[ \n]+valgrind")
    string(APPEND _failures
           " the driver failed on the ${_case} run, but not because the tests did not start"
           " (exit status ${_rc});"
    )
  endif ()
endforeach ()

if (NOT _failures STREQUAL "")
  message(FATAL_ERROR "CallgrindWindowReaderAssertion:${_failures}")
endif ()
message(STATUS "the driver skipped the stop before the probe started and failed the other two")
