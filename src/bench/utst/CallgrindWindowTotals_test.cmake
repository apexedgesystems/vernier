# ==============================================================================
# CallgrindWindowTotals_test.cmake - Holds CallgrindWindowProbe_test.cmake to a
# valid instruction total
#
# Run with: cmake -DDRIVER=<CallgrindWindowProbe_test.cmake>
#                 -DFAKE=<CallgrindWindowTotals_fake_valgrind.sh>
#                 -DWORK_DIR=<dir> -P CallgrindWindowTotals_test.cmake
#
# Runs the window test's driver four times, in its runner case, against a
# stand-in valgrind that writes a profile holding all three of the probe's
# phases and a chosen "totals:" line: a positive number, zero, none, and a
# word. The driver must pass the first and fail the other three because the
# total is not valid, whatever names the profile holds. Each run's exit status
# is checked, and each failure's reason; nothing here needs valgrind.
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
foreach (_case valid zero missing malformed)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env "PATH=${WORK_DIR}/bin:$ENV{PATH}" "FAKE_CALLGRIND_TOTALS=${_case}"
      "${CMAKE_COMMAND}" -DPROBE=/stand-in/CallgrindWindowProbe -DCASE=runner
      "-DWORK_DIR=${WORK_DIR}/${_case}" -P "${DRIVER}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  message(STATUS "${_case}: exit status ${_rc}\n${_out}")
  if (_case STREQUAL "valid")
    if (NOT _rc EQUAL 0)
      string(APPEND _failures " the driver rejected a valid total (exit status ${_rc});")
    endif ()
  elseif (_rc EQUAL 0)
    string(APPEND _failures " the driver accepted the ${_case} total;")
  elseif (NOT _out MATCHES "no[ \n]+valid[ \n]+instruction[ \n]+total")
    string(APPEND _failures
           " the driver failed on the ${_case} total, but not for the total (exit status ${_rc});"
    )
  endif ()
endforeach ()

if (NOT _failures STREQUAL "")
  message(FATAL_ERROR "CallgrindWindowTotals:${_failures}")
endif ()
message(STATUS "the driver passed the valid total and failed zero, missing and malformed ones")
