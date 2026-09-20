# ==============================================================================
# AbiConsumerDriver_test.cmake - Checks that AbiConsumer_test.cmake refuses a
# run whose output carries a sanitizer report
#
# Run with: cmake -DDRIVER=<AbiConsumer_test.cmake> -DWORK_DIR=<dir>
#                 -P AbiConsumerDriver_test.cmake
#
# Needs no sanitizer: a stand-in consumer (a shell script written here) prints
# what a recovering UBSan prints, then the success marker, and exits 0. The
# driver has to fail it for that reason; a stand-in without the report has to
# pass. This covers the driver's reading of the output only; that
# halt_on_error reaches a real sanitizer runtime is outside a plain build.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

function (run_driver _name _report _expect_pass)
  # Written without the execute bit, then copied with it.
  file(WRITE "${WORK_DIR}/plain/${_name}.sh"
       "#!/bin/sh\n${_report}echo 'consumer: constructed profiler stand-in' >&2\nexit 0\n"
  )
  file(
    COPY "${WORK_DIR}/plain/${_name}.sh"
    DESTINATION "${WORK_DIR}"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
  )
  set(_consumer "${WORK_DIR}/${_name}.sh")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -DCONSUMER=${_consumer} -DEXPECT=match
            -DWORK_DIR=${WORK_DIR}/${_name}_work -P "${DRIVER}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  if (_expect_pass AND NOT _rc EQUAL 0)
    message(SEND_ERROR "${_name}: the driver failed a run without a report:\n${_out}")
  elseif (NOT _expect_pass AND (_rc EQUAL 0 OR NOT _out MATCHES "sanitizer report in the output"))
    message(
      SEND_ERROR "${_name}: the driver did not fail the run for its report (rc=${_rc}):\n${_out}"
    )
  else ()
    message(STATUS "PASS ${_name}")
  endif ()
endfunction ()

run_driver(clean "" TRUE)
run_driver(
  recovering_ubsan
  "echo 'probe.cpp:3:52: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented' >&2\n"
  FALSE
)
run_driver(
  asan_summary "echo 'SUMMARY: AddressSanitizer: heap-use-after-free probe.cpp:3' >&2\n" FALSE
)

file(REMOVE_RECURSE "${WORK_DIR}")
