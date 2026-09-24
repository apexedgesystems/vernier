# ==============================================================================
# CallgrindWindowProbe_test.cmake - Runs CallgrindWindowProbe under callgrind
# and checks which of its phases the profile holds
#
# Run with: cmake -DPROBE=<exe> -DCASE=hint|runner -DWORK_DIR=<dir>
#                 -P CallgrindWindowProbe_test.cmake
#
# hint:   run the probe without valgrind, then under the valgrind command the
#         callgrind backend printed, and expect the profile to hold the
#         measured window and nothing that ran before or after it.
# runner: run the probe the way `bench run --profile callgrind` does: the whole
#         process under callgrind, with VERNIER_EXTERNAL_WRAP and
#         VERNIER_EXTERNAL_WRAP_DIR set as tools/rust/src/bench/runner.rs sets
#         them. Expect all three phases: the backend must leave a recording of
#         the whole process alone.
#
# Prints "SKIPPED: <reason>", the tests' skip expression, when valgrind is not
# installed or cannot start the probe (a valgrind older than the compiler's
# debug information gives up before the program runs).
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

set(_probe_args
    --profile
    callgrind
    --gtest_filter=CallgrindWindow.Phases
    --cycles
    2
    --repeats
    1
)

find_program(_valgrind valgrind)
if (NOT _valgrind)
  message(STATUS "SKIPPED: valgrind is not installed")
  return()
endif ()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(_clean_env --unset=VERNIER_EXTERNAL_WRAP --unset=VERNIER_EXTERNAL_WRAP_DIR)

if (CASE STREQUAL "hint")
  # The plain run prints the command, continued over lines ending in '\':
  #   [callgrind]   valgrind --tool=callgrind ... \
  #   [callgrind]     --callgrind-out-file=<file> \
  #   [callgrind]     <this-binary> --profile callgrind [...]
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${_clean_env} "${PROBE}" ${_probe_args}
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE _plain
    ERROR_VARIABLE _plain
  )
  string(FIND "${_plain}" "valgrind --tool=callgrind" _from)
  string(FIND "${_plain}" "<this-binary>" _to)
  if (_from EQUAL -1 OR _to LESS _from)
    message(FATAL_ERROR "the backend printed no valgrind command:\n${_plain}")
  endif ()
  math(EXPR _length "${_to} - ${_from}")
  string(SUBSTRING "${_plain}" ${_from} ${_length} _hint)
  string(REPLACE "[callgrind]" " " _hint "${_hint}")
  string(REPLACE "\\" " " _hint "${_hint}")
  string(REPLACE "\n" " " _hint "${_hint}")
  separate_arguments(_wrap UNIX_COMMAND "${_hint}")
  set(_env ${_clean_env})
  set(_present workInsideWindow)
  set(_absent workBeforeWindow workAfterWindow)
elseif (CASE STREQUAL "runner")
  set(_wrap_dir "${WORK_DIR}/wrap")
  file(MAKE_DIRECTORY "${_wrap_dir}")
  set(_wrap valgrind --tool=callgrind "--callgrind-out-file=${_wrap_dir}/callgrind.out")
  set(_env VERNIER_EXTERNAL_WRAP=callgrind "VERNIER_EXTERNAL_WRAP_DIR=${_wrap_dir}")
  set(_present workBeforeWindow workInsideWindow workAfterWindow)
  set(_absent "")
else ()
  message(FATAL_ERROR "CASE must be hint or runner")
endif ()

set(_profile "")
foreach (_word IN LISTS _wrap)
  if (_word MATCHES "^--callgrind-out-file=(.+)$")
    set(_profile "${CMAKE_MATCH_1}")
  endif ()
endforeach ()
if (_profile STREQUAL "")
  message(FATAL_ERROR "the wrap command names no --callgrind-out-file: ${_wrap}")
endif ()
if (NOT IS_ABSOLUTE "${_profile}")
  set(_profile "${WORK_DIR}/${_profile}")
endif ()

message(STATUS "running: ${_env} ${_wrap} ${PROBE} ${_probe_args}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env ${_env} ${_wrap} "${PROBE}" ${_probe_args}
  WORKING_DIRECTORY "${WORK_DIR}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
message(STATUS "exit status: ${_rc}\n${_out}")

string(FIND "${_out}" "[==========]" _started)
if (_started EQUAL -1)
  message(STATUS "SKIPPED: valgrind could not start the probe")
  return()
endif ()

set(_problems "")
if (NOT _rc EQUAL 0)
  string(APPEND _problems " expected exit 0;")
endif ()
if (NOT EXISTS "${_profile}")
  message(FATAL_ERROR "CallgrindWindowProbe (${CASE}): no profile at ${_profile};${_problems}")
endif ()

file(READ "${_profile}" _data)
string(REGEX MATCH "\ntotals: ([0-9]+)" _totals "${_data}")
set(_totals "${CMAKE_MATCH_1}")
foreach (_phase IN LISTS _present)
  string(FIND "${_data}" "${_phase}" _at)
  if (_at EQUAL -1)
    string(APPEND _problems " ${_phase} is missing from the profile;")
  endif ()
endforeach ()
foreach (_phase IN LISTS _absent)
  string(FIND "${_data}" "${_phase}" _at)
  if (NOT _at EQUAL -1)
    string(APPEND _problems " ${_phase} ran outside the measured window and is in the profile;")
  endif ()
endforeach ()

if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "CallgrindWindowProbe (${CASE}), ${_profile} totals ${_totals}:${_problems}")
endif ()
message(STATUS "profile ${_profile}: totals ${_totals}, ${_present} recorded")
