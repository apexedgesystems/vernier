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
# Prints "SKIPPED: <reason>", the tests' skip expression, in three cases
# only, each with valgrind's own words where it has them: valgrind is not
# installed; valgrind gave up reading debug information before the probe ran
# (a valgrind older than the compiler does); or valgrind ran the probe without
# reading its symbols, so the profile names none of the probe's functions and
# valgrind warned about the probe's debug information. Anything else that
# keeps the probe's tests from starting under valgrind fails with the run's
# output: a failed launch, a signal, an early exit, no output. So does a
# profile that names none of the probe's functions without that warning, and
# a profile of zero instructions.
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

# GoogleTest's banner says the probe reached its tests under valgrind.
# Without it, the one reason that skips is valgrind's own: its debug
# information reader gave up on a file before the program ran. Any other way
# of not getting there is the probe's or the wrap's failure.
string(FIND "${_out}" "[==========]" _started)
if (_started EQUAL -1)
  # The skip quotes valgrind's two lines as it printed them, so what it rests on
  # is valgrind's text, not this script's.
  string(
    REGEX
      MATCH
      "[^\n]*Valgrind: debuginfo reader: [^\n]*\n[^\n]*Valgrind: I can't recover\\.  Giving up\\.[^\n]*"
      _gave_up
      "${_out}"
  )
  if (NOT _gave_up STREQUAL "")
    execute_process(
      COMMAND "${_valgrind}" --version
      OUTPUT_VARIABLE _version
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "SKIPPED: ${_version} gave up reading debug information before the "
                   "probe ran. It printed:\n${_gave_up}"
    )
    return()
  endif ()
  string(REGEX MATCH "Process terminating with default action of signal [0-9]+ \\([A-Z]+\\)"
               _signal "${_out}"
  )
  if (NOT _signal STREQUAL "")
    set(_how "${_signal}")
  elseif (_out STREQUAL "")
    set(_how "no output at all, exit status ${_rc}")
  else ()
    set(_how "exit status ${_rc}")
  endif ()
  message(FATAL_ERROR "CallgrindWindowProbe (${CASE}): the probe's tests did not start "
                      "under valgrind (${_how}); the run's output is printed above"
  )
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

# Whether the profile names any function of the probe: its three phases and
# the test body GoogleTest generates. Where valgrind could not read the
# probe's symbols, the probe's code is recorded under addresses, the phase
# checks below cannot tell a right window from a wrong one, and valgrind says
# so in its own output: skip with that reason. Without the warning, a profile
# that names none of them is a window that missed the probe, and fails.
set(_named "")
foreach (_function workBeforeWindow workInsideWindow workAfterWindow CallgrindWindow_Phases_Test)
  string(FIND "${_data}" "${_function}" _at)
  if (NOT _at EQUAL -1)
    list(APPEND _named ${_function})
  endif ()
endforeach ()
if (_named STREQUAL ""
    AND _problems STREQUAL ""
    AND _totals GREATER 0
)
  get_filename_component(_probe_name "${PROBE}" NAME)
  string(REPLACE "." "\\." _probe_name "${_probe_name}")
  string(REGEX MATCH "When reading debug info from [^\n]*/${_probe_name}:\n[^\n]*-- ([^\n]*)"
               _warning "${_out}"
  )
  if (NOT _warning STREQUAL "")
    set(_reason "${CMAKE_MATCH_1}")
    execute_process(
      COMMAND "${_valgrind}" --version
      OUTPUT_VARIABLE _version
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "SKIPPED: ${_version} could not read the probe's debug information "
                   "(${_reason}), so the profile names none of its functions"
    )
    return()
  endif ()
  string(APPEND _problems " the profile names none of the probe's functions, and"
         " valgrind reported no trouble reading the probe's debug information;"
  )
endif ()

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
