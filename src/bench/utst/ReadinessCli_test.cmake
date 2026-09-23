# ==============================================================================
# ReadinessCli_test.cmake - One command-line readiness case
#
# Run with: cmake -DTARGET=<ReadinessFixtureTarget> -DFIXTURES=<fixtures/readiness>
#                 -DCASE=<Case> -DWORK_DIR=<dir> -P ReadinessCli_test.cmake
#
# The target runs with PATH set to a private directory of fake tools, with
# every setting that could change a decision removed from its environment, and
# with FAKE_LOG naming the file the fakes record their invocations in.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/bin")
set(_log "${WORK_DIR}/fake.log")
set(_problems "")
set(_env "")

# Settings removed from the target's environment, so the case alone decides.
set(_unset
    BENCH_SUDO
    PERF_BPF
    PERF_BPF_SUDO
    PERF_BPF_SCRIPTS
    PERF_BPF_FMT
    PERF_BPF_OUT
    VERNIER_EXTERNAL_WRAP
    VERNIER_EXTERNAL_WRAP_DIR
    FAKE_SUDO_DENY
    LD_PRELOAD
    MALLOC_CONF
)

# Copy fake <template> into the private PATH as <tool>.
function (fake template tool)
  file(
    COPY "${FIXTURES}/${template}"
    DESTINATION "${WORK_DIR}/stage"
    FILE_PERMISSIONS
      OWNER_READ
      OWNER_WRITE
      OWNER_EXECUTE
      GROUP_READ
      GROUP_EXECUTE
      WORLD_READ
      WORLD_EXECUTE
  )
  file(RENAME "${WORK_DIR}/stage/${template}" "${WORK_DIR}/bin/${tool}")
endfunction ()

# Run the target with <args>; sets <prefix>_RC, <prefix>_OUT and <prefix>_ERR.
function (run prefix)
  set(_unset_args "")
  foreach (_name IN LISTS _unset)
    list(APPEND _unset_args "--unset=${_name}")
  endforeach ()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${_unset_args} "PATH=${WORK_DIR}/bin" "FAKE_LOG=${_log}"
            ${_env} "${TARGET}" ${ARGN}
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
  )
  message(
    STATUS "run ${prefix}: ${ARGN}\nexit status: ${_rc}\n--- stdout\n${_out}\n--- stderr\n${_err}"
  )
  set(${prefix}_RC
      "${_rc}"
      PARENT_SCOPE
  )
  set(${prefix}_OUT
      "${_out}"
      PARENT_SCOPE
  )
  set(${prefix}_ERR
      "${_err}"
      PARENT_SCOPE
  )
endfunction ()

# Record a problem unless <text> contains <needle>.
function (expect_has text needle what)
  string(FIND "${text}" "${needle}" _at)
  if (_at EQUAL -1)
    set(_problems
        "${_problems}\n  ${what}: missing '${needle}'"
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# Record a problem if <text> contains <needle>.
function (expect_not text needle what)
  string(FIND "${text}" "${needle}" _at)
  if (NOT _at EQUAL -1)
    set(_problems
        "${_problems}\n  ${what}: unexpected '${needle}'"
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# Record a problem unless <actual> equals <expected>.
function (expect_eq actual expected what)
  if (NOT "${actual}" STREQUAL "${expected}")
    set(_problems
        "${_problems}\n  ${what}: '${actual}' != '${expected}'"
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# Count the occurrences of <needle> in <text> into <out>.
function (count_of out text needle)
  string(LENGTH "${needle}" _len)
  set(_count 0)
  set(_rest "${text}")
  string(FIND "${_rest}" "${needle}" _at)
  while (NOT _at EQUAL -1)
    math(EXPR _count "${_count} + 1")
    math(EXPR _next "${_at} + ${_len}")
    string(SUBSTRING "${_rest}" ${_next} -1 _rest)
    string(FIND "${_rest}" "${needle}" _at)
  endwhile ()
  set(${out}
      ${_count}
      PARENT_SCOPE
  )
endfunction ()

# Fast runs: the cases measure nothing that matters here.
set(_quick --cycles 50 --repeats 2 --warmup 0)

# ------------------------------------------------------------------------------
# Cases
# ------------------------------------------------------------------------------

if (CASE STREQUAL "DoctorLabel")
  # The plain doctor says its rows check each backend's default mode.
  run(doctor --profile-check)
  expect_eq("${doctor_RC}" "0" "doctor exit status")
  expect_has(
    "${doctor_OUT}" "=== Profiler Backend Doctor (default mode of each backend) ==="
    "doctor header"
  )
  expect_has("${doctor_OUT}" "only cases built with the profiler guard create one" "doctor footer")
  expect_not("${doctor_OUT}" "Selected request" "plain doctor")

elseif (CASE STREQUAL "DoctorJsonKeys")
  # backendScope always; selected only with a request, whatever the flag order.
  run(plain --profile-check-json)
  expect_eq("${plain_RC}" "0" "json exit status")
  string(
    JSON
    _scope
    ERROR_VARIABLE
    _json_err
    GET
    "${plain_OUT}"
    backendScope
  )
  expect_eq("${_scope}" "default-mode" "backendScope")
  string(
    JSON
    _none
    ERROR_VARIABLE
    _selected_err
    GET
    "${plain_OUT}"
    selected
  )
  if (NOT _selected_err)
    string(APPEND _problems "\n  a selected row without a request")
  endif ()
  foreach (_order first last)
    if (_order STREQUAL "first")
      run(sel --profile-check-json --profile massif --profile-args x)
    else ()
      run(sel --profile massif --profile-args x --profile-check-json)
    endif ()
    string(
      JSON
      _name
      ERROR_VARIABLE
      _e1
      GET
      "${sel_OUT}"
      selected
      name
    )
    string(
      JSON
      _args
      ERROR_VARIABLE
      _e2
      GET
      "${sel_OUT}"
      selected
      profileArgs
    )
    string(
      JSON
      _status
      ERROR_VARIABLE
      _e3
      GET
      "${sel_OUT}"
      selected
      status
    )
    string(
      JSON
      _scope2
      ERROR_VARIABLE
      _e4
      GET
      "${sel_OUT}"
      backendScope
    )
    expect_eq("${_name}" "massif" "selected name (check flag ${_order})")
    expect_eq("${_args}" "x" "selected profileArgs (check flag ${_order})")
    expect_eq("${_status}" "fail" "selected status (check flag ${_order})")
    expect_eq("${_scope2}" "default-mode" "backendScope with a request")
  endforeach ()

elseif (CASE STREQUAL "SelectedRowMatchesRun")
  # One decision: the run prints the selected row's message and hint, once.
  run(doctor --profile massif --profile-check-json)
  string(
    JSON
    _message
    ERROR_VARIABLE
    _e1
    GET
    "${doctor_OUT}"
    selected
    message
  )
  string(
    JSON
    _hint
    ERROR_VARIABLE
    _e2
    GET
    "${doctor_OUT}"
    selected
    hint
  )
  expect_eq("${_message}" "valgrind binary not found on PATH" "selected message")
  run(text --profile massif --profile-check)
  expect_has(
    "${text_OUT}" "Selected request: --profile massif\n  [FAIL] massif     ${_message}"
    "text selected row"
  )
  run(run --profile massif ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  set(_notice "[FAIL] Profiler 'massif': ${_message}\n   ${_hint}\n   Falling back to no-op")
  expect_has("${run_ERR}" "${_notice}" "run notice")
  count_of(_times "${run_ERR}" "${_notice}")
  expect_eq("${_times}" "1" "notices for two guarded cases")
  expect_not("${run_ERR}" "requested but unavailable" "run notice")

else ()
  message(FATAL_ERROR "unknown CASE '${CASE}'")
endif ()

file(REMOVE_RECURSE "${WORK_DIR}")
if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "ReadinessCli ${CASE}:${_problems}")
endif ()
