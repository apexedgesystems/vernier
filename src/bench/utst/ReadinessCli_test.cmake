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

# Ask the doctor about the request <args> states; sets <prefix>_STATUS,
# <prefix>_MESSAGE and <prefix>_HINT from its JSON's selected row.
function (selected_row prefix)
  run(_row ${ARGN} --profile-check-json)
  string(
    JSON
    _status
    ERROR_VARIABLE
    _e1
    GET
    "${_row_OUT}"
    selected
    status
  )
  string(
    JSON
    _message
    ERROR_VARIABLE
    _e2
    GET
    "${_row_OUT}"
    selected
    message
  )
  string(
    JSON
    _hint
    ERROR_VARIABLE
    _e3
    GET
    "${_row_OUT}"
    selected
    hint
  )
  set(${prefix}_STATUS
      "${_status}"
      PARENT_SCOPE
  )
  set(${prefix}_MESSAGE
      "${_message}"
      PARENT_SCOPE
  )
  set(${prefix}_HINT
      "${_hint}"
      PARENT_SCOPE
  )
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

# The bpftrace fakes, and a scripts directory holding one sched script.
macro (bpf_fakes)
  fake(fake_bpftrace.sh bpftrace)
  fake(fake_sudo.sh sudo)
  fake(fake_kill.sh kill)
  file(WRITE "${WORK_DIR}/scripts/probe_script.bt"
       "tracepoint:sched:sched_switch /pid == {{PID}}/ { @c = count(); }\n"
  )
  list(APPEND _env "PERF_BPF_SCRIPTS=${WORK_DIR}/scripts")
endmacro ()

# The fake log's text ("" when nothing was logged).
function (read_log out)
  set(_text "")
  if (EXISTS "${_log}")
    file(READ "${_log}" _text)
  endif ()
  set(${out}
      "${_text}"
      PARENT_SCOPE
  )
endfunction ()

# Pids of the fake tracers (every "pid=" the log holds) into <out>.
function (tracer_pids out)
  read_log(_text)
  string(REGEX MATCHALL "pid=[0-9]+" _found "${_text}")
  string(REPLACE "pid=" "" _found "${_found}")
  set(${out}
      "${_found}"
      PARENT_SCOPE
  )
endfunction ()

# Record a problem for every fake tracer still running, and for every pid a
# fake kill was asked to signal that is not one of them.
function (expect_owned_and_gone what)
  tracer_pids(_pids)
  foreach (_pid IN LISTS _pids)
    execute_process(
      COMMAND kill -0 ${_pid}
      RESULT_VARIABLE _alive
      OUTPUT_QUIET ERROR_QUIET
    )
    if (_alive EQUAL 0)
      execute_process(COMMAND kill -9 ${_pid} OUTPUT_QUIET ERROR_QUIET)
      set(_problems
          "${_problems}\n  ${what}: tracer ${_pid} survived the run"
          PARENT_SCOPE
      )
    endif ()
  endforeach ()
  read_log(_text)
  string(REGEX MATCHALL "kill -[0-9]+ [0-9]+" _kills "${_text}")
  foreach (_kill IN LISTS _kills)
    string(REGEX REPLACE "kill -[0-9]+ " "" _target "${_kill}")
    if (NOT _target IN_LIST _pids)
      set(_problems
          "${_problems}\n  ${what}: '${_kill}' signalled a pid it does not own"
          PARENT_SCOPE
      )
    endif ()
  endforeach ()
endfunction ()

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

elseif (CASE STREQUAL "BpfNoOptInNeverCallsSudo")
  # No opt-in: the attach runs as the current user, a denial says how to get
  # access, and sudo is never asked, even though it is on PATH.
  bpf_fakes()
  list(APPEND _env FAKE_BPFTRACE_MODE=eperm)
  run(doctor --profile bpftrace --bpf probe_script --profile-check-json)
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
  expect_eq(
    "${_message}"
    "denied: script 'probe_script' could not attach as the current user: ERROR: bpftrace currently only supports running as the root user."
    "selected message"
  )
  expect_has(
    "${_hint}" "Set BENCH_SUDO=1 with a scoped sudoers grant for ${WORK_DIR}/bin/bpftrace"
    "selected hint"
  )
  run(run --profile bpftrace --bpf probe_script ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  expect_has("${run_ERR}" "[FAIL] Profiler 'bpftrace': ${_message}\n   ${_hint}" "run notice")
  read_log(_text)
  expect_not("${_text}" "sudo " "fake log")

elseif (CASE STREQUAL "BpfConflictNamesWinner")
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=0 PERF_BPF_SUDO=1)
  run(doctor --profile bpftrace --bpf probe_script --profile-check-json)
  string(
    JSON
    _status
    ERROR_VARIABLE
    _e1
    GET
    "${doctor_OUT}"
    selected
    status
  )
  string(
    JSON
    _message
    ERROR_VARIABLE
    _e2
    GET
    "${doctor_OUT}"
    selected
    message
  )
  expect_eq("${_status}" "warn" "selected status")
  expect_has(
    "${_message}"
    "BENCH_SUDO=0 and PERF_BPF_SUDO=1 disagree; BENCH_SUDO wins (the probe tool runs as the current user). PERF_BPF_SUDO is deprecated: remove it."
    "selected message"
  )
  read_log(_text)
  expect_not("${_text}" "sudo " "fake log")

elseif (CASE STREQUAL "BpfInvalidValueLaunchesNothing")
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=maybe)
  run(doctor --profile bpftrace --bpf probe_script --profile-check-json)
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
  expect_eq("${_message}" "configuration: BENCH_SUDO='maybe' is not a boolean" "selected message")
  run(run --profile bpftrace --bpf probe_script ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  expect_has("${run_ERR}" "[FAIL] Profiler 'bpftrace': ${_message}" "run notice")
  read_log(_text)
  expect_eq("${_text}" "" "fake log (nothing may run)")

elseif (CASE STREQUAL "BpfProbeRefusedRunDecides")
  # A grant that refuses every -q command. The check's probe copy is not the
  # run's command, so its refusal leaves the run unverified: the doctor warns
  # and names the command sudo refused, and each launch then reports the
  # refusal of its own command; no tracer runs.
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1 FAKE_SUDO_DENY=-q)
  selected_row(row --profile bpftrace --bpf probe_script)
  expect_eq("${row_STATUS}" "warn" "selected status")
  expect_has(
    "${row_MESSAGE}" "unverified: sudo -n refused the probe command ${WORK_DIR}/bin/bpftrace -q "
    "selected message"
  )
  expect_has(
    "${row_MESSAGE}"
    "/probe0.bt: sudo: a password is required; the run executes ${WORK_DIR}/bin/bpftrace -q <capture folder>/probe_script.tmp.bt instead, which only the run can try"
    "selected message"
  )
  expect_not("${row_MESSAGE}" "${WORK_DIR}/scripts/" "selected message (a command not attempted)")
  expect_has("${row_HINT}" "The grant must allow ${WORK_DIR}/bin/bpftrace" "selected hint")
  run(run --profile bpftrace --bpf probe_script ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  count_of(_notices "${run_ERR}"
           "[WARN] Profiler 'bpftrace': unverified: sudo -n refused the probe"
  )
  expect_eq("${_notices}" "1" "run notices")
  foreach (_case First Second)
    expect_has(
      "${run_ERR}"
      "[bpftrace] the tracer exited during its start grace: denied: sudo -n refused ${WORK_DIR}/bin/bpftrace -q ./ReadinessFixture.${_case}.bpf/probe_script.tmp.bt: sudo: a password is required"
      "run report (${_case})"
    )
  endforeach ()
  read_log(_text)
  expect_not("${_text}" "\nbpftrace -q " "fake log (no tracer ran)")

elseif (CASE STREQUAL "BpfRunAllowedProbeRefused")
  # A grant for the run's command that refuses the check's probe copy: the
  # doctor warns, and the run starts both cases' tracers through sudo and
  # stops them there.
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1 FAKE_SUDO_DENY=/probe0.bt)
  selected_row(row --profile bpftrace --bpf probe_script)
  expect_eq("${row_STATUS}" "warn" "selected status")
  run(run --profile bpftrace --bpf probe_script ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  expect_not("${run_ERR}" "[bpftrace]" "run reports")
  read_log(_text)
  count_of(_launches "${_text}" "\nbpftrace -q ./ReadinessFixture.")
  expect_eq("${_launches}" "2" "tracers the run started (one per guarded case)")
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "BpfRunStopsThroughRoute")
  # The run launches the checked tools through sudo, stops each tracer with
  # SIGINT through sudo -n kill, signals only its own tracers, leaves none.
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1)
  run(run --profile bpftrace --bpf probe_script ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  expect_not("${run_ERR}" "Profiler 'bpftrace'" "run notice")
  read_log(_text)
  count_of(_launches "${_text}" "sudo -n -- ${WORK_DIR}/bin/bpftrace -q ./ReadinessFixture.")
  expect_eq("${_launches}" "2" "launches through sudo (one per guarded case)")
  count_of(_interrupts "${_text}" "sudo -n -- ${WORK_DIR}/bin/kill -2 ")
  expect_eq("${_interrupts}" "3" "SIGINT through sudo (the probe and two launches)")
  expect_not("${_text}" "kill -15" "fake log")
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "BpfRunReportsRefusedStop")
  # The run's tracer ignores SIGINT and SIGTERM is refused: both are reported,
  # SIGKILL ends it, and no tracer survives.
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1 FAKE_BPFTRACE_MODE=ignore-int-run "FAKE_SUDO_DENY=kill -15")
  run(run --profile bpftrace --bpf probe_script ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  expect_has(
    "${run_ERR}" "[bpftrace] script 'probe_script': could not deliver SIGTERM to tracer "
    "run report"
  )
  expect_has("${run_ERR}" "-n -- ${WORK_DIR}/bin/kill -15 " "run report names the command")
  expect_has("${run_ERR}" ": sudo: a password is required" "run report names the refusal")
  expect_has(
    "${run_ERR}"
    "[bpftrace] script 'probe_script': the tracer ignored SIGINT and SIGTERM and was killed"
    "run report"
  )
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "OffcpuCurrentUserRun")
  bpf_fakes()
  run(run --profile offcpu ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  read_log(_text)
  expect_not("${_text}" "sudo " "fake log")
  count_of(_launches "${_text}" "bpftrace -e ")
  expect_eq("${_launches}" "3" "offcpu attaches (the probe and two launches)")
  count_of(_written "${run_ERR}" "[offcpu] stacks written to ")
  expect_eq("${_written}" "2" "stacks written, once per case")
  if (NOT EXISTS "${WORK_DIR}/ReadinessFixture.First.offcpu/offcpu.err.txt")
    string(APPEND _problems "\n  offcpu.err.txt was not written")
  endif ()
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "OffcpuRunAllowedProbeRefused")
  # A grant that refuses any program carrying a self-exit interval refuses
  # the check's probe and allows the run's own command: the doctor warns
  # (unverified), and the run starts, stops and writes its stacks.
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1 FAKE_SUDO_DENY=interval:)
  selected_row(row --profile offcpu)
  expect_eq("${row_STATUS}" "warn" "selected status")
  expect_has(
    "${row_MESSAGE}"
    "unverified: sudo -n refused the probe command ${WORK_DIR}/bin/bpftrace -e <the off-CPU script with a 5 s self-exit> "
    "selected message"
  )
  run(run --profile offcpu ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  count_of(_written "${run_ERR}" "[offcpu] stacks written to ")
  expect_eq("${_written}" "2" "stacks written, once per case")
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "OffcpuRefusedStopLeavesNoTracer")
  # Every stop signal refused and the target's exit never seen: the probe's
  # self-exit ends it, and neither the doctor nor the run leaves a tracer.
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1 "FAKE_SUDO_DENY=kill -" FAKE_BPFTRACE_MODE=ignore-target)
  selected_row(row --profile offcpu)
  expect_eq("${row_STATUS}" "fail" "selected status")
  expect_has(
    "${row_MESSAGE}" "denied: cleanup: sudo -n refused ${WORK_DIR}/bin/kill -2 " "selected message"
  )
  expect_not("${row_MESSAGE}" "still runs" "selected message")
  expect_owned_and_gone("doctor")
  run(run --profile offcpu ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  expect_has("${run_ERR}" "[FAIL] Profiler 'offcpu': denied: cleanup: " "run notice")
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "OffcpuNoStacksClaimWhenKilled")
  bpf_fakes()
  list(APPEND _env BENCH_SUDO=1 FAKE_BPFTRACE_MODE=ignore-int-run "FAKE_SUDO_DENY=kill -15")
  run(run --profile offcpu ${_quick})
  expect_has(
    "${run_ERR}"
    "[offcpu] the off-CPU script: the tracer ignored SIGINT and SIGTERM and was killed"
    "run report"
  )
  expect_not("${run_ERR}" "stacks written" "run report")
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "PerfLaunchesTheResolvedPath")
  # The run launches the perf the check ran, by its absolute path.
  fake(fake_perf.sh perf)
  run(run --profile perf ${_quick})
  expect_eq("${run_RC}" "0" "run exit status")
  read_log(_text)
  count_of(
    _launches
    "${_text}"
    "perf ${WORK_DIR}/bin/perf stat -e cpu-cycles,instructions,branches,branch-misses,cache-misses -p "
  )
  expect_eq("${_launches}" "2" "launches of the resolved perf (one per guarded case)")
  expect_not("${_text}" "perf perf " "fake log (perf run by bare name)")
  expect_owned_and_gone("run")

elseif (CASE STREQUAL "PerfBrokenNeverLaunched")
  fake(fake_perf.sh perf)
  list(APPEND _env FAKE_PERF_MODE=broken)
  run(doctor --profile perf --profile-check-json)
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
  expect_has(
    "${_message}"
    "unusable: ${WORK_DIR}/bin/perf --version: exit status 2: WARNING: perf not found for kernel"
    "selected message"
  )
  run(run --profile perf ${_quick})
  expect_has("${run_ERR}" "[FAIL] Profiler 'perf': ${_message}\n   ${_hint}" "run notice")
  read_log(_text)
  expect_not("${_text}" " stat " "fake log (a broken perf must not run)")

elseif (CASE STREQUAL "PerfDeniedMatchesDoctor")
  fake(fake_perf.sh perf)
  list(APPEND _env FAKE_PERF_MODE=denied)
  run(doctor --profile perf --profile-check-json)
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
  expect_has(
    "${_message}" "denied: perf stat cannot open the counters as this user" "selected message"
  )
  run(run --profile perf ${_quick})
  expect_has("${run_ERR}" "[FAIL] Profiler 'perf': ${_message}\n   ${_hint}" "run notice")

elseif (CASE MATCHES "^Gperf")
  # gperf cases need gperftools compiled into libbench.
  run(inventory --profile-check-json)
  set(_gperf_status "")
  if (inventory_OUT MATCHES "\"name\": \"gperf\", \"status\": \"([a-z]+)\"")
    set(_gperf_status "${CMAKE_MATCH_1}")
  endif ()
  if (NOT _gperf_status STREQUAL "ok")
    message(STATUS "READINESS_CLI_SKIPPED: gperf is not usable in this build (${_gperf_status})")
    file(REMOVE_RECURSE "${WORK_DIR}")
    return()
  endif ()
  set(_prof "${WORK_DIR}/ReadinessFixture.First.gperf/cpu.prof")

  if (CASE STREQUAL "GperfAnalyzerFoundIsRun")
    # Only pprof exists: the analysis runs it, by its path, and prints its lines.
    fake(fake_pprof.sh pprof)
    run(run --profile gperf --profile-analyze ${_quick})
    expect_eq("${run_RC}" "0" "run exit status")
    read_log(_text)
    expect_has("${_text}" "pprof ${WORK_DIR}/bin/pprof --text --cum --lines " "fake log")
    expect_not("${_text}" "google-pprof" "fake log")
    expect_has("${run_OUT}" "Analyzer: ${WORK_DIR}/bin/pprof" "analysis header")
    expect_has("${run_OUT}" "fake pprof line 20\n" "cumulative view")
    expect_not("${run_OUT}" "fake pprof line 21\n\n--- Self time" "cumulative view is cut at 20")
    expect_not("${run_OUT}" "using local file" "analysis text (the analyzer's stderr)")
    expect_not("${run_ERR}" "Profiler 'gperf'" "run notice")

  elseif (CASE STREQUAL "GperfAnalyzerMissingIsAnalysisError")
    # The promised analysis without an analyzer is an Error at the analysis
    # stage in the doctor's row, its JSON and the run, which still collects;
    # the same request without the promise is Ok.
    selected_row(on --profile gperf --profile-analyze)
    selected_row(off --profile gperf)
    expect_eq("${on_STATUS}" "fail" "selected status with --profile-analyze")
    expect_has(
      "${on_MESSAGE}" "analysis: missing: --profile-analyze needs google-pprof or pprof"
      "selected message"
    )
    expect_eq("${off_STATUS}" "ok" "selected status without --profile-analyze")
    run(text --profile gperf --profile-analyze --profile-check)
    expect_has(
      "${text_OUT}" "--profile-analyze\n  [FAIL] gperf      analysis: missing: "
      "text selected row"
    )
    run(run --profile gperf --profile-analyze ${_quick})
    expect_eq("${run_RC}" "0" "run exit status")
    set(_notice
        "[FAIL] Profiler 'gperf': ${on_MESSAGE}\n   ${on_HINT}\n   Collection proceeds; the analysis is skipped and the raw capture is kept."
    )
    expect_has("${run_ERR}" "${_notice}" "run notice")
    count_of(_times "${run_ERR}" "${_notice}")
    expect_eq("${_times}" "1" "notices for two guarded cases")
    expect_has(
      "${run_ERR}"
      "[gperf] analysis skipped: no google-pprof or pprof on PATH; raw profile kept at "
      "analysis note"
    )
    if (NOT EXISTS "${_prof}")
      string(APPEND _problems "\n  cpu.prof was not written")
    endif ()
    run(plain --profile gperf ${_quick})
    expect_not("${plain_ERR}" "Profiler 'gperf'" "run notice without --profile-analyze")

  elseif (CASE STREQUAL "GperfAnalyzerBrokenIsAnalysisError")
    # An analyzer that does not run is the same Error, found before the run;
    # the analysis-off twin, with the same broken analyzer on PATH, is Ok.
    fake(fake_pprof.sh google-pprof)
    list(APPEND _env FAKE_PPROF_MODE=fail)
    selected_row(on --profile gperf --profile-analyze)
    selected_row(off --profile gperf)
    expect_eq("${on_STATUS}" "fail" "selected status with --profile-analyze")
    expect_has(
      "${on_MESSAGE}"
      "analysis: unusable: --profile-analyze would run ${WORK_DIR}/bin/google-pprof, which does not run: --help exit status 1: fake pprof: cannot read profile"
      "selected message"
    )
    expect_eq("${off_STATUS}" "ok" "selected status without --profile-analyze")
    run(run --profile gperf --profile-analyze ${_quick})
    expect_eq("${run_RC}" "0" "run exit status")
    expect_has(
      "${run_ERR}"
      "[FAIL] Profiler 'gperf': ${on_MESSAGE}\n   ${on_HINT}\n   Collection proceeds; the analysis is skipped and the raw capture is kept."
      "run notice"
    )
    expect_has(
      "${run_ERR}"
      "[gperf] analysis skipped: ${WORK_DIR}/bin/google-pprof does not run; raw profile kept at "
      "analysis note"
    )
    read_log(_text)
    expect_not("${_text}" " --text " "fake log (a broken analyzer is not run on the profile)")
    if (NOT EXISTS "${_prof}")
      string(APPEND _problems "\n  cpu.prof was not written")
    endif ()

  elseif (CASE STREQUAL "GperfAnalyzerFailsOnTheProfile")
    # An analyzer that runs but fails on this profile is reported when it
    # fails, and the raw capture is kept.
    fake(fake_pprof.sh google-pprof)
    list(APPEND _env FAKE_PPROF_MODE=fail-on-profile)
    selected_row(on --profile gperf --profile-analyze)
    expect_eq("${on_STATUS}" "ok" "selected status (the analyzer answers --help)")
    run(run --profile gperf --profile-analyze ${_quick})
    expect_eq("${run_RC}" "0" "run exit status")
    expect_has(
      "${run_ERR}"
      "[gperf] ${WORK_DIR}/bin/google-pprof failed: exit status 1: fake pprof: cannot read profile; raw profile kept at "
      "analysis failure"
    )
    if (NOT EXISTS "${_prof}")
      string(APPEND _problems "\n  cpu.prof was not kept")
    endif ()

  elseif (CASE STREQUAL "GperfWithoutAnalyzeNeedsNone")
    # The same environment as the missing-analyzer case, without the promise.
    run(doctor --profile gperf --profile-check-json)
    string(
      JSON
      _status
      ERROR_VARIABLE
      _e1
      GET
      "${doctor_OUT}"
      selected
      status
    )
    expect_eq("${_status}" "ok" "selected status")
    run(run --profile gperf ${_quick})
    expect_not("${run_ERR}" "Profiler 'gperf'" "run notice")
    if (NOT EXISTS "${_prof}")
      string(APPEND _problems "\n  cpu.prof was not written")
    endif ()

  elseif (CASE STREQUAL "GperfHeapWithoutSupport")
    run(doctor --profile gperf --profile-args heap --profile-check-json)
    string(
      JSON
      _status
      ERROR_VARIABLE
      _e1
      GET
      "${doctor_OUT}"
      selected
      status
    )
    string(
      JSON
      _message
      ERROR_VARIABLE
      _e2
      GET
      "${doctor_OUT}"
      selected
      message
    )
    if (HEAP_BUILT)
      expect_eq("${_status}" "ok" "selected status (heap compiled in)")
    else ()
      expect_eq("${_status}" "fail" "selected status")
      expect_has("${_message}" "unsupported: heap profiling is not compiled in" "selected message")
      run(run --profile gperf --profile-args heap ${_quick})
      expect_has("${run_ERR}" "[FAIL] Profiler 'gperf': ${_message}" "run notice")
      count_of(_times "${run_ERR}" "-DVERNIER_LINK_TCMALLOC=ON")
      expect_eq("${_times}" "1" "the remedy, once for two guarded cases")
    endif ()
  endif ()

else ()
  message(FATAL_ERROR "unknown CASE '${CASE}'")
endif ()

read_log(_final_log)
file(REMOVE_RECURSE "${WORK_DIR}")
if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "ReadinessCli ${CASE}:${_problems}\n--- fake log\n${_final_log}")
endif ()
