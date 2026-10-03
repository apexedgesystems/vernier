# ==============================================================================
# MeasurementRows_test.cmake - The rows a benchmark's measurements publish
#
# Run with: cmake -DPROBE=<MeasurementRowsProbe> -DCASE=<Publication|CliAgreement>
#                 -DWORK_DIR=<dir> [-DBENCH_CLI=<bench>] -P MeasurementRows_test.cmake
#
# Publication: at --threads 4 the CSV holds one row per completed measurement,
# named as the registry names a test's rows, each with its own measurement's
# median and its case's config; the end-of-run table shows the same names,
# with and without --csv, and its footer counts the rows and the tests that
# published them (one row per test keeps the "<n> tests" line); --gtest_repeat
# writes each repetition's rows once; a measurement that throws publishes
# nothing; under --target-time each case's row carries the cycles of its own
# calibration.
#
# CliAgreement: `bench summary` and `bench compare` read the same names, and a
# baseline written one row per test (the last measurement under the case name)
# compares as a migration: the case names missing, the row names new. Skips
# when the bench CLI was not built.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(_problems "")

# Every row the probe's tests publish at --threads 4, in completion order, as
# <row name>|<threads>|<case name>|<test name>.
set(_expected
    "Rows.SeparateCases/64|1|Rows.SeparateCases/64|Rows.SeparateCases"
    "Rows.SeparateCases/256|1|Rows.SeparateCases/256|Rows.SeparateCases"
    "Rows.SeparateCases/1024|1|Rows.SeparateCases/1024|Rows.SeparateCases"
    "Rows.OneCaseThreeLabels/64|1|Rows.OneCaseThreeLabels|Rows.OneCaseThreeLabels"
    "Rows.OneCaseThreeLabels/256|1|Rows.OneCaseThreeLabels|Rows.OneCaseThreeLabels"
    "Rows.OneCaseThreeLabels/1024|1|Rows.OneCaseThreeLabels|Rows.OneCaseThreeLabels"
    "Rows.SingleThenContention/single|1|Rows.SingleThenContention|Rows.SingleThenContention"
    "Rows.SingleThenContention/contention|4|Rows.SingleThenContention|Rows.SingleThenContention"
    "Rows.RepeatedLabel/x#1|1|Rows.RepeatedLabel|Rows.RepeatedLabel"
    "Rows.RepeatedLabel/y|1|Rows.RepeatedLabel|Rows.RepeatedLabel"
    "Rows.RepeatedLabel/x#2|1|Rows.RepeatedLabel|Rows.RepeatedLabel"
    "Rows.EmptyLabels/#1|1|Rows.EmptyLabels|Rows.EmptyLabels"
    "Rows.EmptyLabels/#2|1|Rows.EmptyLabels|Rows.EmptyLabels"
    "Rows.LabelWithComma/a,b|1|Rows.LabelWithComma|Rows.LabelWithComma"
    "Rows.LabelWithComma/c|1|Rows.LabelWithComma|Rows.LabelWithComma"
    "Rows.OneMeasurement|1|Rows.OneMeasurement|Rows.OneMeasurement"
    "Rows.SecondMeasurementThrows|1|Rows.SecondMeasurementThrows|Rows.SecondMeasurementThrows"
)

set(_names "")
set(_threads "")
set(_cases "")
set(_tests "")
foreach (_entry IN LISTS _expected)
  string(REPLACE "|" ";" _fields "${_entry}")
  list(GET _fields 0 _field)
  list(APPEND _names "${_field}")
  list(GET _fields 1 _field)
  list(APPEND _threads "${_field}")
  list(GET _fields 2 _field)
  list(APPEND _cases "${_field}")
  list(GET _fields 3 _field)
  list(APPEND _tests "${_field}")
endforeach ()
list(LENGTH _names _expected_rows)
set(_measuring_tests ${_tests})
list(REMOVE_DUPLICATES _measuring_tests)
list(LENGTH _measuring_tests _expected_tests)

# Record a problem unless <actual> equals <expected>.
function (expect_eq actual expected what)
  if (NOT "${actual}" STREQUAL "${expected}")
    set(_problems
        "${_problems}\n  ${what}:\n    got      '${actual}'\n    expected '${expected}'"
        PARENT_SCOPE
    )
  endif ()
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

# Run <program> with <args>; sets <prefix>_RC, <prefix>_OUT and <prefix>_ERR and
# keeps both streams in WORK_DIR.
function (run prefix program)
  execute_process(
    COMMAND "${program}" ${ARGN}
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
  )
  file(WRITE "${WORK_DIR}/${prefix}.stdout.txt" "${_out}")
  file(WRITE "${WORK_DIR}/${prefix}.stderr.txt" "${_err}")
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

# Split one CSV line into its test cell (a quoted cell unquoted) and the cells
# after it; sets <prefix>_NAME and <prefix>_REST (a list).
function (split_row prefix line)
  if (line MATCHES "^\"([^\"]*)\",(.*)$")
    set(_name "${CMAKE_MATCH_1}")
    set(_rest "${CMAKE_MATCH_2}")
  elseif (line MATCHES "^([^,]*),(.*)$")
    set(_name "${CMAKE_MATCH_1}")
    set(_rest "${CMAKE_MATCH_2}")
  else ()
    set(_name "${line}")
    set(_rest "")
  endif ()
  string(REPLACE "," ";" _rest "${_rest}")
  set(${prefix}_NAME
      "${_name}"
      PARENT_SCOPE
  )
  set(${prefix}_REST
      "${_rest}"
      PARENT_SCOPE
  )
endfunction ()

# Read the CSV <file>: sets <prefix>_HEADER (column names), <prefix>_LINES (the
# data lines), <prefix>_NAMES (their test cells, in file order), <prefix>_COUNT
# and <prefix>_ROW_<i> (the cells after the test cell of data line i).
function (read_csv prefix file)
  set(_header "")
  set(_data "")
  set(_names "")
  set(_count 0)
  if (EXISTS "${file}")
    file(STRINGS "${file}" _lines)
    list(LENGTH _lines _all)
    if (_all GREATER 0)
      list(GET _lines 0 _header)
      string(REPLACE "," ";" _header "${_header}")
      list(SUBLIST _lines 1 -1 _data)
    endif ()
  endif ()
  foreach (_line IN LISTS _data)
    split_row(_row "${_line}")
    list(APPEND _names "${_row_NAME}")
    set(${prefix}_ROW_${_count}
        "${_row_REST}"
        PARENT_SCOPE
    )
    math(EXPR _count "${_count} + 1")
  endforeach ()
  set(${prefix}_HEADER
      "${_header}"
      PARENT_SCOPE
  )
  set(${prefix}_LINES
      "${_data}"
      PARENT_SCOPE
  )
  set(${prefix}_NAMES
      "${_names}"
      PARENT_SCOPE
  )
  set(${prefix}_COUNT
      "${_count}"
      PARENT_SCOPE
  )
endfunction ()

# The cell of data row <row> of the CSV read as <prefix>, under <column>.
function (cell out prefix row column)
  list(FIND ${prefix}_HEADER "${column}" _index)
  set(_value "<no column ${column}>")
  if (_index GREATER 0)
    math(EXPR _index "${_index} - 1")
    list(LENGTH ${prefix}_ROW_${row} _cells)
    if (_index LESS _cells)
      list(GET ${prefix}_ROW_${row} ${_index} _value)
    else ()
      set(_value "<short row>")
    endif ()
  endif ()
  set(${out}
      "${_value}"
      PARENT_SCOPE
  )
endfunction ()

# The probe's per-measurement lines in <text>: sets <prefix>_MEDIANS,
# <prefix>_CYCLES and <prefix>_BYTES, in the order the measurements completed.
function (probe_lines prefix text)
  string(REGEX MATCHALL "median=[^ \n]+ cycles=[0-9]+ msgBytes=[0-9]+" _lines "${text}")
  set(_medians "")
  set(_cycles "")
  set(_bytes "")
  foreach (_line IN LISTS _lines)
    string(REGEX MATCH "^median=([^ ]+) cycles=([0-9]+) msgBytes=([0-9]+)$" _match "${_line}")
    list(APPEND _medians "${CMAKE_MATCH_1}")
    list(APPEND _cycles "${CMAKE_MATCH_2}")
    list(APPEND _bytes "${CMAKE_MATCH_3}")
  endforeach ()
  set(${prefix}_MEDIANS
      "${_medians}"
      PARENT_SCOPE
  )
  set(${prefix}_CYCLES
      "${_cycles}"
      PARENT_SCOPE
  )
  set(${prefix}_BYTES
      "${_bytes}"
      PARENT_SCOPE
  )
endfunction ()

# The test names of the end-of-run table printed in <text>, in table order.
function (table_names out text)
  set(_names "")
  string(FIND "${text}" "Median (us)" _at)
  if (NOT _at EQUAL -1)
    string(SUBSTRING "${text}" ${_at} -1 _table)
    string(FIND "${_table}" " tests | " _end)
    if (NOT _end EQUAL -1)
      string(SUBSTRING "${_table}" 0 ${_end} _table)
    endif ()
    string(REGEX MATCHALL "\n[^ \n-][^\n]*" _lines "${_table}")
    foreach (_line IN LISTS _lines)
      if (_line MATCHES "^\n([^ ]+) +[-0-9.]+ +[-0-9.]+% ")
        list(APPEND _names "${CMAKE_MATCH_1}")
      endif ()
    endforeach ()
  endif ()
  set(${out}
      "${_names}"
      PARENT_SCOPE
  )
endfunction ()

# The end-of-run table printed in <text>: sets <prefix>_FOOTER (its last line)
# and <prefix>_EXPECT, the footer its rows call for: "<rows> rows from <tests>
# tests | ..." when rows outnumber the <tests> given, "<rows> tests | ..."
# when they do not, the stable and unstable counts being the rows the table
# marks OK and UNSTABLE.
function (table_footer prefix text tests)
  set(_footer "")
  set(_rows 0)
  set(_stable 0)
  set(_unstable 0)
  string(FIND "${text}" "Median (us)" _at)
  if (NOT _at EQUAL -1)
    string(SUBSTRING "${text}" ${_at} -1 _table)
    string(REGEX MATCHALL "  (OK|UNSTABLE)\n" _marks "${_table}")
    foreach (_mark IN LISTS _marks)
      math(EXPR _rows "${_rows} + 1")
      if (_mark MATCHES "UNSTABLE")
        math(EXPR _unstable "${_unstable} + 1")
      else ()
        math(EXPR _stable "${_stable} + 1")
      endif ()
    endforeach ()
    if (_table MATCHES "\n([0-9][^\n]* unstable)\n")
      set(_footer "${CMAKE_MATCH_1}")
    endif ()
  endif ()
  if (tests LESS _rows)
    set(_expect "${_rows} rows from ${tests} tests | ${_stable} stable | ${_unstable} unstable")
  else ()
    set(_expect "${_rows} tests | ${_stable} stable | ${_unstable} unstable")
  endif ()
  set(${prefix}_FOOTER
      "${_footer}"
      PARENT_SCOPE
  )
  set(${prefix}_EXPECT
      "${_expect}"
      PARENT_SCOPE
  )
endfunction ()

# The JSON array at <path...> in <json>: its string elements, or with
# MEMBER=<name> set before the call, that member of each element.
function (json_list out json)
  set(_items "")
  string(JSON _length ERROR_VARIABLE _error LENGTH "${json}" ${ARGN})
  if (_error)
    set(_items "<not JSON: ${_error}>")
  else ()
    set(_i 0)
    while (_i LESS _length)
      string(
        JSON
        _item
        GET
        "${json}"
        ${ARGN}
        ${_i}
        ${MEMBER}
      )
      list(APPEND _items "${_item}")
      math(EXPR _i "${_i} + 1")
    endwhile ()
  endif ()
  set(${out}
      "${_items}"
      PARENT_SCOPE
  )
endfunction ()

# ------------------------------------------------------------------------------
# Publication
# ------------------------------------------------------------------------------

if (CASE STREQUAL "Publication")
  set(_common --threads 4 --cycles 50 --repeats 3)

  # One run to a CSV: the rows, their values and the end-of-run table.
  run(csv "${PROBE}" ${_common} --csv "${WORK_DIR}/rows.csv")
  expect_eq("${csv_RC}" "0" "probe exit status (--csv)")
  read_csv(rows "${WORK_DIR}/rows.csv")
  expect_eq("${rows_NAMES}" "${_names}" "CSV row names")

  probe_lines(probe "${csv_OUT}")
  list(LENGTH probe_MEDIANS _measured)
  expect_eq("${_measured}" "${_expected_rows}" "measurements the probe completed")
  if (rows_COUNT EQUAL _expected_rows AND _measured EQUAL _expected_rows)
    set(_i 0)
    while (_i LESS _expected_rows)
      list(GET _names ${_i} _name)
      list(GET probe_MEDIANS ${_i} _median)
      list(GET probe_CYCLES ${_i} _cycles)
      list(GET probe_BYTES ${_i} _bytes)
      list(GET _threads ${_i} _thread_count)
      cell(_value rows ${_i} wallMedian)
      expect_eq("${_value}" "${_median}" "${_name}: wallMedian against its measurement")
      cell(_value rows ${_i} cycles)
      expect_eq("${_value}" "${_cycles}" "${_name}: cycles against its case")
      cell(_value rows ${_i} msgBytes)
      expect_eq("${_value}" "${_bytes}" "${_name}: msgBytes against its case")
      cell(_value rows ${_i} threads)
      expect_eq("${_value}" "${_thread_count}" "${_name}: threads")
      math(EXPR _i "${_i} + 1")
    endwhile ()
    # The separately named cases keep their own config.
    set(_i 0)
    foreach (_size 64 256 1024)
      cell(_value rows ${_i} msgBytes)
      expect_eq("${_value}" "${_size}" "Rows.SeparateCases/${_size}: msgBytes")
      cell(_value rows ${_i} cycles)
      expect_eq("${_value}" "${_size}" "Rows.SeparateCases/${_size}: cycles")
      math(EXPR _i "${_i} + 1")
    endforeach ()
  endif ()

  table_names(_table "${csv_OUT}")
  expect_eq("${_table}" "${rows_NAMES}" "end-of-run table names against the CSV")
  # The footer counts the rows and the tests that published them.
  table_footer(_footer "${csv_OUT}" ${_expected_tests})
  expect_eq("${_footer_FOOTER}" "${_footer_EXPECT}" "end-of-run table footer")

  # Two repetitions: each writes its own rows once.
  run(repeat "${PROBE}" ${_common} --gtest_repeat=2 --csv "${WORK_DIR}/repeat.csv")
  expect_eq("${repeat_RC}" "0" "probe exit status (--gtest_repeat=2)")
  read_csv(repeat "${WORK_DIR}/repeat.csv")
  expect_eq("${repeat_NAMES}" "${_names};${_names}" "CSV row names, two repetitions")
  math(EXPR _repeated_tests "2 * ${_expected_tests}")
  table_footer(_footer "${repeat_OUT}" ${_repeated_tests})
  expect_eq("${_footer_FOOTER}" "${_footer_EXPECT}" "end-of-run table footer, two repetitions")

  # Without --csv the table still names every row.
  run(plain "${PROBE}" ${_common})
  expect_eq("${plain_RC}" "0" "probe exit status (no --csv)")
  table_names(_table "${plain_OUT}")
  expect_eq("${_table}" "${_names}" "end-of-run table names without --csv")
  table_footer(_footer "${plain_OUT}" ${_expected_tests})
  expect_eq("${_footer_FOOTER}" "${_footer_EXPECT}" "end-of-run table footer without --csv")

  # Two tests of one row each: the footer is the one-row-per-test line.
  run(single "${PROBE}" ${_common} --gtest_filter=Rows.OneMeasurement:Rows.SecondMeasurementThrows)
  expect_eq("${single_RC}" "0" "probe exit status (one row per test)")
  table_footer(_footer "${single_OUT}" 2)
  expect_eq("${_footer_FOOTER}" "${_footer_EXPECT}" "end-of-run table footer, one row per test")
  expect_has("${_footer_EXPECT}" "2 tests | " "the one-row-per-test footer's form")

  # --target-time: each separately named case calibrates its own cycles.
  run(target
      "${PROBE}"
      --threads
      4
      --repeats
      3
      --target-time
      2ms
      --gtest_filter=Rows.SeparateCases
      --csv
      "${WORK_DIR}/target.csv"
  )
  expect_eq("${target_RC}" "0" "probe exit status (--target-time)")
  read_csv(target "${WORK_DIR}/target.csv")
  list(SUBLIST _names 0 3 _separate)
  expect_eq("${target_NAMES}" "${_separate}" "CSV row names (--target-time)")
  string(REGEX MATCHALL "-> cycles=[0-9]+" _calibrations "${target_ERR}")
  list(LENGTH _calibrations _calibrated)
  expect_eq("${_calibrated}" "3" "calibrations printed (--target-time)")
  if (target_COUNT EQUAL 3 AND _calibrated EQUAL 3)
    foreach (_i 0 1 2)
      list(GET _calibrations ${_i} _line)
      string(REGEX REPLACE "^-> cycles=" "" _cycles "${_line}")
      cell(_value target ${_i} cycles)
      list(GET _separate ${_i} _name)
      expect_eq("${_value}" "${_cycles}" "${_name}: cycles against its own calibration")
    endforeach ()
  endif ()

  # ------------------------------------------------------------------------------
  # CliAgreement
  # ------------------------------------------------------------------------------

elseif (CASE STREQUAL "CliAgreement")
  if (NOT BENCH_CLI OR NOT EXISTS "${BENCH_CLI}")
    message(STATUS "MEASUREMENT_ROWS_SKIPPED: the bench CLI was not built ('${BENCH_CLI}')")
    return()
  endif ()

  run(csv
      "${PROBE}"
      --threads
      4
      --cycles
      50
      --repeats
      3
      --csv
      "${WORK_DIR}/rows.csv"
  )
  expect_eq("${csv_RC}" "0" "probe exit status")
  read_csv(rows "${WORK_DIR}/rows.csv")
  expect_eq("${rows_NAMES}" "${_names}" "CSV row names")
  set(_sorted ${_names})
  list(SORT _sorted)

  # bench summary reads every row under its name.
  run(summary "${BENCH_CLI}" summary --json "${WORK_DIR}/rows.csv")
  expect_eq("${summary_RC}" "0" "bench summary exit status")
  set(MEMBER test)
  json_list(_summary "${summary_OUT}")
  list(SORT _summary)
  expect_eq("${_summary}" "${_sorted}" "bench summary names")

  # bench compare of the run against itself compares every row.
  run(self "${BENCH_CLI}" compare --json "${WORK_DIR}/rows.csv" "${WORK_DIR}/rows.csv")
  expect_eq("${self_RC}" "0" "bench compare (run against itself) exit status")
  json_list(_compared "${self_OUT}" results)
  list(SORT _compared)
  expect_eq("${_compared}" "${_sorted}" "bench compare names (run against itself)")
  set(MEMBER "")
  json_list(_alone "${self_OUT}" baseline_only)
  expect_eq("${_alone}" "" "tests only in the baseline (run against itself)")
  json_list(_alone "${self_OUT}" candidate_only)
  expect_eq("${_alone}" "" "tests only in the candidate (run against itself)")

  # A baseline in the one-row-per-test shape: each test's last measurement
  # under its case name.
  file(STRINGS "${WORK_DIR}/rows.csv" _header_line LIMIT_COUNT 1)
  set(_old_lines "${_header_line}")
  set(_old_names "")
  list(LENGTH _tests _rows)
  if (rows_COUNT EQUAL _rows)
    set(_i 0)
    while (_i LESS _rows)
      list(GET _tests ${_i} _test)
      math(EXPR _next "${_i} + 1")
      set(_next_test "")
      if (_next LESS _rows)
        list(GET _tests ${_next} _next_test)
      endif ()
      # The test's last row: the next row belongs to another test, or none follows.
      if (NOT _next_test STREQUAL _test)
        list(GET _cases ${_i} _case)
        list(GET rows_LINES ${_i} _line)
        split_row(_row "${_line}")
        string(REPLACE ";" "," _rest "${_row_REST}")
        list(APPEND _old_lines "${_case},${_rest}")
        list(APPEND _old_names "${_case}")
      endif ()
      set(_i ${_next})
    endwhile ()
  endif ()
  string(REPLACE ";" "\n" _old_text "${_old_lines}")
  file(WRITE "${WORK_DIR}/one_row_per_test.csv" "${_old_text}\n")

  set(_missing "")
  foreach (_name IN LISTS _old_names)
    if (NOT _name IN_LIST _names)
      list(APPEND _missing "${_name}")
    endif ()
  endforeach ()
  set(_new "")
  foreach (_name IN LISTS _names)
    if (NOT _name IN_LIST _old_names)
      list(APPEND _new "${_name}")
    endif ()
  endforeach ()
  list(SORT _missing)
  list(SORT _new)

  run(migration
      "${BENCH_CLI}"
      compare
      --json
      --fail-on-regression
      "${WORK_DIR}/one_row_per_test.csv"
      "${WORK_DIR}/rows.csv"
  )
  expect_eq("${migration_RC}" "1" "bench compare --fail-on-regression exit status (old shape)")
  expect_has(
    "${migration_ERR}" "baseline test(s) missing from the candidate"
    "bench compare --fail-on-regression message (old shape)"
  )
  json_list(_alone "${migration_OUT}" baseline_only)
  list(SORT _alone)
  expect_eq("${_alone}" "${_missing}" "case names missing from the run (old shape)")
  json_list(_alone "${migration_OUT}" candidate_only)
  list(SORT _alone)
  expect_eq("${_alone}" "${_new}" "row names new in the run (old shape)")

else ()
  message(FATAL_ERROR "unknown CASE '${CASE}'")
endif ()

if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "MeasurementRows ${CASE}:${_problems}")
endif ()
