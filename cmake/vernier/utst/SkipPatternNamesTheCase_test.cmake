# ==============================================================================
# SkipPatternNamesTheCase_test.cmake - Test of the skip expression
# vernier_add_gtest gives each GoogleTest case it registers
#
# Run with: cmake -DWORK_DIR=<dir> -DGENERATOR=<name> [-DMAKE_PROGRAM=<path>]
#                 -DCXX_COMPILER=<path> -DGOOGLETEST_DIR=<googletest source>
#                 -P SkipPatternNamesTheCase_test.cmake
#
# Configures and builds a scratch project with GoogleTest (gtest and
# gtest_main, built from GOOGLETEST_DIR) and one program registered through
# vernier_add_gtest, whose cases are: one that prints another case's skip
# lines, as a process test's failure message quotes a child's output, and
# fails; one that prints the skip lines of a longer name that begins with its
# own, and fails; one that skips; one that passes; and a parameterized case
# whose two instances skip. Then:
#
# 1. Every case's SKIP_REGULAR_EXPRESSION (ctest --show-only=json-v1) is the
#    two lines GoogleTest prints for that case alone when it skips: with the
#    time ("[  SKIPPED ] Suite.Case (0 ms)") and at a line's end, the name's
#    regex characters escaped and each * of the parameterized name matching
#    one name part.
# 2. CTest reports each case as it ended: the two that print skip lines and
#    fail Failed, the skipping case and the parameterized case Skipped, the
#    passing case Passed.
# 3. The same with GTEST_PRINT_TIME=0 in the environment, where GoogleTest
#    prints no times.
#
# The work directory is kept when a check fails and removed when all pass.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

foreach (_var WORK_DIR GENERATOR CXX_COMPILER GOOGLETEST_DIR)
  if (NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "SkipPatternNamesTheCase_test.cmake: -D${_var}=... is required")
  endif ()
endforeach ()

set(_toolchain -G "${GENERATOR}" -DCMAKE_CXX_COMPILER=${CXX_COMPILER})
if (DEFINED MAKE_PROGRAM AND NOT MAKE_PROGRAM STREQUAL "")
  list(APPEND _toolchain -DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM})
endif ()

# ------------------------------------------------------------------------------
# Scratch Project
# ------------------------------------------------------------------------------

get_filename_component(CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_src "${WORK_DIR}/src")
set(_build "${WORK_DIR}/build")

# A bracket argument, so only @CMAKE_DIR@ and @GOOGLETEST_DIR@ are substituted.
set(_project
    [=[
cmake_minimum_required(VERSION 3.24)
project(SkipPatternProbe LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "@CMAKE_DIR@")
include(vernier/Core)
include(vernier/Testing)
enable_testing()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory("@GOOGLETEST_DIR@" googletest EXCLUDE_FROM_ALL)
# vernier_add_gtest links gmock beside gtest_main; the probe uses none of it.
add_library(GTest::gmock INTERFACE IMPORTED)

vernier_add_gtest(TARGET Probe SOURCES Probe_uTest.cpp LABELS probe)
]=]
)
string(CONFIGURE "${_project}" _project @ONLY)

set(_probe
    [=[
#include <gtest/gtest.h>

#include <cstdio>

TEST(Probe, FailsQuotingAChild) {
  std::printf("[  SKIPPED ] Other.Case (0 ms)\n[  SKIPPED ] Other.Case\n");
  FAIL() << "the child skipped";
}

TEST(Probe, FailsQuotingALongerName) {
  std::printf("[  SKIPPED ] Probe.FailsQuotingALongerNameToo (0 ms)\n"
              "[  SKIPPED ] Probe.FailsQuotingALongerNameToo\n");
  FAIL() << "the child skipped";
}

TEST(Probe, Skips) { GTEST_SKIP() << "nothing to run here"; }

TEST(Probe, Passes) {}

class ProbeP : public ::testing::TestWithParam<int> {};

TEST_P(ProbeP, Skips) { GTEST_SKIP() << "nothing to run here"; }

INSTANTIATE_TEST_SUITE_P(Sizes, ProbeP, ::testing::Values(1, 2));
]=]
)

file(REMOVE_RECURSE "${WORK_DIR}")
file(WRITE "${_src}/CMakeLists.txt" "${_project}")
file(WRITE "${_src}/Probe_uTest.cpp" "${_probe}")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

set(_failures 0)

# expect(<what> <actual> <wanted>): compare two strings, count a mismatch.
function (expect _what _actual _wanted)
  if ("${_actual}" STREQUAL "${_wanted}")
    message(STATUS "PASS ${_what}")
  else ()
    message(STATUS "FAIL ${_what}: wanted [${_wanted}], got [${_actual}]")
    math(EXPR _count "${_failures} + 1")
    set(_failures
        ${_count}
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# verdicts(<out var> [<env setting>...]): run ctest in the build tree with the
# settings in its environment; "<test>=<Passed|Failed|Skipped|...>", sorted.
function (verdicts _out_var)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${ARGN} "${CMAKE_CTEST_COMMAND}" --timeout 60
    WORKING_DIRECTORY "${_build}"
    OUTPUT_VARIABLE _listing
    ERROR_VARIABLE _listing
  )
  string(REGEX MATCHALL "Test +#[0-9]+: [^ ]+ [ .]+(\\*\\*\\*)?[A-Za-z]+" _lines "${_listing}")
  set(_verdicts "")
  foreach (_line IN LISTS _lines)
    string(REGEX REPLACE "^Test +#[0-9]+: ([^ ]+) [ .]+(\\*\\*\\*)?([A-Za-z]+)$" "\\1=\\3" _verdict
                         "${_line}"
    )
    list(APPEND _verdicts "${_verdict}")
  endforeach ()
  list(SORT _verdicts)
  set(${_out_var}
      "${_verdicts}"
      PARENT_SCOPE
  )
endfunction ()

# ------------------------------------------------------------------------------
# Cases (3)
# ------------------------------------------------------------------------------

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${_src}" -B "${_build}" ${_toolchain}
  RESULT_VARIABLE _rc
  OUTPUT_QUIET
  ERROR_VARIABLE _err
)
expect("configure status" "${_rc}" "0")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${_build}" --parallel 3
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
expect("build status" "${_rc}" "0")
if (NOT _rc EQUAL 0)
  message(STATUS "${_err}${_out}")
endif ()

# 1. Each case's skip expressions name it. Read back from CTest's listing, each
#    as "<test>=<expression>|<expression>", a line feed shown as <LF>.
execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --show-only=json-v1
  WORKING_DIRECTORY "${_build}"
  OUTPUT_VARIABLE _json
)
set(_seen "")
string(JSON _count ERROR_VARIABLE _json_error LENGTH "${_json}" tests)
if (_json_error)
  set(_count 0)
endif ()
set(_i 0)
while (_i LESS _count)
  string(JSON _test GET "${_json}" tests ${_i})
  string(JSON _name GET "${_test}" name)
  string(JSON _property_count LENGTH "${_test}" properties)
  set(_expressions "")
  set(_j 0)
  while (_j LESS _property_count)
    string(JSON _property GET "${_test}" properties ${_j})
    string(JSON _property_name GET "${_property}" name)
    if (_property_name STREQUAL "SKIP_REGULAR_EXPRESSION")
      string(JSON _value_count LENGTH "${_property}" value)
      set(_k 0)
      while (_k LESS _value_count)
        string(JSON _expression GET "${_property}" value ${_k})
        string(REPLACE "\n" "<LF>" _expression "${_expression}")
        string(APPEND _expressions "|${_expression}")
        math(EXPR _k "${_k} + 1")
      endwhile ()
    endif ()
    math(EXPR _j "${_j} + 1")
  endwhile ()
  string(REGEX REPLACE "^\\|" "" _expressions "${_expressions}")
  list(APPEND _seen "${_name}=${_expressions}")
  math(EXPR _i "${_i} + 1")
endwhile ()
list(SORT _seen)
set(_wanted "")
foreach (_case IN ITEMS "*/ProbeP.Skips/*" Probe.FailsQuotingAChild Probe.FailsQuotingALongerName
                        Probe.Passes Probe.Skips
)
  string(REPLACE "." "\\." _escaped "${_case}")
  string(REPLACE "*" "[^ <LF>]*" _escaped "${_escaped}")
  list(APPEND _wanted "${_case}=\\[  SKIPPED \\] ${_escaped} \\(|\\[  SKIPPED \\] ${_escaped}<LF>")
endforeach ()
list(SORT _wanted)
expect("each case's skip expressions" "${_seen}" "${_wanted}")

# 2. CTest's verdicts, with the times GoogleTest prints by default.
set(_expected_verdicts
    "*/ProbeP.Skips/*=Skipped;Probe.FailsQuotingAChild=Failed;Probe.FailsQuotingALongerName=Failed;Probe.Passes=Passed;Probe.Skips=Skipped"
)
verdicts(_seen)
expect("verdicts with times" "${_seen}" "${_expected_verdicts}")

# 3. The same with no times printed.
verdicts(_seen GTEST_PRINT_TIME=0)
expect("verdicts without times" "${_seen}" "${_expected_verdicts}")

if (_failures GREATER 0)
  message(FATAL_ERROR "${_failures} check(s) failed; the scratch project is kept in ${WORK_DIR}")
endif ()
file(REMOVE_RECURSE "${WORK_DIR}")
message(STATUS "All checks passed")
