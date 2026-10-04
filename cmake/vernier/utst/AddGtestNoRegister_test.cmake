# ==============================================================================
# AddGtestNoRegister_test.cmake - Configure-time test of vernier_add_gtest's
# NO_REGISTER option
#
# Run with: cmake -DWORK_DIR=<dir> -DGENERATOR=<name> [-DMAKE_PROGRAM=<path>]
#                 -DCXX_COMPILER=<path> -P AddGtestNoRegister_test.cmake
#
# Configures a scratch project that makes two test programs from one TEST()
# source: an ordinary one, and a NO_REGISTER one whose owner registers a
# controller with add_test. Stand-ins replace the GoogleTest targets and
# nothing is built. What CTest would run is read from its listing (ctest -N):
#
# 1. The ordinary program's case is registered with its label. The NO_REGISTER
#    program registers nothing (its case again would be a duplicate test name,
#    which stops configuration), and the owner's test runs it from bin/tests.
#    No install rule names either program.
# 2. With coverage on, both programs link the coverage flags and are mapped for
#    the report; only the ordinary program gets a coverage run.
# 3. LABELS or TIMING_ALL beside NO_REGISTER stops configuration, naming the
#    option.
#
# The work directory is kept when a check fails and removed when all pass.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

foreach (_var WORK_DIR GENERATOR CXX_COMPILER)
  if (NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "AddGtestNoRegister_test.cmake: -D${_var}=... is required")
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

# A bracket argument, so only @CMAKE_DIR@ is substituted.
set(_project
    [=[
cmake_minimum_required(VERSION 3.24)
project(NoRegisterProbe LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "@CMAKE_DIR@")
include(vernier/Core)
include(vernier/Coverage)
include(vernier/Testing)
enable_testing()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(GTest::gmock INTERFACE IMPORTED)
add_library(probe_lib STATIC Lib.cpp)
if (ENABLE_COVERAGE)
  add_library(vernier_coverage_flags INTERFACE)
  add_library(vernier::coverage_flags ALIAS vernier_coverage_flags)
endif ()

vernier_add_gtest(TARGET Ordinary SOURCES Probe_uTest.cpp LINK probe_lib LABELS probe)
vernier_add_gtest(TARGET Fixture SOURCES Probe_uTest.cpp LINK probe_lib NO_REGISTER)
add_test(NAME FixtureController COMMAND Fixture --gtest_filter=Probe.Case)

if (REFUSED STREQUAL "LABELS")
  vernier_add_gtest(TARGET Refused SOURCES Probe_uTest.cpp NO_REGISTER LABELS probe)
elseif (REFUSED STREQUAL "TIMING_ALL")
  vernier_add_gtest(TARGET Refused SOURCES Probe_uTest.cpp NO_REGISTER TIMING_ALL)
endif ()

get_property(_maps GLOBAL PROPERTY VERNIER_COVERAGE_MAPPINGS)
foreach (_target Ordinary Fixture)
  get_target_property(_links ${_target} LINK_LIBRARIES)
  message(STATUS "PROBE ${_target} links=[${_links}]")
endforeach ()
message(STATUS "PROBE mappings=[${_maps}]")
]=]
)
string(CONFIGURE "${_project}" _project @ONLY)

file(REMOVE_RECURSE "${WORK_DIR}")
file(WRITE "${_src}/CMakeLists.txt" "${_project}")
file(WRITE "${_src}/Probe_uTest.cpp" "TEST(Probe, Case) {}\n")
file(WRITE "${_src}/Lib.cpp" "")

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

# configure(<case> [<cache entries>...]): configure the scratch project into
# build-<case>. Sets _rc, and _out with CMake's line wrapping flattened.
macro (configure _case)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${_src}" -B "${WORK_DIR}/build-${_case}" ${_toolchain} ${ARGN}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  string(REGEX REPLACE "[ \n]+" " " _out "${_out}")
endmacro ()

# contains(<text> <part> <out var>): "yes" or "no".
function (contains _text _part _out_var)
  string(FIND "${_text}" "${_part}" _pos)
  if (_pos EQUAL -1)
    set(${_out_var}
        "no"
        PARENT_SCOPE
    )
  else ()
    set(${_out_var}
        "yes"
        PARENT_SCOPE
    )
  endif ()
endfunction ()

# tests(<case> <out var> [<label regex>]): the names, sorted, of the tests
# `ctest -N` lists in build-<case>, with -L <label regex> when one is given.
function (tests _case _out_var)
  set(_filter "")
  if (ARGC GREATER 2)
    set(_filter -L "${ARGV2}")
  endif ()
  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" -N ${_filter}
    WORKING_DIRECTORY "${WORK_DIR}/build-${_case}"
    OUTPUT_VARIABLE _listing
  )
  string(REGEX MATCHALL "Test +#[0-9]+: [^\n]+" _lines "${_listing}")
  set(_names "")
  foreach (_line IN LISTS _lines)
    string(REGEX REPLACE "^Test +#[0-9]+: " "" _name "${_line}")
    list(APPEND _names "${_name}")
  endforeach ()
  list(SORT _names)
  set(${_out_var}
      "${_names}"
      PARENT_SCOPE
  )
endfunction ()

# ------------------------------------------------------------------------------
# Cases (3)
# ------------------------------------------------------------------------------

# 1. Registration and placement, coverage off.
configure(plain)
expect("plain: configure status" "${_rc}" "0")
tests(plain _all)
expect("plain: registered tests" "${_all}" "FixtureController;Probe.Case")
tests(plain _labelled "^probe$")
expect("plain: tests labelled probe" "${_labelled}" "Probe.Case")
# The program a test runs is resolved when it is built, so CTest's listing of
# an unbuilt tree has no command: read the generated test file instead.
set(_testfile "")
set(_install "")
if (EXISTS "${WORK_DIR}/build-plain/CTestTestfile.cmake")
  file(READ "${WORK_DIR}/build-plain/CTestTestfile.cmake" _testfile)
  file(READ "${WORK_DIR}/build-plain/cmake_install.cmake" _install)
endif ()
contains("${_testfile}"
         "add_test([=[FixtureController]=] \"${WORK_DIR}/build-plain/bin/tests/Fixture\""
         _controller
)
expect("plain: the controller runs bin/tests/Fixture" "${_controller}" "yes")
contains("${_install}" "Fixture" _fixture_rule)
contains("${_install}" "Ordinary" _ordinary_rule)
expect("plain: no install rule names the NO_REGISTER program" "${_fixture_rule}" "no")
expect("plain: no install rule names the ordinary program" "${_ordinary_rule}" "no")

# 2. The coverage policy, coverage on.
configure(coverage -DENABLE_COVERAGE=ON)
expect("coverage: configure status" "${_rc}" "0")
tests(coverage _all)
expect("coverage: registered tests" "${_all}" "FixtureController;Ordinary_coverage;Probe.Case")
tests(coverage _labelled "^Coverage$")
expect("coverage: tests labelled Coverage" "${_labelled}" "Ordinary_coverage")
tests(coverage _labelled "^probe$")
expect("coverage: tests labelled probe" "${_labelled}" "Ordinary_coverage;Probe.Case")
contains("${_out}" "PROBE mappings=[Ordinary:probe_lib;Fixture:probe_lib]" _mapped)
expect("coverage: both programs mapped for the report" "${_mapped}" "yes")
string(REGEX MATCH "PROBE Fixture links=\\[[^ ]*" _fixture_links "${_out}")
contains("${_fixture_links}" "vernier::coverage_flags" _instrumented)
expect("coverage: the NO_REGISTER program links the coverage flags" "${_instrumented}" "yes")

# 3. Registration options are refused beside NO_REGISTER.
foreach (_option LABELS TIMING_ALL)
  configure(refused_${_option} -DREFUSED=${_option})
  expect("refused ${_option}: configure stops" "${_rc}" "1")
  contains("${_out}" "vernier_add_gtest(Refused): ${_option} describes registered tests" _named)
  expect("refused ${_option}: the error names the option" "${_named}" "yes")
endforeach ()

if (_failures GREATER 0)
  message(FATAL_ERROR "${_failures} check(s) failed; the scratch project is kept in ${WORK_DIR}")
endif ()
file(REMOVE_RECURSE "${WORK_DIR}")
message(STATUS "All checks passed")
