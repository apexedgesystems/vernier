# ==============================================================================
# InstallConsumer_test.cmake - Builds and runs programs against an installed
# vernier, the way a project outside this tree uses it
#
# Run with: cmake -DSOURCE_DIR=<vernier root> -DGTEST_SOURCE_DIR=<googletest>
#                 -DCONSUMER_DIR=<consumer project> -DWORK_DIR=<dir>
#                 -DGENERATOR=<name> [-DMAKE_PROGRAM=<path>]
#                 -DC_COMPILER=<path> -DCXX_COMPILER=<path> -DJOBS=<n>
#                 -P InstallConsumer_test.cmake
#
# 1. GTest is built from its source and installed to a prefix of its own: the
#    package finds GTest, so a consumer has one installed.
# 2. This tree is configured with a CMAKE_INSTALL_INCLUDEDIR other than the
#    default, bench and monitor are built and installed to a stage prefix, and
#    the prefix is then moved, so no path recorded at install time is left.
# 3. The installed include directory holds each bench and monitor header once,
#    at the path it has in the source tree, under the vernier root.
# 4. The consumer project finds vernier through CMAKE_PREFIX_PATH and GTest
#    through GTest_ROOT, and builds a PERF_MAIN benchmark and a monitor
#    program in both include styles. Every include directory its compile
#    commands carry lies in the moved prefix or the GTest prefix.
# 5. Each program resolves its vernier library in the moved prefix and runs
#    with LD_LIBRARY_PATH unset: `make test` and `make testp` point that
#    variable at the build tree's libraries, which would otherwise be loaded.
#
# Every step runs without CPATH, C_INCLUDE_PATH, CPLUS_INCLUDE_PATH,
# LIBRARY_PATH and LD_LIBRARY_PATH, which add search paths behind CMake's back.
# The work directory is kept when a check fails and removed when all pass.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

foreach (
  _var
  SOURCE_DIR
  GTEST_SOURCE_DIR
  CONSUMER_DIR
  WORK_DIR
  GENERATOR
  C_COMPILER
  CXX_COMPILER
  JOBS
)
  if (NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "InstallConsumer_test.cmake: -D${_var}=... is required")
  endif ()
endforeach ()

# Two components, and not the default "include".
set(_include_dir "headers/nested")
set(_stage "${WORK_DIR}/stage")
set(_prefix "${WORK_DIR}/moved/prefix")
set(_gtest "${WORK_DIR}/gtest")
set(_logs "${WORK_DIR}/logs")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${_logs}" "${WORK_DIR}/moved" "${WORK_DIR}/run")

# The build under test's generator and compilers. Its sanitizer and coverage
# flags are left out: the programs here would have to link those runtimes. The
# consumer project is C++ only, so the C compiler goes to the other two.
set(_toolchain -G "${GENERATOR}" -DCMAKE_CXX_COMPILER=${CXX_COMPILER})
if (DEFINED MAKE_PROGRAM AND NOT MAKE_PROGRAM STREQUAL "")
  list(APPEND _toolchain -DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM})
endif ()
set(_toolchain_c ${_toolchain} -DCMAKE_C_COMPILER=${C_COMPILER})

set(_clean_env --unset=CPATH --unset=C_INCLUDE_PATH --unset=CPLUS_INCLUDE_PATH --unset=LIBRARY_PATH
               --unset=LD_LIBRARY_PATH
)

# run_step(<name> <command...>): run with the clean environment, keep the
# output in logs/<name>.log, stop the test on a nonzero status.
function (run_step _name)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${_clean_env} ${ARGN}
    WORKING_DIRECTORY "${WORK_DIR}/run"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  file(WRITE "${_logs}/${_name}.log" "${_out}")
  if (NOT _rc EQUAL 0)
    message(FATAL_ERROR "${_name} failed (exit ${_rc}); work directory kept: ${WORK_DIR}\n${_out}")
  endif ()
  set(_step_output
      "${_out}"
      PARENT_SCOPE
  )
endfunction ()

# ------------------------------------------------------------------------------
# 1. GTest installed to its own prefix
# ------------------------------------------------------------------------------

run_step(
  gtest_configure
  "${CMAKE_COMMAND}"
  -S
  "${GTEST_SOURCE_DIR}"
  -B
  "${WORK_DIR}/gtest-build"
  ${_toolchain_c}
  -DBUILD_GMOCK=OFF
  -DINSTALL_GTEST=ON
  -DCMAKE_INSTALL_PREFIX=${_gtest}
)
run_step(gtest_build "${CMAKE_COMMAND}" --build "${WORK_DIR}/gtest-build" --parallel ${JOBS})
run_step(gtest_install "${CMAKE_COMMAND}" --install "${WORK_DIR}/gtest-build")

# ------------------------------------------------------------------------------
# 2. vernier installed with a non-default include directory, then moved
# ------------------------------------------------------------------------------

run_step(
  vernier_configure
  "${CMAKE_COMMAND}"
  -S
  "${SOURCE_DIR}"
  -B
  "${WORK_DIR}/vernier-build"
  ${_toolchain_c}
  -DCMAKE_INSTALL_INCLUDEDIR=${_include_dir}
  -DVERNIER_BUILD_GPU=OFF
  -DVERNIER_BUILD_TOOLS=OFF
  -DPROJECT_BUILD_DOCS=OFF
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${GTEST_SOURCE_DIR}
)
run_step(
  vernier_build
  "${CMAKE_COMMAND}"
  --build
  "${WORK_DIR}/vernier-build"
  --target
  bench
  monitor
  --parallel
  ${JOBS}
)
run_step(vernier_install "${CMAKE_COMMAND}" --install "${WORK_DIR}/vernier-build" --prefix
         "${_stage}"
)
file(RENAME "${_stage}" "${_prefix}")

set(_problems "")

# ------------------------------------------------------------------------------
# 3. One installed header tree
# ------------------------------------------------------------------------------

file(
  GLOB_RECURSE _installed
  RELATIVE "${_prefix}/${_include_dir}"
  "${_prefix}/${_include_dir}/*"
)
foreach (_file IN LISTS _installed)
  if (NOT _file MATCHES "^vernier/(src/[^/]+/inc/[^/]+)$")
    string(APPEND _problems "\n  installed outside vernier/src/<module>/inc/: ${_file}")
    continue()
  endif ()
  if (NOT EXISTS "${SOURCE_DIR}/${CMAKE_MATCH_1}")
    string(APPEND _problems "\n  installed with no source counterpart: ${_file}")
  endif ()
endforeach ()
foreach (_module bench monitor)
  file(
    GLOB _headers
    RELATIVE "${SOURCE_DIR}"
    "${SOURCE_DIR}/src/${_module}/inc/*"
  )
  if (NOT _headers)
    string(APPEND _problems "\n  no headers found in ${SOURCE_DIR}/src/${_module}/inc")
  endif ()
  foreach (_header IN LISTS _headers)
    if (NOT "vernier/${_header}" IN_LIST _installed)
      string(APPEND _problems "\n  not installed: ${_header}")
    endif ()
  endforeach ()
endforeach ()
if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "InstallConsumer: installed header tree:${_problems}\n"
                      "work directory kept: ${WORK_DIR}"
  )
endif ()

# ------------------------------------------------------------------------------
# 4. The consumer project, through find_package(vernier) alone
# ------------------------------------------------------------------------------

# Copied out of the source tree, so nothing next to it can be picked up.
file(COPY "${CONSUMER_DIR}/" DESTINATION "${WORK_DIR}/consumer")
run_step(
  consumer_configure
  "${CMAKE_COMMAND}"
  -S
  "${WORK_DIR}/consumer"
  -B
  "${WORK_DIR}/consumer-build"
  ${_toolchain}
  -DCMAKE_PREFIX_PATH=${_prefix}
  -DGTest_ROOT=${_gtest}
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
run_step(consumer_build "${CMAKE_COMMAND}" --build "${WORK_DIR}/consumer-build" --parallel ${JOBS})

file(STRINGS "${WORK_DIR}/consumer-build/CMakeCache.txt" _found REGEX "^vernier_DIR:")
string(REGEX REPLACE "^vernier_DIR:[A-Z]+=" "" _found "${_found}")
string(FIND "${_found}" "${_prefix}/" _at)
if (NOT _at EQUAL 0)
  string(APPEND _problems "\n  find_package(vernier) used '${_found}', not the moved prefix")
endif ()

# Include directories from the compile commands: the moved prefix's vernier
# root must be among them, and nothing outside the two prefixes may be.
file(READ "${WORK_DIR}/consumer-build/compile_commands.json" _json)
string(JSON _count LENGTH "${_json}")
if (NOT _count EQUAL 4)
  message(FATAL_ERROR "InstallConsumer: expected 4 compile commands, found ${_count}\n"
                      "work directory kept: ${WORK_DIR}"
  )
endif ()
set(_i 0)
while (_i LESS _count)
  string(JSON _file GET "${_json}" ${_i} file)
  string(JSON _command GET "${_json}" ${_i} command)
  separate_arguments(_args UNIX_COMMAND "${_command}")
  set(_dirs "")
  set(_next_is_dir FALSE)
  foreach (_arg IN LISTS _args)
    if (_next_is_dir)
      list(APPEND _dirs "${_arg}")
      set(_next_is_dir FALSE)
    elseif (_arg MATCHES "^-(I|isystem|iquote|idirafter)$")
      set(_next_is_dir TRUE)
    elseif (_arg MATCHES "^-(I|isystem|iquote|idirafter)(.+)$")
      list(APPEND _dirs "${CMAKE_MATCH_2}")
    endif ()
  endforeach ()
  if (NOT "${_prefix}/${_include_dir}/vernier" IN_LIST _dirs)
    string(APPEND _problems "\n  ${_file}: the installed vernier root is not on its include path")
  endif ()
  foreach (_dir IN LISTS _dirs)
    string(FIND "${_dir}" "${_prefix}/" _in_prefix)
    string(FIND "${_dir}" "${_gtest}/" _in_gtest)
    if (NOT _in_prefix EQUAL 0 AND NOT _in_gtest EQUAL 0)
      string(APPEND _problems "\n  ${_file}: include directory outside the installs: ${_dir}")
    endif ()
  endforeach ()
  math(EXPR _i "${_i} + 1")
endwhile ()

# ------------------------------------------------------------------------------
# 5. Each program loads the installed library and runs
# ------------------------------------------------------------------------------

find_program(_ldd ldd)
if (NOT _ldd)
  message(FATAL_ERROR "InstallConsumer: ldd is needed to see which library a program loads")
endif ()

foreach (_style short qualified)
  foreach (_program bench monitor)
    set(_exe "${WORK_DIR}/consumer-build/${_program}_consumer_${_style}")
    if (NOT EXISTS "${_exe}")
      string(APPEND _problems "\n  not built: ${_exe}")
      continue()
    endif ()

    # What the loader resolves for the program in the environment it runs in.
    run_step(ldd_${_program}_${_style} "${_ldd}" "${_exe}")
    string(REGEX MATCHALL "lib${_program}[.]so[^ \t\n]* => [^\n]*" _lines "${_step_output}")
    list(LENGTH _lines _n)
    if (NOT _n EQUAL 1)
      string(APPEND _problems "\n  ${_program}_consumer_${_style}: ldd lists '${_lines}'")
      continue()
    endif ()
    string(REGEX REPLACE "^[^ ]+ => ([^ ]+).*$" "\\1" _loaded "${_lines}")
    string(FIND "${_loaded}" "${_prefix}/" _at)
    if (NOT _at EQUAL 0)
      string(APPEND _problems
             "\n  ${_program}_consumer_${_style}: lib${_program} does not resolve inside the"
             " moved prefix: ${_lines}"
      )
    endif ()
  endforeach ()

  # The benchmark: its case passes and the CSV listener writes its row with
  # the counts given on the command line.
  set(_csv "${WORK_DIR}/run/bench_${_style}.csv")
  run_step(
    run_bench_${_style}
    "${WORK_DIR}/consumer-build/bench_consumer_${_style}"
    --cycles
    20
    --repeats
    2
    --csv
    "${_csv}"
  )
  if (NOT _step_output MATCHES "\\[  PASSED  \\] 1 test")
    string(APPEND _problems "\n  bench_consumer_${_style}: no passing test in its output")
  endif ()
  set(_row "")
  if (EXISTS "${_csv}")
    file(STRINGS "${_csv}" _row REGEX "^InstallConsumer[.]Sum,20,2,")
  endif ()
  if (_row STREQUAL "")
    string(APPEND _problems "\n  bench_consumer_${_style}: no InstallConsumer.Sum row in ${_csv}")
  endif ()

  # The monitor: its file sink, compiled into libmonitor, creates the file.
  set(_samples "${WORK_DIR}/run/monitor_${_style}.vmon")
  run_step(run_monitor_${_style} "${WORK_DIR}/consumer-build/monitor_consumer_${_style}"
           "${_samples}"
  )
  if (NOT EXISTS "${_samples}")
    string(APPEND _problems
           "\n  monitor_consumer_${_style}: the file sink did not create ${_samples}"
    )
  endif ()
endforeach ()

if (NOT _problems STREQUAL "")
  message(FATAL_ERROR "InstallConsumer:${_problems}\nwork directory kept: ${WORK_DIR}")
endif ()

message(STATUS "installed consumers built and ran against ${_prefix}")
file(REMOVE_RECURSE "${WORK_DIR}")
