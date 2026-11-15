# ==============================================================================
# vernier/Testing.cmake - Test infrastructure and coverage support
# ==============================================================================

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# vernier_add_gtest(...)
#
# Add a GoogleTest-based test target with coverage and timing controls.
#
# Coverage is automatic: any project library in LINK is instrumented when
# ENABLE_COVERAGE=ON. Use COVERAGE_FOR only to override auto-detection.
#
# NO_REGISTER builds the program as above but registers nothing with CTest: no
# discovered cases and no coverage run. It is for a process fixture whose cases
# must run only under a controller, which its owner registers with add_test
# beside the target; the options that describe registered tests are refused
# with it. A source that defines main (PERF_MAIN) keeps it: an object's main
# is linked before GoogleTest's. vernier_add_test_program builds a program with
# no GoogleTest in it through this same path, and vernier_add_devtest a
# component test program, placed in bin/dtests.
#
# Every registered test carries one class label, unit or component: the
# program's CLASS (unit unless set; a vernier_add_devtest program's is
# component), or component where COMPONENT_PATTERNS names the test. rig goes
# beside the class on the tests RIG_PATTERNS names, or on a whole program
# through LABELS. The class labels are the helper's, not LABELS', and a
# pattern that names no test of the program is an error. A coverage run, the
# whole program, is component when any of its tests is. The lanes select by
# these labels (mk/test.mk).
#
# WORKER_SOURCES are compiled into the program as SOURCES are, but not read for
# cases: a case defined in one is not registered. It is for a probe case that
# the program's own tests run as a child, which run alone would only skip.
#
# Arguments:
#   TARGET          <n>              required
#   SOURCES         <src...>         required
#   WORKER_SOURCES  <src...>         optional (compiled in, cases not registered)
#   CUDA            <cu...>          optional
#   LINK            <libs...>        optional
#   COVERAGE_FOR    <lib_target>     optional (overrides auto-detection)
#   INC             <dir>            optional
#   LABELS          <labels...>      optional
#   CLASS           <unit|component> optional (the program's class; unit)
#   COMPONENT_PATTERNS <regex...>    optional (its tests that are component)
#   RIG_PATTERNS    <regex...>       optional (its tests that need a rig)
#   WORKING_DIR     <dir>            optional
#   RESOURCE_LOCK   <n>              optional
#   NO_COVERAGE                      optional flag (skip coverage)
#   NO_REGISTER                      optional flag (build only; owner registers)
#   TIMING_ALL                       optional flag
#   TIMING_TESTS    <names...>       optional
#   TIMING_PATTERNS <regex...>       optional
#   REQUIRES_THREADS <n>             optional (min hardware threads required)
# ------------------------------------------------------------------------------
function (vernier_add_gtest)
  # Skip on bare-metal
  if (CMAKE_SYSTEM_NAME STREQUAL "Generic")
    return()
  endif ()

  # _PLAIN_PROGRAM is vernier_add_test_program's: no GoogleTest is linked.
  # _COMPONENT is vernier_add_devtest's: the program goes to bin/dtests.
  cmake_parse_arguments(
    GT "TIMING_ALL;NO_COVERAGE;NO_REGISTER;_PLAIN_PROGRAM;_COMPONENT"
    "TARGET;INC;WORKING_DIR;RESOURCE_LOCK;COVERAGE_FOR;REQUIRES_THREADS;CLASS"
    "SOURCES;WORKER_SOURCES;CUDA;LINK;LABELS;TIMING_TESTS;TIMING_PATTERNS;COMPONENT_PATTERNS;RIG_PATTERNS"
    ${ARGN}
  )
  vernier_require(GT_TARGET GT_SOURCES)

  if (GT_NO_REGISTER)
    foreach (
      _opt
      LABELS
      CLASS
      COMPONENT_PATTERNS
      RIG_PATTERNS
      WORKING_DIR
      RESOURCE_LOCK
      REQUIRES_THREADS
      TIMING_ALL
      TIMING_TESTS
      TIMING_PATTERNS
    )
      if (GT_${_opt})
        message(FATAL_ERROR "${GT_TARGET}: ${_opt} describes registered tests, and ${GT_TARGET} "
                            "registers none: set it on the owner's add_test"
        )
      endif ()
    endforeach ()
  endif ()

  # The program's class; COMPONENT_PATTERNS can make some of its tests
  # component (Class labels, below).
  set(_class unit)
  if (GT__COMPONENT)
    set(_class component)
  endif ()
  if (GT_CLASS)
    if (NOT GT_CLASS MATCHES "^(unit|component)$" OR (GT__COMPONENT AND GT_CLASS STREQUAL "unit"))
      message(FATAL_ERROR "${GT_TARGET}: CLASS ${GT_CLASS}: a program's class is unit or "
                          "component, and a component program's is component"
      )
    endif ()
    set(_class "${GT_CLASS}")
  endif ()
  foreach (_label IN ITEMS unit component)
    if (_label IN_LIST GT_LABELS)
      message(FATAL_ERROR "${GT_TARGET}: ${_label} is a class, set by CLASS or "
                          "COMPONENT_PATTERNS, not a label"
      )
    endif ()
  endforeach ()

  add_executable(${GT_TARGET})
  target_sources(${GT_TARGET} PRIVATE ${GT_SOURCES} ${GT_WORKER_SOURCES})

  if (GT_INC)
    target_include_directories(${GT_TARGET} PRIVATE "${GT_INC}")
  endif ()

  # CUDA sources
  if (GT_CUDA
      AND CUDAToolkit_FOUND
      AND CMAKE_CUDA_COMPILER
  )
    vernier_cuda_sources(${GT_TARGET} FILES ${GT_CUDA})
  endif ()

  # GTest linkage
  set(_gtest_libs "")
  if (NOT GT__PLAIN_PROGRAM)
    if (NOT TARGET GTest::gtest_main)
      find_package(GTest QUIET CONFIG)
      if (NOT TARGET GTest::gtest_main)
        find_package(GTest QUIET)
      endif ()
      if (NOT TARGET GTest::gtest_main)
        message(FATAL_ERROR "vernier_add_gtest: GTest::gtest_main not found")
      endif ()
    endif ()
    set(_gtest_libs GTest::gtest_main GTest::gmock)
  endif ()

  target_link_libraries(${GT_TARGET} PRIVATE ${_gtest_libs} ${GT_LINK})

  set(_out_dir "tests")
  if (GT__COMPONENT)
    set(_out_dir "dtests")
  endif ()
  set_target_properties(
    ${GT_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_out_dir}"
  )

  # Coverage instrumentation (auto-detect from LINK or use COVERAGE_FOR override)
  set(_coverage_libs "")
  if (GT_NO_COVERAGE)
    # Skip coverage entirely (used by vernier_add_ptest)
  elseif (ENABLE_COVERAGE AND TARGET vernier_coverage_flags)
    if (GT_COVERAGE_FOR)
      # Explicit override
      set(_coverage_libs ${GT_COVERAGE_FOR})
    else ()
      # Auto-detect from LINK: filter out imported and aliased targets
      foreach (_lib IN LISTS GT_LINK)
        if (NOT TARGET ${_lib})
          continue()
        endif ()
        # Skip aliased targets (vernier::* or any ALIAS)
        if (_lib MATCHES "^vernier::")
          continue()
        endif ()
        get_target_property(_aliased ${_lib} ALIASED_TARGET)
        if (_aliased)
          continue()
        endif ()
        # Skip imported targets (third-party like fmt::fmt, GTest::*)
        get_target_property(_imported ${_lib} IMPORTED)
        if (_imported)
          continue()
        endif ()
        list(APPEND _coverage_libs ${_lib})
      endforeach ()
    endif ()

    # Instrument test executable and libraries
    if (_coverage_libs)
      target_link_libraries(${GT_TARGET} PRIVATE vernier::coverage_flags)

      foreach (_lib IN LISTS _coverage_libs)
        if (TARGET ${_lib})
          get_target_property(_lib_type ${_lib} TYPE)
          if (_lib_type STREQUAL "INTERFACE_LIBRARY")
            target_link_libraries(${_lib} INTERFACE vernier::coverage_flags)
          else ()
            target_link_libraries(${_lib} PRIVATE vernier::coverage_flags)
          endif ()

          # Register mapping for report generation
          if (COMMAND vernier_coverage_register)
            vernier_coverage_register(${GT_TARGET} ${_lib})
          endif ()
        endif ()
      endforeach ()
    endif ()
  endif ()

  if (GT_NO_REGISTER)
    set(_kind "NO_REGISTER")
    if (GT__PLAIN_PROGRAM)
      set(_kind "test program, no GoogleTest")
    endif ()
    message(STATUS "[Test] target=${GT_TARGET} registered by its owner (${_kind})")
    return()
  endif ()

  # Test discovery
  include(GoogleTest)
  if (GT_WORKING_DIR)
    set(_working_dir "${GT_WORKING_DIR}")
  else ()
    set(_working_dir "${CMAKE_CURRENT_BINARY_DIR}")
  endif ()

  set(_discover_sources ${GT_SOURCES})
  if (GT_CUDA
      AND CUDAToolkit_FOUND
      AND CMAKE_CUDA_COMPILER
  )
    list(APPEND _discover_sources ${GT_CUDA})
  endif ()

  set(_all_tests "")
  gtest_add_tests(
    TARGET ${GT_TARGET}
    SOURCES ${_discover_sources}
    WORKING_DIRECTORY "${_working_dir}"
    TEST_LIST _all_tests
  )
  _vernier_gtest_skip_patterns(${_all_tests})

  # Base labels
  if (GT_LABELS)
    set_property(
      TEST ${_all_tests}
      APPEND
      PROPERTY LABELS "${GT_LABELS}"
    )
  endif ()

  # Class labels: the program's class, or component where a pattern names the
  # test; rig beside it where a pattern names the test.
  set(_component_tests "")
  set(_rig_tests "")
  foreach (_kind IN ITEMS component rig)
    string(TOUPPER "${_kind}" _arg)
    foreach (_rx IN LISTS GT_${_arg}_PATTERNS)
      set(_named "")
      foreach (_name IN LISTS _all_tests)
        if (_name MATCHES "${_rx}")
          list(APPEND _named "${_name}")
        endif ()
      endforeach ()
      if (NOT _named)
        message(FATAL_ERROR "${GT_TARGET}: ${_arg}_PATTERNS '${_rx}' names no test of the program")
      endif ()
      list(APPEND _${_kind}_tests ${_named})
    endforeach ()
  endforeach ()
  foreach (_name IN LISTS _all_tests)
    set(_test_labels "${_class}")
    if (_name IN_LIST _component_tests)
      set(_test_labels component)
    endif ()
    if (_name IN_LIST _rig_tests)
      list(APPEND _test_labels rig)
    endif ()
    set_property(
      TEST "${_name}"
      APPEND
      PROPERTY LABELS ${_test_labels}
    )
  endforeach ()

  # Set working directory for all discovered tests
  set_tests_properties(${_all_tests} PROPERTIES WORKING_DIRECTORY "${_working_dir}")

  # Resource lock for tests that share system resources (ports, files, etc.)
  # Applied to ALL tests in the target, independent of timing sensitivity
  if (GT_RESOURCE_LOCK AND _all_tests)
    set_tests_properties(${_all_tests} PROPERTIES RESOURCE_LOCK "${GT_RESOURCE_LOCK}")
  endif ()

  # Hardware thread requirement - skip tests on systems with insufficient threads
  if (GT_REQUIRES_THREADS AND _all_tests)
    cmake_host_system_information(RESULT _hw_threads QUERY NUMBER_OF_LOGICAL_CORES)
    if (_hw_threads LESS GT_REQUIRES_THREADS)
      message(
        STATUS
          "[Skip] ${GT_TARGET}: requires ${GT_REQUIRES_THREADS} threads, system has ${_hw_threads}"
      )
      set_tests_properties(${_all_tests} PROPERTIES DISABLED TRUE)
    endif ()
  endif ()

  # Timing-sensitive test handling
  set(_timed_tests "")
  if (GT_TIMING_ALL)
    set(_timed_tests "${_all_tests}")
  else ()
    if (GT_TIMING_TESTS)
      foreach (_t IN LISTS GT_TIMING_TESTS)
        list(FIND _all_tests "${_t}" _idx)
        if (_idx GREATER -1)
          list(APPEND _timed_tests "${_t}")
        endif ()
      endforeach ()
    endif ()
    if (GT_TIMING_PATTERNS)
      foreach (_name IN LISTS _all_tests)
        foreach (_rx IN LISTS GT_TIMING_PATTERNS)
          if (_name MATCHES "${_rx}")
            list(APPEND _timed_tests "${_name}")
            break()
          endif ()
        endforeach ()
      endforeach ()
      list(REMOVE_DUPLICATES _timed_tests)
    endif ()
  endif ()

  if (_timed_tests)
    set_property(
      TEST ${_timed_tests}
      APPEND
      PROPERTY LABELS "Timing"
    )
    # Timing tests run serially unless RESOURCE_LOCK was already applied above
    if (NOT GT_RESOURCE_LOCK)
      set_tests_properties(${_timed_tests} PROPERTIES RUN_SERIAL TRUE)
    endif ()
  endif ()

  # Coverage test (when any coverage libraries detected). It runs the whole
  # program: component when any of its tests is, rig when any of them is.
  if (_coverage_libs)
    set(_cov_labels "Coverage")
    if (GT_LABELS)
      list(APPEND _cov_labels ${GT_LABELS})
    endif ()
    if (_component_tests)
      list(APPEND _cov_labels component)
    else ()
      list(APPEND _cov_labels ${_class})
    endif ()
    if (_rig_tests AND NOT "rig" IN_LIST GT_LABELS)
      list(APPEND _cov_labels rig)
    endif ()

    add_test(NAME ${GT_TARGET}_coverage COMMAND $<TARGET_FILE:${GT_TARGET}>)
    set_tests_properties(
      ${GT_TARGET}_coverage
      PROPERTIES LABELS "${_cov_labels}" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" ENVIRONMENT
                 "LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/${GT_TARGET}.coverage.profraw"
    )
  endif ()

  # Summary
  list(LENGTH _all_tests _disc_len)
  list(LENGTH _timed_tests _timed_len)
  set(_req_threads_msg "")
  if (GT_REQUIRES_THREADS)
    set(_req_threads_msg " threads=${GT_REQUIRES_THREADS}")
  endif ()
  list(REMOVE_DUPLICATES _component_tests)
  list(REMOVE_DUPLICATES _rig_tests)
  list(LENGTH _component_tests _component_len)
  list(LENGTH _rig_tests _rig_len)
  if (_class STREQUAL "component")
    set(_component_len ${_disc_len})
  endif ()
  if ("rig" IN_LIST GT_LABELS)
    set(_rig_len ${_disc_len})
  endif ()
  math(EXPR _unit_len "${_disc_len} - ${_component_len}")
  message(STATUS "[Test] target=${GT_TARGET} tests=${_disc_len} timed=${_timed_len} "
                 "lock='${GT_RESOURCE_LOCK}'${_req_threads_msg} unit=${_unit_len} "
                 "component=${_component_len} rig=${_rig_len}"
  )
endfunction ()

# ------------------------------------------------------------------------------
# _vernier_gtest_skip_patterns(<test>...)
#
# Give each GoogleTest case gtest_add_tests registered the skip expression that
# names it, in place of the "\[  SKIPPED \]" gtest_add_tests sets: any skip line
# matches that one, so a case whose failure message quotes a child's output,
# skip line and all, would be reported Skipped and pass the run. The expression
# is the two lines GoogleTest prints for that case alone when it skips, with
# the time ("[  SKIPPED ] Suite.Case (0 ms)") and at a line's end (without the
# time, as under --gtest_print_time=0, and in the run's list of skipped
# cases). The name's regex characters are escaped, and each * of a
# parameterized name matches one part of the name.
# ------------------------------------------------------------------------------
function (_vernier_gtest_skip_patterns)
  foreach (_test IN LISTS ARGN)
    string(REGEX REPLACE "([][+.?()^$\\|])" "\\\\\\1" _name "${_test}")
    string(REPLACE "*" "[^ \n]*" _name "${_name}")
    set_tests_properties(
      "${_test}" PROPERTIES SKIP_REGULAR_EXPRESSION
                            "\\[  SKIPPED \\] ${_name} \\(;\\[  SKIPPED \\] ${_name}\n"
    )
  endforeach ()
endfunction ()

# ------------------------------------------------------------------------------
# vernier_add_test_program(...)
#
# Add a private program that tests run but that is not a GoogleTest program: a
# process fixture or helper with a main of its own, run by a controller that
# its owner registers with add_test beside the target. It is built by
# vernier_add_gtest with NO_REGISTER, without GoogleTest: placed in bin/tests,
# under the same coverage policy, nothing installed, exported or compressed,
# and nothing registered; the options that describe registered tests are
# refused.
#
# Arguments:
#   TARGET          <n>              required
#   SOURCES         <src...>         required
#   CUDA            <cu...>          optional
#   LINK            <libs...>        optional
#   COVERAGE_FOR    <lib_target>     optional (overrides auto-detection)
#   INC             <dir>            optional
#   NO_COVERAGE                      optional flag (skip coverage)
# ------------------------------------------------------------------------------
function (vernier_add_test_program)
  vernier_add_gtest(${ARGN} NO_REGISTER _PLAIN_PROGRAM)
endfunction ()

# ------------------------------------------------------------------------------
# vernier_add_ptest(...)
#
# Add a performance test executable that is built but NOT registered with CTest.
# Performance tests are for benchmarking and should be run manually.
#
# The executable is placed in bin/ptests/ and can be run manually with:
#   ./build/bin/ptests/TestName --csv results.csv
#
# Features:
#   - Output to bin/ptests/
#   - Auto-links benchmarking library if available
#   - Auto-links gperftools if available
#
# Arguments:
#   TARGET          <n>              required
#   SOURCES         <src...>         required
#   CUDA            <cu...>          optional
#   LINK            <libs...>        optional
#   INC             <dir>            optional
# ------------------------------------------------------------------------------
function (vernier_add_ptest)
  # Skip on bare-metal
  if (CMAKE_SYSTEM_NAME STREQUAL "Generic")
    return()
  endif ()

  # Note: LABELS is parsed but ignored (ptests not registered with CTest)
  cmake_parse_arguments(PT "" "TARGET;INC" "SOURCES;CUDA;LINK;LABELS" ${ARGN})
  vernier_require(PT_TARGET PT_SOURCES)

  add_executable(${PT_TARGET})
  target_sources(${PT_TARGET} PRIVATE ${PT_SOURCES})

  if (PT_INC)
    target_include_directories(${PT_TARGET} PRIVATE "${PT_INC}")
  endif ()

  # CUDA sources
  if (PT_CUDA
      AND CUDAToolkit_FOUND
      AND CMAKE_CUDA_COMPILER
  )
    vernier_cuda_sources(${PT_TARGET} FILES ${PT_CUDA})
  endif ()

  # GTest linkage
  if (NOT TARGET GTest::gtest_main)
    find_package(GTest QUIET CONFIG)
    if (NOT TARGET GTest::gtest_main)
      find_package(GTest QUIET)
    endif ()
    if (NOT TARGET GTest::gtest_main)
      message(FATAL_ERROR "vernier_add_ptest: GTest::gtest_main not found")
    endif ()
  endif ()

  # Build link list
  set(_link ${PT_LINK})

  # gperftools (optional)
  find_library(GPERF_PROFILER_LIB NAMES profiler)
  if (GPERF_PROFILER_LIB)
    list(APPEND _link "${GPERF_PROFILER_LIB}")
  endif ()
  # tcmalloc swaps the process allocator: linked only on request
  # (VERNIER_LINK_TCMALLOC), never because the library happens to be installed.
  set(_tcmalloc "off")
  if (VERNIER_LINK_TCMALLOC)
    find_library(GPERF_TCMALLOC_LIB NAMES tcmalloc)
    if (NOT GPERF_TCMALLOC_LIB)
      message(FATAL_ERROR "VERNIER_LINK_TCMALLOC=ON but libtcmalloc was not found")
    endif ()
    list(APPEND _link "${GPERF_TCMALLOC_LIB}")
    set(_tcmalloc "on")
  endif ()

  # Benchmarking library (optional - auto-link if available)
  list(APPEND _link $<$<TARGET_EXISTS:vernier::bench>:vernier::bench>
       $<$<TARGET_EXISTS:bench>:bench>
  )

  target_link_libraries(${PT_TARGET} PRIVATE GTest::gtest_main GTest::gmock ${_link})

  # Output directory
  set_target_properties(
    ${PT_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ptests"
  )

  # RPATH fixup for ptests layout
  set(_ptests_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ptests")
  set(_lib_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  if (NOT _lib_dir)
    set(_lib_dir "${CMAKE_BINARY_DIR}/lib")
  endif ()

  file(RELATIVE_PATH _rel_lib "${_ptests_dir}" "${_lib_dir}")
  string(REPLACE "\\" "/" _rel_lib "${_rel_lib}")

  get_target_property(_cur_rpath ${PT_TARGET} BUILD_RPATH)
  if (NOT _cur_rpath)
    set(_cur_rpath "")
  endif ()

  set_target_properties(
    ${PT_TARGET}
    PROPERTIES BUILD_RPATH "${_cur_rpath};\$ORIGIN/${_rel_lib};${_lib_dir}"
               SKIP_BUILD_RPATH OFF
               BUILD_WITH_INSTALL_RPATH OFF
  )

  set_property(
    TARGET ${PT_TARGET}
    APPEND_STRING
    PROPERTY LINK_FLAGS " -Wl,-rpath-link,${_lib_dir}"
  )

  # NOTE: No gtest_add_tests() - ptests are not registered with CTest

  # Summary
  set(_gperf "off")
  if (GPERF_PROFILER_LIB)
    set(_gperf "on")
  endif ()
  set(_bench "off")
  if (TARGET vernier::bench OR TARGET bench)
    set(_bench "on")
  endif ()
  message(
    STATUS
      "[Perf] target=${PT_TARGET} out=ptests/ benchlib=${_bench} gperftools=${_gperf} tcmalloc=${_tcmalloc} (manual execution only)"
  )
endfunction ()

# ------------------------------------------------------------------------------
# vernier_add_devtest(...)
#
# Add a component test program: GoogleTest cases that drive a whole component
# through its real protocol, too slow for a unit test program. vernier_add_gtest
# builds and registers it, with the same arguments and coverage policy, in
# bin/dtests/, where it also runs by hand:
#   ./build/bin/dtests/TestName --gtest_filter="*Pattern*"
#
# Arguments: those of vernier_add_gtest.
# ------------------------------------------------------------------------------
function (vernier_add_devtest)
  vernier_add_gtest(${ARGN} _COMPONENT)
endfunction ()
