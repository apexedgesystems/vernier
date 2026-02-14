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
# Arguments:
#   TARGET          <n>              required
#   SOURCES         <src...>         required
#   CUDA            <cu...>          optional
#   LINK            <libs...>        optional
#   COVERAGE_FOR    <lib_target>     optional (overrides auto-detection)
#   INC             <dir>            optional
#   LABELS          <labels...>      optional
#   WORKING_DIR     <dir>            optional
#   RESOURCE_LOCK   <n>              optional
#   NO_COVERAGE                      optional flag (skip coverage)
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

  cmake_parse_arguments(
    GT "TIMING_ALL;NO_COVERAGE"
    "TARGET;INC;WORKING_DIR;RESOURCE_LOCK;COVERAGE_FOR;REQUIRES_THREADS"
    "SOURCES;CUDA;LINK;LABELS;TIMING_TESTS;TIMING_PATTERNS" ${ARGN}
  )
  vernier_require(GT_TARGET GT_SOURCES)

  add_executable(${GT_TARGET})
  target_sources(${GT_TARGET} PRIVATE ${GT_SOURCES})

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
  if (NOT TARGET GTest::gtest_main)
    find_package(GTest QUIET CONFIG)
    if (NOT TARGET GTest::gtest_main)
      find_package(GTest QUIET)
    endif ()
    if (NOT TARGET GTest::gtest_main)
      message(FATAL_ERROR "vernier_add_gtest: GTest::gtest_main not found")
    endif ()
  endif ()

  target_link_libraries(${GT_TARGET} PRIVATE GTest::gtest_main GTest::gmock ${GT_LINK})

  set_target_properties(
    ${GT_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
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

  # Base labels
  if (GT_LABELS)
    set_property(
      TEST ${_all_tests}
      APPEND
      PROPERTY LABELS "${GT_LABELS}"
    )
  endif ()

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

  # Coverage test (when any coverage libraries detected)
  if (_coverage_libs)
    set(_cov_labels "Coverage")
    if (GT_LABELS)
      list(APPEND _cov_labels ${GT_LABELS})
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
  message(STATUS "[Test] target=${GT_TARGET} tests=${_disc_len} timed=${_timed_len} "
                 "lock='${GT_RESOURCE_LOCK}'${_req_threads_msg}"
  )
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
  find_library(GPERF_TCMALLOC_LIB NAMES tcmalloc)
  if (GPERF_PROFILER_LIB)
    list(APPEND _link "${GPERF_PROFILER_LIB}")
  endif ()
  if (GPERF_TCMALLOC_LIB)
    list(APPEND _link "${GPERF_TCMALLOC_LIB}")
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
  if (GPERF_PROFILER_LIB OR GPERF_TCMALLOC_LIB)
    set(_gperf "on")
  endif ()
  set(_bench "off")
  if (TARGET vernier::bench OR TARGET bench)
    set(_bench "on")
  endif ()
  message(
    STATUS
      "[Perf] target=${PT_TARGET} out=ptests/ benchlib=${_bench} gperftools=${_gperf} (manual execution only)"
  )
endfunction ()

# ------------------------------------------------------------------------------
# vernier_add_devtest(...)
#
# Add a development/integration test executable that is built but NOT registered
# with CTest. These tests require manual execution (e.g., external data files,
# special hardware).
#
# The executable is placed in bin/dtests/ and can be run manually with:
#   ./build/bin/dtests/TestName --gtest_filter="*Pattern*"
#
# Arguments:
#   TARGET          <n>              required
#   SOURCES         <src...>         required
#   CUDA            <cu...>          optional
#   LINK            <libs...>        optional
#   INC             <dir>            optional
# ------------------------------------------------------------------------------
function (vernier_add_devtest)
  # Skip on bare-metal
  if (CMAKE_SYSTEM_NAME STREQUAL "Generic")
    return()
  endif ()

  cmake_parse_arguments(DT "" "TARGET;INC" "SOURCES;CUDA;LINK" ${ARGN})
  vernier_require(DT_TARGET DT_SOURCES)

  add_executable(${DT_TARGET})
  target_sources(${DT_TARGET} PRIVATE ${DT_SOURCES})

  if (DT_INC)
    target_include_directories(${DT_TARGET} PRIVATE "${DT_INC}")
  endif ()

  # CUDA sources
  if (DT_CUDA
      AND CUDAToolkit_FOUND
      AND CMAKE_CUDA_COMPILER
  )
    vernier_cuda_sources(${DT_TARGET} FILES ${DT_CUDA})
  endif ()

  # GTest linkage
  if (NOT TARGET GTest::gtest_main)
    find_package(GTest QUIET CONFIG)
    if (NOT TARGET GTest::gtest_main)
      find_package(GTest QUIET)
    endif ()
    if (NOT TARGET GTest::gtest_main)
      message(FATAL_ERROR "vernier_add_devtest: GTest::gtest_main not found")
    endif ()
  endif ()

  target_link_libraries(${DT_TARGET} PRIVATE GTest::gtest_main GTest::gmock ${DT_LINK})

  set_target_properties(
    ${DT_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dtests"
  )

  # NOTE: No gtest_add_tests() - dev tests are not registered with CTest

  message(STATUS "[Dev] target=${DT_TARGET} out=dtests/ (manual execution only)")
endfunction ()
