# ==============================================================================
# Unit Tests: bench
#
# Unit tests for benchmarking framework internals.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (13 files)
# ------------------------------------------------------------------------------

set(UNIT_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfStats_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfAbi_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfMetadata_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfTargetTime_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfCsv_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfConfig_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfRegistry_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfListener_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfWatchdog_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerRegistry_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerArtifactDir_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerEnv_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerTcmalloc_uTest.cpp"
)

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestBench
  SOURCES
  ${UNIT_TEST_SOURCES}
  LINK
  bench
  LABELS
  benchmarking
  TIMING_PATTERNS
  "^PerfCaseTargetTimeTest\\."
)

# ------------------------------------------------------------------------------
# Consumer Warning Check
# ------------------------------------------------------------------------------
# Compiles a consumer-style benchmark file against the public headers with
# warnings as errors. The C library marks write(2) and similar calls
# warn_unused_result only in optimized, fortified builds, which is how
# consumers ship, so the check pins that configuration whatever the build type.
# Object library: compiled with every build, never linked or run.
# ------------------------------------------------------------------------------

add_library(
  BenchConsumerWarningCheck OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/ConsumerTranslationUnit.cpp"
)
target_link_libraries(
  BenchConsumerWarningCheck
  PRIVATE bench gtest
)
target_compile_options(
  BenchConsumerWarningCheck
  PRIVATE -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -Werror
)

# ------------------------------------------------------------------------------
# Build Logic Tests (1)
# ------------------------------------------------------------------------------
# Script-mode tests of configure logic; they need no toolkit and no GPU, so the
# CUDA architecture rules are checked on every lane, including CPU-only ones.
# ------------------------------------------------------------------------------

add_test(NAME CudaArchitecturesResolveTest
         COMMAND "${CMAKE_COMMAND}" -P
                 "${PROJECT_SOURCE_DIR}/cmake/vernier/utst/CudaArchitectures_test.cmake"
)
set_tests_properties(CudaArchitecturesResolveTest PROPERTIES LABELS "benchmarking;cmake")

# ------------------------------------------------------------------------------
# Layout Check Consumers (2)
# ------------------------------------------------------------------------------
# The start-up layout check, exercised by separately compiled programs rather
# than by numbers handed to the check: one AbiConsumer built from the tree's
# headers, one from a copy in which PerfConfig has one more member. Both link
# the same libbench.
# ------------------------------------------------------------------------------

set(_abi_headers_root "${CMAKE_CURRENT_BINARY_DIR}/abi_mismatch_headers")
file(GLOB _bench_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/bench/inc/*")
add_custom_command(
  OUTPUT "${_abi_headers_root}/stamp"
  COMMAND "${CMAKE_COMMAND}" -DSRC=${PROJECT_SOURCE_DIR} -DDST=${_abi_headers_root} -P
          "${CMAKE_CURRENT_SOURCE_DIR}/AbiMismatchHeaders.cmake"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_abi_headers_root}/stamp"
  DEPENDS ${_bench_headers} "${CMAKE_CURRENT_SOURCE_DIR}/AbiMismatchHeaders.cmake"
  COMMENT "Generating the mismatched header tree for AbiConsumerMismatch"
  VERBATIM
)
add_custom_target(
  BenchAbiMismatchHeaders
  DEPENDS "${_abi_headers_root}/stamp"
)

add_executable(AbiConsumerMatch "${CMAKE_CURRENT_SOURCE_DIR}/AbiConsumer.cpp")
target_link_libraries(
  AbiConsumerMatch
  PRIVATE bench
)

add_executable(AbiConsumerMismatch "${CMAKE_CURRENT_SOURCE_DIR}/AbiConsumer.cpp")
target_link_libraries(
  AbiConsumerMismatch
  PRIVATE bench
)
target_include_directories(AbiConsumerMismatch BEFORE PRIVATE "${_abi_headers_root}")
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/AbiConsumer.cpp" PROPERTIES OBJECT_DEPENDS
                                                           "${_abi_headers_root}/stamp"
)
add_dependencies(AbiConsumerMismatch BenchAbiMismatchHeaders)

foreach (_case Match Mismatch)
  string(TOLOWER "${_case}" _expect)
  add_test(
    NAME AbiConsumer${_case}Test
    COMMAND
      "${CMAKE_COMMAND}" -DCONSUMER=$<TARGET_FILE:AbiConsumer${_case}> -DEXPECT=${_expect}
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/abi_consumer_${_expect} -P
      "${CMAKE_CURRENT_SOURCE_DIR}/AbiConsumer_test.cmake"
  )
  set_tests_properties(AbiConsumer${_case}Test PROPERTIES LABELS "benchmarking;abi")
endforeach ()
