# ==============================================================================
# Unit Tests: bench
#
# Unit tests for benchmarking framework internals.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (14 files)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerReadiness_uTest.cpp"
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
  PRIVATE bench GTest::gtest
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
# Readiness Fixtures
# ------------------------------------------------------------------------------
# Fake tools (POSIX sh templates) that stand in for sudo, kill and the
# profilers' own tools, so the readiness policy, its routing and its error
# reports are tested without privileges or the real tools. The unit tests copy
# them into a private directory and hand the checks a context whose PATH is it.
# ------------------------------------------------------------------------------

if (TARGET TestBench)
  target_compile_definitions(
    TestBench
    PRIVATE VERNIER_READINESS_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/readiness"
  )
endif ()

# ------------------------------------------------------------------------------
# Layout Check Consumers (2)
# ------------------------------------------------------------------------------
# Separately compiled programs that exercise the start-up layout check, the
# header tree one of them is built from, and the tests that run them.
# ------------------------------------------------------------------------------

include("${CMAKE_CURRENT_LIST_DIR}/AbiConsumers.cmake")
