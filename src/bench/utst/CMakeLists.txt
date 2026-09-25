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
# Host Test on a Stand-in CUPTI (1)
# ------------------------------------------------------------------------------
# TestBenchCuptiDecision: the real CuptiCollector.cu, compiled into the test
# against a counting stand-in for the CUPTI calls it makes (fake_cupti/), so
# every configuration, CPU-only included, checks that the collector applies the
# CUPTI decision before it registers: a setting that stands it down, or one that
# is rejected, registers nothing.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET TestBenchCuptiDecision SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/CuptiCollectorDecision_uTest.cpp" LABELS benchmarking
)
target_include_directories(
  TestBenchCuptiDecision BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_cupti"
                                        "${PROJECT_SOURCE_DIR}"
)

# ------------------------------------------------------------------------------
# GPU Unit Tests (2 files)
# ------------------------------------------------------------------------------
# TestBenchGpu: what the GPU harness measures and publishes. Its tests run a
# kernel, so they need the CUDA toolkit (to build) and a device (to run; they
# skip without one). TestBenchNsight: the Nsight backend, which lives in
# libbench_cuda but makes no CUDA call, so it needs the toolkit and no device.
# Both exist only where libbench_cuda does.
# ------------------------------------------------------------------------------

if (TARGET bench_cuda)
  vernier_add_gtest(
    TARGET
    TestBenchGpu
    SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfGpuHarness_uTest.cu"
    LINK
    bench
    bench_cuda
    LABELS
    benchmarking
    gpu
    TIMING_ALL
  )
  vernier_add_gtest(
    TARGET
    TestBenchNsight
    SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerNsight_uTest.cpp"
    LINK
    bench
    bench_cuda
    LABELS
    benchmarking
    gpu
  )
endif ()

# ------------------------------------------------------------------------------
# Consumer Warning Check
# ------------------------------------------------------------------------------
# Compiles a consumer-style benchmark file against the public headers with
# warnings as errors. The C library marks write(2) and similar calls
# warn_unused_result only in optimized, fortified builds, which is how
# consumers ship, so the check pins that configuration whatever the build type.
# It uses bench alone, as a consumer does: GoogleTest's headers reach it through
# bench's usage requirements. Object library: compiled with every build, never
# linked or run.
# ------------------------------------------------------------------------------

add_library(
  BenchConsumerWarningCheck OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/ConsumerTranslationUnit.cpp"
)
target_link_libraries(
  BenchConsumerWarningCheck
  PRIVATE bench
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
# Separately compiled programs that exercise the start-up layout check, the
# header tree one of them is built from, and the tests that run them.
# ------------------------------------------------------------------------------

include("${CMAKE_CURRENT_LIST_DIR}/AbiConsumers.cmake")

# ------------------------------------------------------------------------------
# Installed Package Consumers (1)
# ------------------------------------------------------------------------------
# An install of this tree used by a project outside it, through
# find_package(vernier) alone.
# ------------------------------------------------------------------------------

include("${CMAKE_CURRENT_LIST_DIR}/InstallConsumers.cmake")
