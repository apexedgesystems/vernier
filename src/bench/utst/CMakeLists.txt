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
# GPU Unit Tests (1 file)
# ------------------------------------------------------------------------------
# What the GPU harness measures and publishes. They run a kernel, so they need
# both the CUDA toolkit (to build) and a device (to run; they skip without
# one) and exist only where libbench_cuda does.
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

# ------------------------------------------------------------------------------
# Callgrind Window Probe (2 tests)
# ------------------------------------------------------------------------------
# A benchmark with work before, inside and after its measured window, run under
# valgrind by CallgrindWindowProbe_test.cmake: the wrap the callgrind backend
# prints must record the window and none of the work around it, the wrap
# `bench run` uses the whole process. A test fixture, never installed; the
# tests skip without valgrind.
# ------------------------------------------------------------------------------

vernier_add_app(
  NAME
  CallgrindWindowProbe
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowProbe.cpp"
  LINK
  bench
  NO_INSTALL
  NO_UPX
)

if (TARGET CallgrindWindowProbe)
  foreach (_case Hint Runner)
    string(TOLOWER "${_case}" _mode)
    add_test(
      NAME CallgrindWindow${_case}Test
      COMMAND
        "${CMAKE_COMMAND}" -DPROBE=$<TARGET_FILE:CallgrindWindowProbe> -DCASE=${_mode}
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/callgrind_window_${_mode} -P
        "${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowProbe_test.cmake"
    )
    set_tests_properties(
      CallgrindWindow${_case}Test PROPERTIES LABELS "benchmarking;callgrind"
                                             SKIP_REGULAR_EXPRESSION "-- SKIPPED: "
    )
  endforeach ()
endif ()
