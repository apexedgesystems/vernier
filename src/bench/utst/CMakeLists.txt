# ==============================================================================
# Unit Tests: bench
#
# Unit tests for benchmarking framework internals.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (15 files)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerReadinessChecks_uTest.cpp"
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
# Host Tests of the CUPTI Collector (2)
# ------------------------------------------------------------------------------
# The real CuptiCollector.cu, compiled into each test, so every configuration,
# CPU-only included, checks it. TestBenchCuptiDecision builds it with CUPTI,
# against a counting stand-in for the CUPTI calls it makes (fake_cupti/): the
# collector applies the CUPTI decision before it registers, and a setting that
# stands it down, or one that is rejected, registers nothing.
# TestBenchCuptiAbsent builds it as a GPU build without CUPTI does
# (COMPAT_CUPTI_AVAILABLE=0): it compiles, never collects and takes the same
# decision. Both link bench, which defines what the decision is built from:
# the shared Boolean parser, the readiness context and the report formatter.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestBenchCuptiDecision
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/CuptiCollectorDecision_uTest.cpp"
  LINK
  bench
  LABELS
  benchmarking
)
target_include_directories(
  TestBenchCuptiDecision BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_cupti"
                                        "${PROJECT_SOURCE_DIR}"
)
target_compile_definitions(TestBenchCuptiDecision PRIVATE COMPAT_CUPTI_AVAILABLE=1)

vernier_add_gtest(
  TARGET
  TestBenchCuptiAbsent
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/CuptiCollectorAbsent_uTest.cpp"
  LINK
  bench
  LABELS
  benchmarking
)
target_include_directories(TestBenchCuptiAbsent BEFORE PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(TestBenchCuptiAbsent PRIVATE COMPAT_CUPTI_AVAILABLE=0)

# ------------------------------------------------------------------------------
# Host Tests of the NVML Readings (2)
# ------------------------------------------------------------------------------
# The GPU harness's NVML readings (src/bench/src/NvmlTelemetry.hpp), compiled
# into each test, so every configuration, CPU-only included, checks them.
# TestBenchNvmlTelemetry builds them with NVML, against a scripted stand-in for
# the NVML calls they make (fake_nvml/): a cell is filled only from readings
# NVML reported, and the statement names the rest. TestBenchNvmlAbsent builds
# them as a GPU build without NVML does (COMPAT_NVML_AVAILABLE=0): they compile
# without any nvml.h and say the build has no NVML.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestBenchNvmlTelemetry
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/NvmlTelemetry_uTest.cpp"
  LINK
  bench
  LABELS
  benchmarking
)
target_include_directories(
  TestBenchNvmlTelemetry BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_nvml"
                                        "${PROJECT_SOURCE_DIR}"
)
target_compile_definitions(TestBenchNvmlTelemetry PRIVATE COMPAT_NVML_AVAILABLE=1)

vernier_add_gtest(
  TARGET
  TestBenchNvmlAbsent
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/NvmlTelemetryAbsent_uTest.cpp"
  LINK
  bench
  LABELS
  benchmarking
)
target_include_directories(TestBenchNvmlAbsent BEFORE PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(TestBenchNvmlAbsent PRIVATE COMPAT_NVML_AVAILABLE=0)

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
# Readiness Fixtures
# ------------------------------------------------------------------------------
# Fake tools (POSIX sh templates) that stand in for sudo, kill and the
# profilers' own tools, so the readiness policy, its routing and its error
# reports are tested without privileges or the real tools. The unit tests copy
# them into a private directory and hand the checks a context whose PATH is it;
# the command-line tests run a fixture benchmark with that PATH.
# ------------------------------------------------------------------------------

if (TARGET TestBench)
  target_compile_definitions(
    TestBench
    PRIVATE VERNIER_READINESS_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/readiness"
  )
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/ReadinessCli.cmake")

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
# tests skip without valgrind, where valgrind cannot read the probe and says
# so, and in a build with the address sanitizer, which the driver learns from
# the build's SANITIZER setting.
# ------------------------------------------------------------------------------

set(_probe_sanitizer "")
if (NOT CMAKE_CROSSCOMPILING)
  set(_probe_sanitizer "${SANITIZER}")
endif ()

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
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/callgrind_window_${_mode}
        -DPROBE_SANITIZER=${_probe_sanitizer} -P
        "${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowProbe_test.cmake"
    )
    set_tests_properties(
      CallgrindWindow${_case}Test PROPERTIES LABELS "benchmarking;callgrind"
                                             SKIP_REGULAR_EXPRESSION "-- SKIPPED: "
    )
  endforeach ()
endif ()

# ------------------------------------------------------------------------------
# Callgrind Window Probe, Instruction Total (1 test)
# ------------------------------------------------------------------------------
# The window tests' driver against a stand-in valgrind that writes a profile
# with every phase named and a chosen "totals:" line: the driver must pass a
# positive total and fail a zero, a missing and a malformed one. Needs neither
# valgrind nor the probe.
# ------------------------------------------------------------------------------

add_test(
  NAME CallgrindWindowTotalsTest
  COMMAND
    "${CMAKE_COMMAND}" -DDRIVER=${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowProbe_test.cmake
    -DFAKE=${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowTotals_fake_valgrind.sh
    -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/callgrind_window_totals -P
    "${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowTotals_test.cmake"
)
set_tests_properties(CallgrindWindowTotalsTest PROPERTIES LABELS "benchmarking;callgrind")

# ------------------------------------------------------------------------------
# Callgrind Window Probe, Startup Fault (1 test)
# ------------------------------------------------------------------------------
# The probe with StartupFaultUnderValgrind.cpp linked in: it runs plainly, and
# under valgrind it aborts before its tests start. The hint test's driver must
# fail on it and say how the run ended. The test passes on that failure, over
# a run that shows the fixture's abort, and skips only where valgrind is
# missing or stops reading debug information, or, in a build with the address
# sanitizer only, on the driver's skip for that build; any other skip would
# mean the driver hides a real startup fault as a valgrind limitation.
# ------------------------------------------------------------------------------

vernier_add_app(
  NAME
  CallgrindWindowProbeStartupFault
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowProbe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/StartupFaultUnderValgrind.cpp"
  LINK
  bench
  NO_INSTALL
  NO_UPX
)

if (TARGET CallgrindWindowProbeStartupFault)
  add_test(
    NAME CallgrindWindowStartupFaultTest
    COMMAND
      "${CMAKE_COMMAND}" -DPROBE=$<TARGET_FILE:CallgrindWindowProbeStartupFault> -DCASE=hint
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/callgrind_window_startup_fault
      -DPROBE_SANITIZER=${_probe_sanitizer} -P
      "${CMAKE_CURRENT_LIST_DIR}/CallgrindWindowProbe_test.cmake"
  )
  # It passes only on the fatal error CMake prints for the driver's startup
  # failure (CMake wraps its text, so words may be split across lines), after
  # the run's output shows the fixture's own abort: the same words printed any
  # other way, a run that valgrind or a sanitizer stopped first, or a skip, do
  # not pass it. It skips only on evidence that is not the driver's wording:
  # "valgrind is not installed", decided before anything runs; valgrind's own
  # words when it stops reading debug information, which the driver prints as
  # valgrind wrote them; and, in a build with the address sanitizer only, the
  # driver's skip for that build.
  set(_fault_skips
      "-- SKIPPED: valgrind is not installed"
      "Valgrind: debuginfo reader: [^\n]*\n[^\n]*Valgrind: I can't recover\\.  Giving up\\."
      "valgrind: m_debuginfo/[^\n]*: Assertion '[^\n]*' failed\\."
  )
  if (_probe_sanitizer STREQUAL "asan")
    list(APPEND _fault_skips
         "-- SKIPPED: the probe is built with the address sanitizer \\(SANITIZER=asan\\)"
    )
  endif ()
  set_tests_properties(
    CallgrindWindowStartupFaultTest
    PROPERTIES
      LABELS
      "benchmarking;callgrind"
      PASS_REGULAR_EXPRESSION
      "StartupFaultUnderValgrind: aborting before the tests start, as intended.*CMake Error at [^\n]*\\(message\\):[ \n]+CallgrindWindowProbe[ \n]+\\(hint\\):[ \n]+the[ \n]+probe's[ \n]+tests[ \n]+did[ \n]+not[ \n]+start[ \n]+under[ \n]+valgrind"
      SKIP_REGULAR_EXPRESSION
      "${_fault_skips}"
  )
endif ()

# ------------------------------------------------------------------------------
# Start Gate Under Helgrind (1 program)
# ------------------------------------------------------------------------------
# contentionRun's start gate spins on two atomic flags, which helgrind cannot
# see as synchronisation; the gate marks them unchecked where valgrind's
# helgrind.h is available. The program runs itself under helgrind on two probe
# cases: workers that add under a mutex must report no error, workers that add
# without one must still report their race. It learns a build it cannot run
# under valgrind from the build's SANITIZER setting. The racy probe skips
# itself outside valgrind, and CTest reports that as skipped; each skip
# expression names its own test, whose failure messages quote the probe runs.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestStartGateHelgrind
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/StartGateHelgrind_uTest.cpp"
  LINK
  bench
  LABELS
  benchmarking
  helgrind
)

if (SANITIZER MATCHES "^(asan|tsan)$" AND NOT CMAKE_CROSSCOMPILING)
  target_compile_definitions(TestStartGateHelgrind PRIVATE START_GATE_SANITIZER="${SANITIZER}")
endif ()

foreach (_start_gate_test StartGateProbe.RacyWorkers StartGateHelgrindTest.GateReportsNothing
                          StartGateHelgrindTest.WorkerRaceIsStillReported
)
  string(REPLACE "." "\\." _start_gate_pattern "${_start_gate_test}")
  set_tests_properties(
    ${_start_gate_test} PROPERTIES SKIP_REGULAR_EXPRESSION
                                   "\\[  SKIPPED \\] ${_start_gate_pattern}"
  )
endforeach ()
