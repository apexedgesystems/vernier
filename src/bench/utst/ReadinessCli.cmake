# ==============================================================================
# ReadinessCli.cmake - The readiness fixture benchmarks and the tests that run them
#
# Included from the bench unit-test CMakeLists.txt. ReadinessFixtureTarget is a
# benchmark with two guarded cases and one case built without the guard;
# ReadinessCustomMainTarget is a benchmark with its own main(); RunStatusTarget
# is a program linked with libbench that ends as its argument says, for how
# bench run reads a benchmark's end under heaptrack. The tests run them under a
# PATH that holds only fake tools (fixtures/readiness/) and check what the
# doctor and a run report for the same request, how the run ends, and what the
# fakes were asked to do. TestBenchReadinessRun's GoogleTest cases start the
# fixtures; each ReadinessCli.<Case> runs ReadinessFixtureTarget through
# ReadinessCli_test.cmake. All are component tests: each runs a benchmark end
# to end.
#
# Targets: ReadinessFixtureTarget, ReadinessCustomMainTarget, RunStatusTarget
#          (process fixtures with their own main, run only by these tests),
#          TestBenchReadinessRun
# Tests:  ReadinessFixtureRun.<Case> (ReadinessFixtureRun_uTest.cpp);
#         ReadinessCli.<Case>, each a run of ReadinessCli_test.cmake
# ==============================================================================

vernier_add_gtest(
  TARGET
  ReadinessFixtureTarget
  SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness/ReadinessFixtureTarget.cpp"
  LINK
  bench
  NO_REGISTER
)

# A benchmark with its own main(), written as the advanced guide shows.
vernier_add_gtest(
  TARGET
  ReadinessCustomMainTarget
  SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness/ReadinessCustomMain.cpp"
  LINK
  bench
  NO_REGISTER
)

# A program that ends as its first argument says (RunStatusTarget.cpp).
vernier_add_test_program(
  TARGET RunStatusTarget SOURCES "${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness/RunStatusTarget.cpp"
  LINK bench
)

# The test helper builds nothing for bare metal; the tests go with it.
if (NOT TARGET ReadinessFixtureTarget
    OR NOT TARGET ReadinessCustomMainTarget
    OR NOT TARGET RunStatusTarget
)
  return()
endif ()

# The GoogleTest cases find the fixtures and the fakes by these paths.
vernier_add_gtest(
  TARGET
  TestBenchReadinessRun
  SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/ReadinessFixtureRun_uTest.cpp"
  LINK
  bench
  LABELS
  benchmarking
  readiness
  CLASS
  component
)
target_compile_definitions(
  TestBenchReadinessRun
  PRIVATE VERNIER_READINESS_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness"
          VERNIER_READINESS_FIXTURE_TARGET="$<TARGET_FILE:ReadinessFixtureTarget>"
          VERNIER_READINESS_CUSTOM_MAIN_TARGET="$<TARGET_FILE:ReadinessCustomMainTarget>"
          VERNIER_RUN_STATUS_TARGET="$<TARGET_FILE:RunStatusTarget>"
          # Where the tools build puts the bench CLI; the case that runs a benchmark through
          # bench run skips where it is not built.
          VERNIER_BENCH_CLI="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tools/rust/bench"
          VERNIER_READINESS_HEAP_BUILT=$<BOOL:${VERNIER_LINK_TCMALLOC}>
)
add_dependencies(
  TestBenchReadinessRun ReadinessFixtureTarget ReadinessCustomMainTarget RunStatusTarget
)

set(_readiness_cli_cases
    DoctorLabel
    DoctorJsonKeys
    SelectedRowMatchesRun
    BpfNoOptInNeverCallsSudo
    BpfConflictNamesWinner
    BpfInvalidValueLaunchesNothing
    BpfProbeRefusedRunDecides
    BpfRunAllowedProbeRefused
    BpfRunStopsThroughRoute
    BpfRunReportsRefusedStop
    BpfBundledScriptFromAnyDirectory
    BpfScriptsFlagSelectsTheDirectory
    OffcpuCurrentUserRun
    OffcpuRunAllowedProbeRefused
    OffcpuRefusedStopLeavesNoTracer
    OffcpuNoStacksClaimWhenKilled
    PerfLaunchesTheResolvedPath
    PerfBrokenNeverLaunched
    PerfDeniedMatchesDoctor
    GperfAnalyzerFoundIsRun
    GperfAnalyzerMissingIsAnalysisError
    GperfAnalyzerBrokenIsAnalysisError
    GperfAnalyzerFailsOnTheProfile
    GperfWithoutAnalyzeNeedsNone
    GperfHeapWithoutSupport
)

foreach (_case IN LISTS _readiness_cli_cases)
  add_test(
    NAME ReadinessCli.${_case}
    COMMAND
      "${CMAKE_COMMAND}" -DTARGET=$<TARGET_FILE:ReadinessFixtureTarget>
      -DFIXTURES=${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness -DCASE=${_case}
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/readiness_cli/${_case}
      -DHEAP_BUILT=${VERNIER_LINK_TCMALLOC} -P "${CMAKE_CURRENT_LIST_DIR}/ReadinessCli_test.cmake"
  )
  set_tests_properties(
    ReadinessCli.${_case} PROPERTIES LABELS "benchmarking;readiness;component"
                                     SKIP_REGULAR_EXPRESSION "READINESS_CLI_SKIPPED"
  )
endforeach ()
