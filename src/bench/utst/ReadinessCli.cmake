# ==============================================================================
# ReadinessCli.cmake - The readiness fixture benchmarks and the tests that run them
#
# Included from the bench unit-test CMakeLists.txt. ReadinessFixtureTarget is a
# benchmark with two guarded cases and one case built without the guard;
# ReadinessCustomMainTarget is a benchmark with its own main(). The tests run
# them under a PATH that holds only fake tools (fixtures/readiness/) and check
# what the doctor and a run report for the same request, how the run ends, and
# what the fakes were asked to do. TestBenchReadinessRun's GoogleTest cases
# start both fixtures; each ReadinessCli.<Case> runs ReadinessFixtureTarget
# through ReadinessCli_test.cmake.
#
# Targets: ReadinessFixtureTarget, ReadinessCustomMainTarget (process fixtures
#          with their own main, run only by these tests), TestBenchReadinessRun
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

# The test helper builds nothing for bare metal; the tests go with it.
if (NOT TARGET ReadinessFixtureTarget OR NOT TARGET ReadinessCustomMainTarget)
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
)
target_compile_definitions(
  TestBenchReadinessRun
  PRIVATE VERNIER_READINESS_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness"
          VERNIER_READINESS_FIXTURE_TARGET="$<TARGET_FILE:ReadinessFixtureTarget>"
          VERNIER_READINESS_CUSTOM_MAIN_TARGET="$<TARGET_FILE:ReadinessCustomMainTarget>"
          VERNIER_READINESS_HEAP_BUILT=$<BOOL:${VERNIER_LINK_TCMALLOC}>
)
add_dependencies(TestBenchReadinessRun ReadinessFixtureTarget ReadinessCustomMainTarget)

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
    ReadinessCli.${_case} PROPERTIES LABELS "benchmarking;readiness" SKIP_REGULAR_EXPRESSION
                                     "READINESS_CLI_SKIPPED"
  )
endforeach ()
