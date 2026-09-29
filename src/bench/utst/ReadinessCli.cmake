# ==============================================================================
# ReadinessCli.cmake - The readiness fixture benchmark and its command-line tests
#
# Included from the bench unit-test CMakeLists.txt. ReadinessFixtureTarget is a
# benchmark with two guarded cases and one case built without the guard;
# ReadinessCustomMainTarget is a benchmark with its own main(). Each test runs
# one under a PATH that holds only fake tools (fixtures/readiness/) and checks
# what the doctor and a run report for the same request, how the run ends, and
# what the fakes were asked to do.
#
# Targets: ReadinessFixtureTarget, ReadinessCustomMainTarget (test fixtures:
#          never installed, no UPX copy)
# Tests:  ReadinessCli.<Case>, each a run of ReadinessCli_test.cmake
# ==============================================================================

vernier_add_app(
  NAME
  ReadinessFixtureTarget
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness/ReadinessFixtureTarget.cpp"
  LINK
  bench
  GTest::gtest
  NO_INSTALL
  NO_UPX
)

# A benchmark with its own main(), written as the advanced guide shows.
vernier_add_app(
  NAME
  ReadinessCustomMainTarget
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness/ReadinessCustomMain.cpp"
  LINK
  bench
  GTest::gtest
  NO_INSTALL
  NO_UPX
)

# The helper skips apps on platforms without POSIX; the tests go with it.
if (NOT TARGET ReadinessFixtureTarget OR NOT TARGET ReadinessCustomMainTarget)
  return()
endif ()

set(_readiness_cli_cases
    DoctorLabel
    DoctorJsonKeys
    SelectedRowMatchesRun
    RunUnknownProfilerFails
    UnprofiledRunExitsZero
    CuptiNeedsNoProfile
    NoProfilerCreatedNotice
    CustomMainReturnsTheRunStatus
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
      -DCUSTOM_TARGET=$<TARGET_FILE:ReadinessCustomMainTarget>
      -DFIXTURES=${CMAKE_CURRENT_LIST_DIR}/fixtures/readiness -DCASE=${_case}
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/readiness_cli/${_case}
      -DHEAP_BUILT=${VERNIER_LINK_TCMALLOC} -P "${CMAKE_CURRENT_LIST_DIR}/ReadinessCli_test.cmake"
  )
  set_tests_properties(
    ReadinessCli.${_case} PROPERTIES LABELS "benchmarking;readiness" SKIP_REGULAR_EXPRESSION
                                     "READINESS_CLI_SKIPPED"
  )
endforeach ()
