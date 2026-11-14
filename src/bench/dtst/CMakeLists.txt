# ==============================================================================
# Component Tests: bench
#
# Backends and readiness processes driven through their real protocol against
# stand-in tools: started, waited on, signalled, stopped and reaped, within the
# bounds a run uses. They take minutes, not seconds, so they are a program of
# their own, beside the unit tests (../utst).
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (3 files)
# ------------------------------------------------------------------------------

set(COMPONENT_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerBpfLifecycle_dTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerPerfLifecycle_dTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfilerReadinessProcesses_dTest.cpp"
)

# ------------------------------------------------------------------------------
# Component Tests
# ------------------------------------------------------------------------------
# The fake tools and the helpers they share with the unit tests come from
# bench_test_support, which ../utst owns.
# ------------------------------------------------------------------------------

vernier_add_devtest(
  TARGET
  TestBenchComponent
  SOURCES
  ${COMPONENT_TEST_SOURCES}
  LINK
  bench
  LABELS
  benchmarking
)

if (TARGET TestBenchComponent)
  target_link_libraries(TestBenchComponent PRIVATE bench_test_support)
endif ()
