# ==============================================================================
# Unit Tests: bench
#
# Unit tests for benchmarking framework internals.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (4 files)
# ------------------------------------------------------------------------------

set(UNIT_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfStats_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfCsv_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfConfig_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerfRegistry_uTest.cpp"
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
)
