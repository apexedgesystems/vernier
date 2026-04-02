# ==============================================================================
# Unit Tests: monitor
#
# Unit tests for runtime performance monitor internals.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (4 files)
# ------------------------------------------------------------------------------

set(UNIT_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/MonitorTag_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MonitorQueue_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Monitor_uTest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MonitorSummary_uTest.cpp"
)

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestMonitor
  SOURCES
  ${UNIT_TEST_SOURCES}
  LINK
  monitor
  LABELS
  monitor
)
