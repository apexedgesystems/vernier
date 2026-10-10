# ==============================================================================
# Unit Tests: demo_example_filter
#
# Tests:
#   TestDemoFilter  - both versions' answers, and the branch misses walkthrough
#                     02 reads from them
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoFilter
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/Filter_uTest.cpp"
  LINK
  demo_example_filter
  LABELS
  demo
)
