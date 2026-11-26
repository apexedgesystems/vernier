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
# FilterBranchTest's cases count branch misses with the machine's hardware
# counters, skipping where they cannot: component tests, labelled rig.
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
  COMPONENT_PATTERNS
  "^FilterBranchTest\\."
  RIG_PATTERNS
  "^FilterBranchTest\\."
)
