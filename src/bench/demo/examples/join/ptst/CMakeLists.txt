# ==============================================================================
# Performance Tests: demo_example_join
#
# Rig checks, built into bin/ptests and never registered with CTest, because a
# timing or a profile's samples belong to the machine:
#   JoinSpeedup             - joinV0 at least three times slower than joinV1 at
#                             demo 01's 1,000 words, the effect walkthrough 01
#                             shows
#   JoinProfileAttribution  - gperftools puts V0's time in what joinV0 calls
#                             and V1's in joinV1's own code, the reading
#                             walkthrough 03 teaches (GperfProfiles.hpp holds
#                             its profiling plumbing)
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Performance Tests (2)
# ------------------------------------------------------------------------------

vernier_add_ptest(
  TARGET JoinSpeedup SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/JoinSpeedup_pTest.cpp" LINK
  demo_example_join
)

vernier_add_ptest(
  TARGET JoinProfileAttribution SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/JoinProfileAttribution_pTest.cpp" LINK demo_example_join
)
