# ==============================================================================
# Demo Examples
#
# Small pieces of code the walkthroughs measure. Each example is a directory
# <name>/{inc,src,utst}: one library the demos link, and unit tests that hold
# its versions to the same answers. Examples are teaching material: they are
# built with the demos and are never installed or exported, so they use plain
# add_library rather than vernier_add_library.
#
# Libraries:
#   demo_example_join  - join a list of strings with a separator, two ways
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (1 example)
# ------------------------------------------------------------------------------

set(CPP_SOURCES_JOIN "${CMAKE_CURRENT_SOURCE_DIR}/join/src/Join.cpp")

set(UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/join/utst/Join_uTest.cpp")

# ------------------------------------------------------------------------------
# Example Libraries (1)
# ------------------------------------------------------------------------------

add_library(demo_example_join STATIC ${CPP_SOURCES_JOIN})
target_include_directories(demo_example_join PUBLIC "${PROJECT_SOURCE_DIR}")
# Line tables in every build type, so a profiler can name the example's source
# lines in an optimized build; -g adds debug sections and changes no code.
target_compile_options(
  demo_example_join
  PRIVATE -g
)

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoExamples
  SOURCES
  ${UNIT_TEST_SOURCES}
  LINK
  demo_example_join
  LABELS
  demo
)
