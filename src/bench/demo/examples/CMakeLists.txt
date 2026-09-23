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
#   demo_example_join   - join a list of strings with a separator, two ways
#   demo_example_saxpy  - y = a*x + y as a CPU loop and two ways of driving one
#                         CUDA kernel; built only with the GPU demos
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (2 examples)
# ------------------------------------------------------------------------------

set(CPP_SOURCES_JOIN "${CMAKE_CURRENT_SOURCE_DIR}/join/src/Join.cpp")
set(CPP_SOURCES_SAXPY "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyCpu.cpp")
set(CUDA_SOURCES_SAXPY "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyGpu.cu")

set(UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/join/utst/Join_uTest.cpp")
set(CUDA_UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/utst/Saxpy_uTest.cu")

# ------------------------------------------------------------------------------
# Example Libraries (2)
# ------------------------------------------------------------------------------

add_library(demo_example_join STATIC ${CPP_SOURCES_JOIN})
target_include_directories(demo_example_join PUBLIC "${PROJECT_SOURCE_DIR}")

set(UNIT_TEST_LINK demo_example_join)

# The saxpy header is one API whose GPU half cannot link without the toolkit,
# and nothing CPU-only uses it, so the library and its tests are built with the
# GPU demos or not at all.
if (TARGET bench_cuda)
  add_library(demo_example_saxpy STATIC ${CPP_SOURCES_SAXPY} ${CUDA_SOURCES_SAXPY})
  target_include_directories(demo_example_saxpy PUBLIC "${PROJECT_SOURCE_DIR}")
  list(APPEND UNIT_TEST_SOURCES ${CUDA_UNIT_TEST_SOURCES})
  list(APPEND UNIT_TEST_LINK demo_example_saxpy)
endif ()

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoExamples
  SOURCES
  ${UNIT_TEST_SOURCES}
  LINK
  ${UNIT_TEST_LINK}
  LABELS
  demo
)
