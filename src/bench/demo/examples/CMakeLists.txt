# ==============================================================================
# Shared Demo Examples: bench
#
# One library per example, shared by the demos and walkthroughs that measure
# it. An example owns the workload and its unit tests; a demo owns how the
# workload is measured. Examples are build-tree only: they are never
# installed or exported, so they use plain add_library rather than the
# project's library helper.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(EXAMPLE_TEST_SOURCES "")

# --- saxpy: y = a*x + y, a CPU loop and two ways of driving one CUDA kernel ---
# Every source of this example is CUDA or is only used beside CUDA, and the
# header declares one API whose GPU half cannot link without the toolkit, so
# the library is built with the GPU demos or not at all.
if (TARGET bench_cuda)
  add_library(
    demo_example_saxpy STATIC "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyCpu.cpp"
                              "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyGpu.cu"
  )
  target_include_directories(demo_example_saxpy PUBLIC "${PROJECT_SOURCE_DIR}")
  list(APPEND EXAMPLE_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/utst/Saxpy_uTest.cu")
endif ()

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------
# One target for every example's tests: they are small, and a reader looking
# for "do the shared workloads still agree with each other" runs one binary.
# ------------------------------------------------------------------------------

if (EXAMPLE_TEST_SOURCES)
  vernier_add_gtest(
    TARGET
    TestDemoExamples
    SOURCES
    ${EXAMPLE_TEST_SOURCES}
    LINK
    $<$<TARGET_EXISTS:demo_example_saxpy>:demo_example_saxpy>
    LABELS
    demo
  )
endif ()
