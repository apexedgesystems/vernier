# ==============================================================================
# Unit Tests: demo_example_saxpy
#
# Tests:
#   TestDemoSaxpyOwnership  - each failure path of the GPU versions releases
#                             what it had acquired, on any machine
#   TestDemoSaxpyDriving    - what each GPU version acquires, copies and
#                             launches per call, on any machine
#   TestDemoSaxpy           - every version answers what the CPU loop answers,
#                             the GPU versions on a device; built only with
#                             the library
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Host Tests on the Stand-in Runtime (2)
# ------------------------------------------------------------------------------
# The real SaxpyGpu.cpp built against a tracked stand-in for the CUDA runtime
# (fake_cuda), so every configuration, CPU-only included, checks it: the facts
# a profiler reads off the GPU versions. Each is a program of its own: it
# defines the same functions as the real saxpy library.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoSaxpyOwnership
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/SaxpyOwnership_uTest.cpp"
  "${SRC_DIR}/SaxpyGpu.cpp"
  LABELS
  demo
)
target_include_directories(
  TestDemoSaxpyOwnership BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_cuda"
                                        "${PROJECT_SOURCE_DIR}"
)

vernier_add_gtest(
  TARGET
  TestDemoSaxpyDriving
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/SaxpyDriving_uTest.cpp"
  "${SRC_DIR}/SaxpyGpu.cpp"
  LABELS
  demo
)
target_include_directories(
  TestDemoSaxpyDriving BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_cuda"
                                      "${PROJECT_SOURCE_DIR}"
)

# ------------------------------------------------------------------------------
# Device Tests (requires CUDA toolkit)
# ------------------------------------------------------------------------------

if (TARGET demo_example_saxpy)
  vernier_add_gtest(
    TARGET
    TestDemoSaxpy
    SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/Saxpy_uTest.cu"
    LINK
    demo_example_saxpy
    LABELS
    demo
  )
endif ()
