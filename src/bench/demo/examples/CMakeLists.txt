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
#   demo_example_join    - join a list of strings with a separator, two ways
#   demo_example_filter  - copy the values above a threshold, with a branch
#                          per value and without one
#   demo_example_saxpy   - y = a*x + y as a CPU loop and two ways of driving
#                          one CUDA kernel; built only with the GPU demos
#
# Tests:
#   TestDemoExamples        - every example's answers (saxpy's on a device),
#                             and what the CPU walkthroughs read from them:
#                             the join's allocations and instructions, the
#                             filter's branch misses
#   TestDemoSaxpyOwnership  - saxpy's resource release, on any machine
#   TestDemoJoinPeakHeap    - the heap one call of each join version holds at
#                             its peak, which walkthrough 14 reads from massif
#
# Check programs, which the demo build registers with CTest:
#   JoinInstructionCounts              - join's instruction counts, callgrind
#   JoinInstructionCountsStartupFault  - the same, aborting under valgrind
#
# Test support:
#   utst/HardwareCounter.hpp - one hardware event counted around a call, for
#                              the tests that read the counters; no demo
#                              includes it
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Sources (3 examples)
# ------------------------------------------------------------------------------

set(CPP_SOURCES_JOIN "${CMAKE_CURRENT_SOURCE_DIR}/join/src/Join.cpp")
set(CPP_SOURCES_FILTER "${CMAKE_CURRENT_SOURCE_DIR}/filter/src/Filter.cpp")
set(CPP_SOURCES_SAXPY "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyCpu.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyGpu.cpp"
)
set(CUDA_SOURCES_SAXPY "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyKernel.cu")

set(UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/join/utst/Join_uTest.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/filter/utst/Filter_uTest.cpp"
)
set(CUDA_UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/utst/Saxpy_uTest.cu")
set(OWNERSHIP_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/utst/SaxpyOwnership_uTest.cpp"
                           "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/src/SaxpyGpu.cpp"
)

# ------------------------------------------------------------------------------
# Example Libraries (3)
# ------------------------------------------------------------------------------

add_library(demo_example_join STATIC ${CPP_SOURCES_JOIN})
target_include_directories(demo_example_join PUBLIC "${PROJECT_SOURCE_DIR}")
# Line tables in every build type, so a profiler can name the example's source
# lines in an optimized build; -g adds debug sections and changes no code.
target_compile_options(
  demo_example_join
  PRIVATE -g
)

add_library(demo_example_filter STATIC ${CPP_SOURCES_FILTER})
target_include_directories(demo_example_filter PUBLIC "${PROJECT_SOURCE_DIR}")

set(UNIT_TEST_LINK demo_example_join demo_example_filter)

# The saxpy header is one API whose GPU half cannot link without the toolkit,
# and nothing CPU-only uses it, so the library and its tests are built with the
# GPU demos or not at all.
if (TARGET bench_cuda)
  add_library(demo_example_saxpy STATIC ${CPP_SOURCES_SAXPY} ${CUDA_SOURCES_SAXPY})
  target_include_directories(demo_example_saxpy PUBLIC "${PROJECT_SOURCE_DIR}")
  # SaxpyGpu.cpp calls the CUDA runtime from host code.
  target_link_libraries(demo_example_saxpy PUBLIC CUDA::cudart)
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

# ------------------------------------------------------------------------------
# Host Ownership Test
# ------------------------------------------------------------------------------
# The real SaxpyGpu.cpp built against a tracked stand-in for the CUDA runtime
# (saxpy/utst/fake_cuda), so every configuration, CPU-only included, checks
# that each failure path releases what the GPU versions had acquired. It is a
# binary of its own: it defines the same functions as the real saxpy library.
# ------------------------------------------------------------------------------

vernier_add_gtest(TARGET TestDemoSaxpyOwnership SOURCES ${OWNERSHIP_TEST_SOURCES} LABELS demo)
target_include_directories(
  TestDemoSaxpyOwnership BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/saxpy/utst/fake_cuda"
                                        "${PROJECT_SOURCE_DIR}"
)

# ------------------------------------------------------------------------------
# Join Instruction-Count Check (2 programs)
# ------------------------------------------------------------------------------
# JoinInstructionCounts runs itself under callgrind as a counting worker and
# holds joinV0 to several times joinV1's instructions per call, the claim
# walkthrough 07 makes. The second program adds StartupFaultUnderValgrind.cpp,
# so its counting worker aborts under valgrind before its tests start. Test
# support, never installed: ../CMakeLists.txt registers both with the other
# demo guards. Both are compiled with the build's sanitizer setting, on which
# the check skips: a sanitizer's instrumentation would be counted too.
# ------------------------------------------------------------------------------

set(JOIN_COUNT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/join/utst/JoinInstructionCounts.cpp")

vernier_add_app(
  NAME
  JoinInstructionCounts
  SRC
  ${JOIN_COUNT_SOURCES}
  LINK
  demo_example_join
  GTest::gtest_main
  NO_INSTALL
  NO_UPX
)

vernier_add_app(
  NAME
  JoinInstructionCountsStartupFault
  SRC
  ${JOIN_COUNT_SOURCES}
  "${PROJECT_SOURCE_DIR}/src/bench/utst/StartupFaultUnderValgrind.cpp"
  LINK
  demo_example_join
  GTest::gtest_main
  NO_INSTALL
  NO_UPX
)

if (SANITIZER MATCHES "^(asan|tsan|ubsan)$" AND NOT CMAKE_CROSSCOMPILING)
  foreach (_check JoinInstructionCounts JoinInstructionCountsStartupFault)
    if (TARGET ${_check})
      target_compile_definitions(${_check} PRIVATE JOIN_COUNTS_SANITIZER="${SANITIZER}")
    endif ()
  endforeach ()
endif ()

# ------------------------------------------------------------------------------
# Join Peak-Heap Check
# ------------------------------------------------------------------------------
# TestDemoJoinPeakHeap holds joinV0 to more than three times the heap joinV1
# holds at its peak, at demo 11's 20,000 words: what walkthrough 14 reads from
# massif. It replaces the global allocation functions to count bytes, so it is
# a program of its own: demo 11, the program massif profiles, carries no
# replacement, and the unit tests above keep theirs, which counts calls. Bytes
# do not depend on load, so CTest runs it under the demo and massif labels.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoJoinPeakHeap
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/join/utst/JoinPeakHeap_uTest.cpp"
  LINK
  demo_example_join
  LABELS
  demo
  massif
)
