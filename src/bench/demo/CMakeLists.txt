# ==============================================================================
# Demos: bench
#
# Educational demonstrations of the benchmarking framework and profiler
# integrations. Each demo is a self-contained executable with an
# accompanying markdown walkthrough in docs/.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Demo Helpers (shared workloads)
# ------------------------------------------------------------------------------

add_library(demo_helpers INTERFACE)
target_include_directories(demo_helpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------------------------------------
# Shared Examples
# ------------------------------------------------------------------------------
# The code the walkthroughs measure, with its unit tests. A demo links the
# example it walks through instead of declaring its own workload.
# ------------------------------------------------------------------------------

add_subdirectory(examples)

# ------------------------------------------------------------------------------
# CPU Demo Sources
# ------------------------------------------------------------------------------

set(CPU_DEMOS
    01_BasicWorkflow
    02_PerfProfiler
    03_GperfProfiler
    04_CacheFriendly
    05_BranchOptimization
    06_ThreadScaling
    07_CallgrindProfiler
    08_RaplProfiler
    09_BpftraceProfiler
    10_NvtxAnnotation
    11_MassifProfiler
    12_MemcheckProfiler
    13_OffCpuProfiler
    14_HelgrindProfiler
    15_HeaptrackProfiler
    16_JemallocProfiler
)

foreach (DEMO ${CPU_DEMOS})
  vernier_add_ptest(
    TARGET
    BenchDemo_${DEMO}
    SOURCES
    cpu/${DEMO}_Demo.cpp
    LINK
    bench
    demo_helpers
  )
endforeach ()

# ------------------------------------------------------------------------------
# GPU Demo Sources
# ------------------------------------------------------------------------------

if (TARGET bench_cuda)
  set(GPU_DEMOS 01_GpuBasicWorkflow 02_NsightProfiler 03_SharedMemoryOpt
                04_ComputeSanitizerProfiler
  )

  foreach (DEMO ${GPU_DEMOS})
    vernier_add_ptest(
      TARGET
      BenchDemo_Gpu_${DEMO}
      SOURCES
      gpu/${DEMO}_Demo.cu
      LINK
      bench
      bench_cuda
      demo_helpers
    )
  endforeach ()
endif ()

# ------------------------------------------------------------------------------
# Demos Built on a Shared Example (4)
# ------------------------------------------------------------------------------
# After both demo lists, so every demo target exists. The GPU demo and its
# example are only built where the GPU demos are.
# ------------------------------------------------------------------------------

target_link_libraries(
  BenchDemo_01_BasicWorkflow
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_07_CallgrindProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_15_HeaptrackProfiler
  PRIVATE demo_example_join
)

if (TARGET BenchDemo_Gpu_01_GpuBasicWorkflow)
  target_link_libraries(BenchDemo_Gpu_01_GpuBasicWorkflow PRIVATE demo_example_saxpy)
endif ()

# ------------------------------------------------------------------------------
# Demo Checks Run by CTest (2)
# ------------------------------------------------------------------------------
# Demo binaries stay out of ctest, so that no timing runs beside a parallel
# suite. Demo 07's check counts instructions under callgrind, which load does
# not change, so ctest runs that one test; it skips where valgrind is missing
# or gives up reading the binary's debug information. The second test runs
# the same check in a copy of demo 07 that aborts under valgrind before its
# tests start (src/bench/utst/StartupFaultUnderValgrind.cpp), and passes only
# when the check fails on it rather than skipping.
# ------------------------------------------------------------------------------

add_test(NAME CallgrindDemoInstructionCountsTest
         COMMAND BenchDemo_07_CallgrindProfiler --gtest_filter=CallgrindProfiler.InstructionCounts
)
set_tests_properties(
  CallgrindDemoInstructionCountsTest PROPERTIES LABELS "callgrind" SKIP_REGULAR_EXPRESSION
                                                "\\[  SKIPPED \\]"
)

vernier_add_ptest(
  TARGET
  CallgrindDemoStartupFault
  SOURCES
  cpu/07_CallgrindProfiler_Demo.cpp
  "${PROJECT_SOURCE_DIR}/src/bench/utst/StartupFaultUnderValgrind.cpp"
  LINK
  bench
  demo_helpers
  demo_example_join
)
add_test(NAME CallgrindDemoStartupFaultTest
         COMMAND CallgrindDemoStartupFault --gtest_filter=CallgrindProfiler.InstructionCounts
)
set_tests_properties(
  CallgrindDemoStartupFaultTest
  PROPERTIES LABELS "callgrind" PASS_REGULAR_EXPRESSION "CountCalls did not start under callgrind"
             SKIP_REGULAR_EXPRESSION
             "valgrind is not installed;could not read this binary's debug information"
)
