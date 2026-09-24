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
# Demos Built on a Shared Example (3)
# ------------------------------------------------------------------------------
# After both demo lists, so every demo target exists. The GPU demo and its
# example are only built where the GPU demos are.
# ------------------------------------------------------------------------------

target_link_libraries(
  BenchDemo_01_BasicWorkflow
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_11_MassifProfiler
  PRIVATE demo_example_join
)

if (TARGET BenchDemo_Gpu_01_GpuBasicWorkflow)
  target_link_libraries(BenchDemo_Gpu_01_GpuBasicWorkflow PRIVATE demo_example_saxpy)
endif ()

# ------------------------------------------------------------------------------
# Demo Effect Guards Run by CTest (1)
# ------------------------------------------------------------------------------
# Demos are ptests: built, not registered with CTest. A guard that checks what
# its demo demonstrates by counting bytes, not by timing, is registered on its
# own, the way gtest_add_tests names and runs a test, under the demo label, so
# ordinary test runs include it and never the demo's timing cases.
# ------------------------------------------------------------------------------

add_test(NAME Massif.JoinPeakHeap COMMAND BenchDemo_11_MassifProfiler
                                          --gtest_filter=Massif.JoinPeakHeap
)
set_tests_properties(
  Massif.JoinPeakHeap PROPERTIES LABELS demo WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
