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
# Demos Built on a Shared Example (5)
# ------------------------------------------------------------------------------
# After both demo lists, so every demo target exists. The GPU demos and their
# example are only built where the GPU demos are.
# ------------------------------------------------------------------------------

target_link_libraries(
  BenchDemo_01_BasicWorkflow
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_03_GperfProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_15_HeaptrackProfiler
  PRIVATE demo_example_join
)

if (TARGET BenchDemo_Gpu_01_GpuBasicWorkflow)
  target_link_libraries(BenchDemo_Gpu_01_GpuBasicWorkflow PRIVATE demo_example_saxpy)
  target_link_libraries(BenchDemo_Gpu_02_NsightProfiler PRIVATE demo_example_saxpy)
  # Demo 02's guard timing, in a source of its own so the host test below builds it.
  target_sources(BenchDemo_Gpu_02_NsightProfiler PRIVATE gpu/02_NsightProfiler_Timing.cpp)
endif ()

# ------------------------------------------------------------------------------
# Host Test on the Stand-in Runtime (1)
# ------------------------------------------------------------------------------
# Demo 02's guard timing built against the tracked stand-in for the CUDA
# runtime (examples/saxpy/utst/fake_cuda), so every configuration, CPU-only
# included, checks that it releases its stream and events on every path,
# exceptions included: TestDemoNsightProfilerTiming.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoNsightProfilerTiming
  SOURCES
  gpu/utst/02_NsightProfiler_Timing_uTest.cpp
  gpu/02_NsightProfiler_Timing.cpp
  LABELS
  demo
)
target_include_directories(
  TestDemoNsightProfilerTiming BEFORE
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples/saxpy/utst/fake_cuda" "${PROJECT_SOURCE_DIR}"
)
