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
# CPU Demo Sources (9 demos)
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
# GPU Demo Sources (3 demos)
# ------------------------------------------------------------------------------

if (TARGET bench_cuda)
  set(GPU_DEMOS 01_GpuBasicWorkflow 02_NsightProfiler 03_SharedMemoryOpt)

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
