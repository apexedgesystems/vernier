# ==============================================================================
# Performance Tests: bench
#
# CPU and GPU performance benchmarks for the benchmarking framework.
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Test Helpers
# ------------------------------------------------------------------------------

add_library(test_helpers INTERFACE)
target_include_directories(test_helpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------------------------------------
# CPU Performance Test Sources (13 files)
# ------------------------------------------------------------------------------

set(CPU_PTEST_SOURCES
    cpu/CoreFeatures_pTest.cpp
    cpu/BranchPrediction_pTest.cpp
    cpu/CacheEffects_pTest.cpp
    cpu/CacheHierarchy_pTest.cpp
    cpu/AllocationOverhead_pTest.cpp
    cpu/Contention_pTest.cpp
    cpu/PayloadScaling_pTest.cpp
    cpu/ThreadScaling_pTest.cpp
    cpu/MemoryBandwidth_pTest.cpp
    cpu/MeasuredAPI_pTest.cpp
    cpu/MeasurementReliability_pTest.cpp
    cpu/Profilers_pTest.cpp
    cpu/SemanticMacros_pTest.cpp
)

# ------------------------------------------------------------------------------
# GPU Performance Test Sources (9 files)
# ------------------------------------------------------------------------------

set(GPU_PTEST_SOURCES
    gpu/GpuTestMain_pTest.cu
    gpu/GpuBasics_pTest.cu
    gpu/GpuMemoryStrategies_pTest.cu
    gpu/GpuBandwidth_pTest.cu
    gpu/GpuMultiDevice_pTest.cu
    gpu/GpuUnifiedMemory_pTest.cu
    gpu/GpuMemoryCoalescing_pTest.cu
    gpu/GpuSharedMemory_pTest.cu
    gpu/GpuProfilers_pTest.cu
)

# ------------------------------------------------------------------------------
# CPU Performance Tests
# ------------------------------------------------------------------------------

vernier_add_ptest(
  TARGET
  BenchmarkCPU_PTEST
  SOURCES
  ${CPU_PTEST_SOURCES}
  LINK
  bench
  test_helpers
  LABELS
  benchmarking
  cpu
)

# ------------------------------------------------------------------------------
# GPU Performance Tests
# ------------------------------------------------------------------------------

if (TARGET bench_cuda)
  vernier_add_ptest(
    TARGET
    BenchmarkGPU_PTEST
    SOURCES
    ${GPU_PTEST_SOURCES}
    LINK
    bench
    bench_cuda
    test_helpers
    LABELS
    benchmarking
    gpu
  )
endif ()
