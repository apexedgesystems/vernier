# ==============================================================================
# Module: bench
#
# Performance benchmarking framework for CPU and GPU workloads.
# Provides profiler integrations, harness infrastructure, and statistics.
#
# Libraries:
#   bench       - CPU profilers and registry (perf, gperftools, callgrind,
#                 bpftrace, RAPL, massif, memcheck, off-cpu, heaptrack,
#                 jemalloc, rocprof) plus shared dispatch/env infrastructure.
#   bench_cuda  - GPU harness, NVTX, Nsight, Compute Sanitizer, CUPTI.
# ==============================================================================

include(vernier/All)
vernier_module(bench)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(CPP_SOURCES
    "${SRC_DIR}/ProfilerRegistry.cpp"
    "${SRC_DIR}/ProfilerPerf.cpp"
    "${SRC_DIR}/ProfilerGperf.cpp"
    "${SRC_DIR}/ProfilerBpftrace.cpp"
    "${SRC_DIR}/ProfilerRAPL.cpp"
    "${SRC_DIR}/ProfilerCallgrind.cpp"
    "${SRC_DIR}/ProfilerMassif.cpp"
    "${SRC_DIR}/ProfilerMemcheck.cpp"
    "${SRC_DIR}/ProfilerHelgrind.cpp"
    "${SRC_DIR}/ProfilerOffCpu.cpp"
    "${SRC_DIR}/ProfilerRocprof.cpp"
    "${SRC_DIR}/ProfilerHeaptrack.cpp"
    "${SRC_DIR}/ProfilerJemalloc.cpp"
)

set(CUDA_SOURCES "${SRC_DIR}/PerfGpuHarness.cu" "${SRC_DIR}/ProfilerNsight.cu"
                 "${SRC_DIR}/ProfilerComputeSanitizer.cu" "${SRC_DIR}/CuptiCollector.cu"
)

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

vernier_add_library(
  NAME
  ${LIB_NAME}
  TYPE
  SHARED
  INC
  "${INC_DIR}"
  SRC
  ${CPP_SOURCES}
  REQUIRES
  POSIX
)

# gperftools (optional)
find_library(GPERF_PROFILER_LIB NAMES profiler)
find_library(GPERF_TCMALLOC_LIB NAMES tcmalloc)
if (GPERF_PROFILER_LIB)
  target_link_libraries(${LIB_NAME} PRIVATE ${GPERF_PROFILER_LIB})
endif ()
if (GPERF_TCMALLOC_LIB)
  target_link_libraries(${LIB_NAME} PRIVATE ${GPERF_TCMALLOC_LIB})
endif ()

# NVTX3 is header-only and exposed via Nvtx.hpp in this library's public
# include directory; downstream consumers that only link bench should
# still get real NVTX ranges when the CUDA toolkit is present.
vernier_nvtx_enable(${LIB_NAME})

# ------------------------------------------------------------------------------
# CUDA Library (requires CUDA toolkit)
# ------------------------------------------------------------------------------
# Optional GPU benchmarking support. Provides GPU harness and Nsight integration.
# Only built when CUDAToolkit_FOUND is true.
# ------------------------------------------------------------------------------

if (CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
  vernier_add_library_cuda(NAME ${LIB_NAME}_cuda CORE ${LIB_NAME} SRC ${CUDA_SOURCES})
  vernier_standard_optins(${LIB_NAME}_cuda)

  # CUPTI is the CUDA Profiling Tools Interface. find_package(CUDAToolkit)
  # exposes it as the CUDA::cupti imported target on any system layout
  # (legacy /usr/local/cuda/extras/CUPTI/ or modern targets/<arch>/), so we
  # never need a path string here. When the target is missing on older
  # toolkits, CuptiCollector compiles to a no-op via __has_include.
  if (TARGET CUDA::cupti)
    target_link_libraries(${LIB_NAME}_cuda PRIVATE CUDA::cupti)
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Opt-ins (coverage, doxygen, UPX)
# ------------------------------------------------------------------------------

vernier_standard_optins(${LIB_NAME})

# ------------------------------------------------------------------------------
# Production documentation (release builds only)
# ------------------------------------------------------------------------------

vernier_install_docs("${CMAKE_CURRENT_SOURCE_DIR}/docs")
vernier_install_docs("${CMAKE_CURRENT_SOURCE_DIR}/demo/docs" DESTINATION "demos")

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

if (PROJECT_IS_TOP_LEVEL)
  if (EXISTS "${UTST_DIR}/CMakeLists.txt")
    add_subdirectory(utst)
  endif ()
  if (EXISTS "${PTST_DIR}/CMakeLists.txt")
    add_subdirectory(ptst)
  endif ()
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo/CMakeLists.txt")
    add_subdirectory(demo)
  endif ()
endif ()
