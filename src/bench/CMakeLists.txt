# ==============================================================================
# Module: bench
#
# Performance benchmarking framework for CPU and GPU workloads.
# Provides profiler integrations, harness infrastructure, and statistics.
#
# Libraries:
#   bench       - CPU profilers (perf, gperftools, bpftrace, RAPL, callgrind)
#   bench_cuda  - GPU harness and Nsight profiler integration
# ==============================================================================

include(vernier/All)
vernier_module(bench)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(CPP_SOURCES
    "${SRC_DIR}/ProfilerPerf.cpp" "${SRC_DIR}/ProfilerGperf.cpp" "${SRC_DIR}/ProfilerBpftrace.cpp"
    "${SRC_DIR}/ProfilerRAPL.cpp" "${SRC_DIR}/ProfilerCallgrind.cpp"
)

set(CUDA_SOURCES "${SRC_DIR}/PerfGpuHarness.cu" "${SRC_DIR}/ProfilerNsight.cu")

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

# ------------------------------------------------------------------------------
# CUDA Library (requires CUDA toolkit)
# ------------------------------------------------------------------------------
# Optional GPU benchmarking support. Provides GPU harness and Nsight integration.
# Only built when CUDAToolkit_FOUND is true.
# ------------------------------------------------------------------------------

if (CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
  vernier_add_library_cuda(NAME ${LIB_NAME}_cuda CORE ${LIB_NAME} SRC ${CUDA_SOURCES})
  vernier_standard_optins(${LIB_NAME}_cuda)
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
