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
    "${SRC_DIR}/PerfAbi.cpp"
    "${SRC_DIR}/HelgrindRequests.cpp"
    "${SRC_DIR}/ProfilerRegistry.cpp"
    "${SRC_DIR}/ProfilerReadiness.cpp"
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

set(CUDA_SOURCES
    "${SRC_DIR}/PerfGpuAbi.cu" "${SRC_DIR}/PerfGpuHarness.cu" "${SRC_DIR}/ProfilerNsight.cu"
    "${SRC_DIR}/ProfilerComputeSanitizer.cu" "${SRC_DIR}/CuptiCollector.cu"
)

# ------------------------------------------------------------------------------
# Shared-library ABI
# ------------------------------------------------------------------------------
# bench and bench_cuda exchange PerfConfig, Stats and PerfGpuConfig objects
# with header code compiled into the benchmark, and bench_cuda embeds Stats in
# the results it returns. The two therefore carry one object layout and one
# SONAME number, stated here once so the GPU library cannot drift from the
# core one. Raise it whenever that layout changes or an exported signature
# does; the loader then refuses a benchmark built against the previous number
# instead of binding it to incompatible code.
# ------------------------------------------------------------------------------

set(BENCH_ABI_SOVERSION 2)

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
  ABI_VERSION
  ${BENCH_ABI_SOVERSION}
)

# GoogleTest is part of the benchmark API: Perf.hpp includes gtest/gtest.h and
# PERF_MAIN() runs the GoogleTest cases. The requirement is for consumers only;
# the library does not use it. The build tree links the fetched GTest::gtest.
# An install names the GTest::gtest that the package's find_dependency(GTest)
# provides; written through TARGET_NAME_IF_EXISTS, the name reaches the export
# unresolved and is resolved in the consumer. A plain name would be resolved
# here, to the fetched target that is not installed, and install(EXPORT) would
# reject it.
target_link_libraries(
  ${LIB_NAME}
  INTERFACE $<BUILD_INTERFACE:GTest::gtest>
            $<INSTALL_INTERFACE:$<TARGET_NAME_IF_EXISTS:GTest::gtest>>
)

# gperftools (optional): the CPU profiler links whenever it is installed.
find_library(GPERF_PROFILER_LIB NAMES profiler)
if (GPERF_PROFILER_LIB)
  target_link_libraries(${LIB_NAME} PRIVATE ${GPERF_PROFILER_LIB})
endif ()

# tcmalloc swaps the process allocator, so it links only on request (see
# VERNIER_LINK_TCMALLOC). The definition is PUBLIC because ProfilerGperf.hpp
# keys the heap profiler's availability on it for every consumer of the header.
if (VERNIER_LINK_TCMALLOC)
  find_library(GPERF_TCMALLOC_LIB NAMES tcmalloc)
  if (NOT GPERF_TCMALLOC_LIB)
    message(FATAL_ERROR "VERNIER_LINK_TCMALLOC=ON but libtcmalloc was not found")
  endif ()
  target_link_libraries(${LIB_NAME} PRIVATE ${GPERF_TCMALLOC_LIB})
  target_compile_definitions(${LIB_NAME} PUBLIC VERNIER_HAS_TCMALLOC=1)
endif ()

# NVTX3 is header-only and exposed via Nvtx.hpp in this library's public
# include directory; downstream consumers that only link bench should
# still get real NVTX ranges when the CUDA toolkit is present.
vernier_nvtx_enable(${LIB_NAME})

# The bpftrace backend reads its bundled scripts from this source tree by
# absolute path, so a run finds them whatever its working directory.
set_source_files_properties(
  "${SRC_DIR}/ProfilerBpftrace.cpp"
  PROPERTIES COMPILE_DEFINITIONS "VERNIER_BPF_SCRIPTS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/bpf\""
)

# ------------------------------------------------------------------------------
# CUDA Library (requires CUDA toolkit)
# ------------------------------------------------------------------------------
# Optional GPU benchmarking support. Provides GPU harness and Nsight integration.
# Only built when CUDAToolkit_FOUND is true.
# ------------------------------------------------------------------------------

if (CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
  # NVML samples GPU clocks, power and temperature for the harness's CSV.
  # COMPAT_NVML_AVAILABLE is 1 when it is found and linked, 0 otherwise
  # (vernier_nvml_enable), and 0 with VERNIER_USE_NVML off, which skips it.
  set(_bench_cuda_nvml "")
  if (NOT VERNIER_USE_NVML)
    set(_bench_cuda_nvml NO_NVML)
  endif ()
  vernier_add_library_cuda(
    NAME
    ${LIB_NAME}_cuda
    CORE
    ${LIB_NAME}
    INC
    "${INC_DIR}"
    SRC
    ${CUDA_SOURCES}
    ABI_VERSION
    ${BENCH_ABI_SOVERSION}
    ${_bench_cuda_nvml}
  )
  if (NOT VERNIER_USE_NVML)
    target_compile_definitions(${LIB_NAME}_cuda PUBLIC COMPAT_NVML_AVAILABLE=0)
  endif ()
  vernier_standard_optins(${LIB_NAME}_cuda)

  # bench_cuda links bench for its symbols and so also receives bench's
  # GoogleTest requirement, which it does not use. Linking as needed keeps the
  # build's GoogleTest library out of bench_cuda's runtime dependencies, also
  # with toolchains that do not link that way by default.
  target_link_options(${LIB_NAME}_cuda PRIVATE "LINKER:--as-needed")

  # CUPTI is the CUDA Profiling Tools Interface. find_package(CUDAToolkit)
  # creates the CUDA::cupti imported target, on any system layout, only when
  # it finds both cupti.h and the library. COMPAT_CUPTI_AVAILABLE is the one
  # switch CuptiCollector.cu builds its collection on: 1 with the library
  # linked, 0 otherwise (VERNIER_USE_CUPTI off, or no CUPTI in the toolkit).
  # The header alone decides nothing: cupti.h sits in the toolkit's main
  # include directory whether or not the library is there. BENCH_CUDA_HAS_CUPTI
  # carries the same decision to the demo directory's checks.
  if (VERNIER_USE_CUPTI AND TARGET CUDA::cupti)
    target_link_libraries(${LIB_NAME}_cuda PRIVATE CUDA::cupti)
    target_compile_definitions(${LIB_NAME}_cuda PRIVATE COMPAT_CUPTI_AVAILABLE=1)
    set(BENCH_CUDA_HAS_CUPTI TRUE)
  else ()
    target_compile_definitions(${LIB_NAME}_cuda PRIVATE COMPAT_CUPTI_AVAILABLE=0)
    set(BENCH_CUDA_HAS_CUPTI FALSE)
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
