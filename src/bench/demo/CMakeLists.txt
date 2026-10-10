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
# The code the walkthroughs measure, with its tests. A demo links the example
# it walks through instead of declaring its own workload. Each example builds
# and registers its own tests, among them the guards of demos 07 and 11, which
# count what those demos show instead of timing it (examples/join/utst).
# ------------------------------------------------------------------------------

add_subdirectory(examples)

# ------------------------------------------------------------------------------
# CPU Demo Sources
# ------------------------------------------------------------------------------

set(CPU_DEMOS
    01_BasicWorkflow
    02_PerfProfiler
    03_GperfProfiler
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
# Demo 12's Private Sources
# ------------------------------------------------------------------------------
# The deliberately wrong join memcheck finds is a source of its own, compiled
# with debug information whatever the build type, so memcheck's report names
# the lines of the write and of the allocation. Nothing else in the build
# changes.
# ------------------------------------------------------------------------------

target_sources(BenchDemo_12_MemcheckProfiler PRIVATE cpu/12_MemcheckProfiler_OffByOne.cpp)
set_source_files_properties(cpu/12_MemcheckProfiler_OffByOne.cpp PROPERTIES COMPILE_OPTIONS "-g")

# ------------------------------------------------------------------------------
# Demo 14's Private Sources
# ------------------------------------------------------------------------------
# The deliberately racy addition helgrind finds is a source of its own,
# compiled with debug information whatever the build type, so helgrind's report
# names the line of the race. Nothing else in the build changes.
# ------------------------------------------------------------------------------

target_sources(BenchDemo_14_HelgrindProfiler PRIVATE cpu/14_HelgrindProfiler_Racy.cpp)
set_source_files_properties(cpu/14_HelgrindProfiler_Racy.cpp PROPERTIES COMPILE_OPTIONS "-g")

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

  # Demo 03's kernels are a source of their own, so the tests beside the demo
  # compile the kernels the demo launches (the sections at the end of this
  # file).
  target_sources(BenchDemo_Gpu_03_SharedMemoryOpt PRIVATE gpu/03_SharedMemoryOpt_Transpose.cu)
endif ()

# ------------------------------------------------------------------------------
# Demos Built on a Shared Example (12)
# ------------------------------------------------------------------------------
# After both demo lists, so every demo target exists. The GPU demos and their
# example are only built where the GPU demos are.
# ------------------------------------------------------------------------------

target_link_libraries(
  BenchDemo_01_BasicWorkflow
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_02_PerfProfiler
  PRIVATE demo_example_join demo_example_filter
)
target_link_libraries(
  BenchDemo_03_GperfProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_06_ThreadScaling
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_07_CallgrindProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_09_BpftraceProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_11_MassifProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_12_MemcheckProfiler
  PRIVATE demo_example_join
)
target_link_libraries(
  BenchDemo_14_HelgrindProfiler
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

# ------------------------------------------------------------------------------
# Walkthrough 19's Column Check (1)
# ------------------------------------------------------------------------------
# TestDemoKernelColumns runs demo 02's two kernel cases with --csv and holds
# each GPU column of their rows to its source: CUDA events, the harness's
# occupancy estimate, CUPTI's records (BENCH_CUDA_HAS_CUPTI says whether the
# build collects them) and NVML's readings, every cell the run cannot fill
# empty and stated. It compiles the SAXPY example's kernel source into itself
# to read the kernel's compiled resource use, so it links no example library.
# Built where the GPU demos are, under the demo and gpu labels; it compares no
# timing. The skip expression names the check, as the memcheck check's does.
# ------------------------------------------------------------------------------

if (TARGET BenchDemo_Gpu_02_NsightProfiler)
  vernier_add_gtest(
    TARGET
    TestDemoKernelColumns
    SOURCES
    gpu/utst/02_NsightProfiler_KernelColumns_uTest.cpp
    gpu/utst/02_NsightProfiler_KernelColumns_Kernel.cu
    LINK
    bench
    bench_cuda
    LABELS
    demo
    gpu
  )
  target_compile_definitions(
    TestDemoKernelColumns
    PRIVATE VERNIER_DEMO_02_BINARY="$<TARGET_FILE:BenchDemo_Gpu_02_NsightProfiler>"
            VERNIER_DEMO_02_HAS_CUPTI=$<BOOL:${BENCH_CUDA_HAS_CUPTI}>
  )
  add_dependencies(TestDemoKernelColumns BenchDemo_Gpu_02_NsightProfiler)
  set_tests_properties(
    KernelColumns.MatchTheirSources PROPERTIES SKIP_REGULAR_EXPRESSION
                                               "\\[  SKIPPED \\] KernelColumns\\.MatchTheirSources"
  )
endif ()

# ------------------------------------------------------------------------------
# Demo Effect Checks Run by CTest (3)
# ------------------------------------------------------------------------------
# A check that runs its demo under a tool and reads the tool's report, not a
# timing, is a test program of its own beside the demo, so the demo's source
# holds only what it teaches. It runs the demo binary, whose path it is given,
# under the demo label and the tool's, so ordinary test runs include it and
# never the demo's timing cases. The skip expression names the check: its
# failure messages quote the child runs' output, whose own GoogleTest lines
# must not turn a failure into a skip.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoMemcheck
  SOURCES
  cpu/utst/12_MemcheckProfiler_uTest.cpp
  LINK
  bench
  demo_helpers
  demo_example_join
  LABELS
  demo
  memcheck
)
target_compile_definitions(
  TestDemoMemcheck PRIVATE VERNIER_DEMO_12_BINARY="$<TARGET_FILE:BenchDemo_12_MemcheckProfiler>"
)
add_dependencies(TestDemoMemcheck BenchDemo_12_MemcheckProfiler)
set_tests_properties(
  Memcheck.FindsTheOffByOne PROPERTIES SKIP_REGULAR_EXPRESSION
                                       "\\[  SKIPPED \\] Memcheck\\.FindsTheOffByOne"
)

vernier_add_gtest(
  TARGET
  TestDemoHelgrind
  SOURCES
  cpu/utst/14_HelgrindProfiler_uTest.cpp
  LINK
  bench
  demo_helpers
  LABELS
  demo
  helgrind
)
target_compile_definitions(
  TestDemoHelgrind
  PRIVATE VERNIER_DEMO_14_BINARY="$<TARGET_FILE:BenchDemo_14_HelgrindProfiler>"
          VERNIER_DEMO_14_RACY_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/cpu/14_HelgrindProfiler_Racy.cpp"
)
add_dependencies(TestDemoHelgrind BenchDemo_14_HelgrindProfiler)
foreach (_helgrind_check Helgrind.FindsTheRace Helgrind.LockedTotalReportsNothing)
  string(REPLACE "." "\\." _helgrind_pattern "${_helgrind_check}")
  set_tests_properties(
    ${_helgrind_check} PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\] ${_helgrind_pattern}"
  )
endforeach ()

# Demo 09's check also counts the demo's writes and totals, with no tool, under
# the demo label; only its traced test runs bpftrace, and carries that label
# too. That test runs alone: its tracer compiles its program and traces every
# write of the process, which timing tests running beside it would feel. Its
# backend bounds each of its waits; the test's own bound is far above their sum.
vernier_add_gtest(
  TARGET
  TestDemoBpftrace
  SOURCES
  cpu/utst/09_BpftraceProfiler_uTest.cpp
  LINK
  bench
  demo_example_join
  LABELS
  demo
)
target_compile_definitions(
  TestDemoBpftrace PRIVATE VERNIER_DEMO_09_BINARY="$<TARGET_FILE:BenchDemo_09_BpftraceProfiler>"
)
add_dependencies(TestDemoBpftrace BenchDemo_09_BpftraceProfiler)
set_property(
  TEST Bpftrace.CountsEveryWrite
  APPEND
  PROPERTY LABELS bpftrace
)
set_tests_properties(
  Bpftrace.CountsEveryWrite PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_REGULAR_EXPRESSION
                                       "\\[  SKIPPED \\] Bpftrace\\.CountsEveryWrite"
)

# ------------------------------------------------------------------------------
# Demo 06's Totals Check Run by CTest
# ------------------------------------------------------------------------------
# What demo 06's two versions cost is a timing, measured on the rig and not
# registered. What the machine's load cannot change is: run through
# contentionRun() as the demo runs them, both versions reach the same total.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoThreadScaling
  SOURCES
  cpu/utst/06_ThreadScaling_uTest.cpp
  LINK
  bench
  demo_example_join
  LABELS
  demo
)

# ------------------------------------------------------------------------------
# Demo Helper Tests
# ------------------------------------------------------------------------------
# The helpers the demos share, under the demo label with the examples' tests.
# The probe case skips itself in an ordinary run, which is the helper working,
# so CTest reports it as skipped rather than passed.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoHelpers
  SOURCES
  helpers/utst/SkipUnlessUnderValgrind_uTest.cpp
  LINK
  bench
  demo_helpers
  LABELS
  demo
)
set_tests_properties(
  SkipUnlessUnderValgrindProbe.RunsOnlyUnderValgrind
  PROPERTIES SKIP_REGULAR_EXPRESSION
             "\\[  SKIPPED \\] SkipUnlessUnderValgrindProbe\\.RunsOnlyUnderValgrind"
)

# ------------------------------------------------------------------------------
# GPU Demo Tests and Effect Checks Run by CTest (3)
# ------------------------------------------------------------------------------
# Demo 03's tests. TestDemoBankConflictsReport holds the bank-conflict check's
# reading of ncu's CSV and its verdict to logs as ncu prints them; it needs no
# GPU, so every configuration runs it, under the check's labels.
# TestDemoTranspose and the check, TestDemoBankConflicts, need a device (the
# check also ncu) and are built where the GPU demos are. The check's skip
# expression names its own case, so the demo's GoogleTest lines in its failure
# output cannot turn a failure into a skip.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoBankConflictsReport
  SOURCES
  gpu/utst/03_SharedMemoryOpt_BankConflictsReport_uTest.cpp
  LABELS
  demo
  ncu
)

if (TARGET bench_cuda)
  vernier_add_gtest(
    TARGET
    TestDemoTranspose
    SOURCES
    gpu/utst/03_SharedMemoryOpt_uTest.cu
    gpu/03_SharedMemoryOpt_Transpose.cu
    LINK
    CUDA::cudart
    LABELS
    demo
  )
  target_include_directories(TestDemoTranspose PRIVATE "${PROJECT_SOURCE_DIR}")

  # The check behind walkthrough 12, beside the demo it runs.
  vernier_add_gtest(
    TARGET
    TestDemoBankConflicts
    SOURCES
    gpu/utst/03_SharedMemoryOpt_BankConflicts_uTest.cpp
    LINK
    bench
    CUDA::cudart
    LABELS
    demo
    ncu
  )
  target_compile_definitions(
    TestDemoBankConflicts
    PRIVATE VERNIER_DEMO_GPU_03_BINARY="$<TARGET_FILE:BenchDemo_Gpu_03_SharedMemoryOpt>"
  )
  add_dependencies(TestDemoBankConflicts BenchDemo_Gpu_03_SharedMemoryOpt)
  set_tests_properties(
    BankConflicts.CountedByNsightCompute
    PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\] BankConflicts\\.CountedByNsightCompute"
  )
endif ()
