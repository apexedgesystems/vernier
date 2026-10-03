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
# Demos Built on a Shared Example (9)
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
  BenchDemo_07_CallgrindProfiler
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
# Demo Effect Guards Run by CTest (3)
# ------------------------------------------------------------------------------
# Demos are ptests: built, not registered with CTest. A guard that checks what
# its demo demonstrates by counting bytes, not by timing, is registered on its
# own, the way gtest_add_tests names and runs a test, under the demo label, so
# ordinary test runs include it and never the demo's timing cases.
#
# Demo 07's guard counts instructions under callgrind, which load does not
# change either. It is a program of the join example's tests,
# JoinInstructionCounts (examples/CMakeLists.txt), so demo 07's source holds
# only what it teaches. It also carries the callgrind label, and skips where
# valgrind is missing or stops reading the program's debug information, and in
# a build with a sanitizer (-DSANITIZER), whose instrumentation it would count.
# The third test, labelled callgrind only, runs the same check in
# JoinInstructionCountsStartupFault, whose counting worker aborts under
# valgrind before its tests start
# (src/bench/utst/StartupFaultUnderValgrind.cpp), and passes only when the
# check fails on that abort rather than skipping.
# ------------------------------------------------------------------------------

add_test(NAME Massif.JoinPeakHeap COMMAND BenchDemo_11_MassifProfiler
                                          --gtest_filter=Massif.JoinPeakHeap
)
set_tests_properties(
  Massif.JoinPeakHeap PROPERTIES LABELS demo WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if (TARGET JoinInstructionCounts)
  add_test(NAME CallgrindDemoInstructionCountsTest
           COMMAND JoinInstructionCounts --gtest_filter=JoinInstructionCounts.UnderCallgrind
  )
  # The skip expression names this test: its failure messages quote the
  # counting runs' output, whose own GoogleTest lines must not turn a failure
  # into a skip.
  set_tests_properties(
    CallgrindDemoInstructionCountsTest
    PROPERTIES LABELS "callgrind;demo" SKIP_REGULAR_EXPRESSION
               "\\[  SKIPPED \\] JoinInstructionCounts\\.UnderCallgrind"
  )
endif ()

if (TARGET JoinInstructionCountsStartupFault)
  add_test(NAME CallgrindDemoStartupFaultTest
           COMMAND JoinInstructionCountsStartupFault
                   --gtest_filter=JoinInstructionCounts.UnderCallgrind
  )
  # It passes only on the check's startup-failure message over a run that shows
  # the fixture's own abort, followed by GoogleTest's FAILED line for the check:
  # a run that valgrind or a sanitizer stopped first does not pass it. It skips
  # only on evidence that is not the check's wording: "valgrind is not
  # installed", decided before anything runs; valgrind's own words when it
  # stops reading debug information, which the check's skip quotes as valgrind
  # wrote them; and, in a build with a sanitizer only, the check's skip for that
  # build, decided at compile time.
  set(_fault_skips
      "valgrind is not installed"
      "Valgrind: debuginfo reader: [^\n]*\n[^\n]*Valgrind: I can't recover\\.  Giving up\\."
      "valgrind: m_debuginfo/[^\n]*: Assertion '[^\n]*' failed\\."
  )
  if (SANITIZER MATCHES "^(asan|tsan|ubsan)$" AND NOT CMAKE_CROSSCOMPILING)
    list(APPEND _fault_skips
         "this program is built with the [a-z-]+ sanitizer \\(SANITIZER=${SANITIZER}\\)"
    )
  endif ()
  set_tests_properties(
    CallgrindDemoStartupFaultTest
    PROPERTIES
      LABELS
      "callgrind"
      PASS_REGULAR_EXPRESSION
      "counting worker did not start under callgrind.*StartupFaultUnderValgrind: aborting before the tests start, as intended.*\\[  FAILED  \\] JoinInstructionCounts\\.UnderCallgrind"
      SKIP_REGULAR_EXPRESSION
      "${_fault_skips}"
  )
endif ()

# ------------------------------------------------------------------------------
# Demo Effect Checks Run by CTest (1)
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
