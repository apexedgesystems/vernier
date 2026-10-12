# ==============================================================================
# Unit Tests: demo_example_join
#
# Tests:
#   TestDemoJoin          - both versions' answers, and what the CPU
#                           walkthroughs read from them: their allocations and
#                           retired instructions
#   TestDemoJoinPeakHeap  - the heap one call of each version holds at its
#                           peak, which walkthrough 14 reads from massif
#
# Process fixtures, registered here through their controller case:
#   JoinInstructionCounts              - instructions per call, under callgrind
#   JoinInstructionCountsStartupFault  - the same, aborting under valgrind
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoJoin
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/Join_uTest.cpp"
  LINK
  demo_example_join
  LABELS
  demo
)

# ------------------------------------------------------------------------------
# Peak-Heap Check
# ------------------------------------------------------------------------------
# TestDemoJoinPeakHeap holds joinV0 to more than three times the heap joinV1
# holds at its peak, at demo 11's 20,000 words: what walkthrough 14 reads from
# massif. It replaces the global allocation functions to count bytes, so it is
# a program of its own: demo 11, the program massif profiles, carries no
# replacement, and TestDemoJoin keeps its own, which counts calls. Bytes do not
# depend on load, so CTest runs it under the demo and massif labels.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestDemoJoinPeakHeap
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/JoinPeakHeap_uTest.cpp"
  LINK
  demo_example_join
  LABELS
  demo
  massif
)

# ------------------------------------------------------------------------------
# Instruction-Count Check (2 programs, 2 tests)
# ------------------------------------------------------------------------------
# Demo 07's guard: JoinInstructionCounts runs its Worker case under callgrind
# and holds joinV0 to several times joinV1's instructions per call, the claim
# walkthrough 07 makes. Only the controller case, UnderCallgrind, is a test;
# the Worker case runs inside its counting runs, so the programs register
# nothing themselves. The second program links the startup-fault fixture, so
# its worker aborts under valgrind before its tests start: the negative
# control. Both carry the build's sanitizer setting, on which the check skips,
# since a sanitizer's instrumentation would be counted too.
# ------------------------------------------------------------------------------

set(JOIN_COUNT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/JoinInstructionCounts.cpp")

vernier_add_gtest(
  TARGET
  JoinInstructionCounts
  SOURCES
  ${JOIN_COUNT_SOURCES}
  LINK
  demo_example_join
  NO_REGISTER
)

vernier_add_gtest(
  TARGET
  JoinInstructionCountsStartupFault
  SOURCES
  ${JOIN_COUNT_SOURCES}
  LINK
  demo_example_join
  NO_REGISTER
)

if (SANITIZER MATCHES "^(asan|tsan|ubsan)$" AND NOT CMAKE_CROSSCOMPILING)
  foreach (_check JoinInstructionCounts JoinInstructionCountsStartupFault)
    if (TARGET ${_check})
      target_compile_definitions(${_check} PRIVATE JOIN_COUNTS_SANITIZER="${SANITIZER}")
    endif ()
  endforeach ()
endif ()

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
  target_link_libraries(JoinInstructionCountsStartupFault PRIVATE StartupFaultUnderValgrind)
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
  # wrote them, its reader's assertion only where that skip quotes it ("It
  # printed:"), which the check allows only with the evidence that valgrind
  # stopped before the program started (src/bench/utst/ValgrindReaderAssertion.hpp);
  # and, in a build with a sanitizer only, the check's skip for that build,
  # decided at compile time.
  set(_fault_skips
      "valgrind is not installed"
      "Valgrind: debuginfo reader: [^\n]*\n[^\n]*Valgrind: I can't recover\\.  Giving up\\."
      "It printed:\nvalgrind: m_debuginfo/[^\n]*: Assertion '[^\n]*' failed\\."
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
