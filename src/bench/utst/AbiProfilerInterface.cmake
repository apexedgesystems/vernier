# ==============================================================================
# AbiProfilerInterface.cmake - The start-up check against a library built
# before a profiler virtual interface change
#
# Included from the bench unit-test CMakeLists.txt. libbench builds the
# profilers that header code compiled into a benchmark calls through, so a
# slot added to Profiler changes the interface between the two binaries
# without changing the size of any struct the check compares. Two small
# shared libraries stand for libbench's side (the check, PerfAbi.cpp, and a
# profiler it builds, AbiProfilerLibrary.cpp): one built from this tree's
# headers, one from a copy as they were before Profiler had its
# abortMeasure() slot. One client, built from this tree's headers, is linked
# against each.
#
# Targets:
#   AbiProfilerLibraryCurrent, AbiProfilerLibraryBeforeSlot  - the libraries
#   AbiProfilerClientMatched, AbiProfilerClientMixed         - the clients
# Tests:
#   AbiProfilerMatchedTest  - the matching pair runs, abortMeasure() included
#   AbiProfilerMixedTest    - the mixed pair stops at the check (status 3)
# ==============================================================================

# ------------------------------------------------------------------------------
# Generated Header Tree
# ------------------------------------------------------------------------------
# One producer, one user, as for the layout consumers' tree: the edited
# Profiler.hpp is a source of AbiProfilerLibraryBeforeSlot alone.
# ------------------------------------------------------------------------------

set(_slot_headers_root "${CMAKE_CURRENT_BINARY_DIR}/abi_before_abort_slot")
set(_slot_edited_header "${_slot_headers_root}/src/bench/inc/Profiler.hpp")
file(GLOB _slot_bench_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/bench/inc/*")
add_custom_command(
  OUTPUT "${_slot_edited_header}"
  COMMAND "${CMAKE_COMMAND}" -DSRC=${PROJECT_SOURCE_DIR} -DDST=${_slot_headers_root} -P
          "${CMAKE_CURRENT_LIST_DIR}/AbiBeforeAbortSlotHeaders.cmake"
  DEPENDS ${_slot_bench_headers} "${CMAKE_CURRENT_LIST_DIR}/AbiBeforeAbortSlotHeaders.cmake"
  COMMENT "Generating the header tree from before Profiler's abortMeasure() slot"
  VERBATIM
)

# ------------------------------------------------------------------------------
# Libraries (2)
# ------------------------------------------------------------------------------
# Test fixtures, not products: never installed or exported.
# ------------------------------------------------------------------------------

add_library(
  AbiProfilerLibraryCurrent SHARED "${CMAKE_CURRENT_LIST_DIR}/AbiProfilerLibrary.cpp"
                                   "${PROJECT_SOURCE_DIR}/src/bench/src/PerfAbi.cpp"
)
add_library(
  AbiProfilerLibraryBeforeSlot SHARED
  "${CMAKE_CURRENT_LIST_DIR}/AbiProfilerLibrary.cpp"
  "${PROJECT_SOURCE_DIR}/src/bench/src/PerfAbi.cpp" "${_slot_edited_header}"
)
# Ahead of every other include directory, the project root included: the
# generated copies have to shadow the tree's headers of the same name.
target_include_directories(AbiProfilerLibraryBeforeSlot BEFORE PRIVATE "${_slot_headers_root}")

# ------------------------------------------------------------------------------
# Clients (2)
# ------------------------------------------------------------------------------

vernier_add_test_program(
  TARGET AbiProfilerClientMatched SOURCES "${CMAKE_CURRENT_LIST_DIR}/AbiProfilerClient.cpp" LINK
  AbiProfilerLibraryCurrent
)
vernier_add_test_program(
  TARGET AbiProfilerClientMixed SOURCES "${CMAKE_CURRENT_LIST_DIR}/AbiProfilerClient.cpp" LINK
  AbiProfilerLibraryBeforeSlot
)

# The helper builds nothing for bare metal; the tests go with it.
if (NOT TARGET AbiProfilerClientMatched OR NOT TARGET AbiProfilerClientMixed)
  return()
endif ()

# ------------------------------------------------------------------------------
# Tests (2)
# ------------------------------------------------------------------------------

foreach (_case Matched Mixed)
  set(_expect accepted)
  if (_case STREQUAL "Mixed")
    set(_expect rejected)
  endif ()
  add_test(NAME AbiProfiler${_case}Test
           COMMAND "${CMAKE_COMMAND}" -DCLIENT=$<TARGET_FILE:AbiProfilerClient${_case}>
                   -DEXPECT=${_expect} -P "${CMAKE_CURRENT_LIST_DIR}/AbiProfilerClient_test.cmake"
  )
  set_tests_properties(AbiProfiler${_case}Test PROPERTIES LABELS "benchmarking;abi")
endforeach ()
