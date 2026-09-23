# ==============================================================================
# AbiConsumers.cmake - Layout-check consumers for the bench unit tests
#
# Included from the bench unit-test CMakeLists.txt. Everything that belongs to
# the two AbiConsumer programs lives here: the generated header tree, the two
# targets, and the tests that run them.
#
# Why they exist: the start-up layout check compares what a benchmark was
# compiled with against what libbench was compiled with. Numbers handed to the
# check from a unit test cannot show what happens when the two really differ,
# so AbiConsumer.cpp is compiled twice against the same libbench: once with the
# tree's headers, once with a copy in which PerfConfig has one more member.
#
# Targets:
#   AbiConsumerMatch     - built from the tree's headers
#   AbiConsumerMismatch  - built from the generated header copy
# Tests:
#   AbiConsumerMatchTest, AbiConsumerMismatchTest, AbiConsumerDriverTest
# ==============================================================================

# ------------------------------------------------------------------------------
# Generated Header Tree
# ------------------------------------------------------------------------------
# One producer, one user. The rule's declared output is the edited
# PerfConfig.hpp, which every generation rewrites, and that file is a source of
# AbiConsumerMismatch and of nothing else. So the rule that deletes, copies and
# edits the tree belongs to that target alone, the target's objects are
# compiled after it, and because the consumer includes that header its object
# is rebuilt in the same build whenever the tree is regenerated. A dependency
# attached to the shared source file would instead put a second copy of the
# rule into AbiConsumerMatch, which a parallel Makefiles build can run
# alongside the first.
# ------------------------------------------------------------------------------

set(_abi_headers_root "${CMAKE_CURRENT_BINARY_DIR}/abi_mismatch_headers")
set(_abi_mismatched_header "${_abi_headers_root}/src/bench/inc/PerfConfig.hpp")
file(GLOB _bench_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/bench/inc/*")
add_custom_command(
  OUTPUT "${_abi_mismatched_header}"
  COMMAND "${CMAKE_COMMAND}" -DSRC=${PROJECT_SOURCE_DIR} -DDST=${_abi_headers_root} -P
          "${CMAKE_CURRENT_LIST_DIR}/AbiMismatchHeaders.cmake"
  DEPENDS ${_bench_headers} "${CMAKE_CURRENT_LIST_DIR}/AbiMismatchHeaders.cmake"
  COMMENT "Generating the mismatched header tree for AbiConsumerMismatch"
  VERBATIM
)

# ------------------------------------------------------------------------------
# Consumer Programs (2)
# ------------------------------------------------------------------------------
# Test fixtures, not products: never installed or exported, no UPX copy.
# ------------------------------------------------------------------------------

vernier_add_app(
  NAME
  AbiConsumerMatch
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/AbiConsumer.cpp"
  LINK
  bench
  NO_INSTALL
  NO_UPX
)

vernier_add_app(
  NAME
  AbiConsumerMismatch
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/AbiConsumer.cpp"
  "${_abi_mismatched_header}"
  LINK
  bench
  NO_INSTALL
  NO_UPX
)

# The helper skips apps on platforms without POSIX; the tests go with them.
if (NOT TARGET AbiConsumerMatch OR NOT TARGET AbiConsumerMismatch)
  return()
endif ()

# Ahead of every other include directory, the project root included: the
# generated copies have to shadow the tree's headers of the same name. The
# helper's INC argument appends, so this one line stays outside it.
target_include_directories(AbiConsumerMismatch BEFORE PRIVATE "${_abi_headers_root}")

# ------------------------------------------------------------------------------
# Tests (3)
# ------------------------------------------------------------------------------

foreach (_case Match Mismatch)
  string(TOLOWER "${_case}" _expect)
  add_test(
    NAME AbiConsumer${_case}Test
    COMMAND
      "${CMAKE_COMMAND}" -DCONSUMER=$<TARGET_FILE:AbiConsumer${_case}> -DEXPECT=${_expect}
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/abi_consumer_${_expect} -P
      "${CMAKE_CURRENT_LIST_DIR}/AbiConsumer_test.cmake"
  )
  set_tests_properties(AbiConsumer${_case}Test PROPERTIES LABELS "benchmarking;abi")
endforeach ()

# The driver's own check: a sanitizer report in a consumer's output fails the
# run, also when the consumer exits 0 (recovering UBSan).
add_test(
  NAME AbiConsumerDriverTest
  COMMAND
    "${CMAKE_COMMAND}" -DDRIVER=${CMAKE_CURRENT_LIST_DIR}/AbiConsumer_test.cmake
    -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/abi_consumer_driver -P
    "${CMAKE_CURRENT_LIST_DIR}/AbiConsumerDriver_test.cmake"
)
set_tests_properties(AbiConsumerDriverTest PROPERTIES LABELS "benchmarking;abi")
