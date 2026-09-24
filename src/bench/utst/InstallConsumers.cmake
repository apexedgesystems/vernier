# ==============================================================================
# InstallConsumers.cmake - Installed-package consumer test for the bench unit
# tests
#
# Included from the bench unit-test CMakeLists.txt. Registers the test that
# installs this tree, moves the install, and builds and runs a PERF_MAIN
# benchmark and a monitor program against it through find_package(vernier)
# alone, in both include styles (InstallConsumer_test.cmake says how).
#
# The include directory an install uses is fixed when a tree is configured, so
# the test configures and builds its own CPU-only bench and monitor with a
# non-default one instead of installing the build under test.
#
# Tests:
#   InstallConsumerTest
# ==============================================================================

# Nothing built by a cross toolchain can run here, and a platform without POSIX
# skips the libraries. The monitor directory is configured after this one, so
# bench, which has the same platform requirement, stands for both.
if (CMAKE_CROSSCOMPILING OR NOT TARGET bench)
  return()
endif ()

# ------------------------------------------------------------------------------
# Tests (1)
# ------------------------------------------------------------------------------
# The nested builds run this many jobs; PROCESSORS reserves them from ctest's
# parallel slots.
# ------------------------------------------------------------------------------

set(_install_consumer_jobs 4)

add_test(
  NAME InstallConsumerTest
  COMMAND
    "${CMAKE_COMMAND}" -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
    -DGTEST_SOURCE_DIR=${googletest_SOURCE_DIR}
    -DCONSUMER_DIR=${CMAKE_CURRENT_LIST_DIR}/InstallConsumer
    -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install_consumer "-DGENERATOR=${CMAKE_GENERATOR}"
    -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DC_COMPILER=${CMAKE_C_COMPILER}
    -DCXX_COMPILER=${CMAKE_CXX_COMPILER} -DJOBS=${_install_consumer_jobs} -P
    "${CMAKE_CURRENT_LIST_DIR}/InstallConsumer_test.cmake"
)
set_tests_properties(
  InstallConsumerTest PROPERTIES LABELS "benchmarking;install" PROCESSORS ${_install_consumer_jobs}
                                 TIMEOUT 600
)
