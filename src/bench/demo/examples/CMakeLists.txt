# ==============================================================================
# Demo Examples
#
# Small pieces of code the walkthroughs measure, one directory per example.
# Each example owns its build: <name>/CMakeLists.txt declares its library and
# adds utst/, its unit tests and the process fixtures they register, and ptst/,
# its performance tests, where it has them. Examples are teaching material:
# they are built with the demos and are never installed or exported, so their
# libraries use plain add_library rather than vernier_add_library.
#
# Examples:
#   join    - demo_example_join: join a list of strings with a separator, two
#             ways
#   filter  - demo_example_filter: copy the values above a threshold, with a
#             branch per value and without one
#   saxpy   - demo_example_saxpy: y = a*x + y as a CPU loop and two ways of
#             driving one CUDA kernel; the library and its device tests are
#             built only with the GPU demos
#
# Test support:
#   utst/HardwareCounter.hpp - one hardware event counted around a call, for
#                              the join and filter tests that read the
#                              counters; no demo includes it
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

add_subdirectory(join)
add_subdirectory(filter)
add_subdirectory(saxpy)
