# ==============================================================================
# InstallConsumer - A project that uses an installed vernier
#
# Built by InstallConsumer_test.cmake against an install of this tree, found
# through find_package(vernier) and nothing else: no include directory or
# library path of its own. Each program is built twice, once per include
# style ("Perf.hpp" and "src/bench/inc/Perf.hpp").
#
# Executables (4):
#   bench_consumer_short, bench_consumer_qualified      PERF_MAIN benchmark
#   monitor_consumer_short, monitor_consumer_qualified  monitor with a file sink
# ==============================================================================

cmake_minimum_required(VERSION 3.24)
project(vernier_install_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vernier REQUIRED)

# ------------------------------------------------------------------------------
# Consumer Programs (4)
# ------------------------------------------------------------------------------
# PERF_MAIN() supplies main() and runs GoogleTest, so the benchmark links
# GTest::gtest, which the package's find_dependency(GTest) provides.
# ------------------------------------------------------------------------------

foreach (_style short qualified)
  add_executable(bench_consumer_${_style} BenchConsumer.cpp)
  target_link_libraries(bench_consumer_${_style} PRIVATE vernier::bench GTest::gtest)

  add_executable(monitor_consumer_${_style} MonitorConsumer.cpp)
  target_link_libraries(monitor_consumer_${_style} PRIVATE vernier::monitor)

  if (_style STREQUAL "qualified")
    target_compile_definitions(bench_consumer_${_style} PRIVATE VERNIER_CONSUMER_QUALIFIED_INCLUDES)
    target_compile_definitions(
      monitor_consumer_${_style} PRIVATE VERNIER_CONSUMER_QUALIFIED_INCLUDES
    )
  endif ()
endforeach ()
