/**
 * @file AbiProfilerLibrary.cpp
 * @brief A bench library's two sides of the virtual interface check: the
 * start-up check (PerfAbi.cpp, linked beside this file) and a profiler the
 * library builds for header code to call through.
 *
 * Compiled into two small shared libraries: one from this tree's headers,
 * one from a copy as they were before Profiler had its abortMeasure() slot
 * (AbiBeforeAbortSlotHeaders.cmake).
 */

#include "src/bench/inc/Profiler.hpp"

/// A profiler built by the library, as libbench builds its backends.
extern "C" vernier::bench::Profiler* abiLibraryProfiler() {
  return new vernier::bench::detail::NoOpProfiler("library-built profiler");
}
