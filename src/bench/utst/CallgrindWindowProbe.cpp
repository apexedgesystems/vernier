/**
 * @file CallgrindWindowProbe.cpp
 * @brief A benchmark whose callgrind profile shows which of its phases were
 * recorded, used to check the callgrind backend's measured window.
 *
 * Its one test does work before its measured window, inside it and after it,
 * each phase in a function of its own, so the function names in a profile say
 * what callgrind recorded. CallgrindWindowProbe_test.cmake runs it under the
 * wrap the backend's hint prints, which must record the measured window and
 * none of the work before or after it, and under the wrap
 * `bench run --profile callgrind` uses, which must record the whole process.
 *
 * Usage: CallgrindWindowProbe --profile callgrind --cycles N --repeats 1
 */

#include <gtest/gtest.h>

#include "src/bench/inc/Perf.hpp"

namespace {

/* ----------------------------- Constants ----------------------------- */

/// Loop steps per phase call: enough instructions to register in a profile.
constexpr unsigned PHASE_STEPS = 20000;

/* ----------------------------- File Helpers ----------------------------- */

// One function per phase. The arithmetic differs so that identical-code
// folding cannot merge two phases under one name, and the volatile keeps each
// loop from being folded away.

[[gnu::noinline]] unsigned workBeforeWindow(unsigned seed) {
  volatile unsigned x = seed;
  for (unsigned i = 0; i < PHASE_STEPS; ++i) {
    x = x * 3u + 1u;
  }
  return x;
}

[[gnu::noinline]] unsigned workInsideWindow(unsigned seed) {
  volatile unsigned x = seed;
  for (unsigned i = 0; i < PHASE_STEPS; ++i) {
    x = x * 5u + 7u;
  }
  return x;
}

[[gnu::noinline]] unsigned workAfterWindow(unsigned seed) {
  volatile unsigned x = seed;
  for (unsigned i = 0; i < PHASE_STEPS; ++i) {
    x = x * 11u + 13u;
  }
  return x;
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/** @brief Work before, inside and after one measured window. */
PERF_TEST(CallgrindWindow, Phases) {
  PERF_GUARD(perf);

  volatile unsigned sink = workBeforeWindow(1u);
  perf.throughputLoop([&] { sink = workInsideWindow(sink); }, "inside");
  sink = workAfterWindow(sink);
}

PERF_MAIN()
