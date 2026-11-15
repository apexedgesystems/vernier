/**
 * @file CallgrindWindowProbe.cpp
 * @brief A benchmark whose callgrind profile shows which of its phases were
 * recorded, used to check the callgrind backend's measured window.
 *
 * Its one test does work before its measured window, inside it and after it,
 * each phase in a function of its own, so the function names in a profile say
 * what callgrind recorded; between two windows a third ends by an exception,
 * and the test's own work after that one is a phase of its own too.
 * CallgrindWindowProbe_test.cmake runs it under the wrap the backend's hint
 * prints, which must record the measured windows and none of the work before,
 * between or after them, and under the wrap `bench run --profile callgrind`
 * uses, which must record the whole process.
 * A second test prints whether libbench has callgrind's client requests,
 * which the driver asks before the cases that need them.
 *
 * Usage: CallgrindWindowProbe --profile callgrind --cycles N --repeats 1
 *        CallgrindWindowProbe --gtest_filter=CallgrindWindow.ClientRequests
 */

#include <gtest/gtest.h>

#include <cstdio>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/src/CallgrindRequests.hpp"

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

[[gnu::noinline]] unsigned workBetweenWindows(unsigned seed) {
  volatile unsigned x = seed;
  for (unsigned i = 0; i < PHASE_STEPS; ++i) {
    x = x * 17u + 19u;
  }
  return x;
}

/// What the window that ends by an exception throws.
struct WindowAborted {};

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @brief Work before a measured window and inside it; then a window that ends
 * by an exception, which the test catches, and the test's own work after it;
 * then a window again, and work after it.
 */
PERF_TEST(CallgrindWindow, Phases) {
  PERF_GUARD(perf);

  volatile unsigned sink = workBeforeWindow(1u);
  perf.throughputLoop([&] { sink = workInsideWindow(sink); }, "inside");
  try {
    perf.measured(
        [&] {
          sink = workInsideWindow(sink);
          throw WindowAborted{};
        },
        "aborted");
  } catch (const WindowAborted&) {
  }
  sink = workBetweenWindows(sink);
  perf.throughputLoop([&] { sink = workInsideWindow(sink); }, "again");
  sink = workAfterWindow(sink);
}

/** @brief Whether libbench has callgrind's client requests: only NVALGRIND compiles them out. */
TEST(CallgrindWindow, ClientRequests) {
  std::printf("callgrind client requests: %s\n",
              vernier::bench::callgrind::requestsBuiltIn() ? "built in" : "compiled out");
}

PERF_MAIN()
