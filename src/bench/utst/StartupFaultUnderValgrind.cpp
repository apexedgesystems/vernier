/**
 * @file StartupFaultUnderValgrind.cpp
 * @brief Test fixture: a program that aborts during static initialization when
 * it runs under valgrind, and works normally otherwise.
 *
 * Linked into a copy of CallgrindWindowProbe and a copy of demo 07, it gives
 * their callgrind checks a program that dies before its tests start, the way a
 * real startup fault would, while the same program run plainly still does its
 * part (the window test's first, unwrapped run; demo 07's own process, which
 * starts the counting runs). The checks must report that death as a failure,
 * not skip it as a limitation of valgrind.
 */

#include <cstdio>
#include <cstdlib>

#include "src/bench/inc/ProfilerEnv.hpp"

namespace {

/* ----------------------------- Constants ----------------------------- */

/// Printed before the abort, so a failure's output shows where it came from.
constexpr const char* FAULT_MESSAGE =
    "StartupFaultUnderValgrind: aborting before the tests start, as intended\n";

/* ----------------------------- Fixture ----------------------------- */

/// Aborts in its constructor when the process runs under valgrind.
struct AbortUnderValgrind {
  AbortUnderValgrind() {
    if (vernier::bench::profiler_env::isRunningUnderValgrind()) {
      std::fputs(FAULT_MESSAGE, stderr);
      std::abort();
    }
  }
};

const AbortUnderValgrind ABORT_UNDER_VALGRIND;

} // namespace
