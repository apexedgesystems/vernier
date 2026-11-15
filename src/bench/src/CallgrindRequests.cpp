/**
 * @file CallgrindRequests.cpp
 * @brief libbench's callgrind client requests: the one place it includes
 *        valgrind's callgrind.h.
 */

#include "src/bench/src/CallgrindRequests.hpp"

#include <atomic>

// Compiled in where this build finds valgrind's callgrind.h. NVALGRIND, defined
// by the build or by valgrind.h itself on a platform valgrind does not
// support, reduces each request to nothing.
#if defined(__has_include)
#if __has_include(<valgrind/callgrind.h>)
#include <valgrind/callgrind.h>
#define VERNIER_CALLGRIND_HEADER_FOUND 1
#endif
#endif

namespace vernier {
namespace bench {
namespace callgrind {

/* ----------------------------- API ----------------------------- */

bool requestsBuiltIn() noexcept {
#if defined(VERNIER_CALLGRIND_HEADER_FOUND) && !defined(NVALGRIND)
  return true;
#else
  return false;
#endif
}

void zeroStatsOnce() noexcept {
  static std::atomic<bool> zeroed{false};
  if (zeroed.exchange(true)) {
    return;
  }
#if defined(VERNIER_CALLGRIND_HEADER_FOUND)
  CALLGRIND_ZERO_STATS;
#endif
}

void startInstrumentation() noexcept {
#if defined(VERNIER_CALLGRIND_HEADER_FOUND)
  CALLGRIND_START_INSTRUMENTATION;
#endif
}

void stopInstrumentation() noexcept {
#if defined(VERNIER_CALLGRIND_HEADER_FOUND)
  CALLGRIND_STOP_INSTRUMENTATION;
#endif
}

} // namespace callgrind
} // namespace bench
} // namespace vernier
