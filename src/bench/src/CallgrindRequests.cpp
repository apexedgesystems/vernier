/**
 * @file CallgrindRequests.cpp
 * @brief libbench's callgrind client requests: the one place it includes
 *        valgrind's callgrind.h, the copy in src/bench/third_party/valgrind.
 */

#include "src/bench/src/CallgrindRequests.hpp"

#include <atomic>

// The copy libbench carries, so every build has the requests whatever its
// machine has installed. NVALGRIND, defined by the build or by valgrind.h
// itself on a platform valgrind does not support, reduces each request to
// nothing.
#include "src/bench/third_party/valgrind/callgrind.h"

namespace vernier {
namespace bench {
namespace callgrind {

/* ----------------------------- API ----------------------------- */

bool requestsBuiltIn() noexcept {
#if defined(NVALGRIND)
  return false;
#else
  return true;
#endif
}

void zeroStatsOnce() noexcept {
  static std::atomic<bool> zeroed{false};
  if (zeroed.exchange(true)) {
    return;
  }
  CALLGRIND_ZERO_STATS;
}

void startInstrumentation() noexcept { CALLGRIND_START_INSTRUMENTATION; }

void stopInstrumentation() noexcept { CALLGRIND_STOP_INSTRUMENTATION; }

} // namespace callgrind
} // namespace bench
} // namespace vernier
