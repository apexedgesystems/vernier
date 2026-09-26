#ifndef VERNIER_DEMO_SKIPUNLESSUNDERVALGRIND_HPP
#define VERNIER_DEMO_SKIPUNLESSUNDERVALGRIND_HPP
/**
 * @file SkipUnlessUnderValgrind.hpp
 * @brief Skip a demo's test case unless the process runs under valgrind.
 *
 * A checker needs a bug to find, so a demo that teaches a valgrind tool can
 * carry a case with a deliberate memory error. That case is for valgrind to
 * find; run anywhere else it would put a memory error into an ordinary test
 * run, so it skips itself unless the process is under valgrind, and says how
 * to run it.
 *
 * Shared by the demos; not part of the installed library.
 */

#include "src/bench/inc/ProfilerEnv.hpp"

#include <string>

#include <gtest/gtest.h>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- Constants ----------------------------- */

/// The skip message a case prints when the process is not under valgrind.
inline constexpr const char* SKIP_UNLESS_UNDER_VALGRIND_REASON =
    "this case makes a memory error for valgrind to find, so it runs only under valgrind: run "
    "this binary under `valgrind --tool=memcheck`, or through `bench run --profile memcheck`";

/* ----------------------------- API ----------------------------- */

/**
 * @brief Why a case that must only run under valgrind is skipped in this
 *        process: SKIP_UNLESS_UNDER_VALGRIND_REASON, or an empty string when
 *        the process runs under valgrind.
 *
 * Decides as the profiler backends do, from a valgrind preload library mapped
 * into the process.
 * @note NOT RT-safe: reads /proc/self/maps.
 */
inline std::string reasonToSkipUnlessUnderValgrind() {
  if (profiler_env::isRunningUnderValgrind()) {
    return {};
  }
  return SKIP_UNLESS_UNDER_VALGRIND_REASON;
}

} // namespace demo
} // namespace bench
} // namespace vernier

/**
 * @brief Skip the current test unless the process runs under valgrind, with
 *        reasonToSkipUnlessUnderValgrind()'s reason. The first statement of a
 *        test body.
 */
#define DEMO_SKIP_UNLESS_UNDER_VALGRIND()                                                          \
  do {                                                                                             \
    const std::string reasonToSkip = ::vernier::bench::demo::reasonToSkipUnlessUnderValgrind();    \
    if (!reasonToSkip.empty()) {                                                                   \
      GTEST_SKIP() << reasonToSkip;                                                                \
    }                                                                                              \
  } while (0)

#endif // VERNIER_DEMO_SKIPUNLESSUNDERVALGRIND_HPP
