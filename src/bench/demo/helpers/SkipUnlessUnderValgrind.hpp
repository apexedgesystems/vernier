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

/// True when this binary was built with an address, thread or memory
/// sanitizer. valgrind does not run such a binary as it runs an ordinary one:
/// with the address sanitizer's runtime linked in, valgrind's own libraries
/// are not mapped into the process (seen with valgrind 3.22 and clang 21), so
/// memcheck sees no allocation and this helper sees no valgrind, and an older
/// valgrind gives up reading the binary before it runs. A test that starts
/// valgrind on this binary skips on it and says so.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
inline constexpr bool BUILT_WITH_A_SANITIZER = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) ||                         \
    __has_feature(memory_sanitizer)
inline constexpr bool BUILT_WITH_A_SANITIZER = true;
#else
inline constexpr bool BUILT_WITH_A_SANITIZER = false;
#endif
#else
inline constexpr bool BUILT_WITH_A_SANITIZER = false;
#endif

/// The skip message a test prints when it would start valgrind on a binary
/// built with a sanitizer.
inline constexpr const char* SANITIZER_UNDER_VALGRIND_REASON =
    "this binary is built with a sanitizer, and valgrind does not run such a binary as it runs "
    "an ordinary one (its libraries are not mapped into the process, so memcheck sees nothing); "
    "use a build without a sanitizer";

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
