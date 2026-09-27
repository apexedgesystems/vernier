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

/// True when this binary was built with the address or the thread sanitizer,
/// whose builds valgrind cannot check. Under valgrind 3.22 (clang 21) an
/// address-sanitizer build ran with none of valgrind's libraries mapped into
/// it, so memcheck saw no allocation and this helper no valgrind, and a
/// thread-sanitizer build never reached main; under valgrind 3.18.1 no
/// address-sanitizer build it was given reached main (GCC 11). Builds with
/// the undefined-behaviour sanitizer ran under valgrind as ordinary ones and
/// are not included. Only a test that starts valgrind on this binary skips on
/// it.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
inline constexpr bool BUILT_WITH_ASAN_OR_TSAN = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
inline constexpr bool BUILT_WITH_ASAN_OR_TSAN = true;
#else
inline constexpr bool BUILT_WITH_ASAN_OR_TSAN = false;
#endif
#else
inline constexpr bool BUILT_WITH_ASAN_OR_TSAN = false;
#endif

/// The skip message a test prints when it would start valgrind on a binary
/// built with the address or the thread sanitizer.
inline constexpr const char* SANITIZER_UNDER_VALGRIND_REASON =
    "this binary is built with the address or the thread sanitizer, whose builds valgrind "
    "cannot check; run this test in a build without either";

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
