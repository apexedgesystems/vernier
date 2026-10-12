#ifndef VERNIER_DEMO_SKIPUNLESSUNDERCOMPUTESANITIZER_HPP
#define VERNIER_DEMO_SKIPUNLESSUNDERCOMPUTESANITIZER_HPP
/**
 * @file SkipUnlessUnderComputeSanitizer.hpp
 * @brief Skip a demo's test case unless the process runs under Compute
 *        Sanitizer.
 *
 * A checker needs a bug to find, so a demo that teaches compute-sanitizer
 * can carry a case that launches a kernel past the end of its buffers. That
 * case is for the tool to find; run anywhere else it would put a device
 * memory error into an ordinary test run, so it skips itself unless the
 * process is under compute-sanitizer, and says how to run it.
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

/// The skip message a case prints when the process is not under the tool.
inline constexpr const char* SKIP_UNLESS_UNDER_COMPUTE_SANITIZER_REASON =
    "this case launches a kernel past the end of its buffers for Compute Sanitizer to find, so "
    "it runs only under it: run this binary under `compute-sanitizer --tool=memcheck`, or "
    "through `bench run --profile compute-sanitizer`";

/* ----------------------------- API ----------------------------- */

/**
 * @brief Why a case that must only run under compute-sanitizer is skipped in
 *        this process: SKIP_UNLESS_UNDER_COMPUTE_SANITIZER_REASON, or an
 *        empty string when the process runs under the tool.
 *
 * Decides as the compute-sanitizer backend does, from what the tool exports
 * to the process it starts and from its libraries mapped into the process.
 * @note NOT RT-safe: reads /proc/self/maps.
 */
inline std::string reasonToSkipUnlessUnderComputeSanitizer() {
  if (profiler_env::isRunningUnderComputeSanitizer()) {
    return {};
  }
  return SKIP_UNLESS_UNDER_COMPUTE_SANITIZER_REASON;
}

} // namespace demo
} // namespace bench
} // namespace vernier

/**
 * @brief Skip the current test unless the process runs under
 *        compute-sanitizer, with reasonToSkipUnlessUnderComputeSanitizer()'s
 *        reason. The first statement of a test body.
 */
#define DEMO_SKIP_UNLESS_UNDER_COMPUTE_SANITIZER()                                                 \
  do {                                                                                             \
    const std::string reasonToSkip =                                                               \
        ::vernier::bench::demo::reasonToSkipUnlessUnderComputeSanitizer();                         \
    if (!reasonToSkip.empty()) {                                                                   \
      GTEST_SKIP() << reasonToSkip;                                                                \
    }                                                                                              \
  } while (0)

#endif // VERNIER_DEMO_SKIPUNLESSUNDERCOMPUTESANITIZER_HPP
