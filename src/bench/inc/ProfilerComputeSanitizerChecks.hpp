#ifndef VERNIER_PROFILERCOMPUTESANITIZERCHECKS_HPP
#define VERNIER_PROFILERCOMPUTESANITIZERCHECKS_HPP
/**
 * @file ProfilerComputeSanitizerChecks.hpp
 * @brief The Compute Sanitizer backend's tool parser and readiness check: CPU
 * code in libbench, so builds without CUDA compile, test and use them.
 *
 * compute-sanitizer checks a process only when it starts it. The check
 * decides in the doctor whether the tool runs here (`compute-sanitizer
 * --version`), and in a run whether it started this process: what it exports
 * to the process it starts, or, for a version that exports nothing, its
 * libraries in the process's memory map. Its checking of the kernels cannot
 * be seen from here, so a request that may proceed is unverified. libbench
 * registers this check with a passive profiler that names the tool and its
 * folder; the CUDA library replaces that registration with the backend.
 *
 * @note NOT RT-safe (heap allocation, fork/exec probes, file reads).
 */

#include <optional>
#include <string>

#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {

/**
 * @brief The compute-sanitizer tool @p profileArgs selects (`memcheck`, the
 * default, `racecheck`, `synccheck` or `initcheck`), or the refusal of a word
 * it does not take or of two tools at once, as `bench run` refuses them.
 * @p tool is the first tool named even when the request is refused, so a
 * backend built directly keeps it; "memcheck" when none is.
 */
[[nodiscard]] std::optional<ReadinessResult> parseSanitizerTool(const std::string& profileArgs,
                                                                std::string& tool);

/** @brief The Compute Sanitizer backend's decision for @p request in @p ctx. */
[[nodiscard]] ReadinessResult checkComputeSanitizerRequest(const ReadinessRequest& request,
                                                           const ReadinessContext& ctx);

/**
 * @brief checkComputeSanitizerRequest() reading @p mapsText as the process's
 * memory map at run time (for tests; the check reads /proc/<pid>/maps).
 */
[[nodiscard]] ReadinessResult checkComputeSanitizerRequestWithMaps(const ReadinessRequest& request,
                                                                   const ReadinessContext& ctx,
                                                                   const std::string& mapsText);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERCOMPUTESANITIZERCHECKS_HPP
