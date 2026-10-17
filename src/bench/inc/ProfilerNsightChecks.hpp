#ifndef VERNIER_PROFILERNSIGHTCHECKS_HPP
#define VERNIER_PROFILERNSIGHTCHECKS_HPP
/**
 * @file ProfilerNsightChecks.hpp
 * @brief The Nsight backend's mode parser and readiness check: CPU code in
 * libbench, so builds without CUDA compile, test and use them.
 *
 * nsys and ncu record a process only when they start it. The check therefore
 * decides from the selected mode's tool and from the session the process
 * shows: in the doctor, whether that tool runs here (`<tool> --version`); in a
 * run, whether nsys or ncu started this process. Neither can show from here
 * that the tool will capture the benchmark's GPU work, so a request that may
 * proceed is unverified. libbench registers this check for `nsight` and `ncu`
 * with a passive profiler that names the tool and its folder; the CUDA
 * library replaces those registrations with the Nsight backend.
 *
 * @note NOT RT-safe (heap allocation, fork/exec probes, file reads).
 */

#include <optional>
#include <string>

#include "src/bench/inc/ProfilerNsight.hpp" // NsightMode, ReplayMetrics
#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {

/**
 * @brief The Nsight mode @p backend ("nsight" or "ncu") selects with
 * @p profileArgs, or the refusal of a word it does not take.
 *
 * nsight: `compute` (or `ncu`) selects Nsight Compute, `replay` its replay
 * metrics, none Nsight Systems. ncu: `replay`, or none for Nsight Compute.
 * The words are split as `bench run` splits them. @p mode is set from the
 * words it takes even when another is refused: the check refuses the request,
 * and a backend built directly keeps the mode those words select.
 */
[[nodiscard]] std::optional<ReadinessResult>
parseNsightMode(const std::string& backend, const std::string& profileArgs, NsightMode& mode);

/** @brief The program @p mode runs under: "nsys" for Systems, "ncu" otherwise. */
[[nodiscard]] const char* nsightModeTool(NsightMode mode);

/** @brief The Nsight backend's decision for @p request (`nsight` or `ncu`) in @p ctx. */
[[nodiscard]] ReadinessResult checkNsightRequest(const ReadinessRequest& request,
                                                 const ReadinessContext& ctx);

/**
 * @brief checkNsightRequest() with the NVIDIA driver's parameter file at
 * @p driverParams, whose `RmProfilingAdminOnly: 1` restricts GPU performance
 * counters to root (for tests; the check reads /proc/driver/nvidia/params).
 */
[[nodiscard]] ReadinessResult checkNsightRequestWith(const ReadinessRequest& request,
                                                     const ReadinessContext& ctx,
                                                     const std::string& driverParams);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERNSIGHTCHECKS_HPP
