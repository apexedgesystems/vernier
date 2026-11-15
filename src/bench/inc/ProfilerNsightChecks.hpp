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
#include <vector>

#include "src/bench/inc/ProfilerNsight.hpp" // NsightMode, ReplayMetrics
#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {

/**
 * @brief What an Nsight request collects: its mode and, under Nsight
 * Compute, the section set, sections and metrics it names.
 */
struct NsightSelection {
  NsightMode mode = NsightMode::Systems; ///< The tool and its report.
  std::string set;                       ///< --set NAME; empty without one.
  std::vector<std::string> sections;     ///< Each --section NAME, in order.
  std::vector<std::string> metrics;      ///< Every --metrics name, in order.

  /**
   * @brief ncu's options for the request, after bench run's own: the set,
   * each section, then one --metrics with every name; replay's metric list
   * (ReplayMetrics) for replay; none under Systems.
   */
  [[nodiscard]] std::vector<std::string> ncuOptions() const;

  /** @brief The report ncu writes: "kernel_replay" for replay, else "kernel_profile". */
  [[nodiscard]] const char* ncuReport() const;

  /**
   * @brief What the request collects, for a doctor row: the replay metrics
   * and their count, or ncuOptions() as given; empty when it names nothing.
   */
  [[nodiscard]] std::string describe() const;
};

/**
 * @brief Read @p profileArgs, a request for @p backend ("nsight" or "ncu"),
 * as Vernier's supported subset of Nsight Compute's options into
 * @p selection, or the refusal of its first problem.
 *
 * Accepted: `replay`; `--set NAME`, once; `--section NAME` and
 * `--metrics NAME`, each as often as needed, a metric list running to the
 * next option or mode word; each option also as `--option=VALUE`; and under
 * nsight the words `compute` and `ncu`. NAME is letters, digits and '_', and
 * a metric name may also hold '.'. A mode word is never a value. Under
 * nsight, none of these is Nsight Systems, and any of them Nsight Compute;
 * replay takes no selection. The words are split as `bench run` splits
 * them, and the refusal's text is the one `bench run` gives (the shared
 * route table's refuse rows hold both to it). @p selection is set from the
 * words it takes even when another is refused: the check refuses the
 * request, and a backend built directly keeps what those words select.
 */
[[nodiscard]] std::optional<ReadinessResult> parseNsightRequest(const std::string& backend,
                                                                const std::string& profileArgs,
                                                                NsightSelection& selection);

/**
 * @brief The Nsight mode @p backend ("nsight" or "ncu") selects with
 * @p profileArgs (parseNsightRequest()), or the refusal of its first problem.
 * @p mode is set from the words it takes even when another is refused.
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
