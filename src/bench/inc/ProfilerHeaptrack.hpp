#ifndef VERNIER_PROFILERHEAPTRACK_HPP
#define VERNIER_PROFILERHEAPTRACK_HPP
/**
 * @file ProfilerHeaptrack.hpp
 * @brief Heaptrack heap profiler backend (low-overhead alternative to massif).
 *
 * Heaptrack uses LD_PRELOAD to intercept malloc / free at runtime and records
 * every call with its call stack. The program otherwise runs natively and
 * pays per allocation, so the slowdown grows with the allocation rate: small
 * for code that seldom allocates, several times for code that allocates
 * millions of times a second. Massif runs every instruction under valgrind,
 * so heaptrack stays usable where massif would be too slow. The trade-off is
 * that heaptrack captures less detail per allocation than massif's full
 * timeline, but it produces the same kind of "where is allocation pressure
 * coming from" picture via heaptrack_print / heaptrack_gui.
 *
 * Heaptrack records only when it runs the whole process; a request run
 * without it fails (exit status 4) and prints the wrap command:
 *
 *   bench run ./MyTest --profile heaptrack --cycles 1000 --gtest_filter='Foo.Bar'
 *
 *   # or by hand:
 *   heaptrack -o run.heaptrack \
 *       ./MyTest --profile heaptrack --cycles 1000 --gtest_filter='Foo.Bar'
 *
 *   heaptrack_print run.heaptrack.* | head -40   # .gz or .zst, whichever was written
 *   heaptrack_gui   run.heaptrack.*             # interactive flamegraph
 *
 * When to reach for which:
 *   - massif       full timeline, lab use, ~20x overhead
 *   - heaptrack    allocation-site rank; cost grows with the allocation rate
 *   - jemalloc     sampling-based, ~5-10% overhead, requires libjemalloc
 *                  available at LD_PRELOAD time (see ProfilerJemalloc.hpp)
 */

#include <memory>
#include <string>
#include <vector>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- HeaptrackPlan ----------------------------- */

/** @brief What the heaptrack check verified, for the launch to use. */
struct HeaptrackPlan final : ReadinessPlan {
  std::string heaptrack;                            ///< The resolved heaptrack (doctor scopes).
  LaunchContext launch = LaunchContext::IN_PROCESS; ///< At the runtime scope: how the wrap began.
};

/* ----------------------------- HeaptrackProfiler ----------------------------- */

class HeaptrackProfiler final : public Profiler {
public:
  /** @brief Decide the request now; prints the decision when it cannot record. */
  HeaptrackProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  HeaptrackProfiler(const PerfConfig& cfg, std::string testName,
                    std::shared_ptr<const HeaptrackPlan> plan);

  ~HeaptrackProfiler() override = default;

  std::string toolName() const noexcept override { return "heaptrack"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const HeaptrackPlan> plan_;
};

/* --------------------------------- API --------------------------------- */

/** @brief How long the doctor's heaptrack probe may run. */
inline constexpr int HEAPTRACK_PROBE_TIMEOUT_MS = 30000;

/**
 * @brief heaptrack's arguments writing its trace into @p dir, before the
 * benchmark: the route `bench run` starts (tools/rust/src/bench/runner.rs),
 * pinned by the shared route table.
 */
std::vector<std::string> heaptrackWrapArguments(const std::string& dir);

/**
 * @brief True when @p mapsText (a /proc/<pid>/maps) shows heaptrack's
 * library mapped: its preload, in a program heaptrack started, or its
 * injection, in one it attached to.
 */
bool heaptrackMapped(const std::string& mapsText);

/**
 * @brief The heaptrack backend's readiness decision for @p request in @p ctx.
 *
 * The doctor's scopes run `heaptrack -o <private directory>/probe /bin/true`,
 * which must exit 0 and write a trace; with libtcmalloc loaded in the process
 * it is a caveat (C++ allocations do not reach heaptrack). A run reads its own
 * memory map for heaptrack's library and fails without it, printing the wrap
 * command. heaptrack takes no mode. --profile-analyze is an analysis-stage
 * error: heaptrack has no automatic analysis (heaptrack_print reads the
 * trace). On success the result's plan is a HeaptrackPlan.
 */
ReadinessResult checkHeaptrackRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/**
 * @brief checkHeaptrackRequest() for a process whose memory map is
 * @p mapsText (the two-argument form reads the map of @p ctx's process).
 */
ReadinessResult checkHeaptrackRequestWithMaps(const ReadinessRequest& request,
                                              const ReadinessContext& ctx,
                                              const std::string& mapsText);

std::unique_ptr<Profiler> makeHeaptrackProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERHEAPTRACK_HPP
