#ifndef VERNIER_PROFILERHEAPTRACK_HPP
#define VERNIER_PROFILERHEAPTRACK_HPP
/**
 * @file ProfilerHeaptrack.hpp
 * @brief Heaptrack heap profiler backend (low-overhead alternative to massif).
 *
 * Heaptrack uses LD_PRELOAD to intercept malloc / free at runtime, which is
 * an order of magnitude cheaper than valgrind massif (~1.5x vs ~20x) and so
 * usable on workloads where massif would be too slow. The trade-off is that
 * heaptrack captures less detail per allocation than massif's full timeline,
 * but it produces the same kind of "where is allocation pressure coming
 * from" picture via heaptrack_print / heaptrack_gui.
 *
 * Wraps the binary externally (same pattern as callgrind / massif):
 *
 *   heaptrack -o run.heaptrack \
 *       ./MyTest --profile heaptrack --cycles 1000 --gtest_filter='Foo.Bar'
 *
 *   heaptrack_print run.heaptrack.zst | head -40
 *   heaptrack_gui   run.heaptrack.zst      # interactive flamegraph
 *
 * When to reach for which:
 *   - massif       full timeline, lab use, ~20x overhead
 *   - heaptrack    production-ish runs, ~1.5x overhead, allocation-site rank
 *   - jemalloc     sampling-based, ~5-10% overhead, requires libjemalloc
 *                  available at LD_PRELOAD time (see ProfilerJemalloc.hpp)
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- HeaptrackProfiler ----------------------------- */

class HeaptrackProfiler final : public Profiler {
public:
  HeaptrackProfiler(const PerfConfig& cfg, std::string testName);
  ~HeaptrackProfiler() override = default;

  std::string toolName() const noexcept override { return "heaptrack"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  bool runningUnderHeaptrack_{false};
};

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeHeaptrackProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERHEAPTRACK_HPP
