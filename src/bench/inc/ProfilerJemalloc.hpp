#ifndef VERNIER_PROFILERJEMALLOC_HPP
#define VERNIER_PROFILERJEMALLOC_HPP
/**
 * @file ProfilerJemalloc.hpp
 * @brief jemalloc heap profiler backend.
 *
 * jemalloc's built-in sampling profiler ("prof") is the lowest-overhead
 * heap profiler in the vernier ladder (~5-10% vs ~20x for massif, ~1.5x
 * for heaptrack). It works without recompiling the benchmark binary: the
 * user LD_PRELOADs libjemalloc.so and sets `MALLOC_CONF=prof:true,...`.
 *
 * Wraps the binary externally (same pattern as heaptrack):
 *
 *   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so \
 *   MALLOC_CONF=prof:true,prof_prefix:./Foo.Bar.jemalloc/jeprof \
 *       ./MyTest --profile jemalloc --cycles 1000 --gtest_filter='Foo.Bar'
 *
 *   jeprof --text ./MyTest ./Foo.Bar.jemalloc/jeprof.*.heap | head -20
 *   jeprof --pdf  ./MyTest ./Foo.Bar.jemalloc/jeprof.*.heap > flame.pdf
 *
 * When to reach for which:
 *   - massif       full timeline, lab use only (~20x)
 *   - heaptrack    moderate runs, allocation-site rank (~1.5x)
 *   - jemalloc     production-ish workloads, longest sampling window (~5-10%)
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- JemallocProfiler ----------------------------- */

class JemallocProfiler final : public Profiler {
public:
  JemallocProfiler(const PerfConfig& cfg, std::string testName);
  ~JemallocProfiler() override = default;

  std::string toolName() const noexcept override { return "jemalloc"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  bool runningUnderJemalloc_{false};
};

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeJemallocProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERJEMALLOC_HPP
