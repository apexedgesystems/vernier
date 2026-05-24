#ifndef VERNIER_PROFILERMASSIF_HPP
#define VERNIER_PROFILERMASSIF_HPP
/**
 * @file ProfilerMassif.hpp
 * @brief Valgrind Massif heap profiler backend.
 *
 * Massif samples heap usage over time, producing a `massif.out.<pid>` file
 * that ms_print renders as a stacked timeline of allocation sites. Complements
 * callgrind (which is CPU/instruction-focused): use massif when the question
 * is "what is allocating, and how much."
 *
 * Wraps the binary externally (same as callgrind):
 *
 *   valgrind --tool=massif --massif-out-file=run.massif.out \
 *       ./MyTest --profile massif --cycles 10 --gtest_filter='Foo.Bar'
 *
 *   ms_print run.massif.out | head -40
 *
 * Tools (selectable via --profile-args):
 *   default               heap allocations only
 *   "pages"               page-level profiling (--pages-as-heap=yes)
 *   "stacks"              include stack allocations (--stacks=yes)
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- MassifProfiler ----------------------------- */

class MassifProfiler final : public Profiler {
public:
  MassifProfiler(const PerfConfig& cfg, std::string testName);
  ~MassifProfiler() override = default;

  std::string toolName() const noexcept override { return "massif"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  bool runningUnderValgrind_{false};
};

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeMassifProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERMASSIF_HPP
