#ifndef VERNIER_PROFILERMEMCHECK_HPP
#define VERNIER_PROFILERMEMCHECK_HPP
/**
 * @file ProfilerMemcheck.hpp
 * @brief Valgrind Memcheck memory error / leak detector backend.
 *
 * Memcheck catches the classic C/C++ memory bugs:
 *  - Use of uninitialized memory
 *  - Reads/writes after free()
 *  - Reads/writes past the end of malloc'd blocks
 *  - Memory leaks (definitely lost / indirectly lost / possibly lost)
 *  - Mismatched malloc/new and free/delete
 *
 * Not a perf profiler per se -- valgrind memcheck slows execution ~20x. The
 * value is running memcheck alongside benchmarks to catch correctness
 * regressions introduced by an optimization pass.
 *
 * Wraps the binary externally (same as callgrind / massif):
 *
 *   valgrind --tool=memcheck --leak-check=full --error-exitcode=1 \
 *       --log-file=run.memcheck.log \
 *       ./MyTest --profile memcheck --cycles 5 --gtest_filter='Foo.Bar'
 *
 * Tools (selectable via --profile-args, passed through to memcheck):
 *   default               leak-check=summary, no track-origins
 *   "leak-full"           leak-check=full
 *   "track-origins"       track-origins=yes (helps locate uninit-read sources)
 *   combinations allowed; profileArgs is a substring match.
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- MemcheckProfiler ----------------------------- */

class MemcheckProfiler final : public Profiler {
public:
  MemcheckProfiler(const PerfConfig& cfg, std::string testName);
  ~MemcheckProfiler() override = default;

  std::string toolName() const noexcept override { return "memcheck"; }
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

std::unique_ptr<Profiler> makeMemcheckProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERMEMCHECK_HPP
