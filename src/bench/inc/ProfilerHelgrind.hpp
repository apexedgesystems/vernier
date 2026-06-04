#ifndef VERNIER_PROFILERHELGRIND_HPP
#define VERNIER_PROFILERHELGRIND_HPP
/**
 * @file ProfilerHelgrind.hpp
 * @brief Valgrind Helgrind / DRD thread-error detector backend.
 *
 * The CPU analog of compute-sanitizer's race/sync checks. Helgrind catches the
 * classic multithreading bugs that on-CPU profilers can't see:
 *  - Data races (unsynchronized access to shared memory)
 *  - Lock-ordering violations (potential deadlocks)
 *  - Misuse of the POSIX pthreads API
 *
 * Complements memcheck (memory errors) and compute-sanitizer (GPU races):
 * run it alongside benchmarks of multithreaded code to catch concurrency
 * regressions introduced by an optimization pass. Slows execution heavily
 * (~20-100x), so use a low --cycles.
 *
 * Wraps the binary externally (same pattern as memcheck / massif):
 *
 *   valgrind --tool=helgrind --log-file=run.helgrind.log \
 *       ./MyTest --profile helgrind --cycles 5 --gtest_filter='Foo.Bar'
 *
 * Tool selection via --profile-args (substring match, passed through):
 *   default    helgrind (data races + lock order + pthread misuse)
 *   "drd"      DRD instead (lower memory, per-thread; also detects races)
 */

#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- HelgrindProfiler ----------------------------- */

class HelgrindProfiler final : public Profiler {
public:
  HelgrindProfiler(const PerfConfig& cfg, std::string testName);
  ~HelgrindProfiler() override = default;

  std::string toolName() const noexcept override { return "helgrind"; }
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

std::unique_ptr<Profiler> makeHelgrindProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERHELGRIND_HPP
