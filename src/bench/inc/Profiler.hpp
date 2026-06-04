#ifndef VERNIER_PROFILER_HPP
#define VERNIER_PROFILER_HPP
/**
 * @file Profiler.hpp
 * @brief Lightweight facade for optional profilers (perf, gperftools, bpftrace, RAPL, callgrind).
 */

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfRegistry.hpp"     // stamp profile metadata
#include "src/bench/inc/ProfilerRegistry.hpp" // backend self-registration

namespace vernier {
namespace bench {

/* ------------------------------- Profiler ------------------------------- */

/**
 * @note NOT RT-safe (virtual dispatch, heap allocation, may spawn subprocesses).
 */
class Profiler {
public:
  virtual ~Profiler() = default;

  /** @return stable tool name ("perf", "gperf", "bpftrace", "rapl", "callgrind") or empty. */
  virtual std::string toolName() const noexcept = 0;

  /** @return directory path where artifacts are written (may be empty for no-op). */
  virtual std::string artifactDir() const noexcept = 0;

  /** Called immediately before the measured window begins. */
  virtual void beforeMeasure() {}

  /** Called immediately after the measured window ends; receives summary stats. */
  virtual void afterMeasure(const Stats& /*s*/) {}

  /**
   * @brief Factory: returns a concrete profiler or a no-op based on cfg.
   * No-Op if cfg.profileTool is empty or unsupported on this platform.
   */
  static std::unique_ptr<Profiler> make(const PerfConfig& cfg, const std::string& testName);
};

/* -------------------------- Detail Implementation -------------------------- */

namespace detail {

class NoOpProfiler final : public Profiler {
public:
  explicit NoOpProfiler(std::string tool = {}, std::string dir = {})
      : tool_(std::move(tool)), dir_(std::move(dir)) {}
  std::string toolName() const noexcept override { return tool_; }
  std::string artifactDir() const noexcept override { return dir_; }

private:
  std::string tool_;
  std::string dir_;
};

} // namespace detail

/* --------------------------------- API --------------------------------- */

inline std::unique_ptr<Profiler> Profiler::make(const PerfConfig& cfg,
                                                const std::string& testName) {
  // Default: no profiling requested.
  if (cfg.profileTool.empty()) {
    return std::make_unique<detail::NoOpProfiler>();
  }
  // Dispatch via registry. Backends self-register at static init via
  // VERNIER_REGISTER_PROFILER_BACKEND in their translation units.
  return ProfilerRegistry::instance().make(cfg.profileTool, cfg, testName);
}

/**
 * @brief Helper to attach profiler hooks to a PerfCase.
 * Creates a profiler instance that lives through both hooks.
 * @note NOT RT-safe (heap allocation via shared_ptr).
 */
inline void attachProfilerHooks(PerfCase& pc, const PerfConfig& cfg) {
  // Keep the profiler alive across both lambdas via shared_ptr.
  auto prof = std::shared_ptr<Profiler>(Profiler::make(cfg, pc.testName()).release());

  pc.setBeforeMeasureHook([prof](const PerfCase&) { prof->beforeMeasure(); });
  pc.setAfterMeasureHook([prof](const PerfCase&, const Stats& s) {
    prof->afterMeasure(s);
    // Stamp CSV metadata for this test
    PerfRegistry::instance().updateProfileMeta(prof->toolName(), prof->artifactDir());
  });
}

/**
 * @brief Factory to create a PerfCase with profiler hooks auto-attached.
 *
 * This is the preferred way to create a PerfCase when profiling may be used.
 * Eliminates the need for manual `attachProfilerHooks()` calls.
 *
 * @param testName Full test name (Suite.Case format)
 * @param cfg Configuration with profiler settings
 * @return PerfCase with hooks attached (no-op if no profiler configured)
 * @note NOT RT-safe (heap allocation, may spawn subprocesses).
 */
inline PerfCase makePerfCaseWithProfiler(std::string testName, const PerfConfig& cfg) {
  PerfCase pc{std::move(testName), cfg};
  attachProfilerHooks(pc, cfg);
  return pc;
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILER_HPP