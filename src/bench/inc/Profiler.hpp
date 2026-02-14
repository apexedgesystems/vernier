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
#include "src/bench/inc/PerfRegistry.hpp" // stamp profile metadata

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

// Forward declarations of backend factories (defined in their respective headers/translation units)
std::unique_ptr<Profiler> makePerfProfiler(const PerfConfig& cfg, const std::string& testName);
std::unique_ptr<Profiler> makeGperfProfiler(const PerfConfig& cfg, const std::string& testName);
std::unique_ptr<Profiler> makeBpftraceProfiler(const PerfConfig& cfg, const std::string& testName);
std::unique_ptr<Profiler> makeRAPLProfiler(const PerfConfig& cfg, const std::string& testName);
std::unique_ptr<Profiler> makeCallgrindProfiler(const PerfConfig& cfg, const std::string& testName);

inline std::unique_ptr<Profiler> Profiler::make(const PerfConfig& cfg,
                                                const std::string& testName) {
  // Default: no profiling requested.
  if (cfg.profileTool.empty()) {
    return std::make_unique<detail::NoOpProfiler>();
  }

  // Helper: warn when a requested profiler is unavailable on this platform.
  auto warnUnavailable = [&](const char* tool, const char* hint) {
    std::fprintf(stderr,
                 "\n[WARN] Profiler '%s' requested but unavailable on this platform.\n"
                 "   %s\n"
                 "   Falling back to no-op (measurements will proceed without profiling).\n\n",
                 tool, hint);
  };

  // Dispatch to known backends.
  if (cfg.profileTool == "perf") {
    if (auto p = makePerfProfiler(cfg, testName))
      return p;
    warnUnavailable("perf", "Install linux-tools-$(uname -r) or run outside Docker.");
    return std::make_unique<detail::NoOpProfiler>("perf", "");
  }
  if (cfg.profileTool == "gperf") {
    if (auto p = makeGperfProfiler(cfg, testName))
      return p;
    warnUnavailable("gperf", "Install libgperftools-dev and rebuild.");
    return std::make_unique<detail::NoOpProfiler>("gperf", "");
  }
  if (cfg.profileTool == "bpftrace") {
    if (auto p = makeBpftraceProfiler(cfg, testName))
      return p;
    warnUnavailable("bpftrace", "Install bpftrace and run with root/sudo.");
    return std::make_unique<detail::NoOpProfiler>("bpftrace", "");
  }
  if (cfg.profileTool == "rapl") {
    if (auto p = makeRAPLProfiler(cfg, testName))
      return p;
    warnUnavailable("rapl", "Requires Intel CPU + 'sudo modprobe msr' + CAP_SYS_RAWIO.");
    return std::make_unique<detail::NoOpProfiler>("rapl", "");
  }
  if (cfg.profileTool == "callgrind") {
    if (auto p = makeCallgrindProfiler(cfg, testName))
      return p;
    warnUnavailable("callgrind", "Install valgrind: apt install valgrind.");
    return std::make_unique<detail::NoOpProfiler>("callgrind", "");
  }

  // Unknown profiler: return a named no-op so CSV can still reflect the request.
  std::fprintf(stderr,
               "\n[WARN] Unknown profiler '%s'. Available: perf, gperf, bpftrace, rapl, "
               "callgrind.\n\n",
               cfg.profileTool.c_str());
  return std::make_unique<detail::NoOpProfiler>(cfg.profileTool, "");
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