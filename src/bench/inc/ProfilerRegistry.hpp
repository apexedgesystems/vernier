#ifndef VERNIER_PROFILERREGISTRY_HPP
#define VERNIER_PROFILERREGISTRY_HPP
/**
 * @file ProfilerRegistry.hpp
 * @brief Self-registration registry for profiler backends.
 *
 * Each backend (perf, gperf, callgrind, bpftrace, rapl, nsight, ...)
 * registers a factory + an availability hint at static-init time via
 * VERNIER_REGISTER_PROFILER_BACKEND. Profiler::make() dispatches by
 * looking up the registered factory rather than a hard-coded if-chain,
 * so new backends slot in by adding a file + one registration line.
 *
 * Threading:
 *  - Registration runs during static init (single-threaded).
 *  - make() / hasBackend() / backendNames() are const lookups, safe
 *    after init completes.
 *
 * @note NOT RT-safe (std::map, std::string, std::function).
 */

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernier {
namespace bench {

// Forward declarations to keep this header light and avoid a circular include
// (PerfConfig.hpp pulls registry-driven diagnostics back in).
struct PerfConfig;
class Profiler;

/* ------------------------------ EnvReport ------------------------------ */

/**
 * @brief Structured result of a backend's environment pre-flight check.
 *
 * Status:
 *  - Ok       backend is fully functional in the current environment
 *  - Warning  backend works with caveats (e.g. kernel-symbol resolution degraded)
 *  - Error    backend cannot run as-is; `hint` describes the fix
 */
struct EnvReport {
  enum class Status { Ok, Warning, Error };

  Status status = Status::Ok;
  std::string message;
  std::string hint;
};

/* ------------------------------ Registry ------------------------------ */

class ProfilerRegistry {
public:
  using Factory = std::function<std::unique_ptr<Profiler>(const PerfConfig&, const std::string&)>;
  using EnvCheck = std::function<EnvReport()>;

  /** @brief Access the process-wide registry singleton. */
  static ProfilerRegistry& instance();

  /**
   * @brief Register a backend factory and its environment check.
   * @param name           Stable backend name (e.g. "perf", "gperf"); also the `--profile` value.
   * @param factory        Callable returning a Profiler or nullptr if unavailable at runtime.
   * @param check          Pre-flight check returning an EnvReport. Pass {} for a default Ok report.
   * @param unavailableHint Actionable hint printed when the factory returns nullptr
   *                       (e.g. "Install linux-tools-$(uname -r) or run outside Docker.").
   *
   * Registration is idempotent: re-registering the same name replaces the prior entry.
   */
  void registerBackend(std::string name, Factory factory, EnvCheck check, std::string unavailableHint);

  /**
   * @brief Construct a profiler for the requested backend, or a named no-op.
   *
   * Returns:
   *  - factory result if backend is registered and available
   *  - named no-op (with warning to stderr + hint) if registered but unavailable
   *  - named no-op (with "unknown profiler" warning) if not registered
   *
   * Never returns nullptr.
   */
  std::unique_ptr<Profiler> make(const std::string& name,
                                 const PerfConfig& cfg,
                                 const std::string& testName) const;

  /** @brief True if a backend with this name has been registered. */
  bool hasBackend(const std::string& name) const noexcept;

  /** @brief Sorted list of registered backend names (for help text, bench doctor, etc.). */
  std::vector<std::string> backendNames() const;

  /**
   * @brief Run a single backend's environment check.
   * @param name Registered backend name.
   * @return EnvReport from the backend, or an Error report if name is unknown.
   */
  EnvReport runCheck(const std::string& name) const;

  /**
   * @brief Run every registered backend's environment check.
   * @return Map of backend name to its report. Iteration order matches backendNames().
   */
  std::vector<std::pair<std::string, EnvReport>> runAllChecks() const;

  /**
   * @brief Print a human-readable doctor report to stdout.
   *
   * For every registered backend, prints one line of the form:
   *   [OK]   perf       perf available, perf_event_paranoid=1
   *   [WARN] callgrind  valgrind available; running in Docker (PID namespace)
   *                     callgrind_control attach will be replaced by direct valgrind wrap.
   *   [FAIL] rapl       RAPL not available (Intel CPU + MSR access required)
   *                     sudo modprobe msr; grant CAP_SYS_RAWIO or run as root.
   *
   * @return Number of FAIL-level backends (0 if all backends are usable).
   */
  int printDoctor() const;

private:
  ProfilerRegistry() = default;

  struct Entry {
    Factory factory;
    EnvCheck check;
    std::string unavailableHint;
  };
  std::map<std::string, Entry> backends_;
};

/* ------------------------- Registration helper ------------------------- */

namespace detail {

/**
 * @brief RAII registrar; one instance per backend at file scope triggers registration.
 */
struct ProfilerRegistrar {
  ProfilerRegistrar(std::string name,
                    ProfilerRegistry::Factory factory,
                    ProfilerRegistry::EnvCheck check,
                    std::string hint) {
    ProfilerRegistry::instance().registerBackend(
        std::move(name), std::move(factory), std::move(check), std::move(hint));
  }
};

} // namespace detail

/**
 * @brief Convenience macro for backend self-registration at file scope.
 *
 * Usage at the bottom of a profiler's translation unit:
 * @code
 * VERNIER_REGISTER_PROFILER_BACKEND(
 *     "perf",
 *     makePerfProfiler,
 *     checkPerfEnvironment,
 *     "Install linux-tools-$(uname -r) or run outside Docker.");
 * @endcode
 *
 * The check function should be a `EnvReport(*)()` (no arguments, returns EnvReport).
 */
#define VERNIER_REGISTER_PROFILER_BACKEND(NAME, FACTORY, CHECK, HINT)                                  \
  namespace {                                                                                          \
  const ::vernier::bench::detail::ProfilerRegistrar UB_REGISTRAR_##__LINE__{                           \
      (NAME), (FACTORY), (CHECK), (HINT)};                                                             \
  }

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERREGISTRY_HPP
