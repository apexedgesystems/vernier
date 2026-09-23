#ifndef VERNIER_PROFILERREGISTRY_HPP
#define VERNIER_PROFILERREGISTRY_HPP
/**
 * @file ProfilerRegistry.hpp
 * @brief Self-registration registry for profiler backends.
 *
 * Each backend (perf, gperf, callgrind, bpftrace, rapl, nsight, ...)
 * registers a factory + an availability hint at static-init time via
 * VERNIER_REGISTER_PROFILER_BACKEND, or a readiness check + a planned
 * factory via VERNIER_REGISTER_READINESS_BACKEND. Profiler::make() dispatches
 * by looking up the registered factory rather than a hard-coded if-chain,
 * so new backends slot in by adding a file + one registration line.
 *
 * One decision per request: the doctor rows, the doctor's selected row and
 * the construction of a profiler all ask checkRequest() the same question,
 * so a run reports exactly what the doctor reports for the same request and
 * context. Backends registered without a readiness check are answered by
 * their zero-argument check through an adapter that never claims more than
 * that check verified.
 *
 * Threading:
 *  - Registration runs during static init (single-threaded); registering or
 *    unregistering is not synchronized against make().
 *  - make() may run on several threads: decisions are memoized per request
 *    and context (ReadinessMemo), computed outside any registry lock.
 *
 * @note NOT RT-safe (std::map, std::string, std::function, fork/exec probes).
 */

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerReadiness.hpp" // EnvReport and the readiness types

namespace vernier {
namespace bench {

// Forward declarations to keep this header light and avoid a circular include
// (PerfConfig.hpp pulls registry-driven diagnostics back in).
struct PerfConfig;
class Profiler;

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
   * @param check          Pre-flight check of the backend's default mode. Pass {} when there
   *                       is none; the doctor then reports the backend as unverified.
   * @param unavailableHint Actionable hint printed when the factory returns nullptr
   *                       (e.g. "Install linux-tools-$(uname -r) or run outside Docker.").
   *
   * Registration is idempotent: re-registering the same name replaces the prior entry.
   */
  void registerBackend(std::string name, Factory factory, EnvCheck check,
                       std::string unavailableHint);

  /**
   * @brief Register a backend that decides whole requests (mode, scripts, analysis).
   * @param name            Stable backend name; also the `--profile` value.
   * @param check           The backend's decision for one request in one context.
   * @param factory         Builds the profiler from that decision, plan included.
   * @param unavailableHint Printed when the factory still returns nullptr.
   * @param contextKeys     Environment variables the check reads beyond PATH,
   *                        BENCH_SUDO and the runner's wrap variables; a change
   *                        to any of them is a new decision.
   *
   * Re-registering a name replaces the prior entry. Startup-only, like registerBackend().
   */
  void registerReadinessBackend(std::string name, ReadinessCheck check, PlannedFactory factory,
                                std::string unavailableHint,
                                std::vector<std::string> contextKeys = {});

  /**
   * @brief Remove a registration; true when there was one.
   *
   * Startup-only, like registration; for tests and for plugins that unload.
   */
  bool unregisterBackend(const std::string& name);

  /**
   * @brief Map user-facing aliases to registered backend names (nsys -> nsight).
   */
  static std::string canonicalName(const std::string& name);

  /**
   * @brief Construct a profiler for the requested backend, or a named no-op.
   *
   * Decides the request first (memoized per request and context):
   *  - Ok: the backend's factory builds the profiler, silently.
   *  - Warning: printed once to stderr; the profiler is built.
   *  - Error at the analysis stage: printed once; the profiler is built,
   *    collects, and skips the analysis it cannot run.
   *  - Error at the collection stage: printed once with its remedy; a named
   *    no-op is returned and the factory is not called.
   * A factory that still returns nullptr (a build or platform guard) and an
   * unregistered name also give a named no-op, with a warning.
   *
   * Never returns nullptr.
   */
  std::unique_ptr<Profiler> make(const std::string& name, const PerfConfig& cfg,
                                 const std::string& testName) const;

  /** @brief make() deciding in @p ctx instead of a snapshot of this process. */
  std::unique_ptr<Profiler> make(const std::string& name, const PerfConfig& cfg,
                                 const std::string& testName, const ReadinessContext& ctx) const;

  /**
   * @brief The one readiness decision for @p request, in a snapshot of this process.
   *
   * Never memoized: the doctor calls it for every row and asks afresh each time.
   * An unregistered backend is an Error.
   */
  ReadinessResult checkRequest(const ReadinessRequest& request) const;

  /** @brief checkRequest() in an explicit context. */
  ReadinessResult checkRequest(const ReadinessRequest& request, const ReadinessContext& ctx) const;

  /**
   * @brief Forget every memoized decision and printed notice.
   *
   * The only reset: a decision is otherwise kept for the life of the
   * process, so a tool, grant or capability changed during a run needs this
   * (or a new process) to be seen.
   */
  void resetReadiness();

  /** @brief True if a backend with this name has been registered. */
  bool hasBackend(const std::string& name) const noexcept;

  /** @brief Sorted list of registered backend names (for help text, bench doctor, etc.). */
  std::vector<std::string> backendNames() const;

  /**
   * @brief Check one backend's default mode (the doctor's inventory row).
   * @param name Registered backend name.
   * @return The backend's report, or an Error report if name is unknown.
   */
  EnvReport runCheck(const std::string& name) const;

  /**
   * @brief Check every registered backend's default mode.
   * @return Map of backend name to its report. Iteration order matches backendNames().
   */
  std::vector<std::pair<std::string, EnvReport>> runAllChecks() const;

  /**
   * @brief Print a human-readable doctor report to stdout.
   *
   * For every registered backend, prints one line of the form:
   *   [OK]   perf       perf available, perf_event_paranoid=1
   *   [WARN] callgrind  valgrind available; running in Docker (PID namespace)
   *                     Run via 'bench run', which wraps valgrind directly (no attach needed).
   *   [FAIL] rapl       RAPL not available (Intel CPU + MSR access required)
   *                     sudo modprobe msr; grant CAP_SYS_RAWIO or run as root.
   * Each row checks the backend's default mode; a footer says what that
   * does and does not cover.
   *
   * @return Number of FAIL-level backends (0 if all backends are usable).
   */
  int printDoctor() const;

  /**
   * @brief printDoctor(), plus the selected row for the request @p cfg states
   * (`--profile` with its `--profile-args`, `--bpf` and `--profile-analyze`),
   * when it names a backend. The selected row does not count toward the
   * returned number.
   */
  int printDoctor(const PerfConfig& cfg) const;

private:
  ProfilerRegistry() = default;

  struct Entry {
    Factory factory;                      ///< Legacy factory.
    EnvCheck check;                       ///< Legacy default-mode check; empty when none.
    ReadinessCheck readiness;             ///< Request check; empty for legacy backends.
    PlannedFactory planned;               ///< Factory that receives the decision.
    std::string unavailableHint;          ///< Printed when a factory returns nullptr.
    std::vector<std::string> contextKeys; ///< Extra decision inputs (memo key).
  };

  ReadinessResult decide(const std::string& name, const Entry& entry,
                         const ReadinessRequest& request, const ReadinessContext& ctx) const;

  std::map<std::string, Entry> backends_;
  std::uint64_t generation_ = 0; ///< Bumped by every (un)registration: new memo keys.
  mutable ReadinessMemo memo_;
};

/* ------------------------- Registration helper ------------------------- */

namespace detail {

/**
 * @brief RAII registrar; one instance per backend at file scope triggers registration.
 */
struct ProfilerRegistrar {
  ProfilerRegistrar(std::string name, ProfilerRegistry::Factory factory,
                    ProfilerRegistry::EnvCheck check, std::string hint) {
    ProfilerRegistry::instance().registerBackend(std::move(name), std::move(factory),
                                                 std::move(check), std::move(hint));
  }
};

/** @brief RAII registrar for VERNIER_REGISTER_READINESS_BACKEND. */
struct ReadinessRegistrar {
  ReadinessRegistrar(std::string name, ReadinessCheck check, PlannedFactory factory,
                     std::string hint, std::vector<std::string> contextKeys) {
    ProfilerRegistry::instance().registerReadinessBackend(std::move(name), std::move(check),
                                                          std::move(factory), std::move(hint),
                                                          std::move(contextKeys));
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
/* Two-level paste so __LINE__ expands before concatenation; a direct
 * NAME##__LINE__ pastes the literal token and collides as soon as one TU
 * registers two backends (e.g. nsight + ncu in ProfilerNsight.cu). */
#define VERNIER_REG_CONCAT_INNER(a, b) a##b
#define VERNIER_REG_CONCAT(a, b) VERNIER_REG_CONCAT_INNER(a, b)

#define VERNIER_REGISTER_PROFILER_BACKEND(NAME, FACTORY, CHECK, HINT)                              \
  namespace {                                                                                      \
  const ::vernier::bench::detail::ProfilerRegistrar VERNIER_REG_CONCAT(UB_REGISTRAR_, __LINE__){   \
      (NAME), (FACTORY), (CHECK), (HINT)};                                                         \
  }

/**
 * @brief Register a backend that decides whole requests, at file scope.
 *
 * @code
 * VERNIER_REGISTER_READINESS_BACKEND(
 *     "perf",
 *     checkPerfRequest,        // ReadinessResult(const ReadinessRequest&, const ReadinessContext&)
 *     makePlannedPerfProfiler, // unique_ptr<Profiler>(cfg, testName, const ReadinessResult&)
 *     "Install linux-tools-$(uname -r).");
 * @endcode
 *
 * Optional trailing arguments name the environment variables the check reads
 * beyond PATH, BENCH_SUDO and the runner's wrap variables.
 */
#define VERNIER_REGISTER_READINESS_BACKEND(NAME, CHECK, FACTORY, HINT, ...)                        \
  namespace {                                                                                      \
  const ::vernier::bench::detail::ReadinessRegistrar VERNIER_REG_CONCAT(UB_READINESS_REGISTRAR_,   \
                                                                        __LINE__){                 \
      (NAME), (CHECK), (FACTORY), (HINT), std::vector<std::string>{__VA_ARGS__}};                  \
  }

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERREGISTRY_HPP
