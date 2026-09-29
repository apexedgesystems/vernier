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
 * One outcome per run: a requested profile that cannot collect, whose
 * promised analysis fails, or whose output does not complete is recorded
 * (reportFailure()), and finishRun() ends the run with a report of every
 * recorded failure and exit status 4 when the tests passed.
 *
 * Threading:
 *  - Registration runs during static init (single-threaded); registering or
 *    unregistering is not synchronized against make().
 *  - make() may run on several threads: decisions are memoized per request
 *    and context (ReadinessMemo), computed outside any registry lock.
 *  - reportFailure() may be called from any thread, including a profiler's
 *    hooks; the record is guarded by its own lock.
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

/* ------------------------------ Outcome ------------------------------ */

/// Exit status of a run whose tests passed and whose requested profile failed.
inline constexpr int BENCH_PROFILE_FAILED_EXIT_CODE = 4;

/** @brief One failed profile request, as the run records and reports it. */
struct ProfileFailure {
  std::string backend;    ///< Canonical backend name.
  std::string test;       ///< The case it concerns; empty for the whole request.
  ReadinessResult result; ///< An Error: its stage, cause, message and remedy.
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
   * @brief Construct a profiler for the requested backend, or a no-op.
   *
   * Decides the request first (memoized per request and context):
   *  - Ok: the backend's factory builds the profiler, silently.
   *  - Warning: printed once to stderr; the profiler is built.
   *  - Error at the analysis stage: printed once and recorded as the run's
   *    failure; the profiler is built, collects, and skips the analysis it
   *    cannot run.
   *  - Error at the collection stage: printed once with its remedy and
   *    recorded; a no-op is returned and the factory is not called.
   * An unregistered name and a factory that still returns nullptr (a build or
   * platform guard) are printed once and recorded, and give a no-op too. Such
   * a no-op names no tool, so the CSV records no profiler for its case.
   * `cupti` is not a backend: it says where the CUPTI columns come from and
   * gives a no-op named `cupti`, recording nothing.
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

  /**
   * @brief Record a failure of the requested profile, for the run's exit status.
   * @param backend Canonical backend name, e.g. "offcpu".
   * @param test    The case the failure concerns (the profiler's test name);
   *                empty when it concerns the whole request.
   * @param failure What failed, built with readinessResult(): COLLECTION when
   *                nothing usable was captured, ANALYSIS for a promised
   *                analysis, COMPLETION when the capture ran and its output is
   *                missing, unreadable or not this run's.
   *
   * A result that is not an Error is ignored. The first report of a failure
   * (same backend, test, stage and message) is recorded and printed to
   * stderr with its remedy; a repeat changes nothing. Callable from any
   * thread, including a profiler's hooks; never throws.
   */
  void reportFailure(const std::string& backend, const std::string& test,
                     const ReadinessResult& failure) const noexcept;

  /** @brief Every failure recorded since the last resetFailures(), in order. */
  std::vector<ProfileFailure> failures() const;

  /**
   * @brief Forget the recorded failures and whether a profiler was created.
   *
   * For tests and for a process that runs more than once; a new process
   * starts empty.
   */
  void resetFailures();

  /**
   * @brief End a run: report its profile outcome and return its exit status.
   * @param cfg        The run's configuration (its `--profile` request).
   * @param testStatus The tests' status, e.g. RUN_ALL_TESTS().
   * @param testsRun   How many tests the run selected to run.
   * @return @p testStatus when it is nonzero; otherwise
   *         BENCH_PROFILE_FAILED_EXIT_CODE when a failure was recorded;
   *         otherwise 0.
   *
   * Prints, to stderr, every recorded failure under one header. An unknown
   * `--profile` name fails here even when no case asked for a profiler.
   * When `--profile` was given, tests ran, nothing failed and no case created
   * a profiler (only cases built with the profiler guard create one), it says
   * that nothing was profiled, or, under `bench run`'s wrap of that tool, that
   * the wrap still recorded the whole process. PERF_MAIN() and PERF_GPU_MAIN()
   * return its result; a benchmark with its own main() calls it the same way.
   */
  int finishRun(const PerfConfig& cfg, int testStatus, int testsRun) const;

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

  ~ProfilerRegistry();
  ProfilerRegistry(const ProfilerRegistry&) = delete;
  ProfilerRegistry& operator=(const ProfilerRegistry&) = delete;

private:
  ProfilerRegistry();

  struct Outcome; ///< The run's recorded failures and whether a profiler was created.

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

  /** @brief Record @p failure; true for its first report (the caller prints it). */
  bool recordFailure(const std::string& backend, const std::string& test,
                     const ReadinessResult& failure) const;

  /** @brief The decision for a name no backend is registered under. */
  ReadinessResult unknownResult(const std::string& name, const ReadinessContext& ctx) const;

  std::map<std::string, Entry> backends_;
  std::uint64_t generation_ = 0; ///< Bumped by every (un)registration: new memo keys.
  mutable ReadinessMemo memo_;
  std::unique_ptr<Outcome> outcome_;
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
