#ifndef VERNIER_PROFILERREADINESS_HPP
#define VERNIER_PROFILERREADINESS_HPP
/**
 * @file ProfilerReadiness.hpp
 * @brief Shared mechanisms behind one readiness decision per profiler request.
 *
 * A backend decides whether one request (its name, mode, scripts and whether
 * an analysis was promised) can run in one context (this user and this
 * environment). The same decision serves the doctor rows and the construction
 * of the profiler, so the two cannot disagree. This header holds only the
 * mechanisms such a decision is built from:
 *  - ReadinessContext: one immutable snapshot of the environment, the
 *    effective uid and the pid;
 *  - ReadinessRequest and ReadinessResult: what was asked and what was
 *    decided, with the snapshot and the backend's plan (the resolved tools
 *    and privilege route its launch and cleanup then use);
 *  - the report formatter (readinessResult), the privilege policy
 *    (decidePrivilege), executable resolution (resolveExecutable), bounded
 *    probes (runBoundedProbe), owned helper processes (OwnedHelper) and the
 *    result memo (ReadinessMemo).
 * Each backend owns its check, its mode parsing and its plan in its own
 * source file; nothing here names a backend, a mode or a tool.
 *
 * Where readiness applies: a profiler is created, and its request checked,
 * only for cases built with the profiler guard (UB_PERF_GUARD / PERF_GUARD),
 * makePerfCaseWithProfiler or attachProfilerHooks. A bare PerfCase never
 * reaches the registry, so --profile does nothing for it.
 *
 * Threading: the value types are immutable once built; the free functions
 * are reentrant; ReadinessMemo may be used from several threads. Nothing here
 * runs inside a measured region.
 *
 * @note NOT RT-safe (heap allocation, fork/exec, blocking waits).
 */

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vernier {
namespace bench {

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

/* ------------------------------ Enumerations ------------------------------ */

/** @brief Which question a readiness decision answers. */
enum class ReadinessScope : std::uint8_t {
  DEFAULT_INVENTORY, ///< The plain doctor: each backend's default mode.
  PREFLIGHT,         ///< The doctor asked about one request before a run.
  RUNTIME            ///< Construction of the profiler for a run.
};

/** @brief How the benchmark process was started relative to the backend's tool. */
enum class LaunchContext : std::uint8_t {
  IN_PROCESS,       ///< The backend runs inside the benchmark (the default).
  RUNNER_WRAPPED,   ///< `bench run` wrapped the process with this backend's tool.
  MANUALLY_WRAPPED, ///< The user started the process under the tool.
  NOT_WRAPPED       ///< The backend needs a wrapper and none is present.
};

/** @brief Which stage of the request a decision concerns. */
enum class ReadinessStage : std::uint8_t {
  COLLECTION, ///< Capturing data; an Error here means nothing is collected.
  ANALYSIS    ///< A promised analysis; an Error here still collects and keeps the raw data.
};

/** @brief Why a decision came out as it did; selects the status and the message prefix. */
enum class ReadinessCause : std::uint8_t {
  READY,          ///< Ok: every required prerequisite was verified.
  CAVEAT,         ///< Warning: runs with a stated limitation.
  UNVERIFIED,     ///< Warning: a prerequisite cannot be checked before the run.
  MISSING,        ///< Error: a required tool, file or build feature is absent.
  UNUSABLE,       ///< Error: present but does not work.
  UNSUPPORTED,    ///< Error: the requested mode or event is not supported here.
  DENIED,         ///< Error: the operation was refused.
  CONFIGURATION,  ///< Error: a setting is invalid.
  MISSING_HELPER, ///< Error: a command the route needs (sudo, kill) is absent.
  INTERNAL        ///< Error: the check itself failed.
};

/* ------------------------------ ReadinessContext ------------------------------ */

/**
 * @brief Immutable snapshot of the facts a decision reads: environment, uid, pid.
 *
 * Captured once per decision. Checks, probes and plans read only the
 * snapshot, never the live process environment, so one decision cannot see
 * two different environments, and tests pass a snapshot instead of calling
 * setenv().
 */
class ReadinessContext {
public:
  /** @brief Snapshot of this process: its environment, effective uid and pid. */
  [[nodiscard]] static ReadinessContext capture();

  ReadinessContext(uid_t euid, pid_t self, std::map<std::string, std::string> environment);

  [[nodiscard]] uid_t euid() const noexcept { return euid_; }
  [[nodiscard]] pid_t self() const noexcept { return self_; }

  /** @brief Value of one environment variable in the snapshot, if set (possibly empty). */
  [[nodiscard]] std::optional<std::string> get(std::string_view name) const;

  /** @brief The whole snapshot environment. */
  [[nodiscard]] const std::map<std::string, std::string>& environment() const noexcept {
    return environment_;
  }

  /**
   * @brief Key of the decision-relevant facts, for memoizing results.
   *
   * Covers the effective uid, PATH, BENCH_SUDO and the runner's wrap variables
   * (VERNIER_EXTERNAL_WRAP, VERNIER_EXTERNAL_WRAP_DIR), plus @p extraKeys:
   * the variables a backend's check reads beyond those.
   */
  [[nodiscard]] std::string fingerprint(const std::vector<std::string>& extraKeys = {}) const;

  /**
   * @brief Environment for a probe child: the snapshot without the variables
   * that inject a tool into a process, and with LC_ALL=C.
   *
   * Removed: LD_PRELOAD, MALLOC_CONF, HEAPTRACK_*, CUDA_INJECTION64_PATH,
   * ROCP_TOOL_LIB, ROCPROFILER_LIBRARY and VERNIER_EXTERNAL_WRAP*. A probe so
   * never runs under the benchmark's own wrap, and tool messages are in the
   * C locale, stable to classify.
   */
  [[nodiscard]] std::vector<std::string> probeEnvironment() const;

private:
  uid_t euid_;
  pid_t self_;
  std::map<std::string, std::string> environment_;
};

/* ------------------------------ Request and Result ------------------------------ */

/** @brief One request, as the command line states it. */
struct ReadinessRequest {
  std::string backend;                 ///< Canonical registered name.
  std::string profileArgs;             ///< Mode text, parsed by the backend's own parser.
  std::vector<std::string> bpfScripts; ///< Script selection, for backends that take scripts.
  bool analyze = false;                ///< --profile-analyze promised an analysis.
  ReadinessScope scope = ReadinessScope::DEFAULT_INVENTORY;
  LaunchContext launch = LaunchContext::IN_PROCESS;
};

/**
 * @brief Backend-owned facts the launch needs (resolved paths, route, scripts).
 *
 * A backend derives its own plan type; the registry hands the whole result,
 * plan included, to the backend's factory, so launch and cleanup use the
 * identity the check verified.
 */
struct ReadinessPlan {
  virtual ~ReadinessPlan();
};

/** @brief One decision: the public report plus what produced it. */
struct ReadinessResult {
  EnvReport report; ///< The verdict, as doctor rows show it.
  ReadinessStage stage = ReadinessStage::COLLECTION;
  ReadinessCause cause = ReadinessCause::READY;
  std::shared_ptr<const ReadinessContext> context; ///< The snapshot the decision used.
  std::shared_ptr<const ReadinessPlan> plan;       ///< Resolved tools and route for launch.

  /** @brief True unless collection itself cannot run (an Error at the collection stage). */
  [[nodiscard]] bool collectionReady() const noexcept {
    return report.status != EnvReport::Status::Error || stage == ReadinessStage::ANALYSIS;
  }
};

/** @brief A backend's decision for one request in one context. */
using ReadinessCheck =
    std::function<ReadinessResult(const ReadinessRequest&, const ReadinessContext&)>;

/** @brief Builds the profiler from the decision, so launch uses the checked plan. */
using PlannedFactory = std::function<std::unique_ptr<Profiler>(
    const PerfConfig&, const std::string&, const ReadinessResult&)>;

/**
 * @brief Build a result: status from @p cause, message prefixed with its cause word.
 *
 * READY gives Ok with @p detail as the message; CAVEAT gives Warning;
 * UNVERIFIED gives Warning prefixed "unverified: "; every other cause gives
 * Error prefixed "missing: ", "unusable: ", "unsupported: ", "denied: ",
 * "configuration: ", "missing helper: " or "internal: ". An ANALYSIS-stage
 * result is further prefixed "analysis: ". @p remedy becomes the hint.
 */
[[nodiscard]] ReadinessResult readinessResult(ReadinessCause cause, std::string detail,
                                              std::string remedy,
                                              ReadinessStage stage = ReadinessStage::COLLECTION);

/**
 * @brief The request a configuration states, for @p scope in @p ctx.
 *
 * The launch context is RUNNER_WRAPPED when the snapshot's
 * VERNIER_EXTERNAL_WRAP names the requested backend, IN_PROCESS otherwise;
 * backends that need a wrapper refine it in their own checks.
 */
[[nodiscard]] ReadinessRequest readinessRequestFor(const PerfConfig& cfg, ReadinessScope scope,
                                                   const ReadinessContext& ctx);

/* ------------------------------ Privilege Policy ------------------------------ */

/** @brief Value of a boolean setting. */
enum class EnvBool : std::uint8_t { ABSENT, FALSE_VALUE, TRUE_VALUE, INVALID };

/**
 * @brief Parse a boolean setting, ignoring case.
 *
 * Unset is ABSENT; "1", "true", "yes", "on" are TRUE_VALUE; "0", "false",
 * "no", "off" and the empty string are FALSE_VALUE; anything else is INVALID.
 */
[[nodiscard]] EnvBool parseEnvBool(const std::optional<std::string>& raw);

/** @brief How a helper tool is run. */
enum class PrivilegeRoute : std::uint8_t {
  CURRENT_USER, ///< As the invoking user (the default).
  ALREADY_ROOT, ///< The process is root; sudo is never called.
  SCOPED_SUDO,  ///< Through `sudo -n`, as the settings opted in.
  CONFIG_ERROR  ///< A setting is invalid; nothing may be launched.
};

/** @brief The effective privilege decision and what the report says about it. */
struct PrivilegeDecision {
  PrivilegeRoute route = PrivilegeRoute::CURRENT_USER;
  bool optIn = false;  ///< The settings ask for `sudo -n`.
  std::string source;  ///< The setting that decided, e.g. "BENCH_SUDO=1".
  std::string warning; ///< Deprecation, conflict or ignored-value notice; empty if none.
  std::string error;   ///< The invalid setting, when route is CONFIG_ERROR.
  std::string remedy;  ///< How to fix the invalid setting.
};

/**
 * @brief The one privilege policy for helper tools.
 *
 * BENCH_SUDO is the opt-in. @p legacyAlias names a deprecated backend-specific
 * setting still honoured below it (empty: none). Precedence: an invalid
 * BENCH_SUDO is a configuration error; a valid BENCH_SUDO decides and a
 * differing valid alias draws a conflict warning, an invalid one an
 * ignored-value warning, an equal one nothing; without BENCH_SUDO a valid
 * alias decides with a deprecation warning and an invalid one is a
 * configuration error. Root never uses sudo.
 */
[[nodiscard]] PrivilegeDecision decidePrivilege(const ReadinessContext& ctx,
                                                std::string_view legacyAlias = {});

/* ------------------------------ Executables ------------------------------ */

/** @brief Where a tool name resolves on the snapshot's PATH. */
struct ResolvedTool {
  std::string name;        ///< The name asked for.
  std::string path;        ///< Absolute path of the match.
  bool executable = false; ///< False: the match is not an executable regular file.
};

/**
 * @brief Resolve @p name on the snapshot's PATH (or as a path when it contains '/').
 *
 * Returns the first executable regular file; failing that, the first
 * candidate that is not one (a plain file without the execute bit, or a
 * directory) with executable=false, so the caller can say why; nullopt when
 * nothing of that name exists.
 */
[[nodiscard]] std::optional<ResolvedTool> resolveExecutable(std::string_view name,
                                                            const ReadinessContext& ctx);

/* ------------------------------ Bounded Probes ------------------------------ */

/** @brief Outcome of one bounded probe. */
struct ProbeResult {
  bool started = false;    ///< The program was executed.
  int spawnErrno = 0;      ///< errno of a failed fork or exec.
  bool exited = false;     ///< Ended normally; exitCode is valid.
  int exitCode = -1;       ///< Exit status when exited.
  int termSignal = 0;      ///< Signal that ended it, 0 if it exited.
  bool timedOut = false;   ///< Killed at the bound.
  std::string output;      ///< stdout (and stderr when MERGED), truncated at PROBE_OUTPUT_LIMIT.
  std::string errorOutput; ///< stderr when SEPARATE, truncated likewise.

  /** @brief Started, exited with status 0 and did not time out. */
  [[nodiscard]] bool succeeded() const noexcept {
    return started && exited && exitCode == 0 && !timedOut;
  }

  /** @brief One phrase: "exit status 2", "killed by signal 9", "timed out after the bound", ... */
  [[nodiscard]] std::string describe() const;
};

/** @brief Bytes of output a probe keeps, per stream. */
inline constexpr std::size_t PROBE_OUTPUT_LIMIT = 64 * 1024;

/** @brief Where a probe's stderr goes. */
enum class ProbeStreams : std::uint8_t {
  MERGED,  ///< Interleaved with stdout in ProbeResult::output.
  SEPARATE ///< Kept apart in ProbeResult::errorOutput.
};

/**
 * @brief Run @p argv once, bounded, and collect its status and output.
 *
 * argv[0] is executed as given (a resolved absolute path; no PATH search),
 * with stdin from /dev/null, in its own process group, in the probe
 * environment of @p ctx. At @p timeoutMs the group is killed. Needs no
 * external helper such as timeout(1).
 */
[[nodiscard]] ProbeResult runBoundedProbe(const std::vector<std::string>& argv, int timeoutMs,
                                          const ReadinessContext& ctx,
                                          ProbeStreams streams = ProbeStreams::MERGED);

/**
 * @brief The last @p maxLines non-empty lines of @p text, joined with " | ".
 *
 * For one-line reports of a tool's output.
 */
[[nodiscard]] std::string outputTail(const std::string& text, std::size_t maxLines = 2);

/* ------------------------------ Owned Helpers ------------------------------ */

/** @brief How an owned helper is stopped: the route and the bounded waits. */
struct HelperStopPolicy {
  PrivilegeRoute route = PrivilegeRoute::CURRENT_USER;
  std::string sudoPath;                            ///< Resolved sudo, used on SCOPED_SUDO.
  std::string killPath;                            ///< Resolved kill, used on SCOPED_SUDO.
  int interruptWaitMs = 2000;                      ///< Wait after SIGINT.
  int terminateWaitMs = 1000;                      ///< Wait after SIGTERM.
  int killWaitMs = 1000;                           ///< Wait after SIGKILL.
  std::shared_ptr<const ReadinessContext> context; ///< Environment for `sudo -n kill`.
};

/** @brief How an owned helper's start went. */
struct HelperStart {
  bool started = false;     ///< The program was executed.
  int spawnErrno = 0;       ///< errno of a failed fork or exec.
  bool exitedEarly = false; ///< It ended within the start grace.
  int waitStatus = 0;       ///< Its wait status when it ended early.
  std::string errorTail;    ///< The end of its stderr when it ended early.

  /** @brief Started and still running after the grace. */
  [[nodiscard]] bool running() const noexcept { return started && !exitedEarly; }
};

/** @brief One signal delivery attempted by OwnedHelper::stop(). */
struct StopDelivery {
  int signal = 0;
  pid_t target = -1;
  bool delivered = false;
  std::string command; ///< "kill(2)" or the exact sudo command line.
  std::string detail;  ///< errno text or the refusing command's output.
};

/** @brief How an owned helper's stop went. */
struct HelperStopResult {
  bool wasRunning = false; ///< The helper was alive when the stop began.
  std::vector<StopDelivery> deliveries;
  bool reaped = false;     ///< The helper's process was reaped.
  int stoppedBy = 0;       ///< The last delivered signal before it ended; 0 if none.
  bool stillAlive = false; ///< Every step was tried and it still runs.
  int waitStatus = 0;      ///< Its wait status when reaped.

  /** @brief True when every delivery succeeded; false names a refused one. */
  [[nodiscard]] bool allDelivered() const noexcept;
};

/**
 * @brief A helper process this benchmark started and alone may signal.
 *
 * start() forks and executes argv[0] directly (an absolute path, often the
 * resolved sudo), with stdout and stderr opened by the child in files the
 * invoking user owns, and waits the start grace, returning at once if the
 * helper ends. stop() delivers SIGINT, then SIGTERM, then SIGKILL through the
 * policy's route, waiting a bounded time after each and reporting every
 * delivery. Only processes this object started are signalled: its direct
 * child while that is not yet reaped, or, on the sudo route, the single child
 * listed under it (sudo may keep a monitor process between the two). An
 * empty or ambiguous list falls back to the direct child.
 */
class OwnedHelper {
public:
  explicit OwnedHelper(HelperStopPolicy policy = {});
  ~OwnedHelper();

  OwnedHelper(const OwnedHelper&) = delete;
  OwnedHelper& operator=(const OwnedHelper&) = delete;
  OwnedHelper(OwnedHelper&& other) noexcept;
  OwnedHelper& operator=(OwnedHelper&& other) noexcept;

  /**
   * @brief Start @p argv; stdout to @p stdoutPath ("" for /dev/null), stderr to
   * @p stderrPath ("" for /dev/null); wait up to @p graceMs.
   * @param probeEnv Null: the helper inherits this process's environment.
   *                 Otherwise it gets that context's probe environment.
   */
  HelperStart start(const std::vector<std::string>& argv, const std::string& stdoutPath,
                    const std::string& stderrPath, int graceMs,
                    const ReadinessContext* probeEnv = nullptr);

  /** @brief Stop through the policy's route; bounded. Safe to call more than once. */
  HelperStopResult stop();

  /** @brief pid of the direct child, or -1. */
  [[nodiscard]] pid_t pid() const noexcept { return child_; }

  /** @brief True while the direct child has not been reaped. */
  [[nodiscard]] bool running();

private:
  HelperStopPolicy policy_;
  pid_t child_ = -1;
  bool reaped_ = true;
  int waitStatus_ = 0;
};

/* ------------------------------ Memo ------------------------------ */

/**
 * @brief Process-lifetime store of readiness results, one per key.
 *
 * getOrCompute() publishes each key through a shared future inserted under
 * one short lock: the first caller computes outside the lock, later callers
 * of the same key wait outside it, and independent keys compute
 * concurrently. Every outcome is kept, errors included, until reset(). A
 * compute that throws is kept as an INTERNAL error. claimNotice() lets
 * exactly one caller per key print a report.
 */
class ReadinessMemo {
public:
  ReadinessMemo();
  ~ReadinessMemo();

  ReadinessMemo(const ReadinessMemo&) = delete;
  ReadinessMemo& operator=(const ReadinessMemo&) = delete;

  /** @brief The result kept for @p key, computing it with @p compute the first time. */
  ReadinessResult getOrCompute(const std::string& key,
                               const std::function<ReadinessResult()>& compute);

  /** @brief True for exactly one caller per @p key until reset(). */
  bool claimNotice(const std::string& key);

  /** @brief Forget every result and claimed notice. */
  void reset();

  /** @brief Number of keys held. */
  [[nodiscard]] std::size_t size() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERREADINESS_HPP
