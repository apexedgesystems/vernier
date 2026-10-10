#ifndef VERNIER_PROFILERBPFTRACE_HPP
#define VERNIER_PROFILERBPFTRACE_HPP
/**
 * @file ProfilerBpftrace.hpp
 * @brief bpftrace backend for the benchmarking profiler facade.
 *
 * Behavior:
 *  - The readiness check (checkBpftraceRequest) decides the privilege route,
 *    resolves bpftrace (and, on the sudo route, sudo and kill) on PATH,
 *    resolves and reads every selected script, refuses a script that uses a
 *    name the capture window reserves or has an iterator probe (which
 *    bpftrace runs only alone), runs `bpftrace --version` as the current
 *    user, and runs a probe copy of each script (with the capture window and
 *    a 5 s self-exit added) through the route for one second, stopping it
 *    with SIGINT; a copy whose stop is refused ends by its self-exit, and the
 *    check waits for it and reaps it. The run's own command reads a copy in
 *    its capture folder, which does not exist yet, so on the sudo route a
 *    grant refusal of the probe copy is unverified rather than denied: the
 *    run's start decides.
 *    The profiler launches and stops with exactly the tools and route that
 *    check verified (BpftracePlan).
 *  - In beforeMeasure(), starts the capture window's arm thread, then one
 *    bpftrace process per selected script (e.g. "write_latency",
 *    "fsync_latency") with {{PID}} replaced by the current PID and the
 *    capture window appended (captureWindowProgram()), bound to this
 *    process, the arm thread and the calling thread, run with -B none, and
 *    returns once every tracer has acknowledged its arm, or failed, within
 *    the plan's armWaitMs. A script's copy and output go to the capture
 *    folder under its file's stem: <stem>.tmp.bt, <stem>.out.<format> and
 *    <stem>.err.txt.
 *  - In afterMeasure(), on the thread that called beforeMeasure(), waits up
 *    to the plan's disarmWaitMs for every armed tracer to acknowledge the
 *    stop, then stops each with SIGINT, then SIGTERM, then SIGKILL through
 *    the same route, and reports each refused delivery, a tracer that ended
 *    by itself before the stop, and one that did not end with status 0.
 *  - captureOutcome() says whether the capture is complete, and whether the
 *    scripts emitted anything: READY for tracers that acknowledged both ends
 *    of the capture with the ids they were bound to, ended their stop with
 *    status 0, as bpftrace does once it has printed its maps, and printed
 *    more than the capture window's lines; CAVEAT when one printed nothing
 *    else.
 *
 * Privileges: bpftrace runs as the current user unless BENCH_SUDO opts in to
 * `sudo -n`; PERF_BPF_SUDO is a deprecated alias that BENCH_SUDO overrides.
 * Root never uses sudo.
 *
 * Notes:
 *  - Linux-only. Safe no-op on other platforms (compile-time guard).
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- bpftrace route ----------------------------- */

/**
 * @brief How bpftrace is started and stopped for one request: the privilege
 * decision and the tools resolved on the snapshot's PATH.
 *
 * Shared by the backends that run bpftrace (bpftrace, offcpu).
 */
struct BpftraceRoute {
  PrivilegeDecision privilege;
  std::string bpftrace; ///< Absolute path of bpftrace.
  std::string sudo;     ///< Absolute path of sudo; set on the sudo route only.
  std::string kill;     ///< Absolute path of kill; set on the sudo route only.

  /** @brief The argv prefix that runs bpftrace through the route. */
  [[nodiscard]] std::vector<std::string> command() const;

  /** @brief Stop policy for a tracer started through the route. */
  [[nodiscard]] HelperStopPolicy stopPolicy(int interruptWaitMs, int terminateWaitMs,
                                            int killWaitMs,
                                            std::shared_ptr<const ReadinessContext> ctx) const;

  /** @brief "as the current user", "as root" or "through sudo -n (BENCH_SUDO=1)". */
  [[nodiscard]] std::string describe() const;
};

namespace bpftrace_tool {

/**
 * @brief Decide the route and resolve its tools.
 * @return The error that stops the request, or nullopt with @p route filled.
 */
std::optional<ReadinessResult> resolveRoute(const ReadinessContext& ctx,
                                            std::string_view legacyAlias, BpftraceRoute& route);

/** @brief `<bpftrace> --version` as the current user: the executable runs. */
std::optional<ReadinessResult> probeExecutable(const BpftraceRoute& route,
                                               const ReadinessContext& ctx);

/** @brief The line a probe prints once bpftrace has attached it (attachLineProgram()). */
inline constexpr const char* PROBE_ATTACH_LINE = "vernier probe attached";

/**
 * @brief The program a probe adds so that the check sees it attach: every
 * 100 ms it prints PROBE_ATTACH_LINE. bpftrace prints what its programs emit
 * only once it has attached every probe (see the capture window below), so
 * the line shows that the probe has attached and reads its output. Run the
 * probe with -B none, so the line reaches the output as it is printed.
 */
std::string attachLineProgram();

/** @brief One attach probe: what runs, how reports name it, and how it relates to the run. */
struct AttachProbe {
  std::vector<std::string> toolArgs; ///< The arguments after bpftrace.
  std::string what;                  ///< How reports name it: "script 'x'", "the off-CPU script".
  std::string commandLine;           ///< The bpftrace command line as reports show it.
  /**
   * Empty when the probe runs the run's own command, so that sudo's answer
   * to it holds for the run. Otherwise how the run's command reads: a grant
   * refusal of the probe is then unverified.
   */
  std::string runCommand;
  /**
   * What the reader changes when the probe ends by itself, status 0, before
   * its stop (a script that exits that soon); empty when the reader cannot
   * change the script, and the hint says to run it by hand.
   */
  std::string selfEndRemedy;
  int graceMs = 1000;      ///< How long the probe runs before its stop, at least.
  int attachWaitMs = 5000; ///< How long after its start the stop waits for its attach line.
  int selfExitMs = 0;      ///< When the probe's own script ends it after its attach; 0: never.
};

/** @brief How long past its self-exit a probe whose stop failed is waited for. */
inline constexpr int PROBE_SELF_EXIT_SLACK_MS = 10000;

/**
 * @brief Run a probe through the route and stop it with the run's first stop signal.
 *
 * Runs route.command() + probe.toolArgs as an owned helper, its output and
 * error output in files in @p scratchDir. The probe's program must carry
 * attachLineProgram(). The call waits the grace, then until the probe prints
 * PROBE_ATTACH_LINE, up to attachWaitMs after its start, and stops it with
 * SIGINT through the route (then SIGTERM and SIGKILL if needed). bpftrace
 * takes SIGINT as a stop only once it reads its output: a SIGINT before then
 * ends a bpftrace that has not set its handler yet, makes 0.23.2 abandon its
 * attach, or goes unnoticed by 0.14.0 until output arrives.
 *
 * How the probe ended decides, not the signal it ended after: one stopped
 * by SIGINT exits with status 0 and nothing on stderr (0.14.0, 0.20.2 and
 * 0.23.2). A probe that ended by itself with status 0 before its stop is
 * refused as one that ended itself after so many ms, with selfEndRemedy as
 * its hint. One that ended otherwise before its stop, or with an error after
 * it, is read from its stderr as one that exited at its start
 * (classifyAttachFailure()). A probe the stop could not end (every signal
 * refused or ignored) ends by its own self-exit: the call waits for that, up
 * to selfExitMs plus PROBE_SELF_EXIT_SLACK_MS after the stop began, and reaps
 * it, so the probe does not outlive the call. Only a probe that outlives even
 * that bound is left, and reported.
 * @return nullopt when it attached and stopped on SIGINT with status 0;
 *         otherwise the Error (refused, unsupported, denied, unusable), or
 *         the Warning (it ignored SIGINT; it had not attached by the bound,
 *         or @p scratchDir is empty, so only the run can show whether it
 *         attaches; or sudo refused a probe command that differs from the
 *         run's).
 */
std::optional<ReadinessResult> probeAttach(const BpftraceRoute& route, const AttachProbe& probe,
                                           const ReadinessContext& ctx,
                                           const std::string& scratchDir);

/**
 * @brief The result for a tracer that exited at start, from its stderr.
 *
 * @p commandLine is the command that was refused, as attempted. On the sudo
 * route, sudo's policy refusal ("a password is required", "is not allowed
 * to execute") depends on the exact arguments: it is DENIED when
 * @p runCommand is empty (the refused command is the run's own), and
 * UNVERIFIED, naming both commands, when @p runCommand says how the run's
 * differing command reads. Any other sudo failure is DENIED whatever the
 * arguments. Its hints hold for every caller, the bpftrace backend's scripts
 * and the offcpu backend's one script alike.
 */
ReadinessResult classifyAttachFailure(const BpftraceRoute& route, const std::string& what,
                                      const std::string& commandLine, const std::string& stderrText,
                                      const ReadinessContext& ctx,
                                      const std::string& runCommand = {});

/**
 * @brief How to get access: root, directly or through a scoped grant.
 *        bpftrace (0.14.0, 0.20.2 and 0.23.2 checked) refuses every effective
 *        user but root, whatever its capabilities.
 */
std::string optInRemedy(const BpftraceRoute& route, const ReadinessContext& ctx);

/** @brief What a scoped sudoers grant must allow. */
std::string grantRemedy(const BpftraceRoute& route, const ReadinessContext& ctx);

/**
 * @brief Judge how a run's tracer ended at its stop, and print it prefixed
 * "[<tag>]".
 *
 * How the tracer ended decides, as for the readiness probe, not the signal it
 * ended after: bpftrace exits with status 0 once it has printed its maps, on
 * SIGINT or SIGTERM. A tracer that so ended, at the stop or before it, is
 * complete, whatever its stderr holds (warnings included). Any other end is
 * not, and is printed: one still running, killed, ended by a signal it did
 * not handle, or exited with another status, named with the line of its
 * error output at @p errorPath ("" when none is kept) that says why, or with
 * "nothing on its stderr". Each refused delivery is printed too, and a tracer
 * that ended only on SIGTERM.
 * @return nullopt when the tracer ended with status 0; otherwise the Error
 *         (UNUSABLE) whose message, after its cause word, is the line printed.
 */
std::optional<ReadinessResult> reportStop(const char* tag, const std::string& what,
                                          const HelperStopResult& stop, const BpftraceRoute& route,
                                          const std::string& errorPath = {});

/* ----------------------------- The capture window ----------------------------- */

// A capture starts when its tracer acknowledges that it sees this process,
// and ends when the tracer acknowledges the stop. Both acknowledgements come
// from a sched_switch program appended to the tracer's script
// (captureWindowProgram()), bound to three ids: the benchmark's pid, the
// thread the backend creates to arm it (an ArmThread, named ARM_THREAD), and
// the thread that runs the measured repeats, which takes the name STOP_THREAD
// once they are over. Each tracer prints its own two lines into its own
// output, which its launch opens empty.
//
// bpftrace prints what its programs emit only once it has attached every
// probe of the script: 0.14.0, 0.20.2 and 0.23.2 read their programs' output
// after both of their attach passes (tracepoints and the like, then kprobes,
// uprobes and the like). An arm line in the output therefore shows that the
// script's probes are attached and that the tracer sees this process's
// threads under the ids the backend knows; it does not confine the script's
// own probes to the measured repeats, which record from their attach to the
// stop.
//
// A backend creates the ArmThread, launches its tracers, then waits
// (waitUntil()) as WAIT_THREAD (ThreadNameScope) while the ArmThread naps,
// reading each output as it grows (WindowWatch) and comparing the arm line's
// ids with the ones it bound. At the stop it reads each output once more (a
// stop line already there came too early), names the measuring thread
// STOP_THREAD and waits the same way. The three thread names are reserved for
// the backend's own threads; the map names starting with WINDOW_MAP_PREFIX and
// the acknowledgement lines are reserved for the program, and reservedNameIn()
// finds a script's use of them. A script with an iterator probe
// (iteratorProbeIn()) cannot take the program, since bpftrace runs such a
// probe only alone.

/// Thread names the capture window reserves.
inline constexpr const char* ARM_THREAD = "vernier-arm";
inline constexpr const char* WAIT_THREAD = "vernier-wait";
inline constexpr const char* STOP_THREAD = "vernier-stop";

/// The prefix of the map names the capture window reserves (it uses @vernier_window).
inline constexpr const char* WINDOW_MAP_PREFIX = "@vernier_";

/// Default bounds of the waits for the arm and the stop acknowledgements.
inline constexpr int ARM_WAIT_MS = 5000;
inline constexpr int DISARM_WAIT_MS = 3000;

/// The nap of every wait, and of the arm thread.
inline constexpr int WINDOW_NAP_MS = 5;

/**
 * @brief The capture-window program for process @p pid, armed by thread
 *        @p armTid and stopped by thread @p stopTid, its two acknowledgements
 *        labelled @p label ("bpftrace", "offcpu").
 *
 * One tracepoint:sched:sched_switch program. A sleeping switch-out
 * (prev_state 1 or 2) of thread @p armTid of @p pid while it is named
 * ARM_THREAD arms it once and prints "<label> armed <pid> <tid>"; one of
 * thread @p stopTid while it is named STOP_THREAD then disarms it once and
 * prints "<label> disarmed <pid> <tid>". The disarm sets the window's state
 * to 2 at once and clears @vernier_window, so a report holds no map of its
 * own: bpftrace clears a map only when its own process handles the clear(),
 * after the program has returned, and the state set at once keeps a later
 * switch-out from printing the line again before then. The program counts
 * nothing. Run it with -B none, so each line reaches the output as it is
 * printed. Ids of 0 make a program that never arms (a readiness probe's).
 */
[[nodiscard]] std::string captureWindowProgram(const std::string& label, long pid, long armTid,
                                               long stopTid);

/** @brief One acknowledgement: the process and thread ids the tracer printed. */
struct WindowAck {
  long pid = -1; ///< The process id it printed.
  long tid = -1; ///< The thread id it printed.
};

/**
 * @brief "<label> armed <pid> <tid>" in @p line, anywhere in it, so a JSON
 *        printf record is read too; nullopt when the line holds none. What
 *        follows the two ids is not read.
 */
[[nodiscard]] std::optional<WindowAck> parseArmAck(std::string_view line, std::string_view label);

/** @brief "<label> disarmed <pid> <tid>" in @p line, anywhere in it; nullopt when none. */
[[nodiscard]] std::optional<WindowAck> parseDisarmAck(std::string_view line,
                                                      std::string_view label);

/**
 * @brief True when @p line is one of the capture window's own: an
 *        acknowledgement labelled @p label, or a line naming a map that
 *        starts with WINDOW_MAP_PREFIX. The rest of a report is the script's.
 */
[[nodiscard]] bool isWindowLine(std::string_view line, std::string_view label);

/**
 * @brief A tracer's output file, read as it grows, and the capture window's
 *        two acknowledgements found in it (the first of each), in order.
 */
class WindowWatch {
public:
  WindowWatch(std::string outputPath, std::string label);

  /**
   * @brief Read what the tracer wrote since the last call, and look for the
   *        acknowledgements in every line it completed.
   * @return 0, or the errno of the failed open or read.
   */
  int poll();

  [[nodiscard]] const std::optional<WindowAck>& armed() const noexcept { return armed_; }
  [[nodiscard]] const std::optional<WindowAck>& disarmed() const noexcept { return disarmed_; }

  /** @brief True when the first stop line came before any arm line. */
  [[nodiscard]] bool disarmedBeforeArmed() const noexcept { return disarmedBeforeArmed_; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
  std::string path_;
  std::string label_;
  std::uintmax_t offset_ = 0;
  std::string partial_;
  std::optional<WindowAck> armed_;
  std::optional<WindowAck> disarmed_;
  bool disarmedBeforeArmed_ = false;
};

/**
 * @brief The first name in @p script that the capture window reserves, as a
 *        report names it ("the map @vernier_window", "the text 'bpftrace
 *        armed'"); empty when there is none. Code is read, not comments: a
 *        map whose name starts with WINDOW_MAP_PREFIX, or a string holding
 *        "<label> armed" or "<label> disarmed".
 */
[[nodiscard]] std::string reservedNameIn(std::string_view script, std::string_view label);

/**
 * @brief The first iterator probe in @p script ("iter:task", or with the
 *        alias "it:"), as the script names it; empty when there is none.
 *        bpftrace runs an iterator probe only as a script's single probe.
 */
[[nodiscard]] std::string iteratorProbeIn(std::string_view script);

/** @brief Names the calling thread @p name until the scope ends, then restores its name. */
class ThreadNameScope {
public:
  explicit ThreadNameScope(const char* name);
  ~ThreadNameScope();

  ThreadNameScope(const ThreadNameScope&) = delete;
  ThreadNameScope& operator=(const ThreadNameScope&) = delete;

private:
  char saved_[16] = {};
  bool renamed_ = false;
};

/**
 * @brief A thread named ARM_THREAD that naps WINDOW_NAP_MS at a time, each
 *        nap a sleeping switch-out, until it is destroyed. The constructor
 *        returns once the thread has named itself.
 */
class ArmThread {
public:
  ArmThread();
  ~ArmThread();

  ArmThread(const ArmThread&) = delete;
  ArmThread& operator=(const ArmThread&) = delete;

  /** @brief Its thread id; -1 where the platform has none to give. */
  [[nodiscard]] long tid() const noexcept { return tid_.load(); }

private:
  std::atomic<bool> stop_{false};
  std::atomic<bool> named_{false};
  std::atomic<long> tid_{-1};
  std::thread thread_;
};

/**
 * @brief Call @p done every WINDOW_NAP_MS until it returns true or
 *        @p boundMs has passed.
 * @return Whether it returned true.
 */
bool waitUntil(const std::function<bool()>& done, int boundMs);

/** @brief The calling thread's id; -1 where the platform has none to give. */
[[nodiscard]] long currentThreadId() noexcept;

/** @brief A PID namespace as /proc/self/ns/pid shows it. */
struct PidNamespaceId {
  bool read = false;       ///< The link could be stat'ed.
  unsigned long inode = 0; ///< Its inode: the namespace's identity.
  std::string link;        ///< What it links to ("pid:[4026532284]"); empty if unreadable.
};

/** @brief This process's PID namespace, read from /proc/self/ns/pid. */
[[nodiscard]] PidNamespaceId readPidNamespace();

/**
 * @brief What @p ns links to ("pid:[4026532284]") when it is a PID namespace
 *        other than the host's, whose inode the kernel fixes at 0xEFFFFFFC
 *        (bpftrace makes the same comparison); "pid:[<inode>]" when the link
 *        text was not read; empty for the host's, and when @p ns was not read.
 */
[[nodiscard]] std::string foreignPidNamespace(const PidNamespaceId& ns);

/** @brief foreignPidNamespace() of readPidNamespace(): this process's. */
[[nodiscard]] std::string foreignPidNamespace();

} // namespace bpftrace_tool

/* ----------------------------- BpftracePlan ----------------------------- */

/** @brief What the bpftrace check verified, for the launch to use. */
struct BpftracePlan final : ReadinessPlan {
  BpftraceRoute route;
  std::vector<std::string> scripts;     ///< Script names as requested.
  std::vector<std::string> scriptPaths; ///< Their resolved files, same order.
  std::string format = "text";          ///< Output format (PERF_BPF_FMT).
  std::string outputDir;                ///< PERF_BPF_OUT, or empty.
  bool envEnabled = false;              ///< PERF_BPF asked for bpftrace.
  std::shared_ptr<const ReadinessContext> context;
  int armWaitMs = bpftrace_tool::ARM_WAIT_MS;       ///< Bound of the wait for every arm.
  int disarmWaitMs = bpftrace_tool::DISARM_WAIT_MS; ///< Bound of the wait for every stop.
  /// Where a run reads this process's PID namespace when no tracer arms, to
  /// name a foreign one (a test substitutes it).
  std::function<bpftrace_tool::PidNamespaceId()> pidNamespace = bpftrace_tool::readPidNamespace;
};

/**
 * @brief The bpftrace backend's readiness decision for @p request in @p ctx.
 *
 * Reads BENCH_SUDO, PERF_BPF_SUDO, PERF_BPF_SCRIPTS, PERF_BPF_FMT, PERF_BPF and
 * PERF_BPF_OUT from the snapshot. A script name without a '/' is looked up as
 * <name>.bt in PERF_BPF_SCRIPTS, or, when that is unset or empty, in the bpf/
 * directory of the source tree this library was built from, by absolute path;
 * a name with a '/' is a path to the script file, absolute or from the working
 * directory. Either may leave out the .bt suffix. A script that uses a name the
 * capture window reserves is a CONFIGURATION error, and one with an iterator
 * probe UNSUPPORTED, before anything runs. On success the result's plan is a
 * BpftracePlan.
 */
ReadinessResult checkBpftraceRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/* ----------------------------- BpftraceProfiler ----------------------------- */

/**
 * @brief bpftrace profiler implementation.
 *
 * Runs bpftrace scripts with PID filtering. Scripts contain {{PID}} placeholder
 * which is replaced with the target process PID before execution.
 */
class BpftraceProfiler final : public Profiler {
public:
  /**
   * @brief Construct bpftrace profiler, deciding the request itself.
   *
   * Captures the environment and runs checkBpftraceRequest(); when the
   * request cannot run, prints why and does nothing in the hooks.
   * @param cfg Configuration with bpfScripts and artifactRoot
   * @param testName Test identifier (e.g., "Suite.Case")
   */
  BpftraceProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Construct from a decision already made (the registry's path). */
  BpftraceProfiler(const PerfConfig& cfg, std::string testName,
                   std::shared_ptr<const BpftracePlan> plan);
  ~BpftraceProfiler() override;

  std::string toolName() const noexcept override { return "bpftrace"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

  /**
   * @brief The capture's outcome: empty until a tracer fails to start or the
   * measured repeats end. Then READY when every tracer captured them and
   * printed data of its own; CAVEAT when one captured them but printed only
   * the capture window's lines; or the Error that leaves the capture
   * incomplete: a tracer that could not start, ended before its arm or its
   * stop acknowledgement, did not acknowledge either within its bound (named
   * UNSUPPORTED, with the PID namespace, in one other than the host's),
   * acknowledged either with other ids or out of order, could not be
   * stopped or did not end its stop with status 0 (reportStop() names how
   * it ended), or left no output, an empty one, or one without the
   * acknowledgements; or measured repeats that ended on another thread than
   * they started on. The first failure is kept, and the capture's files stay.
   */
  [[nodiscard]] const std::optional<ReadinessResult>& captureOutcome() const noexcept;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;

  // Forward declaration of implementation details (defined in .cpp)
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Factory function for bpftrace profiler.
 *
 * Decides the request in a snapshot of this process first.
 * @return Profiler instance, or nullptr if the request cannot run here.
 */
std::unique_ptr<Profiler> makeBpftraceProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERBPFTRACE_HPP
