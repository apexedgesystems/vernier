/**
 * @file ProfilerBpftrace.cpp
 * @brief Implementation of bpftrace profiler backend.
 *
 * The readiness check and the launch share one route (BpftraceRoute): the
 * privilege decision and the resolved bpftrace, sudo and kill. The check
 * attaches each selected script through that route and stops it the way the
 * run will; the runner then launches and stops with the route the check
 * verified. Runner internals stay in an anonymous namespace.
 */

#include "src/bench/inc/ProfilerBpftrace.hpp"

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace vernier {
namespace bench {

/* ----------------------------- BpftraceRoute ----------------------------- */

std::vector<std::string> BpftraceRoute::command() const {
  if (privilege.route == PrivilegeRoute::SCOPED_SUDO) {
    return {sudo, "-n", "--", bpftrace};
  }
  return {bpftrace};
}

HelperStopPolicy BpftraceRoute::stopPolicy(int interruptWaitMs, int terminateWaitMs, int killWaitMs,
                                           std::shared_ptr<const ReadinessContext> ctx) const {
  HelperStopPolicy policy;
  policy.route = privilege.route;
  policy.sudoPath = sudo;
  policy.killPath = kill;
  policy.interruptWaitMs = interruptWaitMs;
  policy.terminateWaitMs = terminateWaitMs;
  policy.killWaitMs = killWaitMs;
  policy.context = std::move(ctx);
  return policy;
}

std::string BpftraceRoute::describe() const {
  switch (privilege.route) {
  case PrivilegeRoute::SCOPED_SUDO:
    return "through sudo -n (" + privilege.source + ")";
  case PrivilegeRoute::ALREADY_ROOT:
    return "as root";
  case PrivilegeRoute::CURRENT_USER:
  case PrivilegeRoute::CONFIG_ERROR:
    break;
  }
  return "as the current user";
}

/* ----------------------------- bpftrace tool ----------------------------- */

namespace bpftrace_tool {

namespace {

bool contains(const std::string& text, const char* needle) {
  return text.find(needle) != std::string::npos;
}

bool containsAny(const std::string& text, std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (contains(text, needle)) {
      return true;
    }
  }
  return false;
}

/** @brief The line that names the failure: sudo's own, else bpftrace's "ERROR". */
std::string failureLine(const std::string& text) {
  std::istringstream in(text);
  std::string line;
  std::string errorLine;
  while (std::getline(in, line)) {
    if (line.rfind("sudo:", 0) == 0 || contains(line, "is not allowed to execute")) {
      return outputTail(line, 1);
    }
    if (errorLine.empty() && contains(line, "ERROR")) {
      errorLine = outputTail(line, 1);
    }
  }
  if (!errorLine.empty()) {
    return errorLine;
  }
  const std::string TAIL = outputTail(text, 2);
  return TAIL.empty() ? std::string{"no message"} : TAIL;
}

const char* signalName(int sig) {
  switch (sig) {
  case SIGINT:
    return "SIGINT";
  case SIGTERM:
    return "SIGTERM";
  case SIGKILL:
    return "SIGKILL";
  default:
    return "a signal";
  }
}

} // namespace

std::string optInRemedy(const BpftraceRoute& route, const ReadinessContext& ctx) {
  std::string killPath = route.kill;
  if (killPath.empty()) {
    const auto KILL = resolveExecutable("kill", ctx);
    killPath = (KILL && KILL->executable) ? KILL->path : std::string{"kill"};
  }
  return "Set BENCH_SUDO=1 with a scoped sudoers grant for " + route.bpftrace + " and " + killPath +
         ", run with CAP_BPF and CAP_PERFMON, or run as root.";
}

std::string grantRemedy(const BpftraceRoute& route, const ReadinessContext& ctx) {
  const std::string USER = ctx.get("USER").value_or("<user>");
  return "The grant must allow " + route.bpftrace + " with the run's script arguments and " +
         route.kill + " with -2, -15 and -9 for the tracer's pid. A grant that allows every " +
         "argument, for example '" + USER + " ALL=(root) NOPASSWD: " + route.bpftrace + ", " +
         route.kill + "', does; an argument-restricted grant must name those arguments. " +
         "vernier never edits sudoers.";
}

std::optional<ReadinessResult> resolveRoute(const ReadinessContext& ctx,
                                            std::string_view legacyAlias, BpftraceRoute& route) {
  route.privilege = decidePrivilege(ctx, legacyAlias);
  if (route.privilege.route == PrivilegeRoute::CONFIG_ERROR) {
    return readinessResult(ReadinessCause::CONFIGURATION, route.privilege.error,
                           route.privilege.remedy);
  }
  const auto TOOL = resolveExecutable("bpftrace", ctx);
  if (!TOOL) {
    return readinessResult(ReadinessCause::MISSING, "bpftrace not found on PATH",
                           "apt install bpftrace (or your distribution's package).");
  }
  if (!TOOL->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, TOOL->path + " is not an executable file",
                           "Reinstall bpftrace, or fix PATH so it finds a working bpftrace.");
  }
  route.bpftrace = TOOL->path;
  if (route.privilege.route != PrivilegeRoute::SCOPED_SUDO) {
    return std::nullopt;
  }
  for (const char* HELPER : {"sudo", "kill"}) {
    const auto FOUND = resolveExecutable(HELPER, ctx);
    if (!FOUND || !FOUND->executable) {
      return readinessResult(ReadinessCause::MISSING_HELPER,
                             std::string{HELPER} + " is not on PATH; " + route.privilege.source +
                                 " runs bpftrace through sudo -n and stops it with sudo -n kill",
                             std::string{"Install "} + HELPER +
                                 ", or unset BENCH_SUDO and run with CAP_BPF and CAP_PERFMON "
                                 "or as root.");
    }
    (std::string{HELPER} == "sudo" ? route.sudo : route.kill) = FOUND->path;
  }
  return std::nullopt;
}

std::optional<ReadinessResult> probeExecutable(const BpftraceRoute& route,
                                               const ReadinessContext& ctx) {
  const ProbeResult VERSION = runBoundedProbe({route.bpftrace, "--version"}, 5000, ctx);
  if (VERSION.succeeded()) {
    return std::nullopt;
  }
  const std::string TAIL = outputTail(VERSION.output);
  return readinessResult(ReadinessCause::UNUSABLE,
                         route.bpftrace + " --version: " + VERSION.describe() +
                             (TAIL.empty() ? std::string{} : ": " + TAIL),
                         "Reinstall bpftrace; this executable does not run.");
}

ReadinessResult classifyAttachFailure(const BpftraceRoute& route, const std::string& what,
                                      const std::string& commandLine, const std::string& stderrText,
                                      const ReadinessContext& ctx, const std::string& runCommand) {
  const std::string LINE = failureLine(stderrText);
  if (route.privilege.route == PrivilegeRoute::SCOPED_SUDO) {
    // sudo's policy answers for the exact command line: a refusal holds for
    // the run only when the refused command is the run's own.
    if (containsAny(stderrText, {"a password is required", "is not allowed to execute"})) {
      if (!runCommand.empty()) {
        return readinessResult(ReadinessCause::UNVERIFIED,
                               "sudo -n refused the probe command " + commandLine + ": " + LINE +
                                   "; the run executes " + runCommand +
                                   " instead, which only the run can try",
                               grantRemedy(route, ctx));
      }
      return readinessResult(ReadinessCause::DENIED, "sudo -n refused " + commandLine + ": " + LINE,
                             grantRemedy(route, ctx));
    }
    // Any other failure of sudo itself does not depend on the arguments.
    if (containsAny(stderrText, {"sudo:"})) {
      return readinessResult(ReadinessCause::DENIED,
                             "sudo -n failed for " + commandLine + ": " + LINE,
                             "sudo cannot run commands as root here, whatever the grant; unset "
                             "BENCH_SUDO and run with CAP_BPF and CAP_PERFMON, or run as root.");
    }
  }
  if (containsAny(stderrText, {"only supports running as the root user", "Operation not permitted",
                               "Permission denied", "EPERM", "EACCES"})) {
    return readinessResult(
        ReadinessCause::DENIED, what + " could not attach " + route.describe() + ": " + LINE,
        route.privilege.route == PrivilegeRoute::CURRENT_USER
            ? optInRemedy(route, ctx)
            : std::string{"bpftrace was refused although it ran as root: check the kernel "
                          "lockdown mode and the capabilities of this container."});
  }
  if (containsAny(stderrText, {"tracepoint not found", "probe not found", "not supported",
                               "No such file or directory", "does not exist"})) {
    return readinessResult(ReadinessCause::UNSUPPORTED, what + ": " + LINE,
                           "The kernel lacks a probe the script uses, or tracefs is not mounted "
                           "(mount -t tracefs tracefs /sys/kernel/tracing); select a script "
                           "whose probes exist here.");
  }
  return readinessResult(ReadinessCause::UNUSABLE, what + " did not stay attached: " + LINE,
                         "Run the script by hand with " + route.bpftrace + " to see why.");
}

std::optional<ReadinessResult> probeAttach(const BpftraceRoute& route, const AttachProbe& probe,
                                           const ReadinessContext& ctx,
                                           const std::string& scratchDir) {
  std::vector<std::string> argv = route.command();
  argv.insert(argv.end(), probe.toolArgs.begin(), probe.toolArgs.end());
  const std::string& what = probe.what;

  OwnedHelper helper(
      route.stopPolicy(2000, 1000, 1000, std::make_shared<const ReadinessContext>(ctx)));
  const auto STARTED_AT = std::chrono::steady_clock::now();
  const HelperStart START = helper.start(
      argv, "", scratchDir.empty() ? "" : scratchDir + "/attach.err", probe.graceMs, &ctx);
  if (!START.started) {
    return readinessResult(ReadinessCause::UNUSABLE, what + ": " + START.errorTail,
                           "Check that " + argv.front() + " can be executed.");
  }
  if (START.exitedEarly) {
    return classifyAttachFailure(route, what, probe.commandLine, START.errorTail, ctx,
                                 probe.runCommand);
  }
  const HelperStopResult STOP = helper.stop();
  std::string lingering;
  if (STOP.stillAlive && probe.selfExitMs > 0) {
    // The stop could not end it; its own self-exit will. Wait for that,
    // bounded, and reap it, so the probe does not outlive the check.
    const auto UNTIL = STARTED_AT + std::chrono::milliseconds(probe.graceMs + probe.selfExitMs +
                                                              PROBE_SELF_EXIT_SLACK_MS);
    while (helper.running() && std::chrono::steady_clock::now() < UNTIL) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  if (STOP.stillAlive && helper.running()) {
    lingering = "; the probe tracer " + std::to_string(helper.pid()) + " still runs" +
                (probe.selfExitMs > 0
                     ? " past its " + std::to_string(probe.selfExitMs / 1000) + " s self-exit"
                     : std::string{});
  }
  for (const StopDelivery& delivery : STOP.deliveries) {
    if (delivery.delivered) {
      continue;
    }
    // The run's stop would be refused the same way, losing the flush on SIGINT.
    if (route.privilege.route == PrivilegeRoute::SCOPED_SUDO) {
      return readinessResult(
          ReadinessCause::DENIED,
          "cleanup: sudo -n refused " + route.kill + " -" + std::to_string(delivery.signal) + " " +
              std::to_string(delivery.target) + ": " + delivery.detail + lingering,
          grantRemedy(route, ctx));
    }
    return readinessResult(ReadinessCause::DENIED,
                           "cleanup: " + delivery.command + " failed: " + delivery.detail +
                               lingering,
                           "The tracer runs as another user; stop it through the same route.");
  }
  if (STOP.stillAlive) {
    return readinessResult(ReadinessCause::UNUSABLE,
                           "the probe tracer " + std::to_string(helper.pid()) +
                               " did not stop on SIGINT, SIGTERM or SIGKILL" + lingering,
                           "Stop it by hand, then check the bpftrace build.");
  }
  if (STOP.wasRunning && STOP.stoppedBy != SIGINT && STOP.stoppedBy != 0) {
    return readinessResult(ReadinessCause::CAVEAT,
                           "the probe tracer ignored SIGINT and stopped on " +
                               std::string{signalName(STOP.stoppedBy)} +
                               "; the run's output may be incomplete",
                           "bpftrace prints its maps on SIGINT; check that this build handles it.");
  }
  if (!STOP.wasRunning && !(WIFEXITED(STOP.waitStatus) && WEXITSTATUS(STOP.waitStatus) == 0)) {
    std::ifstream err(scratchDir + "/attach.err");
    std::stringstream text;
    text << err.rdbuf();
    return classifyAttachFailure(route, what, probe.commandLine, text.str(), ctx, probe.runCommand);
  }
  return std::nullopt;
}

bool reportStop(const char* tag, const std::string& what, const HelperStopResult& stop,
                const BpftraceRoute& route) {
  for (const StopDelivery& delivery : stop.deliveries) {
    if (!delivery.delivered) {
      std::fprintf(stderr, "[%s] %s: could not deliver %s to tracer %d: %s: %s\n", tag,
                   what.c_str(), signalName(delivery.signal), static_cast<int>(delivery.target),
                   delivery.command.c_str(), delivery.detail.c_str());
    }
  }
  if (stop.stillAlive) {
    const std::string MANUAL = route.privilege.route == PrivilegeRoute::SCOPED_SUDO
                                   ? route.sudo + " -n " + route.kill + " -9 <pid>"
                                   : std::string{"kill -9 <pid>"};
    std::fprintf(stderr,
                 "[%s] %s: the tracer is still running and its output may be incomplete; stop "
                 "it with: %s\n",
                 tag, what.c_str(), MANUAL.c_str());
    return false;
  }
  if (!stop.wasRunning) {
    if (WIFEXITED(stop.waitStatus) && WEXITSTATUS(stop.waitStatus) == 0) {
      return true;
    }
    std::fprintf(stderr, "[%s] %s: the tracer ended before the stop (wait status %d)\n", tag,
                 what.c_str(), stop.waitStatus);
    return false;
  }
  if (stop.stoppedBy == SIGKILL) {
    std::fprintf(stderr,
                 "[%s] %s: the tracer ignored SIGINT and SIGTERM and was killed; its output is "
                 "incomplete\n",
                 tag, what.c_str());
    return false;
  }
  if (stop.stoppedBy == SIGTERM) {
    std::fprintf(stderr,
                 "[%s] %s: the tracer ignored SIGINT and stopped on SIGTERM; its output may be "
                 "incomplete\n",
                 tag, what.c_str());
  }
  return true;
}

} // namespace bpftrace_tool

#ifdef __linux__
// ============================================================================
// Runner
// ============================================================================

namespace { // Internal implementation details

constexpr int START_GRACE_MS = 1000; // bpftrace attaches within this before measuring
constexpr int PROBE_SELF_EXIT_S = 5; // a readiness probe copy exits by itself after this
constexpr int INTERRUPT_WAIT_MS = 2000;
constexpr int TERMINATE_WAIT_MS = 1000;
constexpr int KILL_WAIT_MS = 1000;
const char* const DEFAULT_SCRIPTS_DIR = "src/bench/bpf";

/** @brief Where script @p name is looked up: `<dir>/<name>.bt` (an absolute name keeps its path).
 */
std::string scriptPathFor(const std::string& scriptsDir, const std::string& name) {
  return (std::filesystem::path(scriptsDir) / (name + ".bt")).string();
}

/** @brief Read the script at @p path into @p text; 0, or the errno of the failed open or read. */
int readScript(const std::string& path, std::string& text) {
  const int FD = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (FD < 0) {
    return errno;
  }
  text.clear();
  char buf[4096];
  while (true) {
    const ssize_t N = ::read(FD, buf, sizeof(buf));
    if (N < 0 && errno == EINTR) {
      continue;
    }
    if (N < 0) {
      const int ERR = errno;
      ::close(FD);
      return ERR;
    }
    if (N == 0) {
      break;
    }
    text.append(buf, static_cast<std::size_t>(N));
  }
  ::close(FD);
  return 0;
}

std::string errnoText(int err) { return std::system_category().message(err); }

void replacePid(std::string& text, long pid) {
  const std::string TOKEN = "{{PID}}";
  const std::string VALUE = std::to_string(pid);
  for (std::size_t pos = 0; (pos = text.find(TOKEN, pos)) != std::string::npos;
       pos += VALUE.size()) {
    text.replace(pos, TOKEN.size(), VALUE);
  }
}

/** @brief `-q [-f json] <script>`: the arguments every launch passes. */
std::vector<std::string> launchArgs(const std::string& format, const std::string& script) {
  std::vector<std::string> args{"-q"};
  if (format == "json") {
    args.insert(args.end(), {"-f", "json"});
  }
  args.push_back(script);
  return args;
}

/** @brief `<bpftrace> <args...>` as reports show a command. */
std::string commandLineOf(const std::string& bpftrace, const std::vector<std::string>& args) {
  std::string line = bpftrace;
  for (const std::string& arg : args) {
    line += " " + arg;
  }
  return line;
}

/** @brief Where the run writes its copy of script @p name inside capture folder @p outdir. */
std::string runCopyPath(const std::string& outdir, const std::string& name) {
  return (std::filesystem::path(outdir) / (name + ".tmp.bt")).string();
}

/** @brief One tracer for one script, started and stopped through the plan's route. */
class BpfRunner {
public:
  BpfRunner(std::shared_ptr<const BpftracePlan> plan, std::string name, std::string scriptPath,
            std::string outdir)
      : plan_(std::move(plan)), name_(std::move(name)), scriptPath_(std::move(scriptPath)),
        outdir_(std::move(outdir)) {}

  ~BpfRunner() { stop(); }

  BpfRunner(const BpfRunner&) = delete;
  BpfRunner& operator=(const BpfRunner&) = delete;

  bool start(pid_t pid) {
    std::string src;
    if (const int ERR = readScript(scriptPath_, src); ERR != 0) {
      std::fprintf(stderr, "[bpftrace] cannot read script '%s' at %s: %s\n", name_.c_str(),
                   scriptPath_.c_str(), errnoText(ERR).c_str());
      if (ERR == ENOENT) {
        std::fprintf(stderr,
                     "[bpftrace] Pass `--bpf <script>` with a script name that exists under\n"
                     "[bpftrace] `--bpf-scripts <dir>` (default: src/bench/bpf/),\n"
                     "[bpftrace] or pass an absolute path via `--bpf </path/to/script.bt>`.\n");
      }
      return false;
    }
    replacePid(src, static_cast<long>(pid));
    const std::filesystem::path OUTDIR(outdir_);
    std::error_code ec;
    std::filesystem::create_directories(OUTDIR, ec);
    const std::string TEMP_SCRIPT = runCopyPath(outdir_, name_);
    {
      std::ofstream out(TEMP_SCRIPT);
      out << src;
      out.close();
      if (!out) {
        std::fprintf(stderr, "[bpftrace] cannot write the run's copy of script '%s' to %s\n",
                     name_.c_str(), TEMP_SCRIPT.c_str());
        return false;
      }
    }
    const std::string STDOUT_PATH = (OUTDIR / (name_ + ".out." + plan_->format)).string();
    const std::string STDERR_PATH = (OUTDIR / (name_ + ".err.txt")).string();

    const std::vector<std::string> ARGS = launchArgs(plan_->format, TEMP_SCRIPT);
    std::vector<std::string> argv = plan_->route.command();
    argv.insert(argv.end(), ARGS.begin(), ARGS.end());

    // fork+exec of the resolved paths, never a shell: signals land on the
    // process this runner tracks, and the executed tools are the ones the
    // check verified.
    helper_ = std::make_unique<OwnedHelper>(plan_->route.stopPolicy(
        INTERRUPT_WAIT_MS, TERMINATE_WAIT_MS, KILL_WAIT_MS, plan_->context));
    const HelperStart STARTED =
        helper_->start(argv, STDOUT_PATH, STDERR_PATH, START_GRACE_MS, nullptr);
    if (!STARTED.started) {
      std::fprintf(stderr, "[bpftrace] script '%s' could not start: %s\n", name_.c_str(),
                   STARTED.errorTail.c_str());
      helper_.reset();
      return false;
    }
    if (STARTED.exitedEarly) {
      const ReadinessResult WHY = bpftrace_tool::classifyAttachFailure(
          plan_->route, "script '" + name_ + "'", commandLineOf(plan_->route.bpftrace, ARGS),
          STARTED.errorTail, *plan_->context);
      std::fprintf(stderr, "[bpftrace] the tracer exited during its start grace: %s\n",
                   WHY.report.message.c_str());
      if (!WHY.report.hint.empty()) {
        std::fprintf(stderr, "[bpftrace] %s\n", WHY.report.hint.c_str());
      }
      helper_.reset();
      return false;
    }
    return true;
  }

  void stop() noexcept {
    if (!helper_) {
      return;
    }
    const HelperStopResult STOPPED = helper_->stop();
    (void)bpftrace_tool::reportStop("bpftrace", "script '" + name_ + "'", STOPPED, plan_->route);
    helper_.reset();
  }

private:
  std::shared_ptr<const BpftracePlan> plan_;
  std::string name_;
  std::string scriptPath_;
  std::string outdir_;
  std::unique_ptr<OwnedHelper> helper_;
};

} // anonymous namespace

// ============================================================================
// Readiness
// ============================================================================

ReadinessResult checkBpftraceRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  auto plan = std::make_shared<BpftracePlan>();
  plan->context = std::make_shared<const ReadinessContext>(ctx);
  const std::string ENABLE = ctx.get("PERF_BPF").value_or("");
  std::string enable;
  for (const char CH : ENABLE) {
    enable += static_cast<char>(std::tolower(static_cast<unsigned char>(CH)));
  }
  plan->envEnabled = enable == "1" || enable == "true";
  const std::string SCRIPTS_DIR = ctx.get("PERF_BPF_SCRIPTS").value_or(DEFAULT_SCRIPTS_DIR);
  plan->outputDir = ctx.get("PERF_BPF_OUT").value_or("");
  plan->format = ctx.get("PERF_BPF_FMT").value_or("text");
  plan->scripts = request.bpfScripts.empty()
                      ? std::vector<std::string>{"write_latency", "fsync_latency"}
                      : request.bpfScripts;

  if (auto failure = bpftrace_tool::resolveRoute(ctx, "PERF_BPF_SUDO", plan->route)) {
    return *failure;
  }
  // Every selected script is read here, before anything runs: the probe and
  // the run both start from its text.
  std::vector<std::string> sources;
  for (const std::string& name : plan->scripts) {
    const std::string PATH = scriptPathFor(SCRIPTS_DIR, name);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(PATH, ec)) {
      return readinessResult(
          ReadinessCause::MISSING, "bpftrace script '" + name + "' not found at " + PATH,
          "Pass --bpf with a script under --bpf-scripts <dir> (PERF_BPF_SCRIPTS, "
          "default src/bench/bpf/) or an absolute path without the .bt suffix.");
    }
    std::string text;
    if (const int ERR = readScript(PATH, text); ERR != 0) {
      return readinessResult(
          ReadinessCause::UNUSABLE,
          "bpftrace script '" + name + "' at " + PATH + " cannot be read: " + errnoText(ERR),
          "Give this user read access to " + PATH + ", or select a script it can read with --bpf.");
    }
    plan->scriptPaths.push_back(PATH);
    sources.push_back(std::move(text));
  }
  if (auto failure = bpftrace_tool::probeExecutable(plan->route, ctx)) {
    return *failure;
  }

  // Run a copy of each selected script through the route, for the run's start
  // grace, and stop it with the run's first stop signal. The copy also exits
  // by itself, and it lives in a private directory: the run's own copy goes
  // in a capture folder that does not exist before the run.
  const bool SUDO = plan->route.privilege.route == PrivilegeRoute::SCOPED_SUDO;
  const ProbeScratch SCRATCH(ctx);
  std::optional<ReadinessResult> caveat;
  std::string runCommands;
  for (std::size_t i = 0; i < plan->scripts.size(); ++i) {
    const std::string RUN_COMMAND =
        commandLineOf(plan->route.bpftrace,
                      launchArgs(plan->format, runCopyPath("<capture folder>", plan->scripts[i])));
    runCommands += (runCommands.empty() ? "" : ", ") + RUN_COMMAND;
    std::string copy = sources[i];
    replacePid(copy, static_cast<long>(ctx.self()));
    copy += "\ninterval:s:" + std::to_string(PROBE_SELF_EXIT_S) + " { exit(); }\n";
    const std::string COPY_PATH = SCRATCH.write("probe" + std::to_string(i) + ".bt", copy);
    if (COPY_PATH.empty()) {
      const std::string BASE = ctx.get("TMPDIR").value_or("");
      const std::string WHERE =
          SCRATCH.ok() ? SCRATCH.path()
                       : "a new directory under " + (BASE.empty() ? std::string{"/tmp"} : BASE);
      return readinessResult(ReadinessCause::UNUSABLE,
                             "cannot write the probe copy of script '" + plan->scripts[i] +
                                 "' in " + WHERE,
                             "Point TMPDIR at a writable directory, or make /tmp writable.");
    }
    bpftrace_tool::AttachProbe probe;
    probe.toolArgs = launchArgs(plan->format, COPY_PATH);
    probe.what = "script '" + plan->scripts[i] + "'";
    probe.commandLine = commandLineOf(plan->route.bpftrace, probe.toolArgs);
    probe.runCommand = RUN_COMMAND;
    probe.graceMs = START_GRACE_MS;
    probe.selfExitMs = PROBE_SELF_EXIT_S * 1000;
    auto verdict = bpftrace_tool::probeAttach(plan->route, probe, ctx, SCRATCH.path());
    if (verdict && verdict->report.status == EnvReport::Status::Error) {
      return *verdict;
    }
    if (verdict && !caveat) {
      caveat = std::move(verdict);
    }
  }

  // Say what the probe showed and what only the run can show.
  std::string names;
  for (const std::string& name : plan->scripts) {
    names += (names.empty() ? "" : ", ") + name;
  }
  std::string message =
      names + ": " + (plan->scripts.size() == 1 ? "a probe copy" : "a probe copy of each") +
      " with a " + std::to_string(PROBE_SELF_EXIT_S) + " s self-exit stayed running for the " +
      std::to_string(START_GRACE_MS) + " ms start grace " + plan->route.describe() +
      " and stopped on SIGINT" + (SUDO ? " through sudo -n kill" : "") + " (probe with " +
      plan->route.bpftrace + ")";
  if (plan->route.privilege.route == PrivilegeRoute::ALREADY_ROOT && plan->route.privilege.optIn) {
    message += "; running as root; BENCH_SUDO not needed";
  }
  message += "; not checked: " +
             (SUDO ? "the grant for the run's own command (" + runCommands +
                         "), SIGTERM and SIGKILL through sudo, and "
                   : std::string{}) +
             "the run's capture";
  ReadinessResult result;
  if (caveat) {
    result = *caveat;
    if (!plan->route.privilege.warning.empty()) {
      result.report.message += "; " + plan->route.privilege.warning;
    }
  } else if (!plan->route.privilege.warning.empty()) {
    result =
        readinessResult(ReadinessCause::CAVEAT, message + "; " + plan->route.privilege.warning, "");
  } else {
    result = readinessResult(ReadinessCause::READY, message, "");
  }
  result.plan = std::move(plan);
  return result;
}

// ============================================================================
// BpftraceProfiler Implementation
// ============================================================================

class BpftraceProfiler::Impl {
public:
  Impl(const PerfConfig& cfg, const std::string& testName, std::shared_ptr<const BpftracePlan> plan)
      : plan_(std::move(plan)) {
    enabled_ = cfg.profileTool == "bpftrace" || plan_->envEnabled;
    std::string outputDir = plan_->outputDir;
    if (!cfg.artifactRoot.empty()) {
      outputDir = cfg.artifactRoot + "/" + profiler_env::artifactDirName(testName, "bpf");
    }
    // PERF_BPF_OUT or the artifact root already fixed the folder; otherwise
    // the shared rule names it.
    if (outputDir.empty()) {
      artifactDir_ = profiler_env::resolveArtifactDir(cfg.profileTool, "", testName, "bpf");
    } else {
      artifactDir_ = outputDir;
      std::error_code ec;
      std::filesystem::create_directories(artifactDir_, ec);
    }
  }

  void beforeMeasure() {
    if (!enabled_) {
      return;
    }
    const pid_t TARGET = ::getpid();
    for (std::size_t i = 0; i < plan_->scripts.size(); ++i) {
      // One tracer per started script: each runner owns its own process.
      auto runner = std::make_unique<BpfRunner>(plan_, plan_->scripts[i], plan_->scriptPaths[i],
                                                artifactDir_);
      if (runner->start(TARGET)) {
        runners_.push_back(std::move(runner));
      }
    }
  }

  void afterMeasure() {
    for (auto& runner : runners_) {
      runner->stop();
    }
    runners_.clear();
  }

  std::string artifactDir() const { return artifactDir_; }

private:
  std::shared_ptr<const BpftracePlan> plan_;
  bool enabled_ = false;
  std::string artifactDir_;
  std::vector<std::unique_ptr<BpfRunner>> runners_;
};

#else // !__linux__

ReadinessResult checkBpftraceRequest(const ReadinessRequest& /*request*/,
                                     const ReadinessContext& /*ctx*/) {
  return readinessResult(ReadinessCause::UNSUPPORTED, "bpftrace is Linux-only",
                         "Run on Linux or use a different profiler.");
}

#endif // __linux__

// ============================================================================
// Public Interface Implementation
// ============================================================================

namespace {

/** @brief The plan of a decision that lets collection run, or null. */
std::shared_ptr<const BpftracePlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const BpftracePlan>(result.plan);
}

#ifdef __linux__
/** @brief The bpftrace decision for @p cfg in a snapshot of this process. */
ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "bpftrace";
  return checkBpftraceRequest(request, CTX);
}
#endif

} // namespace

BpftraceProfiler::BpftraceProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
#ifdef __linux__
  const ReadinessResult DECISION = decideNow(cfg_);
  auto plan = readyPlan(DECISION);
  if (!plan) {
    std::fprintf(stderr, "[bpftrace] not started: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[bpftrace] %s\n", DECISION.report.hint.c_str());
    }
    return;
  }
  impl_ = std::make_unique<Impl>(cfg_, testName_, std::move(plan));
  artifactDir_ = impl_->artifactDir();
#endif
}

BpftraceProfiler::BpftraceProfiler(const PerfConfig& cfg, std::string testName,
                                   std::shared_ptr<const BpftracePlan> plan)
    : cfg_(cfg), testName_(std::move(testName)) {
#ifdef __linux__
  if (plan) {
    impl_ = std::make_unique<Impl>(cfg_, testName_, std::move(plan));
    artifactDir_ = impl_->artifactDir();
  }
#else
  (void)plan;
#endif
}

BpftraceProfiler::~BpftraceProfiler() = default;

void BpftraceProfiler::beforeMeasure() {
#ifdef __linux__
  if (impl_) {
    impl_->beforeMeasure();
  }
#endif
}

void BpftraceProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (impl_) {
    impl_->afterMeasure();
  }
#endif
}

std::unique_ptr<Profiler> makeBpftraceProfiler(const PerfConfig& cfg, const std::string& testName) {
#ifdef __linux__
  auto plan = readyPlan(decideNow(cfg));
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<BpftraceProfiler>(cfg, testName, std::move(plan));
#else
  (void)cfg;
  (void)testName;
  return nullptr;
#endif
}

namespace {

std::unique_ptr<Profiler> makePlannedBpftraceProfiler(const PerfConfig& cfg,
                                                      const std::string& testName,
                                                      const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<BpftraceProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("bpftrace", ::vernier::bench::checkBpftraceRequest,
                                   ::vernier::bench::makePlannedBpftraceProfiler,
                                   "Install bpftrace; see BENCH_SUDO for privileges.",
                                   "PERF_BPF_SUDO", "PERF_BPF_SCRIPTS", "PERF_BPF_FMT", "PERF_BPF",
                                   "PERF_BPF_OUT")
