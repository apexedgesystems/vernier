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
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif

#include <algorithm>
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

/** @brief How a process with wait status @p waitStatus ended: "exited with status 3". */
std::string endedWith(int waitStatus) {
  if (WIFEXITED(waitStatus)) {
    return "exited with status " + std::to_string(WEXITSTATUS(waitStatus));
  }
  if (WIFSIGNALED(waitStatus)) {
    return "ended on signal " + std::to_string(WTERMSIG(waitStatus));
  }
  return "ended with wait status " + std::to_string(waitStatus);
}

/** @brief The whole of the file at @p path; empty when it cannot be read. */
std::string fileText(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

/** @brief True when @p text holds more than white space. */
bool saysSomething(const std::string& text) {
  return text.find_first_not_of(" \t\r\n") != std::string::npos;
}

/**
 * @brief Wait while @p helper runs, up to @p until, for the probe output at
 * @p outPath to hold PROBE_ATTACH_LINE; true when it does.
 */
bool waitForAttachLine(OwnedHelper& helper, const std::string& outPath,
                       std::chrono::steady_clock::time_point until) {
  while (true) {
    if (fileText(outPath).find(PROBE_ATTACH_LINE) != std::string::npos) {
      return true;
    }
    if (!helper.running() || std::chrono::steady_clock::now() >= until) {
      // One last look: the line may have come just before the end.
      return fileText(outPath).find(PROBE_ATTACH_LINE) != std::string::npos;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

/** @brief True when @p waitStatus is an exit with status 0. */
bool exitedCleanly(int waitStatus) { return WIFEXITED(waitStatus) && WEXITSTATUS(waitStatus) == 0; }

/** @brief Milliseconds from @p from to @p to. */
long long msBetween(std::chrono::steady_clock::time_point from,
                    std::chrono::steady_clock::time_point to) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(to - from).count();
}

/**
 * @brief The refusal of a probe that ended by itself with status 0, @p ms
 * after its start, before the check stopped it: its script exits too soon
 * for the check, and would end before the measured repeats do.
 */
ReadinessResult endedByItself(const BpftraceRoute& route, const AttachProbe& probe, long long ms) {
  return readinessResult(
      ReadinessCause::UNUSABLE,
      probe.what + ": the probe tracer ended by itself with status 0 after " + std::to_string(ms) +
          " ms, before the check stopped it: the check stops a probe once it has attached, " +
          std::to_string(probe.graceMs) +
          " ms after its start at the earliest, and a script that ends itself sooner cannot be "
          "checked",
      probe.selfEndRemedy.empty()
          ? "Run the script by hand with " + route.bpftrace + " to see why it ends."
          : probe.selfEndRemedy);
}

} // namespace

std::string attachLineProgram() {
  return std::string{"interval:ms:100 { printf(\""} + PROBE_ATTACH_LINE + "\\n\"); }\n";
}

std::string optInRemedy(const BpftraceRoute& route, const ReadinessContext& ctx) {
  std::string killPath = route.kill;
  if (killPath.empty()) {
    const auto KILL = resolveExecutable("kill", ctx);
    killPath = (KILL && KILL->executable) ? KILL->path : std::string{"kill"};
  }
  return "Set BENCH_SUDO=1 with a scoped sudoers grant for " + route.bpftrace + " and " + killPath +
         ", or run as root.";
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
                                 ", or unset BENCH_SUDO and run as root.");
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
                             "BENCH_SUDO and run as root.");
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
                           "The kernel lacks a probe the script uses (bpftrace's message names "
                           "it), or tracefs is not mounted: mount -t tracefs tracefs "
                           "/sys/kernel/tracing.");
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
  if (scratchDir.empty()) {
    return readinessResult(ReadinessCause::UNVERIFIED,
                           what + ": the check has no temporary directory for the probe "
                                  "tracer's output, so it cannot see the tracer attach; only "
                                  "the run can show whether it attaches",
                           "Point TMPDIR at a writable directory, or make /tmp writable.");
  }
  const std::string OUT_PATH = scratchDir + "/attach.out";
  const std::string ERR_PATH = scratchDir + "/attach.err";

  OwnedHelper helper(
      route.stopPolicy(2000, 1000, 1000, std::make_shared<const ReadinessContext>(ctx)));
  const auto STARTED_AT = std::chrono::steady_clock::now();
  const HelperStart START = helper.start(argv, OUT_PATH, ERR_PATH, probe.graceMs, &ctx);
  if (!START.started) {
    return readinessResult(ReadinessCause::UNUSABLE, what + ": " + START.errorTail,
                           "Check that " + argv.front() + " can be executed.");
  }
  if (START.exitedEarly) {
    if (exitedCleanly(START.waitStatus)) {
      return endedByItself(route, probe, msBetween(STARTED_AT, std::chrono::steady_clock::now()));
    }
    return classifyAttachFailure(route, what, probe.commandLine, START.errorTail, ctx,
                                 probe.runCommand);
  }
  // A busy machine can keep bpftrace compiling or attaching past the grace;
  // a SIGINT then ends it before it has shown anything, so the stop waits for
  // the attach line, or the tracer's own end.
  const bool ATTACHED = waitForAttachLine(
      helper, OUT_PATH, STARTED_AT + std::chrono::milliseconds(probe.attachWaitMs));
  const auto STOPPING_AT = std::chrono::steady_clock::now();
  const HelperStopResult STOP = helper.stop();
  std::string lingering;
  if (STOP.stillAlive && probe.selfExitMs > 0) {
    // The stop could not end it; its own self-exit will. Wait for that,
    // bounded, and reap it, so the probe does not outlive the check.
    const auto UNTIL =
        STOPPING_AT + std::chrono::milliseconds(probe.selfExitMs + PROBE_SELF_EXIT_SLACK_MS);
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
  // How the tracer ended decides, not the signal it ended after: one stopped
  // by SIGINT exits with status 0 and nothing on stderr (bpftrace 0.14.0,
  // 0.20.2 and 0.23.2 alike). One that ended before the stop, or with an
  // error after it, reads as one that ended at its start, so the result does
  // not depend on how long a busy machine kept it starting.
  const std::string ERR = fileText(ERR_PATH);
  const bool CLEAN = exitedCleanly(STOP.waitStatus);
  if (!STOP.wasRunning || STOP.stoppedBy == 0) {
    if (CLEAN) {
      return endedByItself(route, probe, msBetween(STARTED_AT, STOPPING_AT));
    }
    return classifyAttachFailure(route, what, probe.commandLine, ERR, ctx, probe.runCommand);
  }
  if (!CLEAN && saysSomething(ERR)) {
    return classifyAttachFailure(route, what, probe.commandLine, ERR, ctx, probe.runCommand);
  }
  if (!ATTACHED) {
    return readinessResult(ReadinessCause::UNVERIFIED,
                           what + ": the probe tracer had not attached " +
                               std::to_string(probe.attachWaitMs) +
                               " ms after its start, and the check stopped it; only the run can "
                               "show whether it attaches",
                           "bpftrace starts slowly on a busy machine: check again when it is "
                           "idle.");
  }
  if (!CLEAN && STOP.stoppedBy == SIGINT) {
    return readinessResult(ReadinessCause::UNUSABLE,
                           what + ": the probe tracer " + endedWith(STOP.waitStatus) +
                               " after SIGINT, with nothing on its stderr; a tracer stopped by "
                               "SIGINT exits with status 0",
                           "Run the script by hand with " + route.bpftrace + " to see why.");
  }
  if (STOP.stoppedBy != SIGINT) {
    return readinessResult(ReadinessCause::CAVEAT,
                           "the probe tracer ignored SIGINT and stopped on " +
                               std::string{signalName(STOP.stoppedBy)} +
                               "; the run's output may be incomplete",
                           "bpftrace prints its maps on SIGINT; check that this build handles it.");
  }
  return std::nullopt;
}

std::optional<ReadinessResult> reportStop(const char* tag, const std::string& what,
                                          const HelperStopResult& stop, const BpftraceRoute& route,
                                          const std::string& errorPath) {
  for (const StopDelivery& delivery : stop.deliveries) {
    if (!delivery.delivered) {
      std::fprintf(stderr, "[%s] %s: could not deliver %s to tracer %d: %s: %s\n", tag,
                   what.c_str(), signalName(delivery.signal), static_cast<int>(delivery.target),
                   delivery.command.c_str(), delivery.detail.c_str());
    }
  }
  std::string detail;
  std::string remedy;
  if (stop.stillAlive) {
    const std::string MANUAL = route.privilege.route == PrivilegeRoute::SCOPED_SUDO
                                   ? route.sudo + " -n " + route.kill + " -9 <pid>"
                                   : std::string{"kill -9 <pid>"};
    detail =
        "the tracer is still running and its output may be incomplete; stop it with: " + MANUAL;
    remedy = "Stop it by hand, then run the script by hand to see why it did not stop.";
  } else if (stop.stoppedBy == SIGKILL) {
    detail = "the tracer ignored SIGINT and SIGTERM and was killed; its output is incomplete";
    remedy = "bpftrace prints its maps on SIGINT; check that this build handles it.";
  } else if (!exitedCleanly(stop.waitStatus)) {
    // How the tracer ended decides, as for the readiness probe, not the
    // signal it ended after: bpftrace exits with status 0 once it has printed
    // its maps. Its error output only says why it did not.
    const std::string ERR = errorPath.empty() ? std::string{} : fileText(errorPath);
    std::string after = " during the stop";
    if (!stop.wasRunning) {
      after = " before the stop";
    } else if (stop.stoppedBy != 0) {
      after = std::string{" after "} + signalName(stop.stoppedBy);
    }
    detail = std::string{"the tracer "} +
             (stop.stoppedBy == SIGTERM ? "ignored SIGINT, then " : "") +
             endedWith(stop.waitStatus) + after +
             (saysSomething(ERR) ? ": " + failureLine(ERR)
                                 : std::string{", with nothing on its stderr"}) +
             "; its output may be incomplete";
    remedy = "bpftrace exits with status 0 once it has printed its maps; " +
             (errorPath.empty() ? std::string{"run the script by hand to see why this one did not."}
                                : "its error output is in " + errorPath + ".");
  } else {
    if (stop.stoppedBy == SIGTERM) {
      std::fprintf(stderr,
                   "[%s] %s: the tracer did not stop on SIGINT, and stopped on SIGTERM "
                   "with status 0\n",
                   tag, what.c_str());
    }
    return std::nullopt;
  }
  std::fprintf(stderr, "[%s] %s: %s\n", tag, what.c_str(), detail.c_str());
  return readinessResult(ReadinessCause::UNUSABLE, what + ": " + detail, remedy);
}

/* ----------------------------- The capture window ----------------------------- */

std::string captureWindowProgram(const std::string& label, long pid, long armTid, long stopTid) {
  const std::string PID = std::to_string(pid);
  const std::string ARM = ARM_THREAD;
  const std::string STOP = STOP_THREAD;
  return "// Added by Vernier: the capture window (BPF_SCRIPTS.md). The measured\n"
         "// repeats run between its arm line and its stop line.\n"
         "tracepoint:sched:sched_switch {\n"
         "  if (pid == " +
         PID +
         " && (args->prev_state == 1 || args->prev_state == 2)) {\n"
         "    if (tid == " +
         std::to_string(armTid) + " && comm == \"" + ARM +
         "\") {\n"
         "      if (@vernier_window == 0) {\n"
         "        @vernier_window = 1;\n"
         "        printf(\"" +
         label +
         " armed %d %d\\n\", pid, tid);\n"
         "      }\n"
         "    } else if (tid == " +
         std::to_string(stopTid) + " && comm == \"" + STOP +
         "\") {\n"
         "      if (@vernier_window == 1) {\n"
         "        @vernier_window = 2;\n"
         "        printf(\"" +
         label +
         " disarmed %d %d\\n\", pid, tid);\n"
         "        clear(@vernier_window);\n"
         "      }\n"
         "    }\n"
         "  }\n"
         "}\n";
}

namespace {

/// The decimal number at @p at in @p text, and the index after it; nullopt when no digit is there,
/// or more than an id holds.
std::optional<long> numberAt(std::string_view text, std::size_t& at) {
  constexpr std::size_t MAX_DIGITS = 18;
  const std::size_t FROM = at;
  long value = 0;
  while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
    if (at - FROM == MAX_DIGITS) {
      return std::nullopt;
    }
    value = value * 10 + (text[at] - '0');
    ++at;
  }
  if (at == FROM) {
    return std::nullopt;
  }
  return value;
}

/// "<label> <word> <pid> <tid>", the ids separated by one space, anywhere in @p line.
std::optional<WindowAck> parseAck(std::string_view line, std::string_view label,
                                  std::string_view word) {
  const std::string PREFIX = std::string(label) + " " + std::string(word) + " ";
  for (std::size_t found = line.find(PREFIX); found != std::string_view::npos;
       found = line.find(PREFIX, found + 1)) {
    // The label must start the line or follow a character that cannot be
    // part of a name, so "xbpftrace armed" is not "bpftrace armed".
    if (found > 0) {
      const char BEFORE = line[found - 1];
      if (std::isalnum(static_cast<unsigned char>(BEFORE)) != 0 || BEFORE == '_' || BEFORE == '-') {
        continue;
      }
    }
    std::size_t at = found + PREFIX.size();
    long numbers[2] = {-1, -1};
    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
      if (i > 0) {
        ok = at < line.size() && line[at] == ' ';
        ++at;
      }
      if (ok) {
        const std::optional<long> VALUE = numberAt(line, at);
        ok = VALUE.has_value();
        numbers[i] = VALUE.value_or(-1);
      }
    }
    if (ok) {
      return WindowAck{numbers[0], numbers[1]};
    }
  }
  return std::nullopt;
}

} // namespace

std::optional<WindowAck> parseArmAck(std::string_view line, std::string_view label) {
  return parseAck(line, label, "armed");
}

std::optional<WindowAck> parseDisarmAck(std::string_view line, std::string_view label) {
  return parseAck(line, label, "disarmed");
}

bool isWindowLine(std::string_view line, std::string_view label) {
  if (parseArmAck(line, label) || parseDisarmAck(line, label)) {
    return true;
  }
  // A map the window reserves, as text ("@vernier_window: 2") or JSON
  // ("\"@vernier_window\"") prints it.
  const std::size_t FIRST = line.find_first_not_of(" \t");
  if (FIRST == std::string_view::npos) {
    return false;
  }
  const std::string_view REST = line.substr(FIRST);
  return REST.rfind(WINDOW_MAP_PREFIX, 0) == 0 ||
         line.find(std::string("\"") + WINDOW_MAP_PREFIX) != std::string_view::npos;
}

WindowWatch::WindowWatch(std::string outputPath, std::string label)
    : path_(std::move(outputPath)), label_(std::move(label)) {}

int WindowWatch::poll() {
  const int FD = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (FD < 0) {
    return errno;
  }
  std::string fresh;
  char buf[4096];
  while (true) {
    const ssize_t N = ::pread(FD, buf, sizeof(buf), static_cast<off_t>(offset_));
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
    fresh.append(buf, static_cast<std::size_t>(N));
    offset_ += static_cast<std::uintmax_t>(N);
  }
  ::close(FD);
  partial_ += fresh;
  std::size_t start = 0;
  for (std::size_t end = partial_.find('\n'); end != std::string::npos;
       end = partial_.find('\n', start)) {
    const std::string_view LINE(partial_.data() + start, end - start);
    if (!armed_) {
      armed_ = parseArmAck(LINE, label_);
    }
    if (!disarmed_) {
      disarmed_ = parseDisarmAck(LINE, label_);
      disarmedBeforeArmed_ = disarmed_.has_value() && !armed_.has_value();
    }
    start = end + 1;
  }
  partial_.erase(0, start);
  return 0;
}

namespace {

/// A script's code apart from its comments, and its string literals' contents.
struct ScriptCode {
  std::string code;                 ///< Comments as one space, each string literal as "".
  std::vector<std::string> strings; ///< The string literals' contents, escapes as written.
};

/// Split @p script as bpftrace reads it: // and /* */ comments, "strings".
ScriptCode codeOf(std::string_view script) {
  ScriptCode out;
  std::size_t i = 0;
  while (i < script.size()) {
    const char CH = script[i];
    const char NEXT = i + 1 < script.size() ? script[i + 1] : '\0';
    if (CH == '/' && NEXT == '/') {
      while (i < script.size() && script[i] != '\n') {
        ++i;
      }
      out.code += ' ';
    } else if (CH == '/' && NEXT == '*') {
      const std::size_t CLOSE = script.find("*/", i + 2);
      i = CLOSE == std::string_view::npos ? script.size() : CLOSE + 2;
      out.code += ' ';
    } else if (CH == '"') {
      std::string text;
      ++i;
      while (i < script.size() && script[i] != '"') {
        if (script[i] == '\\' && i + 1 < script.size()) {
          text += script[i];
          ++i;
        }
        text += script[i];
        ++i;
      }
      ++i;
      out.strings.push_back(std::move(text));
      out.code += "\"\"";
    } else {
      out.code += CH;
      ++i;
    }
  }
  return out;
}

bool identifierChar(char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

} // namespace

std::string reservedNameIn(std::string_view script, std::string_view label) {
  const ScriptCode CODE = codeOf(script);
  const std::size_t MAP = CODE.code.find(WINDOW_MAP_PREFIX);
  if (MAP != std::string::npos) {
    std::size_t end = MAP + 1;
    while (end < CODE.code.size() && identifierChar(CODE.code[end])) {
      ++end;
    }
    return "the map " + CODE.code.substr(MAP, end - MAP);
  }
  for (const std::string& text : CODE.strings) {
    for (const char* WORD : {" disarmed", " armed"}) {
      const std::string MARKER = std::string(label) + WORD;
      if (text.find(MARKER) != std::string::npos) {
        return "the text '" + MARKER + "'";
      }
    }
  }
  return "";
}

std::string iteratorProbeIn(std::string_view script) {
  const std::string CODE = codeOf(script).code;
  int depth = 0;
  for (std::size_t i = 0; i < CODE.size(); ++i) {
    const char CH = CODE[i];
    if (CH == '{') {
      ++depth;
    } else if (CH == '}') {
      depth = std::max(depth - 1, 0);
    } else if (depth == 0 && identifierChar(CH) && (i == 0 || !identifierChar(CODE[i - 1]))) {
      std::size_t end = i;
      while (end < CODE.size() && identifierChar(CODE[end])) {
        ++end;
      }
      const std::string WORD = CODE.substr(i, end - i);
      if ((WORD == "iter" || WORD == "it") && end < CODE.size() && CODE[end] == ':') {
        std::size_t stop = end;
        while (stop < CODE.size() && std::isspace(static_cast<unsigned char>(CODE[stop])) == 0 &&
               CODE[stop] != ',' && CODE[stop] != '{' && CODE[stop] != '/') {
          ++stop;
        }
        return CODE.substr(i, stop - i);
      }
      i = end - 1;
    }
  }
  return "";
}

ThreadNameScope::ThreadNameScope(const char* name) {
#ifdef __linux__
  renamed_ = ::pthread_getname_np(::pthread_self(), saved_, sizeof(saved_)) == 0 &&
             ::pthread_setname_np(::pthread_self(), name) == 0;
#else
  (void)name;
#endif
}

ThreadNameScope::~ThreadNameScope() {
#ifdef __linux__
  if (renamed_) {
    (void)::pthread_setname_np(::pthread_self(), saved_);
  }
#endif
}

ArmThread::ArmThread()
    : thread_([this] {
#ifdef __linux__
        (void)::pthread_setname_np(::pthread_self(), ARM_THREAD);
#endif
        tid_ = currentThreadId();
        named_ = true;
        while (!stop_) {
          std::this_thread::sleep_for(std::chrono::milliseconds(WINDOW_NAP_MS));
        }
      }) {
  while (!named_.load()) {
    std::this_thread::yield();
  }
}

ArmThread::~ArmThread() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool waitUntil(const std::function<bool()>& done, int boundMs) {
  const auto UNTIL =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(boundMs, 0));
  while (true) {
    if (done()) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= UNTIL) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(WINDOW_NAP_MS));
  }
}

long currentThreadId() noexcept {
#ifdef __linux__
  return static_cast<long>(::syscall(SYS_gettid));
#else
  return -1;
#endif
}

PidNamespaceId readPidNamespace() {
  constexpr const char* LINK = "/proc/self/ns/pid";
  PidNamespaceId ns;
  struct stat info{};
  if (::stat(LINK, &info) != 0) {
    return ns;
  }
  ns.read = true;
  ns.inode = static_cast<unsigned long>(info.st_ino);
  char target[64] = {};
  const ssize_t LENGTH = ::readlink(LINK, target, sizeof(target) - 1);
  if (LENGTH > 0) {
    ns.link.assign(target, static_cast<std::size_t>(LENGTH));
  }
  return ns;
}

std::string foreignPidNamespace(const PidNamespaceId& ns) {
  constexpr unsigned long HOST_PID_NAMESPACE_INODE = 0xEFFFFFFCUL;
  if (!ns.read || ns.inode == HOST_PID_NAMESPACE_INODE) {
    return "";
  }
  return ns.link.empty() ? "pid:[" + std::to_string(ns.inode) + "]" : ns.link;
}

std::string foreignPidNamespace() { return foreignPidNamespace(readPidNamespace()); }

} // namespace bpftrace_tool

#ifdef __linux__
// ============================================================================
// Runner
// ============================================================================

namespace { // Internal implementation details

constexpr int PROBE_RUN_MS = 1000;   // how long a readiness probe copy runs before its stop
constexpr int PROBE_SELF_EXIT_S = 5; // a readiness probe copy exits by itself after this
constexpr const char* WINDOW_LABEL = "bpftrace"; // the capture window's acknowledgement label
constexpr int INTERRUPT_WAIT_MS = 2000;
constexpr int TERMINATE_WAIT_MS = 1000;
constexpr int KILL_WAIT_MS = 1000;

#ifndef VERNIER_BPF_SCRIPTS_DIR
#error "VERNIER_BPF_SCRIPTS_DIR must name the bundled scripts (src/bench/CMakeLists.txt sets it)"
#endif
// The bundled scripts, in the source tree this library was built from: an
// absolute path, so the lookup does not depend on the working directory.
const char* const DEFAULT_SCRIPTS_DIR = VERNIER_BPF_SCRIPTS_DIR;

/** @brief PERF_BPF_SCRIPTS when set and not empty, else the bundled scripts' directory. */
std::string scriptsDirectory(const ReadinessContext& ctx) {
  const std::string DIR = ctx.get("PERF_BPF_SCRIPTS").value_or("");
  return DIR.empty() ? std::string{DEFAULT_SCRIPTS_DIR} : DIR;
}

/** @brief True when @p name ends in the ".bt" suffix. */
bool hasBtSuffix(const std::string& name) {
  return name.size() >= 3 && name.compare(name.size() - 3, 3, ".bt") == 0;
}

/**
 * @brief Where script @p name is: a name with a '/' is a path to the file,
 * absolute or from the working directory; any other name is a script in
 * @p scriptsDir. Either may leave out the ".bt" suffix.
 */
std::string scriptPathFor(const std::string& scriptsDir, const std::string& name) {
  const std::string FILE = hasBtSuffix(name) ? name : name + ".bt";
  if (name.find('/') != std::string::npos) {
    return FILE;
  }
  return (std::filesystem::path(scriptsDir) / FILE).string();
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

/**
 * @brief `-q -B none [-f json] <script>`: the arguments every launch passes.
 * -B none writes each printed line out as it is printed, so the capture
 * window's acknowledgements reach the output at once.
 */
std::vector<std::string> launchArgs(const std::string& format, const std::string& script) {
  std::vector<std::string> args{"-q", "-B", "none"};
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

/**
 * @brief The name a script's run files take in the capture folder: its file's
 * stem, so a script given by path writes there too, not beside the script.
 */
std::string outputStem(const std::string& scriptPath) {
  return std::filesystem::path(scriptPath).stem().string();
}

/** @brief Where the run writes its copy of a script with stem @p stem in capture folder @p outdir.
 */
std::string runCopyPath(const std::string& outdir, const std::string& stem) {
  return (std::filesystem::path(outdir) / (stem + ".tmp.bt")).string();
}

/**
 * @brief What a tracer's stop says about its capture: READY when the tracer
 * ended with status 0 at the stop, having printed its maps; else the Error
 * that leaves the capture incomplete, @p incomplete as reportStop() judged and
 * printed it. A tracer that ended with status 0 before the stop ran its own
 * exit() while the measured repeats went on; that case is printed here.
 */
ReadinessResult stopOutcome(const std::string& what, const HelperStopResult& stop,
                            const std::optional<ReadinessResult>& incomplete) {
  if (!stop.stillAlive && !stop.wasRunning && WIFEXITED(stop.waitStatus) &&
      WEXITSTATUS(stop.waitStatus) == 0) {
    const std::string DETAIL = what +
                               " ended by itself before the measured repeats finished; its output "
                               "covers only the part before it ended";
    std::fprintf(stderr, "[bpftrace] %s\n", DETAIL.c_str());
    return readinessResult(ReadinessCause::UNUSABLE, DETAIL,
                           "End the script only on the traced process's exit "
                           "(sched_process_exit filtered on tid == {{PID}}); the backend stops it "
                           "when the measured repeats finish.");
  }
  if (incomplete) {
    return *incomplete;
  }
  return readinessResult(ReadinessCause::READY, what + " stopped and flushed its output", "");
}

/**
 * @brief One tracer for one script: launched through the plan's route with the
 * capture window appended to its copy, bound to the benchmark's pid, the arm
 * thread and the measuring thread, armed and disarmed through the window, and
 * stopped through the route. Every failure is printed as a "[bpftrace]" line,
 * stops the tracer at once and is kept as its outcome.
 */
class BpfRunner {
public:
  BpfRunner(std::shared_ptr<const BpftracePlan> plan, std::string name, std::string scriptPath,
            std::string outdir)
      : plan_(std::move(plan)), name_(std::move(name)), scriptPath_(std::move(scriptPath)),
        stem_(outputStem(scriptPath_)), outdir_(std::move(outdir)), what_("script '" + name_ + "'"),
        stdoutPath_((std::filesystem::path(outdir_) / (stem_ + ".out." + plan_->format)).string()),
        stderrPath_((std::filesystem::path(outdir_) / (stem_ + ".err.txt")).string()) {}

  ~BpfRunner() { stopTracer(); }

  BpfRunner(const BpfRunner&) = delete;
  BpfRunner& operator=(const BpfRunner&) = delete;

  /**
   * @brief Write the run's copy (the script with {{PID}} filled in and the
   * capture window, bound to @p pid, @p armTid and @p stopTid, appended) and
   * start its tracer, without waiting for it. The script file itself is only
   * read.
   * @return False, with the outcome kept and printed, when it did not start.
   */
  bool launch(pid_t pid, long armTid, long stopTid) {
    std::string src;
    if (const int ERR = readScript(scriptPath_, src); ERR != 0) {
      std::fprintf(stderr, "[bpftrace] cannot read script '%s' at %s: %s\n", name_.c_str(),
                   scriptPath_.c_str(), errnoText(ERR).c_str());
      outcome_ =
          readinessResult(ReadinessCause::UNUSABLE,
                          what_ + " cannot be read at " + scriptPath_ + ": " + errnoText(ERR),
                          "Keep the script where the check found it until the run ends.");
      return false;
    }
    replacePid(src, static_cast<long>(pid));
    src += "\n" + bpftrace_tool::captureWindowProgram(WINDOW_LABEL, static_cast<long>(pid), armTid,
                                                      stopTid);
    std::error_code ec;
    std::filesystem::create_directories(outdir_, ec);
    const std::string TEMP_SCRIPT = runCopyPath(outdir_, stem_);
    {
      std::ofstream out(TEMP_SCRIPT);
      out << src;
      out.close();
      if (!out) {
        std::fprintf(stderr, "[bpftrace] cannot write the run's copy of script '%s' to %s\n",
                     name_.c_str(), TEMP_SCRIPT.c_str());
        outcome_ =
            readinessResult(ReadinessCause::UNUSABLE,
                            "the run's copy of " + what_ + " cannot be written to " + TEMP_SCRIPT,
                            "Make the capture folder writable, or select another root "
                            "with --profile-output-dir.");
        return false;
      }
    }
    const std::vector<std::string> ARGS = launchArgs(plan_->format, TEMP_SCRIPT);
    commandLine_ = commandLineOf(plan_->route.bpftrace, ARGS);
    std::vector<std::string> argv = plan_->route.command();
    argv.insert(argv.end(), ARGS.begin(), ARGS.end());

    // fork+exec of the resolved paths, never a shell: signals land on the
    // process this runner tracks, and the executed tools are the ones the
    // check verified. No start grace: the capture window's arm is awaited.
    helper_ = std::make_unique<OwnedHelper>(plan_->route.stopPolicy(
        INTERRUPT_WAIT_MS, TERMINATE_WAIT_MS, KILL_WAIT_MS, plan_->context));
    const HelperStart STARTED = helper_->start(argv, stdoutPath_, stderrPath_, 0, nullptr);
    if (!STARTED.started) {
      std::fprintf(stderr, "[bpftrace] script '%s' could not start: %s\n", name_.c_str(),
                   STARTED.errorTail.c_str());
      outcome_ = readinessResult(ReadinessCause::UNUSABLE,
                                 what_ + " could not start: " + STARTED.errorTail,
                                 "Check that " + argv.front() + " can be executed.");
      helper_.reset();
      return false;
    }
    watch_ = std::make_unique<bpftrace_tool::WindowWatch>(stdoutPath_, WINDOW_LABEL);
    if (STARTED.exitedEarly) {
      endedBeforeArm();
      return false;
    }
    phase_ = Phase::LAUNCHED;
    return true;
  }

  /**
   * @brief One look while the arm is awaited: true once the tracer has
   * acknowledged its arm, from @p armTid in process @p pid, or failed.
   */
  bool armSettled(pid_t pid, long armTid) {
    if (phase_ != Phase::LAUNCHED) {
      return true;
    }
    if (const int ERR = watch_->poll(); ERR != 0) {
      fail(unreadableOutput(ERR));
      return true;
    }
    if (watch_->disarmedBeforeArmed()) {
      fail(outOfOrder("before its arm", *watch_->disarmed()));
      return true;
    }
    if (const std::optional<bpftrace_tool::WindowAck>& ACK = watch_->armed()) {
      if (ACK->pid != static_cast<long>(pid) || ACK->tid != armTid) {
        fail(wrongTarget("arm probe", *ACK, pid, armTid, "the arm thread"));
      } else {
        phase_ = Phase::ARMED;
      }
      return true;
    }
    if (!helper_->running()) {
      endedBeforeArm();
      return true;
    }
    return false;
  }

  /** @brief After the arm's wait: a tracer that neither acknowledged nor failed has failed. */
  void endArmWait() {
    if (phase_ == Phase::LAUNCHED) {
      fail(noArmAcknowledgement());
    }
  }

  /** @brief True once the tracer acknowledged its arm, while nothing has failed. */
  [[nodiscard]] bool armed() const noexcept { return phase_ == Phase::ARMED; }

  /**
   * @brief Before the measuring thread takes the stop's name: read what the
   * tracer wrote during the measured repeats. A stop line there came before
   * the stop was asked for.
   */
  void beginStop() {
    if (phase_ != Phase::ARMED) {
      return;
    }
    if (const int ERR = watch_->poll(); ERR != 0) {
      fail(unreadableOutput(ERR));
      return;
    }
    if (watch_->disarmed()) {
      fail(outOfOrder("before the measured repeats finished", *watch_->disarmed()));
    }
  }

  /** @brief The measured repeats ended on a thread the window's stop is not bound to. */
  void endedOnAnotherThread(long endTid, long stopTid) {
    if (phase_ != Phase::ARMED) {
      return;
    }
    fail(readinessResult(ReadinessCause::UNUSABLE,
                         what_ + ": the measured repeats ended on thread " +
                             std::to_string(endTid) + ", not on thread " + std::to_string(stopTid) +
                             ", where they started and which the capture window's stop names, "
                             "so the stop cannot be acknowledged",
                         "Call the profiler's beforeMeasure() and afterMeasure() from one "
                         "thread, as the harness does."));
  }

  /**
   * @brief One look while the stop is awaited: true once an armed tracer has
   * acknowledged the stop, ended, or failed, and for any other.
   */
  bool disarmSettled() {
    if (phase_ != Phase::ARMED) {
      return true;
    }
    if (const int ERR = watch_->poll(); ERR != 0) {
      fail(unreadableOutput(ERR));
      return true;
    }
    return watch_->disarmed().has_value() || !helper_->running();
  }

  /**
   * @brief After the stop's wait: judge the capture, the stop acknowledged by
   * @p stopTid in process @p pid, and stop the tracer.
   */
  void finish(pid_t pid, long stopTid) {
    if (phase_ != Phase::ARMED) {
      stopTracer();
      return;
    }
    const std::optional<bpftrace_tool::WindowAck> ACK = watch_->disarmed();
    if (ACK && (ACK->pid != static_cast<long>(pid) || ACK->tid != stopTid)) {
      fail(wrongTarget("stop", *ACK, pid, stopTid, "the stopping thread"));
      return;
    }
    if (!ACK && helper_->running()) {
      fail(readinessResult(ReadinessCause::UNUSABLE,
                           what_ + " did not acknowledge the stop within " +
                               std::to_string(plan_->disarmWaitMs) +
                               " ms, so the capture's validity could not be established",
                           "Run the script by hand with " + plan_->route.bpftrace +
                               "; its errors are in " + stderrPath_ + "."));
      return;
    }
    // The stop acknowledged, or the tracer ended before it: stop it, judge
    // how it ended, then the output it left.
    const HelperStopResult STOPPED = helper_->stop();
    const std::optional<ReadinessResult> INCOMPLETE =
        bpftrace_tool::reportStop("bpftrace", what_, STOPPED, plan_->route, stderrPath_);
    helper_.reset();
    phase_ = Phase::DONE;
    ReadinessResult result = stopOutcome(what_, STOPPED, INCOMPLETE);
    if (result.report.status == EnvReport::Status::Ok) {
      result = judgeOutput(pid);
    }
    outcome_ = std::move(result);
  }

  /** @brief How this tracer's capture ended: empty while it runs, set by a failure or the stop. */
  [[nodiscard]] const std::optional<ReadinessResult>& outcome() const noexcept { return outcome_; }

private:
  enum class Phase { IDLE, LAUNCHED, ARMED, FAILED, DONE };

  /** @brief Stop a tracer still running; the outcome of a stop nothing judged yet. */
  void stopTracer() noexcept {
    if (!helper_) {
      return;
    }
    const HelperStopResult STOPPED = helper_->stop();
    const std::optional<ReadinessResult> INCOMPLETE =
        bpftrace_tool::reportStop("bpftrace", what_, STOPPED, plan_->route, stderrPath_);
    if (!outcome_) {
      outcome_ = stopOutcome(what_, STOPPED, INCOMPLETE);
    }
    helper_.reset();
    phase_ = Phase::DONE;
  }

  /** @brief Print @p why, keep it, and stop the tracer. */
  void fail(ReadinessResult why) {
    std::fprintf(stderr, "[bpftrace] %s\n", why.report.message.c_str());
    if (!why.report.hint.empty()) {
      std::fprintf(stderr, "[bpftrace] %s\n", why.report.hint.c_str());
    }
    outcome_ = std::move(why);
    phase_ = Phase::FAILED;
    if (helper_) {
      const HelperStopResult STOPPED = helper_->stop();
      (void)bpftrace_tool::reportStop("bpftrace", what_, STOPPED, plan_->route, stderrPath_);
      helper_.reset();
    }
  }

  /** @brief The tracer ended before it acknowledged its arm: reap it and say why. */
  void endedBeforeArm() {
    const HelperStopResult ENDED = helper_->stop();
    helper_.reset();
    std::string errorText;
    (void)readScript(stderrPath_, errorText);
    ReadinessResult why =
        WIFEXITED(ENDED.waitStatus) && WEXITSTATUS(ENDED.waitStatus) == 0
            ? readinessResult(ReadinessCause::UNUSABLE,
                              what_ + " ended by itself before it acknowledged its arm probe",
                              "End the script only on the traced process's exit "
                              "(sched_process_exit filtered on tid == {{PID}}).")
            : bpftrace_tool::classifyAttachFailure(plan_->route, what_, commandLine_, errorText,
                                                   *plan_->context);
    std::fprintf(stderr, "[bpftrace] the tracer exited before it acknowledged its arm probe: %s\n",
                 why.report.message.c_str());
    if (!why.report.hint.empty()) {
      std::fprintf(stderr, "[bpftrace] %s\n", why.report.hint.c_str());
    }
    outcome_ = std::move(why);
    phase_ = Phase::FAILED;
  }

  /** @brief No arm acknowledgement within the plan's bound; the PID namespace named when foreign.
   */
  ReadinessResult noArmAcknowledgement() const {
    const std::string DETAIL = what_ + " did not acknowledge its arm probe within " +
                               std::to_string(plan_->armWaitMs) + " ms";
    const std::string NAMESPACE = bpftrace_tool::foreignPidNamespace(
        plan_->pidNamespace ? plan_->pidNamespace() : bpftrace_tool::readPidNamespace());
    if (!NAMESPACE.empty()) {
      return readinessResult(ReadinessCause::UNSUPPORTED,
                             DETAIL + ": this process runs in PID namespace " + NAMESPACE +
                                 ", not the host's, where bpftrace does not see its threads "
                                 "under the ids it knows",
                             "Run the benchmark on the host, or in a container started with "
                             "--pid=host.");
    }
    return readinessResult(
        ReadinessCause::UNUSABLE, DETAIL + ", so nothing shows that it sees this process",
        "Run the script by hand with " + plan_->route.bpftrace +
            " to see whether it attaches; its errors are in " + stderrPath_ + ".");
  }

  /** @brief A stop line that came @p when ("before its arm", ...). */
  ReadinessResult outOfOrder(const char* when, const bpftrace_tool::WindowAck& ack) const {
    return readinessResult(ReadinessCause::UNUSABLE,
                           what_ + " acknowledged a stop " + when + ", for pid " +
                               std::to_string(ack.pid) + " thread " + std::to_string(ack.tid) +
                               ", so its capture window did not cover the measured repeats",
                           "A thread of the benchmark may carry a name the capture window "
                           "reserves (vernier-arm, vernier-wait, vernier-stop); rename it.");
  }

  /** @brief An acknowledgement that names another process or thread. */
  ReadinessResult wrongTarget(const char* of, const bpftrace_tool::WindowAck& ack, pid_t pid,
                              long tid, const char* whose) const {
    return readinessResult(ReadinessCause::UNUSABLE,
                           what_ + " acknowledged its " + of + " for pid " +
                               std::to_string(ack.pid) + " thread " + std::to_string(ack.tid) +
                               ", not for this process, pid " + std::to_string(pid) + ", and " +
                               whose + ", " + std::to_string(tid),
                           "bpftrace numbers this process's threads otherwise than the process "
                           "does; run the benchmark in the host's PID namespace.");
  }

  ReadinessResult unreadableOutput(int err) const {
    return readinessResult(ReadinessCause::UNUSABLE,
                           "the output of " + what_ + " cannot be read at " + stdoutPath_ + ": " +
                               errnoText(err),
                           "Keep the capture folder and its files until the run ends.");
  }

  /**
   * @brief After a flushed stop: the output must be there and still hold the
   * two acknowledgements; READY when the script printed something else too,
   * CAVEAT, printed, when it printed nothing but the window's lines.
   */
  ReadinessResult judgeOutput(pid_t pid) {
    const int FD = ::open(stdoutPath_.c_str(), O_RDONLY | O_CLOEXEC);
    if (FD < 0) {
      ReadinessResult why = unreadableOutput(errno);
      std::fprintf(stderr, "[bpftrace] %s\n", why.report.message.c_str());
      return why;
    }
    ::close(FD);
    std::ifstream in(stdoutPath_);
    std::string line;
    bool any = false;
    bool armLine = false;
    bool stopLine = false;
    bool data = false;
    while (std::getline(in, line)) {
      any = true;
      armLine = armLine || bpftrace_tool::parseArmAck(line, WINDOW_LABEL).has_value();
      stopLine = stopLine || bpftrace_tool::parseDisarmAck(line, WINDOW_LABEL).has_value();
      data = data || (line.find_first_not_of(" \t\r") != std::string::npos &&
                      !bpftrace_tool::isWindowLine(line, WINDOW_LABEL));
    }
    if (!any) {
      ReadinessResult why =
          readinessResult(ReadinessCause::UNUSABLE,
                          what_ + " stopped, but its output at " + stdoutPath_ + " is empty",
                          "Keep the capture folder and its files until the run ends.");
      std::fprintf(stderr, "[bpftrace] %s\n", why.report.message.c_str());
      return why;
    }
    if (!armLine || !stopLine) {
      ReadinessResult why =
          readinessResult(ReadinessCause::UNUSABLE,
                          what_ + " stopped, but its output at " + stdoutPath_ +
                              " lacks the capture window's two lines, so it is not the output that "
                              "acknowledged them",
                          "Keep the capture folder and its files until the run ends.");
      std::fprintf(stderr, "[bpftrace] %s\n", why.report.message.c_str());
      return why;
    }
    const std::string CAPTURED = "its tracer acknowledged their start and their end for pid " +
                                 std::to_string(pid) + " and flushed its output";
    if (!data) {
      ReadinessResult why = readinessResult(
          ReadinessCause::CAVEAT,
          what_ + " captured the measured repeats (" + CAPTURED +
              "), but printed no data of its own: its output holds only the capture window's "
              "lines",
          "Its probes recorded nothing it prints for this process; check the script's filters, "
          "and that the test makes the calls the script traces.");
      std::fprintf(stderr, "[bpftrace] %s\n", why.report.message.c_str());
      return why;
    }
    return readinessResult(ReadinessCause::READY,
                           what_ + " captured the measured repeats: " + CAPTURED, "");
  }

  std::shared_ptr<const BpftracePlan> plan_;
  std::string name_;
  std::string scriptPath_;
  std::string stem_;
  std::string outdir_;
  std::string what_;
  std::string stdoutPath_;
  std::string stderrPath_;
  std::string commandLine_;
  Phase phase_ = Phase::IDLE;
  std::unique_ptr<OwnedHelper> helper_;
  std::unique_ptr<bpftrace_tool::WindowWatch> watch_;
  std::optional<ReadinessResult> outcome_;
};

} // anonymous namespace

// ============================================================================
// Readiness
// ============================================================================

ReadinessResult checkBpftraceRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  auto plan = std::make_shared<BpftracePlan>();
  plan->context = std::make_shared<const ReadinessContext>(ctx);
  // PERF_BPF turns tracing on without --profile bpftrace, for a profiler made
  // directly; read with the grammar of every boolean setting.
  const std::optional<std::string> ENABLE_RAW = ctx.get("PERF_BPF");
  const EnvBool ENABLE = parseEnvBool(ENABLE_RAW);
  if (ENABLE == EnvBool::INVALID) {
    return readinessResult(ReadinessCause::CONFIGURATION,
                           "PERF_BPF='" + *ENABLE_RAW + "' is not a boolean",
                           "Use 1, true, yes or on to trace with bpftrace without --profile "
                           "bpftrace; 0, false, no, off or an empty value to trace only with it.");
  }
  plan->envEnabled = ENABLE == EnvBool::TRUE_VALUE;
  const std::string SCRIPTS_DIR = scriptsDirectory(ctx);
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
      return readinessResult(ReadinessCause::MISSING,
                             "bpftrace script '" + name + "' not found at " + PATH,
                             "--bpf takes a script name, looked up as <name>.bt in " + SCRIPTS_DIR +
                                 " (set by --bpf-scripts DIR or PERF_BPF_SCRIPTS), or a path "
                                 "to a script file, with or without .bt.");
    }
    std::string text;
    if (const int ERR = readScript(PATH, text); ERR != 0) {
      return readinessResult(
          ReadinessCause::UNUSABLE,
          "bpftrace script '" + name + "' at " + PATH + " cannot be read: " + errnoText(ERR),
          "Give this user read access to " + PATH + ", or select a script it can read with --bpf.");
    }
    // The run appends the capture window to its copy: a script must leave the
    // window's names to it, and be able to run beside it.
    if (const std::string RESERVED = bpftrace_tool::reservedNameIn(text, WINDOW_LABEL);
        !RESERVED.empty()) {
      return readinessResult(ReadinessCause::CONFIGURATION,
                             "bpftrace script '" + name + "' at " + PATH + " uses " + RESERVED +
                                 ", which the capture window the run appends reserves",
                             "Rename it: maps named @vernier_... and the lines 'bpftrace armed' "
                             "and 'bpftrace disarmed' belong to the capture window "
                             "(BPF_SCRIPTS.md).");
    }
    if (const std::string ITERATOR = bpftrace_tool::iteratorProbeIn(text); !ITERATOR.empty()) {
      return readinessResult(ReadinessCause::UNSUPPORTED,
                             "bpftrace script '" + name + "' at " + PATH +
                                 " has an iterator probe (" + ITERATOR +
                                 "), which bpftrace runs only as a script's single probe, so the "
                                 "run cannot append the capture window that times its capture",
                             "Run the script by hand with " + plan->route.bpftrace +
                                 "; --profile bpftrace takes scripts whose probes run beside a "
                                 "sched_switch tracepoint.");
    }
    plan->scriptPaths.push_back(PATH);
    sources.push_back(std::move(text));
  }
  // A script's run files take its stem in the capture folder, so two scripts
  // with one stem would overwrite each other's.
  for (std::size_t i = 0; i < plan->scriptPaths.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      const std::string STEM = outputStem(plan->scriptPaths[i]);
      if (STEM == outputStem(plan->scriptPaths[j])) {
        return readinessResult(
            ReadinessCause::CONFIGURATION,
            "bpftrace scripts '" + plan->scripts[j] + "' and '" + plan->scripts[i] +
                "' would both write the capture folder's " + STEM + ".tmp.bt, " + STEM + ".out." +
                plan->format + " and " + STEM + ".err.txt",
            "Select each script once, and give scripts from different directories different "
            "file names.");
      }
    }
  }
  if (auto failure = bpftrace_tool::probeExecutable(plan->route, ctx)) {
    return *failure;
  }

  // Run a copy of each selected script, with the capture window the run
  // appends, through the route until it has attached, a second at least, and
  // stop it with the run's first stop signal. The copy also prints its attach
  // line and exits by itself, and it lives in a private directory: the run's
  // own copy goes in a capture folder that does not exist before the run.
  const bool SUDO = plan->route.privilege.route == PrivilegeRoute::SCOPED_SUDO;
  const ProbeScratch SCRATCH(ctx);
  std::optional<ReadinessResult> caveat;
  std::string runCommands;
  for (std::size_t i = 0; i < plan->scripts.size(); ++i) {
    const std::string RUN_COMMAND =
        commandLineOf(plan->route.bpftrace,
                      launchArgs(plan->format, runCopyPath("<capture folder>",
                                                           outputStem(plan->scriptPaths[i]))));
    runCommands += (runCommands.empty() ? "" : ", ") + RUN_COMMAND;
    std::string copy = sources[i];
    replacePid(copy, static_cast<long>(ctx.self()));
    // The window the run appends, bound to no thread: the probe copy never arms.
    copy += "\n" +
            bpftrace_tool::captureWindowProgram(WINDOW_LABEL, static_cast<long>(ctx.self()), 0, 0);
    copy += "\n" + bpftrace_tool::attachLineProgram();
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
    probe.selfEndRemedy = "Make the script run longer: end it only on the traced process's exit "
                          "(sched_process_exit filtered on tid == {{PID}}), as the bundled "
                          "scripts do; the backend stops it when the measured repeats end.";
    probe.graceMs = PROBE_RUN_MS;
    probe.selfExitMs = PROBE_SELF_EXIT_S * 1000;
    auto verdict = bpftrace_tool::probeAttach(plan->route, probe, ctx, SCRATCH.path());
    if (verdict && verdict->report.status == EnvReport::Status::Error) {
      // The window adds a dependency of its own: say so when it is the one
      // refused.
      if (verdict->report.message.find("sched_switch") != std::string::npos &&
          sources[i].find("sched_switch") == std::string::npos) {
        verdict->report.message += "; the capture window the run appends to every script needs "
                                   "tracepoint:sched:sched_switch, which bpftrace refused here";
      }
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
      " with a " + std::to_string(PROBE_SELF_EXIT_S) + " s self-exit stayed running for " +
      std::to_string(PROBE_RUN_MS) + " ms " + plan->route.describe() + " and stopped on SIGINT" +
      (SUDO ? " through sudo -n kill" : "") + " (probe with " + plan->route.bpftrace + ")";
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

  /**
   * @brief Start the capture window's arm thread, launch every script's
   * tracer with the window bound to this process, that thread and the calling
   * thread, then wait, as the window's WAIT_THREAD while the arm thread naps,
   * until each tracer has acknowledged its arm or failed, up to the plan's
   * armWaitMs. Every tracer is launched before the wait for their
   * acknowledgements begins.
   */
  void beforeMeasure() {
    if (!enabled_) {
      return;
    }
    const pid_t TARGET = ::getpid();
    stopTid_ = bpftrace_tool::currentThreadId();
    const bpftrace_tool::ArmThread ARM;
    for (std::size_t i = 0; i < plan_->scripts.size(); ++i) {
      // One tracer per started script: each runner owns its own process.
      auto runner = std::make_unique<BpfRunner>(plan_, plan_->scripts[i], plan_->scriptPaths[i],
                                                artifactDir_);
      if (runner->launch(TARGET, ARM.tid(), stopTid_)) {
        runners_.push_back(std::move(runner));
      } else {
        record(runner->outcome());
      }
    }
    if (runners_.empty()) {
      return;
    }
    {
      const bpftrace_tool::ThreadNameScope WAITING(bpftrace_tool::WAIT_THREAD);
      (void)bpftrace_tool::waitUntil(
          [&] {
            bool settled = true;
            for (auto& runner : runners_) {
              settled = runner->armSettled(TARGET, ARM.tid()) && settled;
            }
            return settled;
          },
          plan_->armWaitMs);
    }
    for (auto& runner : runners_) {
      runner->endArmWait();
      if (!runner->armed()) {
        record(runner->outcome());
      }
    }
  }

  /**
   * @brief On the thread that started the capture: read what each armed
   * tracer wrote during the measured repeats, then wait, as the capture
   * window's STOP_THREAD, until every armed tracer has acknowledged the stop
   * or ended, up to the plan's disarmWaitMs; then stop and judge each.
   */
  void afterMeasure() {
    if (runners_.empty()) {
      return;
    }
    const pid_t TARGET = ::getpid();
    const long STOPPING = bpftrace_tool::currentThreadId();
    for (auto& runner : runners_) {
      if (STOPPING != stopTid_) {
        runner->endedOnAnotherThread(STOPPING, stopTid_);
      } else {
        runner->beginStop();
      }
    }
    {
      const bpftrace_tool::ThreadNameScope STOPPING_NAME(bpftrace_tool::STOP_THREAD);
      (void)bpftrace_tool::waitUntil(
          [&] {
            bool settled = true;
            for (auto& runner : runners_) {
              settled = runner->disarmSettled() && settled;
            }
            return settled;
          },
          plan_->disarmWaitMs);
    }
    for (auto& runner : runners_) {
      runner->finish(TARGET, STOPPING);
      record(runner->outcome());
    }
    runners_.clear();
  }

  std::string artifactDir() const { return artifactDir_; }

  [[nodiscard]] const std::optional<ReadinessResult>& outcome() const noexcept { return outcome_; }

private:
  /**
   * @brief Keep the capture's gravest outcome, the first of its kind: a
   * failure over a caveat, a caveat over READY.
   */
  void record(const std::optional<ReadinessResult>& result) {
    if (!result) {
      return;
    }
    if (!outcome_ || result->report.status > outcome_->report.status) {
      outcome_ = *result;
    }
  }

  std::shared_ptr<const BpftracePlan> plan_;
  bool enabled_ = false;
  std::string artifactDir_;
  std::vector<std::unique_ptr<BpfRunner>> runners_;
  long stopTid_ = -1; ///< The thread that started the capture, which the window's stop names.
  std::optional<ReadinessResult> outcome_;
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

const std::optional<ReadinessResult>& BpftraceProfiler::captureOutcome() const noexcept {
  static const std::optional<ReadinessResult> NONE;
#ifdef __linux__
  if (impl_) {
    return impl_->outcome();
  }
#endif
  return NONE;
}

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
