/**
 * @file ProfilerOffCpu.cpp
 * @brief Off-CPU profiling via bpftrace on the sched tracepoints.
 */

#include "src/bench/inc/ProfilerOffCpu.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

// Probes ride the sched tracepoints (stable kernel ABI) rather than
// finish_task_switch kprobes: the scheduler function inlines on modern
// kernels, leaving an attachable symbol that target switches never
// traverse -- probes look live and record nothing.
//
// One sched_switch program holds both sides of a switch. At switch-out the
// current task IS the thread being descheduled, so its user stack is the
// blocking site. Only a thread of this process that goes to sleep is counted:
// prev_state 1 (S, interruptible) or 2 (D, uninterruptible). A task preempted
// on its way back to user space reports 0, one preempted inside the kernel 256
// (the preempted flag), and an exiting thread 16 (EXIT_DEAD); none of them
// waits for anything. The time from a sleeping switch-out to the thread's
// next switch-in joins through the per-tid start map; it holds the sleep and
// the wait for a CPU after the wakeup.
//
// Nothing is recorded outside an armed window, which the capture opens and
// closes through threads of its own, bound by id and name as the shared
// capture window binds them (bpftrace_tool::captureWindowProgram()), so that
// the tracer's acknowledgements show it saw this process from the start of
// the measured region to its end. offCpuScript() fills in the two thread ids
// and the window's three reserved names:
//  - a sleeping switch-out of the arm thread, the backend's own, while it
//    carries the arm name, arms it and prints "offcpu armed <pid> <tid>";
//  - one of the thread that runs the measured repeats, once they are over
//    and it carries the stop name, disarms it, forgets the start of every
//    thread still asleep (counted once, with no time) and prints
//    "offcpu disarmed <pid> <tid> <n>", n the switch-outs it recorded;
//  - no thread that carries one of the three names, the capture's own waits
//    among them, is recorded.
// Ids of 0 make a script that never arms (a readiness probe's). @recorded
// counts with ++ because printf takes no count() map; increments racing on
// two CPUs can only make n smaller than the dump's counts. @armed stays 2
// after the disarm and is the first map printed, so an output without
// "@armed: 2" holds no dump, whatever its acknowledgements say.
//
// No END block on purpose: bpftrace auto-prints every map at exit (SIGINT,
// exit(), or target death via the self-exit probe), and distro builds are
// often stripped, which breaks BEGIN/END trigger symbols outright. Consumers
// read @offcpu_blocks / @offcpu_ns.
//
// The self-exit probe matches the main thread alone (tid == $1):
// sched_process_exit fires for every exiting thread, and a test that starts
// threads must not end its own trace when the first of them finishes.
constexpr const char* OFFCPU_SCRIPT = R"BT(
tracepoint:sched:sched_switch {
  if (@armed == 1 && @start[args->next_pid]) {
    @offcpu_ns[args->next_pid] = sum(nsecs - @start[args->next_pid]);
    delete(@start[args->next_pid]);
  }
  if (pid == $1 && (args->prev_state == 1 || args->prev_state == 2)) {
    if (tid == {{ARM_TID}} && comm == "{{ARM_NAME}}") {
      if (@armed == 0) {
        @armed = 1;
        printf("offcpu armed %d %d\n", pid, tid);
      }
    } else if (tid == {{STOP_TID}} && comm == "{{STOP_NAME}}") {
      if (@armed == 1) {
        @armed = 2;
        clear(@start);
        printf("offcpu disarmed %d %d %d\n", pid, tid, @recorded);
      }
    } else if (@armed == 1 && comm != "{{ARM_NAME}}" && comm != "{{WAIT_NAME}}" &&
               comm != "{{STOP_NAME}}") {
      @start[args->prev_pid] = nsecs;
      @offcpu_blocks[ustack, comm] = count();
      @recorded++;
    }
  }
}
tracepoint:sched:sched_process_exit /tid == $1/ {
  exit();
}
)BT";

/// The label of the script's acknowledgements: "offcpu armed <pid> <tid>".
constexpr const char* LABEL = "offcpu";

// How long the check's probe must stay running: older, stripped bpftrace
// builds take on the order of a second to attach.
constexpr int START_GRACE_MS = 1500;
constexpr int INTERRUPT_WAIT_MS = 4000;
constexpr int TERMINATE_WAIT_MS = 2000;
constexpr int KILL_WAIT_MS = 2000;
const char* const WHAT = "the off-CPU script";

/// The stage of a failure found once the tracer armed: what the capture wrote.
constexpr ReadinessStage CAPTURE_STAGE = ReadinessStage::COLLECTION;

const char* const NAMESPACE_REMEDY =
    "Run the benchmark on the host, or in a container started with --pid=host.";

/** @brief @p text with every @p token replaced by @p value. */
std::string filled(std::string text, const std::string& token, const std::string& value) {
  for (std::size_t at = text.find(token); at != std::string::npos;
       at = text.find(token, at + value.size())) {
    text.replace(at, token.size(), value);
  }
  return text;
}

/**
 * @brief The off-CPU script, armed by thread @p armTid and stopped by thread
 * @p stopTid of the process $1 names, under the capture window's reserved
 * thread names; ids of 0 make a script that never arms.
 */
std::string offCpuScript(long armTid, long stopTid) {
  std::string script = OFFCPU_SCRIPT;
  script = filled(std::move(script), "{{ARM_TID}}", std::to_string(armTid));
  script = filled(std::move(script), "{{STOP_TID}}", std::to_string(stopTid));
  script = filled(std::move(script), "{{ARM_NAME}}", bpftrace_tool::ARM_THREAD);
  script = filled(std::move(script), "{{WAIT_NAME}}", bpftrace_tool::WAIT_THREAD);
  return filled(std::move(script), "{{STOP_NAME}}", bpftrace_tool::STOP_THREAD);
}

/**
 * @brief `-B none -e <script> <pid>`: the arguments every run passes, the
 * script bound to @p armTid and @p stopTid; $1 binds the pid.
 */
std::vector<std::string> launchArgs(long pid, long armTid, long stopTid) {
  return {"-B", "none", "-e", offCpuScript(armTid, stopTid), std::to_string(pid)};
}

/** @brief The run's command line as reports show it, the embedded script named. */
std::string commandLine(const BpftraceRoute& route, long pid) {
  return route.bpftrace + " -B none -e <the off-CPU script> " + std::to_string(pid);
}

#ifdef __linux__
/** @brief How many seconds a readiness probe runs before its script ends it. */
constexpr int PROBE_SELF_EXIT_S = 5;

/** @brief The self-exit a check's tracer carries: an interval probe that ends it. */
std::string selfExitProgram() {
  return "interval:s:" + std::to_string(PROBE_SELF_EXIT_S) + " { exit(); }\n";
}

/**
 * @brief The readiness probe's script: the launch's, bound to no thread,
 * plus the attach line the check waits for and an interval probe that ends
 * it by itself. A tracer whose stop is refused so ends within the bound
 * whatever it traces; the check waits for that and reaps it.
 */
std::string probeScript() {
  return offCpuScript(0, 0) + bpftrace_tool::attachLineProgram() + selfExitProgram();
}

/**
 * @brief The check's second copy: the launch's script, armed by thread
 * @p armTid and stopped by none, and the probe's self-exit. Its own arm
 * line, not an attach line, is what the check waits for.
 */
std::string witnessScript(long armTid) { return offCpuScript(armTid, 0) + selfExitProgram(); }
#endif

std::shared_ptr<const OffCpuPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const OffCpuPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "offcpu";
  return checkOffCpuRequest(request, CTX);
}

/** @brief How a process ended, as reports word it, from a status read here. */
std::string statusText(int waitStatus) {
  if (WIFEXITED(waitStatus)) {
    return "exit status " + std::to_string(WEXITSTATUS(waitStatus));
  }
  if (WIFSIGNALED(waitStatus)) {
    return "killed by signal " + std::to_string(WTERMSIG(waitStatus));
  }
  return "wait status " + std::to_string(waitStatus);
}

/**
 * @brief What a tracer that ended with nothing on its stderr showed of its
 * end: its status, or, when this process could not read it, why.
 */
std::string endedSilently(const HelperStopResult& end) {
  if (!end.statusObtained) {
    return "its end could not be observed (" + describeStatusLoss(end.statusLoss) +
           "), and it printed nothing on stderr";
  }
  return "it ended with " + statusText(end.waitStatus) + " and printed nothing on stderr";
}

/** @brief A file's text, or why it could not be read. */
struct FileText {
  bool read = false;
  std::string text;
  std::string error;
};

FileText readFile(const std::string& path) {
  FileText file;
  const int FD = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (FD < 0) {
    file.error = std::generic_category().message(errno);
    return file;
  }
  std::string chunk(1U << 16U, '\0');
  for (;;) {
    const ssize_t N = ::read(FD, chunk.data(), chunk.size());
    if (N > 0) {
      file.text.append(chunk.data(), static_cast<std::size_t>(N));
    } else if (N < 0 && errno == EINTR) {
      continue;
    } else {
      if (N < 0) {
        file.error = std::generic_category().message(errno);
      }
      break;
    }
  }
  ::close(FD);
  file.read = file.error.empty();
  return file;
}

/**
 * @brief Wait up to @p boundMs, under the name the capture window reserves
 * for its waits, until @p watch holds the arm line, or a stop line before it
 * when @p stopFirstEnds, or @p tracer has ended; then end @p arm under the
 * same name. Joining the arm thread waits out its last nap, a sleep of this
 * thread once the script may have armed: under that name the script leaves
 * it out, as it leaves out the wait.
 * @return Whether the tracer still ran at the last look. Whether it runs is
 *         read before the output, so a line written just before it ended
 *         still counts. A read that fails reads nothing: the bound decides.
 */
bool waitForTheArm(OwnedHelper& tracer, bpftrace_tool::WindowWatch& watch,
                   std::optional<bpftrace_tool::ArmThread>& arm, int boundMs, bool stopFirstEnds) {
  bool running = true;
  const bpftrace_tool::ThreadNameScope WAITING(bpftrace_tool::WAIT_THREAD);
  (void)bpftrace_tool::waitUntil(
      [&] {
        running = tracer.running();
        (void)watch.poll();
        return watch.armed().has_value() || (stopFirstEnds && watch.disarmedBeforeArmed()) ||
               !running;
      },
      boundMs);
  arm.reset();
  return running;
}

/**
 * @brief The outcome of a tracer that did not arm for this process: in a PID
 * namespace other than the host's, that namespace is why.
 */
ReadinessResult notSeen(const std::string& detail, const std::string& remedy) {
  const std::string NAMESPACE = bpftrace_tool::foreignPidNamespace();
  if (!NAMESPACE.empty()) {
    return readinessResult(ReadinessCause::UNSUPPORTED,
                           detail + "; this process runs in PID namespace " + NAMESPACE +
                               ", not the host's, and offcpu traces only from the host's PID "
                               "view",
                           NAMESPACE_REMEDY);
  }
  return readinessResult(ReadinessCause::UNUSABLE, detail, remedy);
}

/** @brief What a stop could not deliver, if anything: the first refused delivery. */
const StopDelivery* refusedDelivery(const HelperStopResult& stop) {
  for (const StopDelivery& delivery : stop.deliveries) {
    if (!delivery.delivered) {
      return &delivery;
    }
  }
  return nullptr;
}

/**
 * @brief The count the script's first stop line carries after its two ids
 * ("offcpu disarmed <pid> <tid> <n>"); nullopt when @p text holds no stop
 * line, or one without a count.
 */
std::optional<long> stopCount(const std::string& text) {
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    const std::optional<bpftrace_tool::WindowAck> ACK = bpftrace_tool::parseDisarmAck(line, LABEL);
    if (!ACK) {
      continue;
    }
    const std::string IDS = std::string{LABEL} + " disarmed " + std::to_string(ACK->pid) + " " +
                            std::to_string(ACK->tid) + " ";
    const std::size_t AT = line.find(IDS);
    const std::string COUNT =
        AT == std::string::npos ? std::string{} : line.substr(AT + IDS.size());
    if (COUNT.empty() || COUNT.size() > 18 ||
        COUNT.find_first_not_of("0123456789") != std::string::npos) {
      return std::nullopt;
    }
    return std::stol(COUNT);
  }
  return std::nullopt;
}

/** @brief What a map dump holds of the capture's own counts. */
struct DumpCounts {
  bool disarmed = false;  ///< The dump holds "@armed: 2": the maps as the disarm left them.
  bool wellFormed = true; ///< Every @offcpu_blocks entry ends with its count.
  long blocks = 0;        ///< The sum of the @offcpu_blocks counts.
  long recorded = -1;     ///< The @recorded line's value; -1 when there is none.
};

/**
 * @brief Read the @armed line, the @offcpu_blocks entries and the @recorded
 * line of a dump.
 *
 * bpftrace prints its maps in the order of their names, so @armed comes
 * first and @recorded after @offcpu_blocks and @offcpu_ns: an output without
 * @armed holds no dump at all, and a dump cut short loses @recorded. The
 * block counts are not compared with it: count() creates a key with an
 * insert that fails when another CPU created the same key at the same
 * moment, so threads that first sleep at one stack together can leave that
 * entry short.
 */
DumpCounts countsIn(const std::string& dump) {
  DumpCounts counts;
  bool inEntry = false;
  std::istringstream lines(dump);
  std::string line;
  while (std::getline(lines, line)) {
    if (line == "@armed: 2") {
      counts.disarmed = true;
      continue;
    }
    if (line.rfind("@offcpu_blocks[", 0) == 0) {
      counts.wellFormed = counts.wellFormed && !inEntry;
      inEntry = true;
      continue;
    }
    const std::size_t CLOSE = line.rfind("]: ");
    if (inEntry && line.rfind(", ", 0) == 0 && CLOSE != std::string::npos) {
      const std::string COUNT = line.substr(CLOSE + 3);
      counts.wellFormed = counts.wellFormed && !COUNT.empty() && COUNT.size() <= 18 &&
                          COUNT.find_first_not_of("0123456789") == std::string::npos;
      if (counts.wellFormed) {
        counts.blocks += std::stol(COUNT);
      }
      inEntry = false;
    } else if (line.rfind("@recorded: ", 0) == 0) {
      const std::string VALUE = line.substr(11);
      if (!VALUE.empty() && VALUE.size() <= 18 &&
          VALUE.find_first_not_of("0123456789") == std::string::npos) {
        counts.recorded = std::stol(VALUE);
      }
    }
  }
  counts.wellFormed = counts.wellFormed && !inEntry;
  return counts;
}

#ifdef __linux__
/**
 * @brief How long the check waits for its copy of the script to arm: less
 * than the copy's self-exit, which counts from its attach and would
 * otherwise end it first on a tracer that attaches at once.
 */
constexpr int CHECK_ARM_WAIT_MS = PROBE_SELF_EXIT_S * 1000 - 1000;

/**
 * @brief The check's witness: a copy of the launch's script with the
 * probe's self-exit and unbuffered output, through @p route, armed by a
 * thread of this process it starts first. The context is a snapshot of this
 * process, so the thread belongs to the script's target, the context's pid.
 * @return The Error that stops the request, a Warning the check reports, or
 *         nullopt with @p seen saying what the tracer acknowledged.
 */
std::optional<ReadinessResult> witnessArm(const BpftraceRoute& route, const ReadinessContext& ctx,
                                          const std::string& scratchDir,
                                          const std::string& runCommand, std::string& seen) {
  const long PID = static_cast<long>(ctx.self());
  std::optional<bpftrace_tool::ArmThread> arm;
  try {
    arm.emplace();
  } catch (const std::system_error& error) {
    return readinessResult(ReadinessCause::UNUSABLE,
                           std::string{WHAT} +
                               ": the thread that arms the tracer could not start: " + error.what(),
                           "");
  }
  const long ARM_TID = arm->tid();
  std::vector<std::string> argv = route.command();
  const std::vector<std::string> ARGS = {"-B", "none", "-e", witnessScript(ARM_TID),
                                         std::to_string(PID)};
  argv.insert(argv.end(), ARGS.begin(), ARGS.end());
  const std::string COMMAND = route.bpftrace + " -B none -e <the off-CPU script with a " +
                              std::to_string(PROBE_SELF_EXIT_S) + " s self-exit> " +
                              std::to_string(PID);
  const std::string OUT = scratchDir + "/witness.out";
  const std::string ERR = scratchDir + "/witness.err";
  OwnedHelper tracer(
      route.stopPolicy(2000, 1000, 1000, std::make_shared<const ReadinessContext>(ctx)));
  const auto STARTED_AT = std::chrono::steady_clock::now();
  const HelperStart START = tracer.start(argv, OUT, ERR, 0, &ctx);
  if (!START.started) {
    return readinessResult(ReadinessCause::UNUSABLE, std::string{WHAT} + ": " + START.errorTail,
                           "Check that " + argv.front() + " can be executed.");
  }
  bpftrace_tool::WindowWatch watch(OUT, LABEL);
  const bool RUNNING = waitForTheArm(tracer, watch, arm, CHECK_ARM_WAIT_MS, false);
  const HelperStopResult STOP = tracer.stop();
  if (STOP.stillAlive) {
    // The stop could not end it; its own self-exit will, as for the attach probe.
    const auto UNTIL =
        STARTED_AT + std::chrono::milliseconds(PROBE_SELF_EXIT_S * 1000 +
                                               bpftrace_tool::PROBE_SELF_EXIT_SLACK_MS);
    while (tracer.running() && std::chrono::steady_clock::now() < UNTIL) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  const std::optional<bpftrace_tool::WindowAck>& ARMED = watch.armed();
  if (!ARMED && !RUNNING) {
    std::string text = readFile(ERR).text;
    if (outputTail(text).empty()) {
      text = endedSilently(STOP);
    }
    return bpftrace_tool::classifyAttachFailure(route, WHAT, COMMAND, text, ctx, runCommand);
  }
  if (!ARMED) {
    return notSeen(std::string{WHAT} + " ran " + route.describe() +
                       " but did not acknowledge its arm probe within " +
                       std::to_string(CHECK_ARM_WAIT_MS) + " ms",
                   "bpftrace did not see this process's arm thread go to sleep; run the "
                   "off-CPU script by hand with " +
                       route.bpftrace + " to see why.");
  }
  if (ARMED->pid != PID || ARMED->tid != ARM_TID) {
    return notSeen(std::string{WHAT} + " armed for pid " + std::to_string(ARMED->pid) + " thread " +
                       std::to_string(ARMED->tid) + ", not for this process's arm thread (pid " +
                       std::to_string(PID) + " thread " + std::to_string(ARM_TID) + ")",
                   "bpftrace numbers this process's threads differently from the process "
                   "itself; offcpu traces only from the host's PID view.");
  }
  if (tracer.running()) {
    return readinessResult(ReadinessCause::UNUSABLE,
                           "the check's copy of the script, tracer " +
                               std::to_string(tracer.pid()) +
                               ", still runs past its stop and its self-exit",
                           "Stop it by hand, then check the bpftrace build.");
  }
  seen = "a copy with unbuffered output armed on this process's arm thread (pid " +
         std::to_string(PID) + ", thread " + std::to_string(ARM_TID) + ")";
  return std::nullopt;
}
#endif

} // namespace

/* ----------------------------- Readiness ----------------------------- */

ReadinessResult checkOffCpuRequest(const ReadinessRequest& /*request*/,
                                   const ReadinessContext& ctx) {
#ifdef __linux__
  auto plan = std::make_shared<OffCpuPlan>();
  plan->context = std::make_shared<const ReadinessContext>(ctx);
  if (auto failure = bpftrace_tool::resolveRoute(ctx, {}, plan->route)) {
    return *failure;
  }
  if (auto failure = bpftrace_tool::probeExecutable(plan->route, ctx)) {
    return *failure;
  }
  // The launch's script, with an attach line and a self-exit added, through
  // the route until it has attached, the run's start grace at least, on this
  // process; stopped with the run's first stop signal. -B none brings the
  // attach line out as it is printed. The additions make it a command the run
  // never runs, so a grant's refusal of it is unverified, and the self-exit
  // bounds a tracer whose stop is refused: the check waits for that end and
  // reaps it.
  const bool SUDO = plan->route.privilege.route == PrivilegeRoute::SCOPED_SUDO;
  const std::string SELF = std::to_string(static_cast<long>(ctx.self()));
  const std::string RUN_COMMAND =
      plan->route.bpftrace + " -B none -e <the off-CPU script> <benchmark pid>";
  const ProbeScratch SCRATCH(ctx);
  bpftrace_tool::AttachProbe probe;
  probe.toolArgs = {"-B", "none", "-e", probeScript(), SELF};
  probe.what = WHAT;
  probe.commandLine = plan->route.bpftrace +
                      " -B none -e <the off-CPU script with an attach line and a " +
                      std::to_string(PROBE_SELF_EXIT_S) + " s self-exit> " + SELF;
  probe.runCommand = RUN_COMMAND;
  probe.graceMs = START_GRACE_MS;
  probe.selfExitMs = PROBE_SELF_EXIT_S * 1000;
  auto verdict = bpftrace_tool::probeAttach(plan->route, probe, ctx, SCRATCH.path());
  if (verdict && verdict->report.status == EnvReport::Status::Error) {
    return *verdict;
  }
  // The capture's own evidence that bpftrace sees this process's threads
  // under the ids the process knows: a copy of the script arms on a thread
  // of this process. Not tried after an unverified probe: the grant refused
  // it and would refuse the copy alike, or only the run can show whether the
  // tracer attaches here, or how it ends.
  std::string seen;
  if (!verdict || verdict->cause != ReadinessCause::UNVERIFIED) {
    auto witness = witnessArm(plan->route, ctx, SCRATCH.path(), RUN_COMMAND, seen);
    if (witness && witness->report.status == EnvReport::Status::Error) {
      return *witness;
    }
    if (witness && !verdict) {
      verdict = std::move(witness);
    }
  }
  std::string message =
      std::string{WHAT} + ", with a " + std::to_string(PROBE_SELF_EXIT_S) +
      " s self-exit added, stayed running for the " + std::to_string(START_GRACE_MS) +
      " ms start grace " + plan->route.describe() + " and stopped on SIGINT" +
      (SUDO ? " through sudo -n kill" : "") + " (probe with " + plan->route.bpftrace + ")";
  if (!seen.empty()) {
    message += ", and " + seen;
  }
  if (plan->route.privilege.route == PrivilegeRoute::ALREADY_ROOT && plan->route.privilege.optIn) {
    message += "; running as root; BENCH_SUDO not needed";
  }
  message += "; not checked: " +
             (SUDO ? "the grant for the run's own command (" + RUN_COMMAND +
                         "), SIGTERM and SIGKILL through sudo, and "
                   : std::string{}) +
             "the run's capture";
  ReadinessResult result;
  if (verdict) {
    result = *verdict;
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
#else
  (void)ctx;
  return readinessResult(ReadinessCause::UNSUPPORTED, "off-CPU profiling is Linux-only",
                         "Run on Linux or use a different profiler.");
#endif
}

/* ----------------------------- OffCpuProfiler ----------------------------- */

struct OffCpuProfiler::Capture {
  Capture(HelperStopPolicy policy, const std::string& outputPath, long stopThread)
      : tracer(std::move(policy)), watch(outputPath, LABEL), stopTid(stopThread) {}

  OwnedHelper tracer;
  bpftrace_tool::WindowWatch watch;
  long stopTid; ///< The thread that started the capture, which the script's stop names.
};

OffCpuProfiler::OffCpuProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  const ReadinessResult DECISION = decideNow(cfg_);
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    std::fprintf(stderr, "[offcpu] not started: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[offcpu] %s\n", DECISION.report.hint.c_str());
    }
    outcome_ = DECISION;
    return;
  }
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "offcpu");
  outputPath_ = artifactDir_ + "/offcpu.txt";
  errorPath_ = artifactDir_ + "/offcpu.err.txt";
}

OffCpuProfiler::OffCpuProfiler(const PerfConfig& cfg, std::string testName,
                               std::shared_ptr<const OffCpuPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "offcpu");
  outputPath_ = artifactDir_ + "/offcpu.txt";
  errorPath_ = artifactDir_ + "/offcpu.err.txt";
}

OffCpuProfiler::~OffCpuProfiler() { stopBpftrace(); }

void OffCpuProfiler::beforeMeasure() {
  if (plan_) {
    spawnBpftrace();
  }
}

void OffCpuProfiler::afterMeasure(const Stats& /*s*/) { stopBpftrace(); }

void OffCpuProfiler::spawnBpftrace() {
  if (capture_) {
    stopBpftrace();
  }
  outcome_.reset();
  const long PID = static_cast<long>(::getpid());
  // The script names the thread that arms it and the one that will stop it,
  // the calling thread, which runs the measured repeats: the arm thread
  // exists before the launch, and naps until the arm wait ends.
  const long STOP_TID = bpftrace_tool::currentThreadId();
  std::optional<bpftrace_tool::ArmThread> arm;
  try {
    arm.emplace();
  } catch (const std::system_error& error) {
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           std::string{"no capture: the thread that arms the tracer could not "
                                       "start: "} +
                               error.what(),
                           ""));
    return;
  }
  const long ARM_TID = arm->tid();
  const std::vector<std::string> ARGS = launchArgs(PID, ARM_TID, STOP_TID);
  std::vector<std::string> argv = plan_->route.command();
  argv.insert(argv.end(), ARGS.begin(), ARGS.end());
  // The capture files are opened by the child as the invoking user, so they
  // stay the user's whatever the tracer runs as.
  auto capture = std::make_unique<Capture>(
      plan_->route.stopPolicy(INTERRUPT_WAIT_MS, TERMINATE_WAIT_MS, KILL_WAIT_MS, plan_->context),
      outputPath_, STOP_TID);
  const HelperStart STARTED = capture->tracer.start(argv, outputPath_, errorPath_, 0, nullptr);
  if (!STARTED.started) {
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           "could not start bpftrace: " + STARTED.errorTail,
                           "Check that " + argv.front() + " can be executed."));
    return;
  }
  // The script arms when the arm thread goes to sleep; this thread waits for
  // that, and ends the arm thread, under a name the script leaves out.
  const bool RUNNING = waitForTheArm(capture->tracer, capture->watch, arm, plan_->armWaitMs, true);
  const std::optional<bpftrace_tool::WindowAck>& ARMED = capture->watch.armed();
  if (!capture->watch.disarmedBeforeArmed() && !ARMED && !RUNNING) {
    const HelperStopResult ENDED = capture->tracer.stop();
    std::string text = readFile(errorPath_).text;
    if (text.size() > 4096) {
      text.erase(0, text.size() - 4096);
    }
    if (outputTail(text).empty()) {
      text = endedSilently(ENDED);
    }
    finish(bpftrace_tool::classifyAttachFailure(plan_->route, WHAT, commandLine(plan_->route, PID),
                                                text, *plan_->context),
           "the tracer ended before it armed the capture: ");
    return;
  }
  if (capture->watch.disarmedBeforeArmed() || !ARMED || ARMED->pid != PID ||
      ARMED->tid != ARM_TID) {
    const HelperStopResult STOPPED = capture->tracer.stop();
    if (STOPPED.wasRunning) {
      (void)bpftrace_tool::reportStop("offcpu", WHAT, STOPPED, plan_->route, errorPath_,
                                      outputPath_);
    }
    if (capture->watch.disarmedBeforeArmed()) {
      const bpftrace_tool::WindowAck EARLY = *capture->watch.disarmed();
      finish(readinessResult(ReadinessCause::UNUSABLE,
                             "no capture: the tracer acknowledged a stop before its arm, for pid " +
                                 std::to_string(EARLY.pid) + " thread " +
                                 std::to_string(EARLY.tid) + "; its output is in " + artifactDir_,
                             "The script prints its stop line only once it has armed; see "
                             "offcpu.err.txt there, and run the off-CPU script by hand with " +
                                 plan_->route.bpftrace + "."));
    } else if (!ARMED) {
      finish(notSeen("no capture: the tracer did not acknowledge its arm probe within " +
                         std::to_string(plan_->armWaitMs) + " ms; its output is in " + artifactDir_,
                     "bpftrace ran but did not see this process's arm thread go to sleep; see "
                     "offcpu.err.txt there, and run --profile-check --profile offcpu."));
    } else {
      finish(notSeen("no capture: the tracer armed for pid " + std::to_string(ARMED->pid) +
                         " thread " + std::to_string(ARMED->tid) +
                         ", not for this process's arm thread (pid " + std::to_string(PID) +
                         " thread " + std::to_string(ARM_TID) + "); its output is in " +
                         artifactDir_,
                     "bpftrace numbers this process's threads differently from the process "
                     "itself; offcpu traces only from the host's PID view."));
    }
    return;
  }
  capture_ = std::move(capture);
}

void OffCpuProfiler::stopBpftrace() {
  if (!capture_) {
    return;
  }
  const std::unique_ptr<Capture> CAPTURE = std::move(capture_);
  const long PID = static_cast<long>(::getpid());
  const long STOPPING = bpftrace_tool::currentThreadId();
  const std::string VALIDITY = "capture validity could not be established: ";
  // What the tracer wrote during the measured repeats: a stop line there came
  // before the stop was asked for. The stop is bound to the thread that
  // started the capture, and only that thread can take its name.
  (void)CAPTURE->watch.poll();
  std::optional<ReadinessResult> early;
  if (const std::optional<bpftrace_tool::WindowAck>& EARLY = CAPTURE->watch.disarmed()) {
    early = readinessResult(ReadinessCause::UNUSABLE,
                            VALIDITY +
                                "the tracer acknowledged a stop before the measured repeats "
                                "finished, for pid " +
                                std::to_string(EARLY->pid) + " thread " +
                                std::to_string(EARLY->tid) + "; its output is in " + artifactDir_,
                            std::string{"The thread that runs the measured repeats may carry the "
                                        "name "} +
                                bpftrace_tool::STOP_THREAD +
                                ", which the capture reserves; give it another.",
                            CAPTURE_STAGE);
  } else if (STOPPING != CAPTURE->stopTid) {
    early = readinessResult(ReadinessCause::UNUSABLE,
                            VALIDITY + "the measured repeats ended on thread " +
                                std::to_string(STOPPING) + ", not on thread " +
                                std::to_string(CAPTURE->stopTid) +
                                ", where they started and which the script's stop names; its "
                                "output is in " +
                                artifactDir_,
                            "Call the profiler's beforeMeasure() and afterMeasure() from one "
                            "thread, as the harness does.",
                            CAPTURE_STAGE);
  }
  if (early) {
    finish(*early);
    const HelperStopResult STOPPED = CAPTURE->tracer.stop();
    (void)bpftrace_tool::reportStop("offcpu", WHAT, STOPPED, plan_->route, errorPath_, outputPath_);
    return;
  }
  {
    const bpftrace_tool::ThreadNameScope STOPPING_NAME(bpftrace_tool::STOP_THREAD);
    (void)bpftrace_tool::waitUntil(
        [&CAPTURE] {
          const bool RUNNING = CAPTURE->tracer.running();
          (void)CAPTURE->watch.poll();
          return CAPTURE->watch.disarmed().has_value() || !RUNNING;
        },
        plan_->disarmWaitMs);
  }
  // SIGINT makes bpftrace print its maps and exit with status 0. The stop
  // escalates through the plan's route and reports every delivery it could
  // not make; reportStop() judges how the tracer ended, at the stop or before
  // it, and prints every end but one seen to be status 0, whose stacks are
  // not claimed. An end this process could not observe is never status 0.
  const HelperStopResult STOPPED = CAPTURE->tracer.stop();
  const std::optional<ReadinessResult> INCOMPLETE =
      bpftrace_tool::reportStop("offcpu", WHAT, STOPPED, plan_->route, errorPath_, outputPath_);
  // At the stop bpftrace prints the lines it still had queued, then its
  // maps: a disarm line that did not arrive in time can arrive there.
  if (!CAPTURE->watch.disarmed()) {
    (void)CAPTURE->watch.poll();
  }
  const std::optional<bpftrace_tool::WindowAck>& DISARMED = CAPTURE->watch.disarmed();
  const bool SUDO = plan_->route.privilege.route == PrivilegeRoute::SCOPED_SUDO;
  const StopDelivery* const REFUSED = refusedDelivery(STOPPED);
  const std::string KEPT = "; the capture in " + artifactDir_ + " is incomplete";
  if (STOPPED.stillAlive || STOPPED.stoppedBy == SIGKILL) {
    if (REFUSED != nullptr) {
      finish(readinessResult(ReadinessCause::DENIED,
                             std::string{"the tracer "} +
                                 (STOPPED.stillAlive ? "could not be stopped and still runs"
                                                     : "was killed before it printed its maps") +
                                 ": " + REFUSED->command + " failed: " + REFUSED->detail + KEPT,
                             SUDO ? bpftrace_tool::grantRemedy(plan_->route, *plan_->context)
                                  : std::string{},
                             CAPTURE_STAGE));
    } else {
      finish(readinessResult(ReadinessCause::UNUSABLE,
                             std::string{"the tracer "} +
                                 (STOPPED.stillAlive
                                      ? "did not stop on SIGINT, SIGTERM or SIGKILL and still runs"
                                      : "ignored SIGINT and SIGTERM and was killed before it "
                                        "printed its maps") +
                                 KEPT,
                             "bpftrace prints its maps on SIGINT; check that this build handles "
                             "it.",
                             CAPTURE_STAGE));
    }
    return;
  }
  if (INCOMPLETE) {
    // Not seen to end with status 0: reportStop() printed how it ended, or
    // why that could not be observed, naming the files kept; its verdict is
    // the capture's.
    ReadinessResult incomplete = *INCOMPLETE;
    incomplete.stage = CAPTURE_STAGE;
    if (!incomplete.report.hint.empty()) {
      std::fprintf(stderr, "[offcpu] %s\n", incomplete.report.hint.c_str());
    }
    record(std::move(incomplete));
    return;
  }
  // Seen to end with status 0: before the stop, its script ended itself.
  if (!STOPPED.wasRunning && !DISARMED) {
    std::string detail =
        "the tracer ended by itself before the stop (" + statusText(STOPPED.waitStatus) + ")";
    const std::string TAIL = outputTail(readFile(errorPath_).text);
    if (!TAIL.empty()) {
      detail += ": " + TAIL;
    }
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           detail + "; it misses the end of the measured region" + KEPT,
                           "The script ends by itself when this process's main thread exits; "
                           "anything else is bpftrace failing: see offcpu.err.txt there.",
                           CAPTURE_STAGE));
    return;
  }
  if (!DISARMED) {
    finish(readinessResult(
        ReadinessCause::UNUSABLE,
        VALIDITY + "the tracer did not acknowledge the stop within " +
            std::to_string(plan_->disarmWaitMs) +
            " ms, nor in the lines it printed at the stop; its output is in " + artifactDir_,
        "The tracer ran but did not see the stopping thread go to sleep; see offcpu.err.txt "
        "there.",
        CAPTURE_STAGE));
    return;
  }
  if (DISARMED->pid != PID || DISARMED->tid != CAPTURE->stopTid) {
    finish(readinessResult(
        ReadinessCause::UNUSABLE,
        VALIDITY + "the tracer acknowledged the stop for pid " + std::to_string(DISARMED->pid) +
            " thread " + std::to_string(DISARMED->tid) +
            ", not for this process's stopping thread (pid " + std::to_string(PID) + " thread " +
            std::to_string(CAPTURE->stopTid) + "); its output is in " + artifactDir_,
        "", CAPTURE_STAGE));
    return;
  }
  const FileText DUMP = readFile(outputPath_);
  if (!DUMP.read) {
    finish(
        readinessResult(ReadinessCause::UNUSABLE,
                        "the tracer's output " + outputPath_ + " could not be read: " + DUMP.error,
                        "", CAPTURE_STAGE));
    return;
  }
  const std::optional<long> STOP_COUNT = stopCount(DUMP.text);
  if (!STOP_COUNT) {
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           VALIDITY +
                               "the tracer's stop line carries no count of the switch-outs it "
                               "recorded; its output is in " +
                               artifactDir_,
                           "", CAPTURE_STAGE));
    return;
  }
  const DumpCounts COUNTS = countsIn(DUMP.text);
  const long N = *STOP_COUNT;
  std::string cut;
  if (!COUNTS.wellFormed) {
    cut = "an @offcpu_blocks entry of its dump is cut short";
  } else if (!COUNTS.disarmed) {
    cut = "it holds no map dump (no @armed: 2 line after the acknowledgements)";
  } else if (N > 0 && COUNTS.recorded < N) {
    cut = "the tracer recorded " + std::to_string(N) +
          " sleeping switch-outs, and its dump ends before the @recorded line that counts them";
  } else if (N > 0 && COUNTS.blocks == 0) {
    cut = "the tracer recorded " + std::to_string(N) +
          " sleeping switch-outs, and its dump holds no @offcpu_blocks entry";
  }
  if (!cut.empty()) {
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           "the tracer's output " + outputPath_ + " is incomplete: " + cut, "",
                           CAPTURE_STAGE));
    return;
  }
  const long RECORDED = COUNTS.blocks;
  if (RECORDED == 0) {
    finish(readinessResult(ReadinessCause::READY,
                           "no thread of this process went to sleep while the capture was armed; "
                           "the tracer's output is in " +
                               outputPath_,
                           ""));
    return;
  }
  finish(readinessResult(ReadinessCause::READY,
                         "stacks written to " + outputPath_ + " (" + std::to_string(RECORDED) +
                             " sleeping switch-out" + (RECORDED == 1 ? "" : "s") + ")",
                         ""));
}

void OffCpuProfiler::finish(ReadinessResult outcome, const std::string& context) {
  std::fprintf(stderr, "[offcpu] %s%s\n", context.c_str(), outcome.report.message.c_str());
  if (outcome.report.status == EnvReport::Status::Error && !outcome.report.hint.empty()) {
    std::fprintf(stderr, "[offcpu] %s\n", outcome.report.hint.c_str());
  }
  record(std::move(outcome));
}

void OffCpuProfiler::record(ReadinessResult outcome) {
  // A failure is also the run's: the registry records it for this case and
  // fails the run with status 4 once the tests pass.
  if (outcome.report.status == EnvReport::Status::Error) {
    ProfilerRegistry::instance().reportFailure("offcpu", testName_, outcome);
  }
  outcome_ = std::move(outcome);
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeOffCpuProfiler(const PerfConfig& cfg, const std::string& testName) {
  auto plan = readyPlan(decideNow(cfg));
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<OffCpuProfiler>(cfg, testName, std::move(plan));
}

namespace {

std::unique_ptr<Profiler> makePlannedOffCpuProfiler(const PerfConfig& cfg,
                                                    const std::string& testName,
                                                    const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<OffCpuProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("offcpu", ::vernier::bench::checkOffCpuRequest,
                                   ::vernier::bench::makePlannedOffCpuProfiler,
                                   "Install bpftrace; see BENCH_SUDO for privileges.")
