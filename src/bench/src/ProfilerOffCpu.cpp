/**
 * @file ProfilerOffCpu.cpp
 * @brief Off-CPU profiling via bpftrace on the sched tracepoints.
 */

#include "src/bench/inc/ProfilerOffCpu.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
// closes through threads of its own, so that the tracer's acknowledgements
// show it saw this process from the start of the measured region to its end:
//  - a sleeping switch-out of a thread named vernier-arm that is not the main
//    thread arms it and prints "offcpu armed <pid> <tid>"; the capture checks
//    the ids against its own, and a tracer that numbers this process's
//    threads differently, as bpftrace does inside a PID namespace of its own,
//    never arms;
//  - a sleeping switch-out of a thread named vernier-stop disarms it, forgets
//    the start of every thread still asleep (counted once, with no time) and
//    prints "offcpu disarmed <pid> <tid> <n>", n the switch-outs it recorded;
//  - the capture's own waits run under the name vernier-wait, not recorded.
// @recorded counts with ++ because printf takes no count() map; increments
// racing on two CPUs can only make n smaller than the dump's counts.
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
    if (comm == "vernier-arm") {
      if (@armed == 0 && tid != $1) {
        @armed = 1;
        printf("offcpu armed %d %d\n", pid, tid);
      }
    } else if (comm == "vernier-stop") {
      if (@armed == 1) {
        @armed = 2;
        clear(@start);
        printf("offcpu disarmed %d %d %d\n", pid, tid, @recorded);
      }
    } else if (@armed == 1 && comm != "vernier-wait") {
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

/// The names the capture gives its own threads; the script tests them.
constexpr const char* ARM_THREAD_NAME = "vernier-arm";
constexpr const char* WAIT_THREAD_NAME = "vernier-wait";
constexpr const char* STOP_THREAD_NAME = "vernier-stop";

/// The script's acknowledgements, up to their numbers.
constexpr const char* ARMED_LINE = "offcpu armed ";
constexpr const char* DISARMED_LINE = "offcpu disarmed ";

// How long the check's probe must stay running: older, stripped bpftrace
// builds take on the order of a second to attach.
constexpr int START_GRACE_MS = 1500;
constexpr int INTERRUPT_WAIT_MS = 4000;
constexpr int TERMINATE_WAIT_MS = 2000;
constexpr int KILL_WAIT_MS = 2000;
/// How often a wait for an acknowledgement reads the tracer's output.
constexpr int POLL_MS = 5;
const char* const WHAT = "the off-CPU script";

/// The stage of a failure found once the tracer armed: what the capture wrote.
constexpr ReadinessStage CAPTURE_STAGE = ReadinessStage::COLLECTION;

/// The PID namespace every process starts in, as /proc/self/ns/pid names it.
constexpr const char* INITIAL_PID_NAMESPACE = "pid:[4026531836]";

const char* const NAMESPACE_REMEDY =
    "Run the benchmark on the host, or in a container started with --pid=host.";

/** @brief `-B none -e <script> <pid>`: the arguments every run passes; $1 binds the pid. */
std::vector<std::string> launchArgs(long pid) {
  return {"-B", "none", "-e", OFFCPU_SCRIPT, std::to_string(pid)};
}

/** @brief The run's command line as reports show it, the embedded script named. */
std::string commandLine(const BpftraceRoute& route, long pid) {
  return route.bpftrace + " -B none -e <the off-CPU script> " + std::to_string(pid);
}

#ifdef __linux__
/** @brief How many seconds a readiness probe runs before its script ends it. */
constexpr int PROBE_SELF_EXIT_S = 5;

/**
 * @brief The readiness probe's script: the launch's, plus an interval probe
 * that ends it by itself. A tracer whose stop is refused so ends within the
 * bound whatever it traces; the check waits for that and reaps it.
 */
std::string probeScript() {
  return std::string{OFFCPU_SCRIPT} + "interval:s:" + std::to_string(PROBE_SELF_EXIT_S) +
         " { exit(); }\n";
}
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

/** @brief How a process ended, as reports word it. */
std::string statusText(int waitStatus) {
  if (WIFEXITED(waitStatus)) {
    return "exit status " + std::to_string(WEXITSTATUS(waitStatus));
  }
  if (WIFSIGNALED(waitStatus)) {
    return "killed by signal " + std::to_string(WTERMSIG(waitStatus));
  }
  return "wait status " + std::to_string(waitStatus);
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

/** @brief The calling thread's id. */
long threadId() {
#ifdef __linux__
  return static_cast<long>(::syscall(SYS_gettid));
#else
  return 0;
#endif
}

/** @brief Gives the calling thread a name for the tracer, and its own name back at the end. */
class ThreadNameScope {
public:
  explicit ThreadNameScope(const char* name) {
#ifdef __linux__
    (void)::prctl(PR_GET_NAME, saved_.data(), 0UL, 0UL, 0UL);
    (void)::prctl(PR_SET_NAME, name, 0UL, 0UL, 0UL);
#else
    (void)name;
#endif
  }
  ~ThreadNameScope() {
#ifdef __linux__
    (void)::prctl(PR_SET_NAME, saved_.data(), 0UL, 0UL, 0UL);
#endif
  }
  ThreadNameScope(const ThreadNameScope&) = delete;
  ThreadNameScope& operator=(const ThreadNameScope&) = delete;

private:
  std::array<char, 17> saved_{}; ///< PR_GET_NAME writes at most 16 bytes.
};

/**
 * @brief A thread named for the tracer that sleeps in POLL_MS steps until it
 * is released: the tracer arms on its first sleep. Its id is stored before
 * it takes the name, and it keeps the name until it ends, so no sleep of its
 * own is recorded under another name.
 */
class ArmThread {
public:
  ArmThread()
      : thread_([this] {
          id_ = threadId();
#ifdef __linux__
          (void)::prctl(PR_SET_NAME, ARM_THREAD_NAME, 0UL, 0UL, 0UL);
#endif
          while (!released_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));
          }
        }) {
  }
  ~ArmThread() {
    released_ = true;
    thread_.join();
  }
  ArmThread(const ArmThread&) = delete;
  ArmThread& operator=(const ArmThread&) = delete;

  /** @brief The thread's id, once it has started. */
  [[nodiscard]] long id() const noexcept { return id_; }

private:
  std::atomic<bool> released_{false};
  std::atomic<long> id_{0};
  std::thread thread_; ///< Last: it starts once the flags exist.
};

/** @brief The lines a tracer appends to its output file, read as they arrive. */
class OutputLines {
public:
  explicit OutputLines(std::string path) : path_(std::move(path)) {}

  /** @brief The complete lines written since the last call. */
  std::vector<std::string> next() {
    std::vector<std::string> lines;
    std::ifstream in(path_, std::ios::binary);
    if (!in || !in.seekg(offset_)) {
      return lines;
    }
    std::array<char, 4096> buffer{};
    while (in.read(buffer.data(), buffer.size()) || in.gcount() > 0) {
      pending_.append(buffer.data(), static_cast<std::size_t>(in.gcount()));
      offset_ += in.gcount();
    }
    std::size_t start = 0;
    for (std::size_t end = pending_.find('\n'); end != std::string::npos;
         end = pending_.find('\n', start)) {
      lines.push_back(pending_.substr(start, end - start));
      start = end + 1;
    }
    pending_.erase(0, start);
    return lines;
  }

private:
  std::string path_;
  std::streamoff offset_ = 0;
  std::string pending_;
};

/** @brief One acknowledgement's numbers: pid, thread id and, for the stop, the count. */
struct Ack {
  long pid = 0;
  long tid = 0;
  long recorded = 0;
};

std::optional<Ack> parseAck(const std::string& line, const char* prefix, bool withCount) {
  if (line.rfind(prefix, 0) != 0) {
    return std::nullopt;
  }
  std::istringstream fields(line.substr(std::strlen(prefix)));
  Ack ack;
  if (!(fields >> ack.pid >> ack.tid) || (withCount && !(fields >> ack.recorded))) {
    return std::nullopt;
  }
  std::string rest;
  if (fields >> rest) {
    return std::nullopt;
  }
  return ack;
}

enum class AckState : std::uint8_t { ACKNOWLEDGED, ENDED, TIMED_OUT };

struct AckWait {
  AckState state = AckState::TIMED_OUT;
  Ack ack;
};

/**
 * @brief Wait up to @p boundMs for a line starting with @p prefix in the
 * tracer's output, reading it every POLL_MS while the tracer runs.
 */
AckWait waitForAck(OutputLines& output, OwnedHelper& tracer, const char* prefix, bool withCount,
                   int boundMs) {
  const auto DEADLINE = std::chrono::steady_clock::now() + std::chrono::milliseconds(boundMs);
  for (;;) {
    // Whether it runs is read first: a line written just before it ended
    // still counts.
    const bool RUNNING = tracer.running();
    for (const std::string& line : output.next()) {
      if (const std::optional<Ack> ACK = parseAck(line, prefix, withCount)) {
        return {AckState::ACKNOWLEDGED, *ACK};
      }
    }
    if (!RUNNING) {
      return {AckState::ENDED, {}};
    }
    if (std::chrono::steady_clock::now() >= DEADLINE) {
      return {AckState::TIMED_OUT, {}};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));
  }
}

/** @brief The target of this process's /proc/self/ns/pid link, "" if unreadable. */
std::string ownPidNamespace() {
  std::error_code ec;
  const std::filesystem::path LINK = std::filesystem::read_symlink("/proc/self/ns/pid", ec);
  return ec ? std::string{} : LINK.string();
}

/**
 * @brief The outcome of a tracer that did not arm for this process: in a PID
 * namespace other than the initial one, that namespace is why.
 */
ReadinessResult notSeen(const std::string& detail, const std::string& remedy) {
  const std::string NOTE = offCpuPidNamespaceNote(ownPidNamespace());
  if (!NOTE.empty()) {
    return readinessResult(ReadinessCause::UNSUPPORTED, detail + "; " + NOTE, NAMESPACE_REMEDY);
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

/** @brief What a map dump holds of the capture's own counts. */
struct DumpCounts {
  bool wellFormed = true; ///< Every @offcpu_blocks entry ends with its count.
  long blocks = 0;        ///< The sum of the @offcpu_blocks counts.
  long recorded = -1;     ///< The @recorded line's value; -1 when there is none.
};

/**
 * @brief Read the @offcpu_blocks entries and the @recorded line of a dump.
 *
 * bpftrace prints its maps in the order of their names, so @recorded comes
 * after @offcpu_blocks and @offcpu_ns: a dump cut short loses it. The block
 * counts are not compared with it: count() creates a key with an insert
 * that fails when another CPU created the same key at the same moment, so
 * threads that first sleep at one stack together can leave that entry short.
 */
DumpCounts countsIn(const std::string& dump) {
  DumpCounts counts;
  bool inEntry = false;
  std::istringstream lines(dump);
  std::string line;
  while (std::getline(lines, line)) {
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

} // namespace

std::string offCpuPidNamespaceNote(const std::string& namespaceLink) {
  if (namespaceLink.empty() || namespaceLink == INITIAL_PID_NAMESPACE) {
    return {};
  }
  return "this process runs in PID namespace " + namespaceLink + ", not in the initial one (" +
         INITIAL_PID_NAMESPACE + "), and offcpu traces only from the host's PID view";
}

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
  // The launch's script, with a self-exit added, through the route for the
  // run's start grace, on this process; stopped with the run's first stop
  // signal. The added interval makes it a command the run never runs, so a
  // grant's refusal of it is unverified, and it bounds a tracer whose stop is
  // refused: the check waits for that end and reaps it.
  const bool SUDO = plan->route.privilege.route == PrivilegeRoute::SCOPED_SUDO;
  const std::string SELF = std::to_string(static_cast<long>(ctx.self()));
  const std::string RUN_COMMAND =
      plan->route.bpftrace + " -B none -e <the off-CPU script> <benchmark pid>";
  const ProbeScratch SCRATCH(ctx);
  bpftrace_tool::AttachProbe probe;
  probe.toolArgs = {"-e", probeScript(), SELF};
  probe.what = WHAT;
  probe.commandLine = plan->route.bpftrace + " -e <the off-CPU script with a " +
                      std::to_string(PROBE_SELF_EXIT_S) + " s self-exit> " + SELF;
  probe.runCommand = RUN_COMMAND;
  probe.graceMs = START_GRACE_MS;
  probe.selfExitMs = PROBE_SELF_EXIT_S * 1000;
  auto verdict = bpftrace_tool::probeAttach(plan->route, probe, ctx, SCRATCH.path());
  if (verdict && verdict->report.status == EnvReport::Status::Error) {
    return *verdict;
  }
  std::string message =
      std::string{WHAT} + ", with a " + std::to_string(PROBE_SELF_EXIT_S) +
      " s self-exit added, stayed running for the " + std::to_string(START_GRACE_MS) +
      " ms start grace " + plan->route.describe() + " and stopped on SIGINT" +
      (SUDO ? " through sudo -n kill" : "") + " (probe with " + plan->route.bpftrace + ")";
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
  Capture(HelperStopPolicy policy, std::string outputPath)
      : tracer(std::move(policy)), output(std::move(outputPath)) {}

  OwnedHelper tracer;
  OutputLines output;
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
  const std::vector<std::string> ARGS = launchArgs(PID);
  std::vector<std::string> argv = plan_->route.command();
  argv.insert(argv.end(), ARGS.begin(), ARGS.end());
  // The capture files are opened by the child as the invoking user, so they
  // stay the user's whatever the tracer runs as.
  auto capture = std::make_unique<Capture>(
      plan_->route.stopPolicy(INTERRUPT_WAIT_MS, TERMINATE_WAIT_MS, KILL_WAIT_MS, plan_->context),
      outputPath_);
  const HelperStart STARTED = capture->tracer.start(argv, outputPath_, errorPath_, 0, nullptr);
  if (!STARTED.started) {
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           "could not start bpftrace: " + STARTED.errorTail,
                           "Check that " + argv.front() + " can be executed."));
    return;
  }
  // The script arms when a thread of this process named for it goes to
  // sleep; this thread waits for that under a name the script leaves out,
  // and so does its wait for the arm thread to end.
  AckWait armed;
  long armTid = 0;
  std::string threadError;
  {
    const ThreadNameScope WAITING(WAIT_THREAD_NAME);
    try {
      const ArmThread ARM;
      armed = waitForAck(capture->output, capture->tracer, ARMED_LINE, false, plan_->armWaitMs);
      armTid = ARM.id();
    } catch (const std::system_error& error) {
      threadError = error.what();
    }
  }
  if (!threadError.empty()) {
    (void)capture->tracer.stop();
    finish(readinessResult(
        ReadinessCause::UNUSABLE,
        "no capture: the thread that arms the tracer could not start: " + threadError, ""));
    return;
  }
  if (armed.state == AckState::ENDED) {
    const HelperStopResult ENDED = capture->tracer.stop();
    std::string text = readFile(errorPath_).text;
    if (text.size() > 4096) {
      text.erase(0, text.size() - 4096);
    }
    if (outputTail(text).empty()) {
      text = "it ended with " + statusText(ENDED.waitStatus) + " and printed nothing on stderr";
    }
    finish(bpftrace_tool::classifyAttachFailure(plan_->route, WHAT, commandLine(plan_->route, PID),
                                                text, *plan_->context),
           "the tracer ended before it armed the capture: ");
    return;
  }
  if (armed.state == AckState::TIMED_OUT || armed.ack.pid != PID || armed.ack.tid != armTid) {
    const HelperStopResult STOPPED = capture->tracer.stop();
    if (STOPPED.wasRunning) {
      (void)bpftrace_tool::reportStop("offcpu", WHAT, STOPPED, plan_->route);
    }
    if (armed.state == AckState::TIMED_OUT) {
      finish(notSeen("no capture: the tracer did not acknowledge its arm probe within " +
                         std::to_string(plan_->armWaitMs) + " ms; its output is in " + artifactDir_,
                     "bpftrace ran but did not see this process's arm thread go to sleep; see "
                     "offcpu.err.txt there, and run --profile-check --profile offcpu."));
    } else {
      finish(notSeen("no capture: the tracer armed for pid " + std::to_string(armed.ack.pid) +
                         " thread " + std::to_string(armed.ack.tid) +
                         ", not for this process's arm thread (pid " + std::to_string(PID) +
                         " thread " + std::to_string(armTid) + "); its output is in " +
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
  AckWait disarmed;
  long stopTid = 0;
  {
    const ThreadNameScope STOPPING(STOP_THREAD_NAME);
    stopTid = threadId();
    disarmed =
        waitForAck(CAPTURE->output, CAPTURE->tracer, DISARMED_LINE, true, plan_->disarmWaitMs);
  }
  // SIGINT makes bpftrace print its maps; the stop escalates through the
  // plan's route and reports every delivery it could not make.
  const HelperStopResult STOPPED = CAPTURE->tracer.stop();
  if (STOPPED.wasRunning) {
    (void)bpftrace_tool::reportStop("offcpu", WHAT, STOPPED, plan_->route);
  }
  const bool SUDO = plan_->route.privilege.route == PrivilegeRoute::SCOPED_SUDO;
  const StopDelivery* const REFUSED = refusedDelivery(STOPPED);
  const std::string KEPT = "; the capture in " + artifactDir_ + " is incomplete";
  const bool CLEAN_END = WIFEXITED(STOPPED.waitStatus) && WEXITSTATUS(STOPPED.waitStatus) == 0;
  if (disarmed.state == AckState::ENDED ||
      (!STOPPED.wasRunning && !(disarmed.state == AckState::ACKNOWLEDGED && CLEAN_END))) {
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
  if (!CLEAN_END) {
    finish(readinessResult(ReadinessCause::UNUSABLE,
                           "the tracer did not end cleanly after the stop (" +
                               statusText(STOPPED.waitStatus) + ")" + KEPT,
                           "bpftrace exits 0 once it has printed its maps; see offcpu.err.txt "
                           "there.",
                           CAPTURE_STAGE));
    return;
  }
  const std::string VALIDITY = "capture validity could not be established: ";
  if (disarmed.state == AckState::TIMED_OUT) {
    finish(readinessResult(
        ReadinessCause::UNUSABLE,
        VALIDITY + "the tracer did not acknowledge the stop within " +
            std::to_string(plan_->disarmWaitMs) + " ms; its output is in " + artifactDir_,
        "The tracer ran but did not see the stopping thread go to sleep; see offcpu.err.txt "
        "there.",
        CAPTURE_STAGE));
    return;
  }
  if (disarmed.ack.pid != PID || disarmed.ack.tid != stopTid) {
    finish(readinessResult(
        ReadinessCause::UNUSABLE,
        VALIDITY + "the tracer acknowledged the stop for pid " + std::to_string(disarmed.ack.pid) +
            " thread " + std::to_string(disarmed.ack.tid) +
            ", not for this process's stopping thread (pid " + std::to_string(PID) + " thread " +
            std::to_string(stopTid) + "); its output is in " + artifactDir_,
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
  const DumpCounts COUNTS = countsIn(DUMP.text);
  const long N = disarmed.ack.recorded;
  std::string cut;
  if (!COUNTS.wellFormed) {
    cut = "an @offcpu_blocks entry of its dump is cut short";
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
