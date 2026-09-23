/**
 * @file ProfilerOffCpu.cpp
 * @brief Off-CPU profiling via bpftrace on the sched tracepoints.
 */

#include "src/bench/inc/ProfilerOffCpu.hpp"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
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
// At switch-out the current task IS the thread being descheduled, so its
// user stack is the blocking site; prev_state != 0 keeps only genuine
// blocks (preemption excluded). Wait time joins at switch-in through the
// per-tid start map. No END block on purpose: bpftrace auto-prints every
// map at exit (SIGINT, exit(), or target death via the self-exit probe),
// and distro builds are often stripped, which breaks BEGIN/END trigger
// symbols outright. Consumers read @offcpu_blocks / @offcpu_ns.
constexpr const char* OFFCPU_SCRIPT = R"BT(
tracepoint:sched:sched_switch /pid == $1 && args->prev_state != 0/ {
  @start[args->prev_pid] = nsecs;
  @offcpu_blocks[ustack, comm] = count();
}
tracepoint:sched:sched_switch /@start[args->next_pid]/ {
  @offcpu_ns[args->next_pid] = sum(nsecs - @start[args->next_pid]);
  delete(@start[args->next_pid]);
}
tracepoint:sched:sched_process_exit /pid == $1/ {
  exit();
}
)BT";

// Older, stripped bpftrace builds take on the order of a second to arm
// their probes; a short grace silently yields empty maps.
constexpr int START_GRACE_MS = 1500;
constexpr int INTERRUPT_WAIT_MS = 4000;
constexpr int TERMINATE_WAIT_MS = 2000;
constexpr int KILL_WAIT_MS = 2000;
const char* const WHAT = "the off-CPU script";

/** @brief `-e <script> <pid>`: the arguments every launch passes; $1 binds the pid. */
std::vector<std::string> launchArgs(const std::string& script, long pid) {
  return {"-e", script, std::to_string(pid)};
}

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
  // The launch's own argument shape, on a script that also exits by itself.
  const ProbeScratch SCRATCH(ctx);
  auto verdict = bpftrace_tool::probeAttach(
      plan->route,
      launchArgs(std::string{OFFCPU_SCRIPT} + "interval:s:5 { exit(); }\n",
                 static_cast<long>(ctx.self())),
      WHAT, START_GRACE_MS, ctx, SCRATCH.path());
  if (verdict && verdict->report.status == EnvReport::Status::Error) {
    return *verdict;
  }
  std::string message = std::string{WHAT} + " attached for " + std::to_string(START_GRACE_MS) +
                        " ms " + plan->route.describe() + " and stopped on SIGINT (probe with " +
                        plan->route.bpftrace + ")";
  if (plan->route.privilege.route == PrivilegeRoute::ALREADY_ROOT && plan->route.privilege.optIn) {
    message += "; running as root; BENCH_SUDO not needed";
  }
  ReadinessResult result;
  if (verdict) {
    result = *verdict;
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

OffCpuProfiler::OffCpuProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  const ReadinessResult DECISION = decideNow(cfg_);
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    std::fprintf(stderr, "[offcpu] not started: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[offcpu] %s\n", DECISION.report.hint.c_str());
    }
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
  const std::vector<std::string> ARGS = launchArgs(OFFCPU_SCRIPT, static_cast<long>(::getpid()));
  std::vector<std::string> argv = plan_->route.command();
  argv.insert(argv.end(), ARGS.begin(), ARGS.end());
  // The capture files are opened by the child as the invoking user, so they
  // stay the user's whatever the tracer runs as.
  helper_ = std::make_unique<OwnedHelper>(
      plan_->route.stopPolicy(INTERRUPT_WAIT_MS, TERMINATE_WAIT_MS, KILL_WAIT_MS, plan_->context));
  const HelperStart STARTED =
      helper_->start(argv, outputPath_, errorPath_, START_GRACE_MS, nullptr);
  if (!STARTED.started) {
    std::fprintf(stderr, "[offcpu] could not start bpftrace: %s\n", STARTED.errorTail.c_str());
    helper_.reset();
    return;
  }
  if (STARTED.exitedEarly) {
    std::string commandLine = plan_->route.bpftrace + " -e <script> " + ARGS.back();
    const ReadinessResult WHY = bpftrace_tool::classifyAttachFailure(
        plan_->route, WHAT, commandLine, STARTED.errorTail, *plan_->context);
    std::fprintf(stderr, "[offcpu] the tracer exited during its start grace: %s\n",
                 WHY.report.message.c_str());
    if (!WHY.report.hint.empty()) {
      std::fprintf(stderr, "[offcpu] %s\n", WHY.report.hint.c_str());
    }
    helper_.reset();
  }
}

void OffCpuProfiler::stopBpftrace() {
  if (!helper_) {
    return;
  }
  // SIGINT makes bpftrace print its maps; the stop escalates through the
  // plan's route and reports every delivery it could not make.
  const HelperStopResult STOPPED = helper_->stop();
  const bool FLUSHED = bpftrace_tool::reportStop("offcpu", WHAT, STOPPED, plan_->route);
  helper_.reset();
  if (FLUSHED) {
    std::fprintf(stderr, "[offcpu] stacks written to %s\n", outputPath_.c_str());
  }
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
