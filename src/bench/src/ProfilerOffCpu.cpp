/**
 * @file ProfilerOffCpu.cpp
 * @brief Off-CPU profiling via bpftrace attached to finish_task_switch.
 */

#include "src/bench/inc/ProfilerOffCpu.hpp"

#ifdef __linux__
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isBpftraceOnPath() { return std::system("command -v bpftrace >/dev/null 2>&1") == 0; }

#ifdef __linux__
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
#endif

} // namespace

/* ----------------------------- OffCpuProfiler ----------------------------- */

OffCpuProfiler::OffCpuProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  artifactDir_ = cfg_.artifactRoot.empty() ? "./" + testName_ + ".offcpu"
                                           : cfg_.artifactRoot + "/" + testName_ + ".offcpu";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;
  outputPath_ = artifactDir_ + "/offcpu.txt";
}

void OffCpuProfiler::beforeMeasure() {
#ifdef __linux__
  viaSudo_ = profiler_env::benchSudoActive();
  if (geteuid() != 0 && !viaSudo_) {
    std::fprintf(stderr, "\n[offcpu] not running as root; bpftrace cannot attach kprobes.\n"
                         "[offcpu] Re-run with sudo, or set BENCH_SUDO=1 with a scoped sudoers\n"
                         "[offcpu] grant for bpftrace and kill (tests then stay unprivileged and\n"
                         "[offcpu] only the probe tooling elevates).\n"
                         "[offcpu] Measurement will proceed without profiling.\n\n");
    return;
  }
  if (!isBpftraceOnPath()) {
    std::fprintf(stderr, "\n[offcpu] bpftrace not found on PATH; cannot collect off-CPU stacks.\n"
                         "[offcpu]   apt install bpftrace (or equivalent), then re-run.\n\n");
    return;
  }
  spawnBpftrace();
#endif
}

void OffCpuProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  stopBpftrace();
#endif
}

#ifdef __linux__

void OffCpuProfiler::spawnBpftrace() {
  const pid_t SELF_PID = ::getpid();
  childPid_ = ::fork();
  if (childPid_ < 0) {
    std::fprintf(stderr, "[offcpu] fork failed: %s\n", std::strerror(errno));
    childPid_ = -1;
    return;
  }
  if (childPid_ == 0) {
    // Child: redirect bpftrace stdout to artifact file, then exec.
    std::freopen(outputPath_.c_str(), "w", stdout);
    // bpftrace -e '<script>' $TARGET_PID
    // $1 inside the script is bound to the first positional argument
    // (the target PID), which lets the kprobe filter only this process.
    std::string pidStr = std::to_string(SELF_PID);
    char arg0[] = "bpftrace";
    char argE[] = "-e";
    // The script must be writable for exec.
    std::string scriptBuf = OFFCPU_SCRIPT;
    if (viaSudo_) {
      // BENCH_SUDO: elevate only the probe tool; the test process (and its
      // artifacts) stay owned by the invoking user.
      char sudo0[] = "sudo";
      char sudoN[] = "-n";
      ::execlp("sudo", sudo0, sudoN, arg0, argE, scriptBuf.c_str(), pidStr.c_str(),
               static_cast<char*>(nullptr));
    } else {
      ::execlp("bpftrace", arg0, argE, scriptBuf.c_str(), pidStr.c_str(),
               static_cast<char*>(nullptr));
    }
    std::fprintf(stderr, "[offcpu] execlp(bpftrace) failed: %s\n", std::strerror(errno));
    std::_Exit(127);
  }
  // Parent: give bpftrace time to arm its probes before measurement.
  // Older/stripped builds (0.14-era) take on the order of a second to
  // resolve wildcard kprobes; a short grace silently yields empty maps.
  // (A real readiness handshake is the ticketed follow-up.)
  ::usleep(1500 * 1000);
}

void OffCpuProfiler::stopBpftrace() {
  if (childPid_ <= 0)
    return;
  // SIGINT triggers bpftrace's END block, flushing the @offcpu_ns map.
  // Under BENCH_SUDO the child runs as root, so delivery goes through
  // `sudo -n kill` (a plain ::kill would EPERM and lose the flush).
  const pid_t TARGET = profiler_env::tracerPid(childPid_);
  const bool SIGNALED =
      viaSudo_ ? profiler_env::sudoKill(TARGET, SIGINT) : (::kill(TARGET, SIGINT) == 0);
  if (!SIGNALED) {
    std::fprintf(stderr, "[offcpu] could not signal bpftrace (sudo -n kill unavailable?); "
                         "output may be incomplete.\n");
  }
  // Bounded reap: never hang the test on a stuck tracer. The child is our
  // fork, so waitpid works regardless of its uid.
  int status = 0;
  for (int i = 0; i < 40; ++i) { // ~4s at 100ms
    if (::waitpid(childPid_, &status, WNOHANG) != 0)
      break;
    ::usleep(100 * 1000);
  }
  if (::waitpid(childPid_, &status, WNOHANG) == 0 && profiler_env::processAlive(childPid_)) {
    std::fprintf(stderr, "[offcpu] bpftrace did not exit after SIGINT; sending SIGTERM.\n");
    (void)(viaSudo_ ? profiler_env::sudoKill(TARGET, SIGTERM) : (::kill(TARGET, SIGTERM) == 0));
    for (int i = 0; i < 20; ++i) { // ~2s
      if (::waitpid(childPid_, &status, WNOHANG) != 0)
        break;
      ::usleep(100 * 1000);
    }
  }
  // Last resort: SIGKILL always lands (sudo -n kill as root), and the
  // final reap stays bounded -- a stuck tracer must never hang a test.
  if (::waitpid(childPid_, &status, WNOHANG) == 0 && profiler_env::processAlive(childPid_)) {
    std::fprintf(stderr, "[offcpu] bpftrace unresponsive; sending SIGKILL.\n");
    (void)(viaSudo_ ? profiler_env::sudoKill(TARGET, SIGKILL) : (::kill(TARGET, SIGKILL) == 0));
    for (int i = 0; i < 20; ++i) {
      if (::waitpid(childPid_, &status, WNOHANG) != 0)
        break;
      ::usleep(100 * 1000);
    }
  }
  childPid_ = -1;
  std::fprintf(stderr, "[offcpu] stacks written to %s\n", outputPath_.c_str());
}

#endif // __linux__

/* ----------------------------- Env check ----------------------------- */

EnvReport checkOffCpuEnvironment() {
#ifdef __linux__
  if (!isBpftraceOnPath()) {
    return EnvReport{EnvReport::Status::Error, "bpftrace not found on PATH",
                     "apt install bpftrace."};
  }
  if (geteuid() != 0) {
    if (profiler_env::benchSudoActive() && profiler_env::sudoBpftraceUsable()) {
      if (!profiler_env::bpftraceAttachViable(true)) {
        return EnvReport{EnvReport::Status::Warning,
                         "sudo -n bpftrace runs but cannot attach a tracepoint script",
                         "Check tracefs (mount -t tracefs tracefs /sys/kernel/tracing) and "
                         "that the bpftrace build is not broken (stripped BEGIN/END)."};
      }
      return EnvReport{EnvReport::Status::Ok,
                       "bpftrace available via BENCH_SUDO (tracepoint attach verified)", ""};
    }
    return EnvReport{EnvReport::Status::Warning, "bpftrace available but not running as root",
                     "Run with sudo, or set BENCH_SUDO=1 with a scoped sudoers grant "
                     "(bpftrace + kill)."};
  }
  if (!profiler_env::bpftraceAttachViable(false)) {
    return EnvReport{EnvReport::Status::Warning,
                     "bpftrace runs as root but cannot attach a tracepoint script",
                     "Check tracefs (mount -t tracefs tracefs /sys/kernel/tracing)."};
  }
  return EnvReport{EnvReport::Status::Ok, "bpftrace available, running as root (attach verified)",
                   ""};
#else
  return EnvReport{EnvReport::Status::Error, "off-CPU profiling is Linux-only",
                   "Run on Linux or use a different profiler."};
#endif
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeOffCpuProfiler(const PerfConfig& cfg, const std::string& testName) {
#ifdef __linux__
  if (!isBpftraceOnPath())
    return nullptr;
  return std::make_unique<OffCpuProfiler>(cfg, testName);
#else
  (void)cfg;
  (void)testName;
  return nullptr;
#endif
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("offcpu", ::vernier::bench::makeOffCpuProfiler,
                                  ::vernier::bench::checkOffCpuEnvironment,
                                  "apt install bpftrace and run with sudo (CAP_BPF).")
