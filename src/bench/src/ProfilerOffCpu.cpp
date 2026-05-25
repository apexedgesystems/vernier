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

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isBpftraceOnPath() { return std::system("command -v bpftrace >/dev/null 2>&1") == 0; }

#ifdef __linux__
constexpr const char* OFFCPU_SCRIPT = R"BT(
kprobe:finish_task_switch* /pid == $1/ {
  @start[tid] = nsecs;
}
kretprobe:finish_task_switch* /@start[tid]/ {
  @offcpu_ns[ustack, comm] = sum(nsecs - @start[tid]);
  delete(@start[tid]);
}
END {
  print(@offcpu_ns);
  clear(@offcpu_ns);
  clear(@start);
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
  if (geteuid() != 0) {
    std::fprintf(stderr, "\n[offcpu] not running as root; bpftrace cannot attach kprobes.\n"
                         "[offcpu] Re-run with sudo (or grant CAP_BPF) to collect off-CPU stacks.\n"
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
  const pid_t selfPid = ::getpid();
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
    std::string pidStr = std::to_string(selfPid);
    char arg0[] = "bpftrace";
    char argE[] = "-e";
    // The script must be writable for exec.
    std::string scriptBuf = OFFCPU_SCRIPT;
    ::execlp("bpftrace", arg0, argE, scriptBuf.c_str(), pidStr.c_str(),
             static_cast<char*>(nullptr));
    std::fprintf(stderr, "[offcpu] execlp(bpftrace) failed: %s\n", std::strerror(errno));
    std::_Exit(127);
  }
  // Parent: give bpftrace a moment to attach the probes before measurement.
  ::usleep(200 * 1000);
}

void OffCpuProfiler::stopBpftrace() {
  if (childPid_ <= 0)
    return;
  // SIGINT triggers bpftrace's END block, flushing the @offcpu_ns map.
  ::kill(childPid_, SIGINT);
  int status = 0;
  ::waitpid(childPid_, &status, 0);
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
    return EnvReport{EnvReport::Status::Warning, "bpftrace available but not running as root",
                     "Run with sudo or grant CAP_BPF; off-CPU kprobes need kernel privileges."};
  }
  return EnvReport{EnvReport::Status::Ok, "bpftrace available, running as root", ""};
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
