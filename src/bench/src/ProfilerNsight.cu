/**
 * @file ProfilerNsight.cu
 * @brief NVIDIA Nsight profiler integration for GPU benchmarks.
 */

#include "src/bench/inc/ProfilerNsight.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <thread>
#include <chrono>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

namespace vernier {
namespace bench {

NsightProfiler::NsightProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {

  if (!cfg_.artifactRoot.empty()) {
    artifactDir_ = cfg_.artifactRoot + "/" + testName_ + ".nsight";
  } else {
    artifactDir_ = "./" + testName_ + ".nsight";
  }

  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);

  if (cfg_.profileArgs.find("replay") != std::string::npos) {
    mode_ = NsightMode::ComputeReplay;
    useReplayMode_ = true;
  } else if (cfg_.profileArgs.find("ncu") != std::string::npos ||
             cfg_.profileArgs.find("compute") != std::string::npos) {
    mode_ = NsightMode::Compute;
  } else {
    mode_ = NsightMode::Systems;
  }
}

NsightProfiler::~NsightProfiler() { stopProfiler(); }

bool NsightProfiler::isNsysAvailable() const {
  return (std::system("command -v nsys >/dev/null 2>&1") == 0);
}

bool NsightProfiler::isNcuAvailable() const {
  return (std::system("command -v ncu >/dev/null 2>&1") == 0);
}

void NsightProfiler::beforeMeasure() {
  // Inside a container PID namespace, `nsys profile -p <pid>` / `ncu -p <pid>`
  // attach modes cannot reach this process reliably. Print the wrap-externally
  // hint and skip the attach attempt; the user runs nsys/ncu around the binary
  // instead (same pattern callgrind / compute-sanitizer use).
  if (profiler_env::isInContainer()) {
    const char* tool = (mode_ == NsightMode::Systems) ? "nsys" : "ncu";
    std::fprintf(stderr,
                 "\n[nsight] running inside a container; attach-by-pid is unreliable.\n"
                 "[nsight] Wrap externally instead:\n"
                 "[nsight]   %s profile -o %s/profile <this-binary> --profile nsight [...]\n"
                 "[nsight] Skipping internal attach for this run.\n\n",
                 tool, artifactDir_.c_str());
  } else if (mode_ == NsightMode::Systems && isNsysAvailable()) {
    launchNsys();
  } else if (mode_ == NsightMode::Compute && isNcuAvailable()) {
    launchNcu();
  } else if (mode_ == NsightMode::ComputeReplay && isNcuAvailable()) {
    launchNcuReplay();
  }
  // Auto-emit an NVTX range named after the test so the measured window
  // appears as a labeled region in the nsys timeline. Pop in afterMeasure.
#if COMPAT_NVTX_AVAILABLE
  nvtxRangePushA(testName_.c_str());
  nvtxRangePush_ = true;
#endif
}

void NsightProfiler::afterMeasure(const Stats& /*s*/) {
#if COMPAT_NVTX_AVAILABLE
  if (nvtxRangePush_) {
    nvtxRangePop();
    nvtxRangePush_ = false;
  }
#endif
  stopProfiler();

  if (useReplayMode_) {
    parseReplayMetrics();
  }
}

void NsightProfiler::launchNsys() {
  std::string outputPath = artifactDir_ + "/profile";
  std::string cmd = "nsys profile -o " + outputPath + " -t cuda,nvtx";

  if (!cfg_.profileArgs.empty()) {
    std::string args = cfg_.profileArgs;
    if (args.find("nsys") == 0) {
      args = args.substr(4);
    } else if (args.find("systems") == 0) {
      args = args.substr(7);
    }
    while (!args.empty() && (args.front() == ' ' || args.front() == '\t')) {
      args.erase(args.begin());
    }
    if (!args.empty()) {
      cmd += " " + args;
    }
  }

  pid_t targetPid = ::getpid();
  cmd += " -p " + std::to_string(targetPid);

  std::string stdoutPath = artifactDir_ + "/nsys.out.txt";
  std::string stderrPath = artifactDir_ + "/nsys.err.txt";

  std::string shellCmd = "sh -c '" + cmd + " >" + stdoutPath + " 2>" + stderrPath + " & echo $!'";

  FILE* pipe = ::popen(shellCmd.c_str(), "r");
  if (!pipe) {
    return;
  }

  std::array<char, 64> buf{};
  if (::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    childPid_ = static_cast<pid_t>(std::strtol(buf.data(), nullptr, 10));
  }
  ::pclose(pipe);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void NsightProfiler::launchNcu() {
  std::string outputPath = artifactDir_ + "/kernel_profile";
  std::string cmd = "ncu -o " + outputPath;

  if (!cfg_.profileArgs.empty()) {
    std::string args = cfg_.profileArgs;
    if (args.find("ncu") == 0) {
      args = args.substr(3);
    } else if (args.find("compute") == 0) {
      args = args.substr(7);
    }
    while (!args.empty() && (args.front() == ' ' || args.front() == '\t')) {
      args.erase(args.begin());
    }
    if (!args.empty()) {
      cmd += " " + args;
    }
  }

  cmd += " --target-processes all";

  pid_t targetPid = ::getpid();
  cmd += " -p " + std::to_string(targetPid);

  std::string stdoutPath = artifactDir_ + "/ncu.out.txt";
  std::string stderrPath = artifactDir_ + "/ncu.err.txt";

  std::string shellCmd = "sh -c '" + cmd + " >" + stdoutPath + " 2>" + stderrPath + " & echo $!'";

  FILE* pipe = ::popen(shellCmd.c_str(), "r");
  if (!pipe) {
    return;
  }

  std::array<char, 64> buf{};
  if (::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    childPid_ = static_cast<pid_t>(std::strtol(buf.data(), nullptr, 10));
  }
  ::pclose(pipe);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void NsightProfiler::launchNcuReplay() {
  std::string outputPath = artifactDir_ + "/kernel_replay";
  std::string cmd = "ncu --mode=launch-and-attach --replay-mode kernel";

  std::string metricsStr = replayMetrics_.toNcuMetricString();
  if (!metricsStr.empty()) {
    cmd += " --metrics " + metricsStr;
  }

  cmd += " -o " + outputPath;

  pid_t targetPid = ::getpid();
  cmd += " --target-processes all -p " + std::to_string(targetPid);

  std::string stdoutPath = artifactDir_ + "/ncu_replay.out.txt";
  std::string stderrPath = artifactDir_ + "/ncu_replay.err.txt";

  std::string shellCmd = "sh -c '" + cmd + " >" + stdoutPath + " 2>" + stderrPath + " & echo $!'";

  FILE* pipe = ::popen(shellCmd.c_str(), "r");
  if (!pipe) {
    return;
  }

  std::array<char, 64> buf{};
  if (::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    childPid_ = static_cast<pid_t>(std::strtol(buf.data(), nullptr, 10));
  }
  ::pclose(pipe);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void NsightProfiler::stopProfiler() {
  if (childPid_ <= 0) {
    return;
  }

  ::kill(childPid_, SIGINT);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  if (::kill(childPid_, 0) == 0) {
    ::kill(childPid_, SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  int status = 0;
  ::waitpid(childPid_, &status, WNOHANG);
  childPid_ = -1;

  // For Nsight Systems runs, auto-extract the four canonical reports
  // (kernel summary, API summary, mem size summary, mem time summary).
  // Reports are written next to the .nsys-rep so the user can grep them
  // without an extra cli round-trip.
  if (mode_ == NsightMode::Systems) {
    extractNsysStats();
  }
}

void NsightProfiler::extractNsysStats() {
  const std::string repPath = artifactDir_ + "/profile.nsys-rep";
  std::error_code ec;
  if (!std::filesystem::exists(repPath, ec)) {
    return; // nsys never produced a report (likely Docker attach failure)
  }
  static const char* const REPORTS[] = {
      "cuda_gpu_kern_sum",
      "cuda_api_sum",
      "cuda_gpu_mem_size_sum",
      "cuda_gpu_mem_time_sum",
  };
  for (const char* report : REPORTS) {
    const std::string outPath = artifactDir_ + "/" + report + ".txt";
    const std::string cmd = "nsys stats --report " + std::string(report) + " '" + repPath +
                            "' > '" + outPath + "' 2>/dev/null";
    [[maybe_unused]] int rc = std::system(cmd.c_str());
  }
  std::fprintf(stderr,
               "\n[nsight] auto-extracted nsys stats reports to %s:\n"
               "[nsight]   cuda_gpu_kern_sum.txt    -- per-kernel time distribution\n"
               "[nsight]   cuda_api_sum.txt         -- CUDA API call overhead\n"
               "[nsight]   cuda_gpu_mem_size_sum.txt -- H2D/D2H byte totals\n"
               "[nsight]   cuda_gpu_mem_time_sum.txt -- transfer time totals\n\n",
               artifactDir_.c_str());
}

void NsightProfiler::parseReplayMetrics() {
  std::string stdoutPath = artifactDir_ + "/ncu_replay.out.txt";

  std::ifstream in(stdoutPath);
  if (!in) {
    std::fprintf(stderr, "Warning: Could not open ncu replay output for parsing\n");
    return;
  }

  std::string line;
  std::printf("\n=== Kernel Replay Metrics (Nsight Compute) ===\n");

  while (std::getline(in, line)) {
    if (line.find("throughput") != std::string::npos ||
        line.find("occupancy") != std::string::npos ||
        line.find("efficiency") != std::string::npos || line.find("warps") != std::string::npos) {
      std::printf("%s\n", line.c_str());
    }
  }

  std::printf("\nFull report: %s\n", (artifactDir_ + "/kernel_replay.ncu-rep").c_str());
  std::printf("View with: ncu-ui %s\n", (artifactDir_ + "/kernel_replay.ncu-rep").c_str());
}

std::unique_ptr<Profiler> makeNsightProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<NsightProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

namespace vernier {
namespace bench {

EnvReport checkNsightEnvironment() {
  const bool nsys = std::system("command -v nsys >/dev/null 2>&1") == 0;
  const bool ncu  = std::system("command -v ncu  >/dev/null 2>&1") == 0;
  if (!nsys && !ncu) {
    return EnvReport{EnvReport::Status::Error,
                     "neither nsys nor ncu found on PATH",
                     "Install CUDA toolkit + Nsight (devtools repo on Ubuntu)."};
  }
  if (!nsys) {
    return EnvReport{EnvReport::Status::Warning,
                     "ncu present but nsys missing",
                     "Install nsight-systems-cli for timeline profiling."};
  }
  if (!ncu) {
    return EnvReport{EnvReport::Status::Warning,
                     "nsys present but ncu missing",
                     "Install nsight-compute for kernel analysis."};
  }
  // Both present; Docker PID namespace still breaks attach-by-pid (handled later).
  if (std::system("grep -q docker /proc/1/cgroup 2>/dev/null") == 0) {
    return EnvReport{EnvReport::Status::Warning,
                     "nsys + ncu available; running in Docker (PID namespace)",
                     "attach-by-pid will be replaced by direct nsys/ncu wrap."};
  }
  return EnvReport{EnvReport::Status::Ok, "nsys + ncu available", ""};
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND(
    "nsight",
    ::vernier::bench::makeNsightProfiler,
    ::vernier::bench::checkNsightEnvironment,
    "Install NVIDIA Nsight tools (nsys/ncu) and ensure a CUDA-capable GPU is visible.")