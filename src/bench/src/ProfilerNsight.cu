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
  if (mode_ == NsightMode::Systems && isNsysAvailable()) {
    launchNsys();
  } else if (mode_ == NsightMode::Compute && isNcuAvailable()) {
    launchNcu();
  } else if (mode_ == NsightMode::ComputeReplay && isNcuAvailable()) {
    launchNcuReplay();
  }
}

void NsightProfiler::afterMeasure(const Stats& /*s*/) {
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