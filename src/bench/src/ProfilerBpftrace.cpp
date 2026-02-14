/**
 * @file ProfilerBpftrace.cpp
 * @brief Implementation of bpftrace profiler backend.
 *
 * This file consolidates the entire BPF subsystem (formerly BpfConfig, BpfProbes, BpfRunner)
 * into a single implementation file, as these are implementation details of the bpftrace profiler.
 */

#include "src/bench/inc/ProfilerBpftrace.hpp"

#ifdef __linux__
#include <array>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <chrono>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace vernier {
namespace bench {

#ifdef __linux__
// ============================================================================
// BPF Configuration (formerly BpfConfig.hpp)
// ============================================================================

namespace { // Internal implementation details

struct BpfConfig {
  bool enabled = false;
  std::string scriptsDir = "src/utilities/benchmarking/bpf";
  std::vector<std::string> scripts;
  std::string outputDir;
  std::string format = "text";
  bool requireSudo = true;
  int startGraceMs = 100;
  int stopGraceMs = 200;
};

void populateFromEnv(BpfConfig& cfg) {
  if (const char* v = std::getenv("PERF_BPF")) {
    std::string val = v;
    for (char& ch : val) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    cfg.enabled = (val == "1" || val == "true");
  }
  if (const char* v = std::getenv("PERF_BPF_SCRIPTS")) {
    cfg.scriptsDir = v;
  }
  if (const char* v = std::getenv("PERF_BPF_OUT")) {
    cfg.outputDir = v;
  }
  if (const char* v = std::getenv("PERF_BPF_FMT")) {
    cfg.format = v;
  }
  if (const char* v = std::getenv("PERF_BPF_SUDO")) {
    std::string val = v;
    for (char& ch : val) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    cfg.requireSudo = !(val == "0" || val == "false");
  }
}

void populateFromPerfConfig(BpfConfig& out, const PerfConfig& perf, const std::string& testName) {
  if (perf.profileTool == "bpftrace") {
    out.enabled = true;
  }

  if (!perf.bpfScripts.empty()) {
    out.scripts = perf.bpfScripts;
  } else if (out.scripts.empty()) {
    out.scripts = {"write_latency", "fsync_latency"};
  }

  if (!perf.artifactRoot.empty()) {
    out.outputDir = perf.artifactRoot + "/" + testName + ".bpf";
  }
}

// ============================================================================
// BPF Script Specs (formerly BpfProbes.hpp - simplified)
// ============================================================================

struct BpfSpec {
  std::string name;
  std::string scriptPath;
  std::vector<std::string> args;
};

// ============================================================================
// BPF Runner (formerly BpfRunner.hpp)
// ============================================================================

class BpfRunner {
public:
  explicit BpfRunner(BpfConfig cfg) : cfg_(std::move(cfg)) {}

  [[nodiscard]] bool available() const noexcept {
    return (std::system("command -v bpftrace >/dev/null 2>&1") == 0);
  }

  [[nodiscard]] bool canRun() const noexcept {
    if (!available()) {
      return false;
    }
    if (!cfg_.requireSudo) {
      return true;
    }
    if (::geteuid() == 0) {
      return true;
    }
    return (std::system("sudo -n true >/dev/null 2>&1") == 0);
  }

  bool start(const BpfSpec& spec, int pid) {
    if (!cfg_.enabled) {
      return false;
    }
    if (!canRun()) {
      return false;
    }

    std::filesystem::path outdir = cfg_.outputDir.empty() ? std::filesystem::current_path()
                                                          : std::filesystem::path(cfg_.outputDir);
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);

    std::filesystem::path scriptPath =
        spec.scriptPath.empty() ? (std::filesystem::path(cfg_.scriptsDir) / (spec.name + ".bt"))
                                : std::filesystem::path(spec.scriptPath);

    std::ifstream in(scriptPath);
    if (!in) {
      return false;
    }
    std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Replace {{PID}}
    std::string pidToken = "{{PID}}";
    std::string pidString = std::to_string(pid);
    for (std::size_t pos = 0; (pos = src.find(pidToken, pos)) != std::string::npos;
         pos += pidString.size()) {
      src.replace(pos, pidToken.size(), pidString);
    }

    tempScript_ = (outdir / (spec.name + ".tmp.bt")).string();
    {
      std::ofstream out(tempScript_);
      out << src;
    }

    stdoutPath_ = (outdir / (spec.name + ".out." + cfg_.format)).string();
    stderrPath_ = (outdir / (spec.name + ".err.txt")).string();

    std::string cmd = "bpftrace -q '" + tempScript_ + "'";
    if (cfg_.format == "json") {
      cmd += " -f json";
    }
    if (cfg_.requireSudo && ::geteuid() != 0) {
      cmd = std::string("sudo -n ") + cmd;
    }

    std::string shellCmd =
        "sh -c '" + cmd + " >" + stdoutPath_ + " 2>" + stderrPath_ + " & echo $!'";
    FILE* pipe = ::popen(shellCmd.c_str(), "r");
    if (!pipe) {
      return false;
    }

    std::array<char, 64> buf{};
    if (!::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
      ::pclose(pipe);
      return false;
    }
    ::pclose(pipe);

    childPid_ = static_cast<pid_t>(std::strtol(buf.data(), nullptr, 10));
    if (childPid_ <= 0) {
      childPid_ = -1;
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.startGraceMs));
    return true;
  }

  bool startByName(const std::string& scriptName, int pid) {
    BpfSpec spec;
    spec.name = scriptName;
    spec.scriptPath.clear();
    return start(spec, pid);
  }

  void stop() noexcept {
    if (childPid_ <= 0) {
      return;
    }
    ::kill(childPid_, SIGINT);
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.stopGraceMs));
    if (::kill(childPid_, 0) == 0) {
      ::kill(childPid_, SIGTERM);
    }
    int status = 0;
    (void)::waitpid(childPid_, &status, WNOHANG);
    childPid_ = -1;
  }

  ~BpfRunner() { stop(); }

  [[nodiscard]] std::string stdoutPath() const { return stdoutPath_; }
  [[nodiscard]] std::string stderrPath() const { return stderrPath_; }

private:
  BpfConfig cfg_{};
  pid_t childPid_ = -1;
  std::string tempScript_{};
  std::string stdoutPath_{};
  std::string stderrPath_{};
};

} // anonymous namespace

// ============================================================================
// BpftraceProfiler Implementation
// ============================================================================

class BpftraceProfiler::Impl {
public:
  Impl(const PerfConfig& cfg, const std::string& testName) : cfg_(cfg), testName_(testName) {
    populateFromEnv(bpfCfg_);
    populateFromPerfConfig(bpfCfg_, cfg_, testName_);

    if (bpfCfg_.outputDir.empty()) {
      artifactDir_ = "./" + testName_ + ".bpf";
    } else {
      artifactDir_ = bpfCfg_.outputDir;
    }
    std::error_code ec;
    std::filesystem::create_directories(artifactDir_, ec);
  }

  void beforeMeasure() {
    if (!bpfCfg_.enabled) {
      return;
    }

    pid_t targetPid = ::getpid();
    for (const auto& name : bpfCfg_.scripts) {
      BpfConfig perScriptCfg = bpfCfg_;
      perScriptCfg.outputDir = artifactDir_;
      auto runner = std::make_unique<BpfRunner>(std::move(perScriptCfg));
      if (runner->startByName(name, static_cast<int>(targetPid))) {
        runners_.push_back(std::move(runner));
      }
    }
  }

  void afterMeasure() {
    for (auto& r : runners_) {
      r->stop();
    }
    runners_.clear();
  }

  std::string artifactDir() const { return artifactDir_; }

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  BpfConfig bpfCfg_{};
  std::vector<std::unique_ptr<BpfRunner>> runners_;
};

#endif // __linux__

// ============================================================================
// Public Interface Implementation
// ============================================================================

BpftraceProfiler::BpftraceProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
#ifdef __linux__
  impl_ = std::make_unique<Impl>(cfg_, testName_);
  artifactDir_ = impl_->artifactDir();
#else
  (void)cfg_;
  (void)testName_;
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

// Factory implementation
std::unique_ptr<Profiler> makeBpftraceProfiler(const PerfConfig& cfg, const std::string& testName) {
#ifdef __linux__
  return std::make_unique<BpftraceProfiler>(cfg, testName);
#else
  (void)cfg;
  (void)testName;
  return nullptr;
#endif
}

} // namespace bench
} // namespace vernier