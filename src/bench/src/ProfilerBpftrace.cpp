/**
 * @file ProfilerBpftrace.cpp
 * @brief Implementation of bpftrace profiler backend.
 *
 * Holds the BpfConfig + BpfRunner internals behind an anonymous namespace so
 * the bpftrace integration does not export any extra symbols beyond
 * ProfilerBpftrace itself.
 */

#include "src/bench/inc/ProfilerBpftrace.hpp"

#include "src/bench/inc/ProfilerEnv.hpp"

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
// BPF Configuration
// ============================================================================

namespace { // Internal implementation details

struct BpfConfig {
  bool enabled = false;
  std::string scriptsDir = "src/bench/bpf";
  std::vector<std::string> scripts;
  std::string outputDir;
  std::string format = "text";
  bool requireSudo = true;
  int startGraceMs = 1000;
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
// BPF Script Specs
// ============================================================================

struct BpfSpec {
  std::string name;
  std::string scriptPath;
  std::vector<std::string> args;
};

// ============================================================================
// BPF Runner
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
    // Probe the actual capability: scoped sudoers grants authorize
    // bpftrace specifically, so `sudo -n true` would false-negative.
    return profiler_env::sudoBpftraceUsable();
  }

  bool start(const BpfSpec& spec, int pid) {
    if (!cfg_.enabled) {
      return false;
    }
    if (!canRun()) {
      std::fprintf(stderr,
                   "[bpftrace] cannot run %s: bpftrace not on PATH or no usable privileges.\n"
                   "[bpftrace] Re-run with sudo, or set BENCH_SUDO=1 with a scoped sudoers\n"
                   "[bpftrace] grant (bpftrace + kill).\n",
                   spec.name.c_str());
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
      // The script was not found under cfg_.scriptsDir. The default
      // (src/bench/bpf/) resolves when running from the vernier tree; a
      // consumer points --bpf-scripts / PERF_BPF_SCRIPTS at its own copy.
      // Silently skipping looks like the profile ran fine, so be loud.
      std::fprintf(stderr,
                   "[bpftrace] script not found: %s\n"
                   "[bpftrace] Pass `--bpf <script>` with a script name that exists under\n"
                   "[bpftrace] `--bpf-scripts <dir>` (default: src/bench/bpf/),\n"
                   "[bpftrace] or pass an absolute path via `--bpf </path/to/script.bt>`.\n",
                   scriptPath.c_str());
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

    viaSudo_ = cfg_.requireSudo && ::geteuid() != 0;

    // fork+exec, not popen/sh-backgrounding: signals must land on the
    // process we actually track. A shell's $! hands back a wrapper pid,
    // and SIGKILL on a sudo wrapper merely orphans the root tracer under
    // it. A direct child receives delivery for real and waitpid can reap
    // it. (Same proven pattern as the offcpu backend.)
    const pid_t CHILD = ::fork();
    if (CHILD < 0) {
      return false;
    }
    if (CHILD == 0) {
      std::freopen(stdoutPath_.c_str(), "w", stdout);
      std::freopen(stderrPath_.c_str(), "w", stderr);
      const bool JSON = (cfg_.format == "json");
      if (viaSudo_) {
        if (JSON) {
          ::execlp("sudo", "sudo", "-n", "bpftrace", "-q", "-f", "json", tempScript_.c_str(),
                   static_cast<char*>(nullptr));
        } else {
          ::execlp("sudo", "sudo", "-n", "bpftrace", "-q", tempScript_.c_str(),
                   static_cast<char*>(nullptr));
        }
      } else {
        if (JSON) {
          ::execlp("bpftrace", "bpftrace", "-q", "-f", "json", tempScript_.c_str(),
                   static_cast<char*>(nullptr));
        } else {
          ::execlp("bpftrace", "bpftrace", "-q", tempScript_.c_str(), static_cast<char*>(nullptr));
        }
      }
      std::_Exit(127);
    }
    // One tracer per started script: a single slot would leak every tracer
    // but the last when a spec list runs several scripts back to back.
    children_.push_back(CHILD);

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
    // Under the sudo prefix tracers run as root: a plain ::kill from an
    // unprivileged caller EPERMs silently, losing the flush and leaking the
    // tracer past the run. Route delivery through `sudo -n kill` and
    // escalate SIGINT -> SIGTERM -> SIGKILL with bounded reaps per tracer;
    // the tracers are direct fork children, so waitpid works regardless of
    // their uid.
    for (const pid_t CHILD : children_) {
      const pid_t TARGET = profiler_env::tracerPid(CHILD);
      const auto DELIVER = [this, TARGET](int sig) {
        return viaSudo_ ? profiler_env::sudoKill(TARGET, sig) : (::kill(TARGET, sig) == 0);
      };
      const auto WAIT_GONE = [CHILD](int ms) {
        int status = 0;
        for (int i = 0; i < ms / 50; ++i) {
          if (::waitpid(CHILD, &status, WNOHANG) != 0)
            return true;
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return ::waitpid(CHILD, &status, WNOHANG) != 0;
      };
      if (!DELIVER(SIGINT)) {
        std::fprintf(stderr,
                     "[bpftrace] could not signal tracer %d (sudo -n kill "
                     "unavailable?); output may be incomplete.\n",
                     static_cast<int>(CHILD));
      }
      if (WAIT_GONE(cfg_.stopGraceMs * 10))
        continue;
      DELIVER(SIGTERM);
      if (WAIT_GONE(cfg_.stopGraceMs * 5))
        continue;
      std::fprintf(stderr, "[bpftrace] tracer %d unresponsive; sending SIGKILL.\n",
                   static_cast<int>(CHILD));
      DELIVER(SIGKILL);
      (void)WAIT_GONE(cfg_.stopGraceMs * 5);
    }
    children_.clear();
  }

  ~BpfRunner() { stop(); }

  [[nodiscard]] std::string stdoutPath() const { return stdoutPath_; }
  [[nodiscard]] std::string stderrPath() const { return stderrPath_; }

private:
  BpfConfig cfg_{};
  std::vector<pid_t> children_{};
  bool viaSudo_ = false;
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

namespace vernier {
namespace bench {

EnvReport checkBpftraceEnvironment() {
#ifdef __linux__
  if (std::system("command -v bpftrace >/dev/null 2>&1") != 0) {
    return EnvReport{EnvReport::Status::Error, "bpftrace binary not found on PATH",
                     "apt install bpftrace."};
  }
  if (geteuid() != 0) {
    if (profiler_env::benchSudoActive() && profiler_env::sudoBpftraceUsable()) {
      if (!profiler_env::bpftraceKprobeViable(true)) {
        return EnvReport{EnvReport::Status::Warning,
                         "sudo -n bpftrace runs but cannot attach a kprobe script",
                         "Check tracefs (mount -t tracefs tracefs /sys/kernel/tracing) and "
                         "that the bpftrace build is not broken (stripped BEGIN/END)."};
      }
      return EnvReport{EnvReport::Status::Ok,
                       "bpftrace available via BENCH_SUDO (kprobe attach verified)", ""};
    }
    return EnvReport{EnvReport::Status::Warning, "bpftrace available but not running as root",
                     "Run with sudo, or set BENCH_SUDO=1 with a scoped sudoers grant "
                     "(bpftrace + kill)."};
  }
  if (!profiler_env::bpftraceKprobeViable(false)) {
    return EnvReport{EnvReport::Status::Warning,
                     "bpftrace runs as root but cannot attach a kprobe script",
                     "Check tracefs (mount -t tracefs tracefs /sys/kernel/tracing)."};
  }
  return EnvReport{EnvReport::Status::Ok, "bpftrace available, running as root (attach verified)",
                   ""};
#else
  return EnvReport{EnvReport::Status::Error, "bpftrace is Linux-only",
                   "Run on Linux or use a different profiler."};
#endif
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("bpftrace", ::vernier::bench::makeBpftraceProfiler,
                                  ::vernier::bench::checkBpftraceEnvironment,
                                  "Install bpftrace and run with root/sudo.")