/**
 * @file ProfilerRocprof.cpp
 * @brief AMD ROCm rocprof backend implementation.
 *
 * rocprof wraps the binary externally; the backend itself stays passive
 * unless detected wrapping, mirroring compute-sanitizer and callgrind.
 */

#include "src/bench/inc/ProfilerRocprof.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isRocprofOnPath() { return std::system("command -v rocprof >/dev/null 2>&1") == 0; }

bool isRocmRuntimePresent() {
  // /opt/rocm/lib/libhsa-runtime64.so is the canonical install marker;
  // /sys/class/kfd/kfd/topology/nodes is the kernel-driver marker. Either
  // is good enough to claim "AMD GPU stack is installed on this host".
  if (std::FILE* f = std::fopen("/opt/rocm/lib/libhsa-runtime64.so", "rb")) {
    std::fclose(f);
    return true;
  }
  if (std::FILE* f = std::fopen("/sys/class/kfd/kfd/topology/nodes", "rb")) {
    std::fclose(f);
    return true;
  }
  return false;
}

// rocprof sets ROCP_TOOL_LIB / ROCPROFILER_LIBRARY to inject its tracer
// before the process starts. Use either env signal as the "wrapping detected"
// marker; both are stable across rocprof v1 and v2.
bool detectUnderRocprof() {
  if (std::getenv("ROCP_TOOL_LIB") != nullptr)
    return true;
  if (std::getenv("ROCPROFILER_LIBRARY") != nullptr)
    return true;
  const char* preload = std::getenv("LD_PRELOAD");
  return preload && std::strstr(preload, "rocprof") != nullptr;
}

std::string modeFromArgs(const std::string& args) {
  if (args.find("stats") != std::string::npos)
    return "stats";
  if (args.find("hsa-trace") != std::string::npos)
    return "hsa-trace";
  if (args.find("hip-trace") != std::string::npos)
    return "hip-trace";
  return "default";
}

std::string modeFlag(const std::string& mode) {
  if (mode == "stats")
    return "--stats";
  if (mode == "hsa-trace")
    return "--hsa-trace";
  if (mode == "hip-trace")
    return "--hip-trace";
  return ""; // default: no extra flag
}

} // namespace

/* ----------------------------- RocprofProfiler ----------------------------- */

RocprofProfiler::RocprofProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  mode_ = modeFromArgs(cfg_.profileArgs);
  runningUnderRocprof_ = detectUnderRocprof();

  artifactDir_ = cfg_.artifactRoot.empty() ? testName_ + ".rocprof"
                                           : cfg_.artifactRoot + "/" + testName_ + ".rocprof";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;
}

void RocprofProfiler::beforeMeasure() {
  if (runningUnderRocprof_) {
    std::fprintf(stderr,
                 "[rocprof] mode=%s -- wrapping detected; reports written by rocprof at\n"
                 "[rocprof] process exit. Artifact directory: %s\n",
                 mode_.c_str(), artifactDir_.c_str());
    return;
  }
  // Not wrapped: print the precise rocprof invocation, matching the
  // hint pattern the other wrap-externally backends use
  // (compute-sanitizer / callgrind / nsight).
  const std::string FLAG = modeFlag(mode_);
  std::fprintf(stderr,
               "\n[rocprof] NOT running under rocprof; this measurement will execute\n"
               "[rocprof] normally but no profile is collected. To collect:\n"
               "[rocprof]   rocprof%s%s -o %s/results.csv \\\n"
               "[rocprof]       <this-binary> --profile rocprof --profile-args %s [...]\n\n",
               FLAG.empty() ? "" : " ", FLAG.c_str(), artifactDir_.c_str(), mode_.c_str());
}

void RocprofProfiler::afterMeasure(const Stats& /*s*/) {
  // rocprof writes results.{csv,json} at process exit when wrapping; nothing
  // to do per-measure on the in-process side.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkRocprofEnvironment() {
  const bool TOOL = isRocprofOnPath();
  const bool RUNTIME = isRocmRuntimePresent();
  if (!TOOL && !RUNTIME) {
    return EnvReport{EnvReport::Status::Error,
                     "ROCm not detected (no rocprof on PATH, no /opt/rocm)",
                     "Install ROCm + roctracer (https://rocm.docs.amd.com)."};
  }
  if (!TOOL) {
    return EnvReport{EnvReport::Status::Error, "ROCm runtime present but rocprof binary missing",
                     "apt install rocprofiler (or your distro's equivalent)."};
  }
  if (!RUNTIME) {
    return EnvReport{EnvReport::Status::Warning,
                     "rocprof present but no ROCm runtime / GPU kernel driver detected",
                     "rocprof will run but cannot attach to AMD GPUs on this host."};
  }
  return EnvReport{EnvReport::Status::Ok, "rocprof + ROCm runtime available", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeRocprofProfiler(const PerfConfig& cfg, const std::string& testName) {
  if (!isRocprofOnPath()) {
    return nullptr;
  }
  return std::make_unique<RocprofProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND(
    "rocprof", ::vernier::bench::makeRocprofProfiler, ::vernier::bench::checkRocprofEnvironment,
    "Install ROCm + rocprof (apt install rocprofiler on Debian/Ubuntu).")
