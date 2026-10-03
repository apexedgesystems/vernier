/**
 * @file ProfilerComputeSanitizer.cu
 * @brief NVIDIA Compute Sanitizer backend implementation.
 */

#include "src/bench/inc/ProfilerComputeSanitizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerComputeSanitizerChecks.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------- File helpers ----------------------- */

namespace {

bool isComputeSanitizerOnPath() {
  return std::system("command -v compute-sanitizer >/dev/null 2>&1") == 0;
}

// Heuristic detection that this process is running under compute-sanitizer.
// compute-sanitizer injects a launch hook via this env var. Not guaranteed
// to be stable across CUDA releases, but reliable on 2025.x.
bool detectUnderSanitizer() {
  const char* p = std::getenv("CUDA_INJECTION64_PATH");
  if (p && (std::strstr(p, "sanitizer") != nullptr || std::strstr(p, "Sanitizer") != nullptr))
    return true;
  // Fallback: the injection library is mapped into the process whenever
  // compute-sanitizer is actually instrumenting. The env-var name/value has
  // drifted across CUDA releases, so scanning /proc/self/maps is the reliable
  // signal (and avoids a false "NOT running" hint when the auto-wrap ran).
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp)
    return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strstr(line, "sanitizer") || std::strstr(line, "Sanitizer")) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

} // namespace

/* ----------------------- ComputeSanitizerProfiler ----------------------- */

ComputeSanitizerProfiler::ComputeSanitizerProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // The tool, by the readiness check's own parser. The registry refuses a
  // request holding a word the parser does not take before it builds this;
  // built directly, the tool named decides (memcheck when none is).
  (void)parseSanitizerTool(cfg_.profileArgs, sanitizerTool_);
  runningUnderSanitizer_ = detectUnderSanitizer();

  artifactDir_ = profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_,
                                                  "compute-sanitizer");
}

void ComputeSanitizerProfiler::beforeMeasure() {
  if (runningUnderSanitizer_) {
    std::fprintf(stderr,
                 "[compute-sanitizer] tool=%s -- wrapping detected; errors will be reported on "
                 "stderr at process exit. Artifact directory: %s\n",
                 sanitizerTool_.c_str(), artifactDir_.c_str());
    return;
  }
  // Not wrapped: print the exact invocation the user should run instead.
  // We DO NOT re-exec the parent here; that would surprise long-running test
  // binaries. The friendly hint is more predictable.
  std::fprintf(stderr,
               "\n[compute-sanitizer] NOT running under compute-sanitizer; this measurement\n"
               "[compute-sanitizer] will execute normally but no checking happens. To check:\n"
               "[compute-sanitizer]   compute-sanitizer --tool=%s --log-file=%s/sanitizer.log \\\n"
               "[compute-sanitizer]       <this-binary> --profile compute-sanitizer "
               "--profile-args %s [...]\n\n",
               sanitizerTool_.c_str(), artifactDir_.c_str(), sanitizerTool_.c_str());
}

void ComputeSanitizerProfiler::afterMeasure(const Stats& /*s*/) {
  // Nothing to do post-measure: compute-sanitizer reports at process exit when
  // it is wrapping the binary. When not wrapped, this backend is a no-op.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeComputeSanitizerProfiler(const PerfConfig& cfg,
                                                       const std::string& testName) {
  if (!isComputeSanitizerOnPath()) {
    return nullptr;
  }
  return std::make_unique<ComputeSanitizerProfiler>(cfg, testName);
}

namespace {

/** @brief The backend for a request the readiness check let collect. */
std::unique_ptr<Profiler> makePlannedComputeSanitizerProfiler(const PerfConfig& cfg,
                                                              const std::string& testName,
                                                              const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::make_unique<ComputeSanitizerProfiler>(cfg, testName);
}

} // namespace

} // namespace bench
} // namespace vernier

// The check is libbench's (ProfilerComputeSanitizerChecks.cpp), which
// registers it with a passive profiler; this registration replaces that one
// with the backend in every build that has it.
VERNIER_REGISTER_READINESS_BACKEND("compute-sanitizer",
                                   ::vernier::bench::checkComputeSanitizerRequest,
                                   ::vernier::bench::makePlannedComputeSanitizerProfiler,
                                   "Install the CUDA toolkit; compute-sanitizer ships with it.",
                                   "NV_SANITIZER_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH")
