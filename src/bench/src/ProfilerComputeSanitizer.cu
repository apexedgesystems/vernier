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
  if (!p) return false;
  // The path usually ends in libcompute-sanitizer.so or similar.
  return std::strstr(p, "sanitizer") != nullptr || std::strstr(p, "Sanitizer") != nullptr;
}

std::string sanitizerToolFromArgs(const std::string& profileArgs) {
  static const char* const kTools[] = {"memcheck", "racecheck", "synccheck", "initcheck"};
  for (const char* tool : kTools) {
    if (profileArgs.find(tool) != std::string::npos) {
      return tool;
    }
  }
  return "memcheck"; // default
}

} // namespace

/* ----------------------- ComputeSanitizerProfiler ----------------------- */

ComputeSanitizerProfiler::ComputeSanitizerProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  sanitizerTool_ = sanitizerToolFromArgs(cfg_.profileArgs);
  runningUnderSanitizer_ = detectUnderSanitizer();

  // Artifact directory mirrors the convention used by other backends
  // (e.g. <Suite.Case>.compute-sanitizer/).
  artifactDir_ = cfg_.artifactRoot.empty()
                     ? testName_ + ".compute-sanitizer"
                     : cfg_.artifactRoot + "/" + testName_ + ".compute-sanitizer";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;
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

/* ----------------------------- Env check ----------------------------- */

EnvReport checkComputeSanitizerEnvironment() {
  if (!isComputeSanitizerOnPath()) {
    return EnvReport{EnvReport::Status::Error,
                     "compute-sanitizer not found on PATH",
                     "Install the CUDA toolkit; compute-sanitizer ships with it."};
  }
  return EnvReport{EnvReport::Status::Ok, "compute-sanitizer available", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeComputeSanitizerProfiler(const PerfConfig& cfg,
                                                       const std::string& testName) {
  if (!isComputeSanitizerOnPath()) {
    return nullptr;
  }
  return std::make_unique<ComputeSanitizerProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND(
    "compute-sanitizer",
    ::vernier::bench::makeComputeSanitizerProfiler,
    ::vernier::bench::checkComputeSanitizerEnvironment,
    "Install CUDA toolkit; compute-sanitizer ships with it.")
