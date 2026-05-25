/**
 * @file ProfilerHeaptrack.cpp
 * @brief Heaptrack heap-profiler backend implementation.
 *
 * Same wrap-externally pattern as callgrind / massif / compute-sanitizer:
 * the backend stays passive unless heaptrack's LD_PRELOAD has been injected,
 * in which case it just records the artifact path. When unwrapped, it prints
 * the precise heaptrack invocation including the per-test artifact subdir.
 */

#include "src/bench/inc/ProfilerHeaptrack.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isHeaptrackOnPath() { return std::system("command -v heaptrack >/dev/null 2>&1") == 0; }

/// heaptrack injects libheaptrack_preload.so via LD_PRELOAD; that env var is
/// the most reliable detection signal across distros.
bool detectUnderHeaptrack() {
  const char* preload = std::getenv("LD_PRELOAD");
  if (preload && std::strstr(preload, "heaptrack"))
    return true;
  // Alternative signal heaptrack >=1.4 sets:
  if (std::getenv("HEAPTRACK_OUTPUT") != nullptr)
    return true;
  return false;
}

} // namespace

/* ----------------------------- HeaptrackProfiler ----------------------------- */

HeaptrackProfiler::HeaptrackProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  runningUnderHeaptrack_ = detectUnderHeaptrack();
  artifactDir_ = cfg_.artifactRoot.empty() ? "./" + testName_ + ".heaptrack"
                                           : cfg_.artifactRoot + "/" + testName_ + ".heaptrack";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;
}

void HeaptrackProfiler::beforeMeasure() {
  if (runningUnderHeaptrack_) {
    std::fprintf(stderr,
                 "[heaptrack] wrapping detected; heap profile will be written at process\n"
                 "[heaptrack] exit. Artifact directory: %s\n",
                 artifactDir_.c_str());
    return;
  }
  std::fprintf(stderr,
               "\n[heaptrack] NOT running under heaptrack; this measurement will execute\n"
               "[heaptrack] normally but no heap profile is collected. To collect:\n"
               "[heaptrack]   heaptrack -o %s/run.heaptrack \\\n"
               "[heaptrack]       <this-binary> --profile heaptrack [...]\n"
               "[heaptrack] Then: heaptrack_print %s/run.heaptrack.zst | head -40\n\n",
               artifactDir_.c_str(), artifactDir_.c_str());
}

void HeaptrackProfiler::afterMeasure(const Stats& /*s*/) {
  // heaptrack writes its .zst file at process exit; nothing to do per-measure.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkHeaptrackEnvironment() {
  if (!isHeaptrackOnPath()) {
    return EnvReport{EnvReport::Status::Error, "heaptrack binary not found on PATH",
                     "apt install heaptrack (and optionally heaptrack-gui)."};
  }
  return EnvReport{EnvReport::Status::Ok, "heaptrack available", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeHeaptrackProfiler(const PerfConfig& cfg,
                                                const std::string& testName) {
  if (!isHeaptrackOnPath())
    return nullptr;
  return std::make_unique<HeaptrackProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("heaptrack", ::vernier::bench::makeHeaptrackProfiler,
                                  ::vernier::bench::checkHeaptrackEnvironment,
                                  "apt install heaptrack (low-overhead heap profiler).")
