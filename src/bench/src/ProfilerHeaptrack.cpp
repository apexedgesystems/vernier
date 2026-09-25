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

/// True when a shared object whose path contains `needle` is mapped into
/// this process.
bool isLibraryMapped(const char* needle) {
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp)
    return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strstr(line, needle)) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

/// Detect whether the current process is running under heaptrack.
///
/// Older heaptrack (~1.2) exported `LD_PRELOAD=...libheaptrack_preload.so`
/// to the child; newer releases (1.5+) inject the library via a GDB-style
/// hand-off so the child's environment is empty. `HEAPTRACK_OUTPUT` is
/// likewise inconsistent. The reliable cross-version signal is the mapped
/// preload library itself in `/proc/self/maps`, which is present whenever
/// heaptrack actually instrumented the process.
bool detectUnderHeaptrack() {
  const char* preload = std::getenv("LD_PRELOAD");
  if (preload && std::strstr(preload, "heaptrack"))
    return true;
  if (std::getenv("HEAPTRACK_OUTPUT") != nullptr)
    return true;
  return isLibraryMapped("libheaptrack_");
}

/// heaptrack's preload interposes the malloc family only. An allocator that
/// exports its own operator new (tcmalloc does) serves C++ allocations
/// without ever calling malloc, so they never reach the trace.
bool isOperatorNewReplaced() { return isLibraryMapped("libtcmalloc"); }

} // namespace

/* ----------------------------- HeaptrackProfiler ----------------------------- */

HeaptrackProfiler::HeaptrackProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  runningUnderHeaptrack_ = detectUnderHeaptrack();
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "heaptrack");
}

void HeaptrackProfiler::beforeMeasure() {
  if (isOperatorNewReplaced()) {
    std::fprintf(
        stderr,
        "\n[heaptrack] WARNING: libtcmalloc is loaded and replaces operator new.\n"
        "[heaptrack] heaptrack hooks the malloc family only, so C++ allocations will\n"
        "[heaptrack] be MISSING from the trace. Use a build without tcmalloc\n"
        "[heaptrack] (-DVERNIER_LINK_TCMALLOC=OFF, the default) and do not preload it.\n\n");
  }
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
               "[heaptrack] Then: heaptrack_print %s/run.heaptrack.* | head -40\n\n",
               artifactDir_.c_str(), artifactDir_.c_str());
}

void HeaptrackProfiler::afterMeasure(const Stats& /*s*/) {
  // heaptrack writes its trace at process exit; nothing to do per-measure.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkHeaptrackEnvironment() {
  if (!isHeaptrackOnPath()) {
    return EnvReport{EnvReport::Status::Error, "heaptrack binary not found on PATH",
                     "apt install heaptrack (and optionally heaptrack-gui)."};
  }
  // bench doctor runs this inside the benchmark binary (--profile-check), so
  // the maps checked here are those of the process heaptrack would record.
  if (isOperatorNewReplaced()) {
    return EnvReport{
        EnvReport::Status::Warning,
        "heaptrack available, but libtcmalloc is loaded: C++ allocations will be missing",
        "Use a build without tcmalloc (-DVERNIER_LINK_TCMALLOC=OFF, the default) and do not "
        "preload it."};
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
