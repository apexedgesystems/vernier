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

// Whether compute-sanitizer started this process: decided as the shared
// helper decides it, from what the tool exports and maps, never from a name.
bool detectUnderSanitizer() { return profiler_env::isRunningUnderComputeSanitizer(); }

std::string sanitizerToolFromArgs(const std::string& profileArgs) {
  static const char* const TOOLS[] = {"memcheck", "racecheck", "synccheck", "initcheck"};
  for (const char* tool : TOOLS) {
    if (profileArgs.find(tool) != std::string::npos) {
      return tool;
    }
  }
  return "memcheck"; // default
}

// @p path with each '%' written "%%": compute-sanitizer expands %p, %q{VAR}
// and %% in its --log-file name and refuses any other '%'.
std::string escapePercent(const std::string& path) {
  std::string out;
  for (const char CH : path) {
    out += CH;
    if (CH == '%') {
      out += '%';
    }
  }
  return out;
}

// The mode argument the commands carry: none for memcheck, the default the
// registry checks; a tool asked for by name is passed on.
std::string toolArguments(const std::string& tool) {
  return tool == "memcheck" ? std::string() : " --profile-args " + tool;
}

// What the backend prints when the process is not under the tool: the ways
// to check it with the tool it was asked for. `bench run --profile
// compute-sanitizer` wraps the process with memcheck whatever --profile-args
// says, so it is offered for memcheck only, first, since it makes the folder
// its wrap logs into; any tool runs by hand. The tool opens its log before
// this program starts and drops it silently when the folder is missing, so
// the by-hand command makes the folder first.
std::string notWrappedHint(const std::string& tool, const std::string& artifactDir) {
  const char* const ROUTES =
      tool == "memcheck"
          ? "[compute-sanitizer]   bench run <this-binary> --profile compute-sanitizer -- [...]\n"
            "[compute-sanitizer] or by hand, making the folder first (the tool opens its log "
            "before this program starts):\n"
          : "[compute-sanitizer] by hand, making the folder first (the tool opens its log before "
            "this program starts):\n";
  return "\n[compute-sanitizer] not running under compute-sanitizer: this measurement runs "
         "unchecked. To check it:\n" +
         std::string(ROUTES) + "[compute-sanitizer]   mkdir -p " + artifactDir +
         " && compute-sanitizer --tool=" + tool + " --log-file=" + escapePercent(artifactDir) +
         "/sanitizer.log \\\n"
         "[compute-sanitizer]       <this-binary> --profile compute-sanitizer" +
         toolArguments(tool) + " [...]\n\n";
}

// What the backend prints when the process is under the tool, which reports
// at process exit into its --log-file, or on its stdout without one.
std::string wrappedNotice(const std::string& tool, const std::string& artifactDir) {
  return "[compute-sanitizer] tool=" + tool +
         " -- wrapping detected; compute-sanitizer reports at process exit, in its --log-file "
         "or on its stdout. Artifact directory: " +
         artifactDir + "\n";
}

} // namespace

/* ----------------------- ComputeSanitizerProfiler ----------------------- */

ComputeSanitizerProfiler::ComputeSanitizerProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  sanitizerTool_ = sanitizerToolFromArgs(cfg_.profileArgs);
  runningUnderSanitizer_ = detectUnderSanitizer();

  artifactDir_ = profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_,
                                                  "compute-sanitizer");
}

void ComputeSanitizerProfiler::beforeMeasure() {
  if (runningUnderSanitizer_) {
    std::fputs(wrappedNotice(sanitizerTool_, artifactDir_).c_str(), stderr);
    return;
  }
  // Not wrapped: print the exact invocations the user should run instead.
  // The parent is not re-executed here; that would surprise long-running
  // test binaries.
  std::fputs(notWrappedHint(sanitizerTool_, artifactDir_).c_str(), stderr);
}

void ComputeSanitizerProfiler::afterMeasure(const Stats& /*s*/) {
  // Nothing to do post-measure: compute-sanitizer reports at process exit when
  // it is wrapping the binary. When not wrapped, this backend is a no-op.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkComputeSanitizerEnvironment() {
  if (!isComputeSanitizerOnPath()) {
    return EnvReport{EnvReport::Status::Error, "compute-sanitizer not found on PATH",
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

VERNIER_REGISTER_PROFILER_BACKEND("compute-sanitizer",
                                  ::vernier::bench::makeComputeSanitizerProfiler,
                                  ::vernier::bench::checkComputeSanitizerEnvironment,
                                  "Install CUDA toolkit; compute-sanitizer ships with it.")
