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

// Whether compute-sanitizer started this process: decided as the shared
// helper decides it, from what the tool exports and maps, never from a name.
bool detectUnderSanitizer() { return profiler_env::isRunningUnderComputeSanitizer(); }

using detail::escapePercent;
using detail::shellQuote;

// The mode argument the commands carry: none for memcheck, the default the
// registry checks; a tool asked for by name is passed on.
std::string toolArguments(const std::string& tool) {
  return tool == "memcheck" ? std::string() : " --profile-args " + tool;
}

// What the backend prints when the process is not under the tool: the ways
// to check it with the tool it was asked for. `bench run --profile
// compute-sanitizer` wraps the process with the tool --profile-args names, so
// it is offered first, with that request, since it makes the folder its wrap
// logs into; the tool also runs by hand. The tool opens its log before this
// program starts and drops it silently when the folder is missing, so the
// by-hand command makes the folder first. The folder and the log are one
// shell word each, quoted where they need it; the log's '%' is written "%%"
// for the tool before it is quoted for the shell.
std::string notWrappedHint(const std::string& tool, const std::string& artifactDir) {
  const std::string ROUTES =
      "[compute-sanitizer]   bench run <this-binary> --profile compute-sanitizer" +
      toolArguments(tool) +
      " -- [...]\n"
      "[compute-sanitizer] or by hand, making the folder first (the tool opens its log before "
      "this program starts):\n";
  return "\n[compute-sanitizer] not running under compute-sanitizer: this measurement runs "
         "unchecked. To check it:\n" +
         ROUTES + "[compute-sanitizer]   mkdir -p " + shellQuote(artifactDir) +
         " && compute-sanitizer --tool=" + tool +
         " --log-file=" + shellQuote(escapePercent(artifactDir) + "/sanitizer.log") +
         " \\\n"
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
