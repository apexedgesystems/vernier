/**
 * @file ProfilerHeaptrack.cpp
 * @brief Heaptrack heap-profiler backend implementation.
 *
 * heaptrack records the whole process when it runs it; the backend only
 * reports where the artifacts belong and warns when an allocator hides C++
 * allocations from it. Whether heaptrack runs the process is the readiness
 * check's decision, read from the process's memory map.
 */

#include "src/bench/inc/ProfilerHeaptrack.hpp"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

namespace vernier {
namespace bench {

namespace {

/** @brief The memory map of process @p pid, or "" when it cannot be read. */
std::string mapsOf(pid_t pid) {
  std::ifstream maps("/proc/" + std::to_string(static_cast<long>(pid)) + "/maps");
  std::stringstream text;
  text << maps.rdbuf();
  return text.str();
}

/// heaptrack's preload interposes the malloc family only. An allocator that
/// exports its own operator new (tcmalloc does) serves C++ allocations
/// without ever calling malloc, so they never reach the trace.
bool operatorNewReplaced(const std::string& mapsText) {
  return mapsText.find("libtcmalloc") != std::string::npos;
}

ReadinessResult heaptrackMissing() {
  return readinessResult(ReadinessCause::MISSING, "heaptrack not found on PATH",
                         "apt install heaptrack (and optionally heaptrack-gui).");
}

/** @brief The doctor's probe: heaptrack records /bin/true into a private directory. */
ReadinessResult probeHeaptrack(const ReadinessContext& ctx, std::string& heaptrackPath) {
  const auto TOOL = resolveExecutable("heaptrack", ctx);
  if (!TOOL) {
    return heaptrackMissing();
  }
  if (!TOOL->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, TOOL->path + " is not an executable file",
                           "Reinstall heaptrack, or fix PATH so it finds a working heaptrack.");
  }
  heaptrackPath = TOOL->path;
  const auto TRUE_PROGRAM = resolveExecutable("/bin/true", ctx);
  if (!TRUE_PROGRAM || !TRUE_PROGRAM->executable) {
    return readinessResult(ReadinessCause::MISSING_HELPER,
                           "/bin/true, the program heaptrack's probe records, is missing",
                           "Restore /bin/true (coreutils).");
  }
  const ProbeScratch SCRATCH(ctx);
  if (!SCRATCH.ok()) {
    return readinessResult(ReadinessCause::INTERNAL,
                           "no private directory could be created for heaptrack's probe",
                           "Check that TMPDIR (or /tmp) is writable.");
  }
  const std::string OUTPUT = SCRATCH.path() + "/probe";
  const std::vector<std::string> ARGV{TOOL->path, "-o", OUTPUT, TRUE_PROGRAM->path};
  const std::string COMMAND = TOOL->path + " -o <private directory>/probe " + TRUE_PROGRAM->path;
  const ProbeResult PROBE = runBoundedProbe(ARGV, HEAPTRACK_PROBE_TIMEOUT_MS, ctx);
  if (!PROBE.succeeded()) {
    const std::string TAIL = outputTail(PROBE.output);
    return readinessResult(ReadinessCause::UNUSABLE,
                           "heaptrack does not record /bin/true: " + COMMAND + ": " +
                               PROBE.describe() + (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Run heaptrack /bin/true by hand to see why; reinstall heaptrack if its "
                           "libraries are missing.");
  }
  // heaptrack appends the suffix of the compression it was built with.
  for (const char* SUFFIX : {".zst", ".gz"}) {
    std::error_code ec;
    if (std::filesystem::file_size(OUTPUT + SUFFIX, ec) > 0 && !ec) {
      return readinessResult(ReadinessCause::READY,
                             "heaptrack records /bin/true (probe: " + COMMAND +
                                 ", which wrote probe" + SUFFIX + ")",
                             "");
    }
  }
  return readinessResult(ReadinessCause::UNUSABLE,
                         "heaptrack exited 0 but wrote no trace: " + COMMAND,
                         "Run heaptrack /bin/true by hand to see why.");
}

ReadinessResult decideHeaptrack(const ReadinessRequest& request, const ReadinessContext& ctx,
                                const std::string* mapsText) {
  for (const std::string& WORD : valgrind_tool::modeWords(request.profileArgs)) {
    return valgrind_tool::refusedWord("heaptrack", WORD, {});
  }
  const std::string MAPS = mapsText != nullptr ? *mapsText : mapsOf(ctx.self());
  auto plan = std::make_shared<HeaptrackPlan>();
  ReadinessResult result;
  if (request.scope == ReadinessScope::RUNTIME) {
    if (!heaptrackMapped(MAPS)) {
      plan->launch = LaunchContext::NOT_WRAPPED;
      if (!resolveExecutable("heaptrack", ctx)) {
        result = heaptrackMissing();
      } else {
        result = readinessResult(
            ReadinessCause::MISSING,
            "heaptrack collects only when heaptrack runs the process, and heaptrack does not run "
            "this one",
            "Wrap it: heaptrack -o ./run <this-binary> --profile heaptrack [...]; or run it with "
            "bench run --profile heaptrack, which wraps it.");
      }
    } else {
      const bool BY_RUNNER = request.launch == LaunchContext::RUNNER_WRAPPED;
      plan->launch = BY_RUNNER ? LaunchContext::RUNNER_WRAPPED : LaunchContext::MANUALLY_WRAPPED;
      result = readinessResult(ReadinessCause::READY,
                               BY_RUNNER ? "heaptrack runs this process, under bench run's wrap"
                                         : "heaptrack runs this process, under a wrap started by "
                                           "hand",
                               "");
    }
  } else {
    result = probeHeaptrack(ctx, plan->heaptrack);
    // The doctor runs inside the benchmark binary (--profile-check), so its
    // map is the map of the process heaptrack would record.
    if (result.report.status == EnvReport::Status::Ok && operatorNewReplaced(MAPS)) {
      result = readinessResult(
          ReadinessCause::CAVEAT,
          "heaptrack records /bin/true, but libtcmalloc is loaded in this process: C++ "
          "allocations will be missing from its trace",
          "Use a build without tcmalloc (-DVERNIER_LINK_TCMALLOC=OFF, the default) and do not "
          "preload it.");
    }
  }
  result.plan = std::move(plan);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: heaptrack has no automatic analysis; the capture still "
                      "runs and its trace is kept",
                      "Read the trace with heaptrack_print (or heaptrack_gui) after the process "
                      "exits, and drop --profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

std::shared_ptr<const HeaptrackPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const HeaptrackPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "heaptrack";
  return checkHeaptrackRequest(request, CTX);
}

} // namespace

/* ----------------------------- Check ----------------------------- */

std::vector<std::string> heaptrackWrapArguments(const std::string& dir) {
  return {"-o", dir + "/run"};
}

bool heaptrackMapped(const std::string& mapsText) {
  // libheaptrack_preload.so in a program heaptrack started (1.3 to 1.5 map it
  // whether or not they export LD_PRELOAD to it), libheaptrack_inject.so in
  // one it attached to.
  return mapsText.find("libheaptrack_") != std::string::npos;
}

ReadinessResult checkHeaptrackRequest(const ReadinessRequest& request,
                                      const ReadinessContext& ctx) {
  return decideHeaptrack(request, ctx, nullptr);
}

ReadinessResult checkHeaptrackRequestWithMaps(const ReadinessRequest& request,
                                              const ReadinessContext& ctx,
                                              const std::string& mapsText) {
  return decideHeaptrack(request, ctx, &mapsText);
}

/* ----------------------------- HeaptrackProfiler ----------------------------- */

HeaptrackProfiler::HeaptrackProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "heaptrack");
  const ReadinessResult DECISION = decideNow(cfg_);
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    std::fprintf(stderr, "[heaptrack] no heap profile: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[heaptrack] %s\n", DECISION.report.hint.c_str());
    }
  }
}

HeaptrackProfiler::HeaptrackProfiler(const PerfConfig& cfg, std::string testName,
                                     std::shared_ptr<const HeaptrackPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "heaptrack");
}

void HeaptrackProfiler::beforeMeasure() {
  if (operatorNewReplaced(mapsOf(::getpid()))) {
    std::fprintf(
        stderr,
        "\n[heaptrack] WARNING: libtcmalloc is loaded and replaces operator new.\n"
        "[heaptrack] heaptrack hooks the malloc family only, so C++ allocations will\n"
        "[heaptrack] be MISSING from the trace. Use a build without tcmalloc\n"
        "[heaptrack] (-DVERNIER_LINK_TCMALLOC=OFF, the default) and do not preload it.\n\n");
  }
  if (plan_ && plan_->launch == LaunchContext::RUNNER_WRAPPED) {
    std::fprintf(stderr,
                 "[heaptrack] wrapping detected; heap profile will be written at process\n"
                 "[heaptrack] exit. Artifact directory: %s\n",
                 artifactDir_.c_str());
  } else if (plan_) {
    std::fprintf(stderr, "[heaptrack] wrapping detected; heaptrack writes its trace at process\n"
                         "[heaptrack] exit, where its -o option points.\n");
  }
}

void HeaptrackProfiler::afterMeasure(const Stats& /*s*/) {
  // heaptrack writes its trace at process exit; nothing to do per-measure.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeHeaptrackProfiler(const PerfConfig& cfg,
                                                const std::string& testName) {
  return std::make_unique<HeaptrackProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedHeaptrackProfiler(const PerfConfig& cfg,
                                                       const std::string& testName,
                                                       const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<HeaptrackProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("heaptrack", ::vernier::bench::checkHeaptrackRequest,
                                   ::vernier::bench::makePlannedHeaptrackProfiler,
                                   "apt install heaptrack (low-overhead heap profiler).")
