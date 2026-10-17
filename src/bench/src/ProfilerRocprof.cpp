/**
 * @file ProfilerRocprof.cpp
 * @brief Backend for AMD's legacy rocprof.
 *
 * rocprof wraps the binary externally; the backend only reports where the
 * artifacts belong. Whether rocprof's injection is present is the readiness
 * check's decision, which is never Ok: the integration is not validated.
 */

#include "src/bench/inc/ProfilerRocprof.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

namespace vernier {
namespace bench {

namespace {

const char* const NOT_VALIDATED = "AMD collection is not validated (legacy rocprof)";

ReadinessResult rocprofMissing() {
  return readinessResult(ReadinessCause::MISSING, "rocprof not found on PATH",
                         "Install ROCm's rocprofiler (apt install rocprofiler on Debian and "
                         "Ubuntu).");
}

/** @brief rocprof's injection markers in the snapshot, named; "" when there is none. */
std::string injectionMarkers(const ReadinessContext& ctx) {
  std::string found;
  for (const char* NAME : {"ROCP_TOOL_LIB", "ROCPROFILER_LIBRARY"}) {
    if (ctx.get(NAME)) {
      found += (found.empty() ? "" : ", ") + std::string{NAME};
    }
  }
  const auto PRELOAD = ctx.get("LD_PRELOAD");
  if (PRELOAD && PRELOAD->find("rocprof") != std::string::npos) {
    found += (found.empty() ? "" : ", ") + std::string{"rocprof in LD_PRELOAD"};
  }
  return found;
}

std::string wrapRemedy(const std::vector<std::string>& flags, const std::string& profileArgs) {
  std::string command = "rocprof";
  for (const std::string& FLAG : flags) {
    command += " " + FLAG;
  }
  std::string words;
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    words += (words.empty() ? "" : ",") + WORD;
  }
  return "Wrap it: " + command + " -o ./results.csv <this-binary> --profile rocprof" +
         (words.empty() ? std::string{} : " --profile-args " + words) +
         " [...]; bench run does not wrap rocprof.";
}

std::shared_ptr<const RocprofPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const RocprofPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "rocprof";
  return checkRocprofRequest(request, CTX);
}

} // namespace

/* ----------------------------- Mode and Check ----------------------------- */

std::optional<ReadinessResult> parseRocprofMode(const std::string& profileArgs,
                                                std::vector<std::string>& flags) {
  flags.clear();
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    if (WORD != "stats" && WORD != "hsa-trace" && WORD != "hip-trace") {
      return valgrind_tool::refusedWord("rocprof", WORD, {"stats", "hsa-trace", "hip-trace"});
    }
    const std::string FLAG = "--" + WORD;
    if (std::find(flags.begin(), flags.end(), FLAG) == flags.end()) {
      flags.push_back(FLAG);
    }
  }
  return std::nullopt;
}

ReadinessResult checkRocprofRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  auto plan = std::make_shared<RocprofPlan>();
  if (auto refused = parseRocprofMode(request.profileArgs, plan->flags)) {
    return *refused;
  }
  ReadinessResult result;
  if (request.scope == ReadinessScope::RUNTIME) {
    const std::string MARKERS = injectionMarkers(ctx);
    if (!MARKERS.empty()) {
      plan->launch = LaunchContext::MANUALLY_WRAPPED;
      result =
          readinessResult(ReadinessCause::UNVERIFIED,
                          "rocprof's injection is present (" + MARKERS + "); " + NOT_VALIDATED, "");
    } else {
      plan->launch = LaunchContext::NOT_WRAPPED;
      result = !resolveExecutable("rocprof", ctx)
                   ? rocprofMissing()
                   : readinessResult(ReadinessCause::MISSING,
                                     "rocprof collects only when rocprof runs the process, and "
                                     "rocprof does not run this one",
                                     wrapRemedy(plan->flags, request.profileArgs));
    }
  } else {
    const auto TOOL = resolveExecutable("rocprof", ctx);
    if (!TOOL) {
      result = rocprofMissing();
    } else if (!TOOL->executable) {
      result = readinessResult(ReadinessCause::UNUSABLE, TOOL->path + " is not an executable file",
                               "Reinstall rocprofiler, or fix PATH so it finds a working "
                               "rocprof.");
    } else {
      plan->rocprof = TOOL->path;
      result = readinessResult(ReadinessCause::UNVERIFIED,
                               std::string{NOT_VALIDATED} + ": rocprof is " + TOOL->path +
                                   "; no AMD device access or capture is checked",
                               "");
    }
  }
  result.plan = std::move(plan);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: rocprof has no automatic analysis; its reports are read "
                      "as they are, and the capture still runs",
                      "Read the reports rocprof writes where its -o option points, and drop "
                      "--profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

/* ----------------------------- RocprofProfiler ----------------------------- */

RocprofProfiler::RocprofProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Reported as the registry reports its own decisions. An empty folder is
  // not capture evidence: a request that cannot collect creates none.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (!ProfilerRegistry::instance().reportDecision("rocprof", DECISION)) {
    return;
  }
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    return;
  }
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "rocprof");
}

RocprofProfiler::RocprofProfiler(const PerfConfig& cfg, std::string testName,
                                 std::shared_ptr<const RocprofPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "rocprof");
}

void RocprofProfiler::beforeMeasure() {
  // rocprof records the whole process when its injection is present.
}

void RocprofProfiler::afterMeasure(const Stats& /*s*/) {
  // rocprof writes its reports at process exit when wrapping; nothing to do
  // per-measure on the in-process side.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeRocprofProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<RocprofProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedRocprofProfiler(const PerfConfig& cfg,
                                                     const std::string& testName,
                                                     const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<RocprofProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND(
    "rocprof", ::vernier::bench::checkRocprofRequest, ::vernier::bench::makePlannedRocprofProfiler,
    "Install ROCm + rocprof (apt install rocprofiler on Debian/Ubuntu).", "ROCP_TOOL_LIB",
    "ROCPROFILER_LIBRARY", "LD_PRELOAD")
