/**
 * @file ProfilerHelgrind.cpp
 * @brief Valgrind Helgrind / DRD thread-error detector implementation.
 *
 * The selected tool checks the whole process when valgrind runs it; the
 * backend only reports where the artifacts belong. Whether that tool runs
 * the process is the readiness check's decision.
 */

#include "src/bench/inc/ProfilerHelgrind.hpp"

#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Mode and Check ----------------------------- */

std::optional<ReadinessResult> parseHelgrindMode(const std::string& profileArgs,
                                                 valgrind_tool::ValgrindMode& mode) {
  mode = valgrind_tool::ValgrindMode{};
  mode.tool = "helgrind";
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    if (WORD != "drd") {
      return valgrind_tool::refusedWord("helgrind", WORD, {"drd"});
    }
    mode.tool = "drd";
  }
  return std::nullopt;
}

ReadinessResult checkHelgrindRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  valgrind_tool::ValgrindMode mode;
  if (auto refused = parseHelgrindMode(request.profileArgs, mode)) {
    return *refused;
  }
  ReadinessResult result = valgrind_tool::decideCollection(
      "helgrind", mode, request, ctx,
      valgrind_tool::wrapRemedy("helgrind", mode, request.profileArgs));
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: " + mode.tool +
                          " has no automatic analysis; its log is the report, and the check "
                          "still runs",
                      "Read " + mode.tool +
                          "'s log after the process exits, and drop "
                          "--profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

/* ----------------------------- HelgrindProfiler ----------------------------- */

namespace {

std::shared_ptr<const valgrind_tool::ValgrindPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const valgrind_tool::ValgrindPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "helgrind";
  return checkHelgrindRequest(request, CTX);
}

} // namespace

HelgrindProfiler::HelgrindProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Decided first and reported as the registry reports its own decisions; a
  // request that cannot collect creates nothing.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (ProfilerRegistry::instance().reportDecision("helgrind", DECISION)) {
    plan_ = readyPlan(DECISION);
    artifactDir_ = profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_,
                                                    "helgrind");
  }
}

HelgrindProfiler::HelgrindProfiler(const PerfConfig& cfg, std::string testName,
                                   std::shared_ptr<const valgrind_tool::ValgrindPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "helgrind");
}

void HelgrindProfiler::beforeMeasure() {
  // Helgrind/DRD instrument continuously; nothing to toggle. Logs at exit.
}

void HelgrindProfiler::afterMeasure(const Stats& /*s*/) {
  // Valgrind writes its log at process exit when running under the tool.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeHelgrindProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<HelgrindProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedHelgrindProfiler(const PerfConfig& cfg,
                                                      const std::string& testName,
                                                      const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<HelgrindProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("helgrind", ::vernier::bench::checkHelgrindRequest,
                                   ::vernier::bench::makePlannedHelgrindProfiler,
                                   "apt install valgrind (helgrind + drd ship with it).")
