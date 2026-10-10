/**
 * @file ProfilerMemcheck.cpp
 * @brief Valgrind Memcheck implementation.
 *
 * Memcheck checks the whole process when valgrind runs it; the backend only
 * reports where the artifacts belong. Whether valgrind's memcheck runs the
 * process is the readiness check's decision.
 */

#include "src/bench/inc/ProfilerMemcheck.hpp"

#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Mode and Check ----------------------------- */

std::optional<ReadinessResult> parseMemcheckMode(const std::string& profileArgs,
                                                 valgrind_tool::ValgrindMode& mode) {
  mode = valgrind_tool::ValgrindMode{};
  mode.tool = "memcheck";
  // The full leak check is the default; errors are the log's, not the status's.
  mode.options = {"--leak-check=full", "--error-exitcode=0"};
  bool trackOrigins = false;
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    if (WORD == "track-origins") {
      trackOrigins = true;
    } else if (WORD != "leak-full") {
      return valgrind_tool::refusedWord("memcheck", WORD, {"leak-full", "track-origins"});
    }
  }
  if (trackOrigins) {
    mode.options.push_back("--track-origins=yes");
    mode.unseen.push_back("--track-origins=yes");
  }
  return std::nullopt;
}

ReadinessResult checkMemcheckRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  valgrind_tool::ValgrindMode mode;
  if (auto refused = parseMemcheckMode(request.profileArgs, mode)) {
    return *refused;
  }
  ReadinessResult result = valgrind_tool::decideCollection(
      "memcheck", mode, request, ctx,
      valgrind_tool::wrapRemedy("memcheck", mode, request.profileArgs));
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: memcheck has no automatic analysis; its log is the "
                      "report, and the check still runs",
                      "Read memcheck's log after the process exits, and drop --profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

/* ----------------------------- MemcheckProfiler ----------------------------- */

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
  request.backend = "memcheck";
  return checkMemcheckRequest(request, CTX);
}

} // namespace

MemcheckProfiler::MemcheckProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Decided first and reported as the registry reports its own decisions; a
  // request that cannot collect creates nothing.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (ProfilerRegistry::instance().reportDecision("memcheck", DECISION)) {
    plan_ = readyPlan(DECISION);
    artifactDir_ = profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_,
                                                    "memcheck");
  }
}

MemcheckProfiler::MemcheckProfiler(const PerfConfig& cfg, std::string testName,
                                   std::shared_ptr<const valgrind_tool::ValgrindPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "memcheck");
}

void MemcheckProfiler::beforeMeasure() {
  // Memcheck runs continuously; nothing to toggle. Logs at process exit.
}

void MemcheckProfiler::afterMeasure(const Stats& /*s*/) {
  // Memcheck writes its log at process exit when running under valgrind.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeMemcheckProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<MemcheckProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedMemcheckProfiler(const PerfConfig& cfg,
                                                      const std::string& testName,
                                                      const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<MemcheckProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("memcheck", ::vernier::bench::checkMemcheckRequest,
                                   ::vernier::bench::makePlannedMemcheckProfiler,
                                   "apt install valgrind (memcheck is its default tool).")
