/**
 * @file ProfilerMassif.cpp
 * @brief Valgrind Massif heap profiler implementation.
 *
 * Massif records the whole process when valgrind runs it; the backend only
 * reports where the artifacts belong. Whether valgrind's massif runs the
 * process is the readiness check's decision.
 */

#include "src/bench/inc/ProfilerMassif.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Mode and Check ----------------------------- */

std::optional<ReadinessResult> parseMassifMode(const std::string& profileArgs,
                                               valgrind_tool::ValgrindMode& mode) {
  mode = valgrind_tool::ValgrindMode{};
  mode.tool = "massif";
  bool pages = false;
  bool stacks = false;
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    if (WORD == "pages") {
      pages = true;
    } else if (WORD == "stacks") {
      stacks = true;
    } else {
      return valgrind_tool::refusedWord("massif", WORD, {"pages", "stacks"});
    }
  }
  if (pages && stacks) {
    return readinessResult(ReadinessCause::CONFIGURATION,
                           "valgrind's massif cannot combine pages (--pages-as-heap=yes) with "
                           "stacks (--stacks=yes); choose one",
                           "Use --profile-args pages or --profile-args stacks.");
  }
  if (pages) {
    mode.options.push_back("--pages-as-heap=yes");
  }
  if (stacks) {
    mode.options.push_back("--stacks=yes");
  }
  mode.unseen = mode.options;
  return std::nullopt;
}

ReadinessResult checkMassifRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  valgrind_tool::ValgrindMode mode;
  if (auto refused = parseMassifMode(request.profileArgs, mode)) {
    return *refused;
  }
  ReadinessResult result = valgrind_tool::decideCollection(
      "massif", mode, request, ctx, valgrind_tool::wrapRemedy("massif", mode, request.profileArgs));
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: massif has no automatic analysis; the capture still "
                      "runs and its profile is kept",
                      "Read the profile with ms_print after the process exits, and drop "
                      "--profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

/* ----------------------------- MassifProfiler ----------------------------- */

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
  request.backend = "massif";
  return checkMassifRequest(request, CTX);
}

} // namespace

MassifProfiler::MassifProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "massif");
  const ReadinessResult DECISION = decideNow(cfg_);
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    std::fprintf(stderr, "[massif] no heap profile: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[massif] %s\n", DECISION.report.hint.c_str());
    }
  }
}

MassifProfiler::MassifProfiler(const PerfConfig& cfg, std::string testName,
                               std::shared_ptr<const valgrind_tool::ValgrindPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "massif");
}

void MassifProfiler::beforeMeasure() {
  // Massif samples continuously while running under valgrind; nothing to toggle.
}

void MassifProfiler::afterMeasure(const Stats& /*s*/) {
  // Massif writes its output file at process exit when running under valgrind.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeMassifProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<MassifProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedMassifProfiler(const PerfConfig& cfg,
                                                    const std::string& testName,
                                                    const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<MassifProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("massif", ::vernier::bench::checkMassifRequest,
                                   ::vernier::bench::makePlannedMassifProfiler,
                                   "apt install valgrind (massif ships with it).")
