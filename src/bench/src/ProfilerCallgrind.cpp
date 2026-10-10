/**
 * @file ProfilerCallgrind.cpp
 * @brief Implementation of Valgrind Callgrind profiler backend.
 *
 * Under a manual `valgrind --tool=callgrind --instr-atstart=no` wrap, switches
 * instrumentation on for each measured window and off after it with
 * callgrind_control. A recording made by bench run's wrap covers the whole
 * process and is left alone. Whether valgrind's callgrind runs the process,
 * and how the wrap began, is the readiness check's decision.
 */

#include "src/bench/inc/ProfilerCallgrind.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Mode and Check ----------------------------- */

std::optional<ReadinessResult> parseCallgrindMode(const std::string& profileArgs,
                                                  valgrind_tool::ValgrindMode& mode) {
  mode = valgrind_tool::ValgrindMode{};
  mode.tool = "callgrind";
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    return valgrind_tool::refusedWord("callgrind", WORD, {});
  }
  return std::nullopt;
}

namespace {

ReadinessResult decideCallgrind(const ReadinessRequest& request, const ReadinessContext& ctx,
                                const valgrind_tool::ValgrindIdentity* identity) {
  valgrind_tool::ValgrindMode mode;
  if (auto refused = parseCallgrindMode(request.profileArgs, mode)) {
    return *refused;
  }
  // Without callgrind_control the window cannot be switched on, so the wrap
  // command leaves instrumentation on from the start.
  const auto CONTROL = resolveExecutable("callgrind_control", ctx);
  const bool HAS_CONTROL = CONTROL && CONTROL->executable;
  const std::vector<std::string> WINDOW =
      HAS_CONTROL ? std::vector<std::string>{"--instr-atstart=no"} : std::vector<std::string>{};
  ReadinessResult result = valgrind_tool::decideCollection(
      "callgrind", mode, request, ctx,
      valgrind_tool::wrapRemedy("callgrind", mode, request.profileArgs, WINDOW), identity);

  const auto PLAN = std::dynamic_pointer_cast<const valgrind_tool::ValgrindPlan>(result.plan);
  const bool RUNTIME = request.scope == ReadinessScope::RUNTIME;
  if (RUNTIME && result.collectionReady() && PLAN &&
      PLAN->launch == LaunchContext::MANUALLY_WRAPPED) {
    auto plan = std::make_shared<valgrind_tool::ValgrindPlan>(*PLAN);
    plan->canToggle = HAS_CONTROL;
    result.plan = plan;
    if (!HAS_CONTROL && result.report.status == EnvReport::Status::Ok) {
      ReadinessResult caveat = readinessResult(
          ReadinessCause::CAVEAT,
          "valgrind's callgrind runs this process, but callgrind_control is not on PATH: the "
          "measured window cannot be switched on and off, so the profile holds what the wrap "
          "records (nothing, for a wrap started with --instr-atstart=no)",
          "Put valgrind's callgrind_control on PATH, or run it with bench run --profile "
          "callgrind, which records the whole process.");
      caveat.plan = std::move(plan);
      result = std::move(caveat);
    }
  }
  if (!request.analyze) {
    return result;
  }

  // valgrind writes the profile when the process exits, after the last hook:
  // the annotation is bench run's, after that exit.
  if (RUNTIME) {
    if (PLAN && PLAN->launch == LaunchContext::RUNNER_WRAPPED) {
      if (result.report.status == EnvReport::Status::Ok) {
        result.report.message += "; bench run annotates the profile after the process exits";
      }
      return result;
    }
    return valgrind_tool::withAnalysis(
        std::move(result),
        readinessResult(ReadinessCause::UNSUPPORTED,
                        "--profile-analyze: valgrind writes the profile when this process "
                        "exits, after the benchmark's last hook, so the benchmark cannot "
                        "annotate it; the capture still runs",
                        "Run callgrind_annotate on the profile after the process exits, or run "
                        "it with bench run --profile callgrind --profile-analyze, which does.",
                        ReadinessStage::ANALYSIS));
  }
  const auto ANNOTATE = resolveExecutable("callgrind_annotate", ctx);
  if (!ANNOTATE || !ANNOTATE->executable) {
    return valgrind_tool::withAnalysis(
        std::move(result),
        readinessResult(ReadinessCause::MISSING,
                        "--profile-analyze needs callgrind_annotate, which ships with valgrind, "
                        "and it is not on PATH; the capture still runs",
                        "Install valgrind's callgrind_annotate, or drop --profile-analyze.",
                        ReadinessStage::ANALYSIS));
  }
  if (result.report.status == EnvReport::Status::Ok) {
    result.report.message +=
        "; bench run annotates the profile after the process exits with " + ANNOTATE->path;
  }
  return result;
}

} // namespace

ReadinessResult checkCallgrindRequest(const ReadinessRequest& request,
                                      const ReadinessContext& ctx) {
  return decideCallgrind(request, ctx, nullptr);
}

ReadinessResult checkCallgrindRequestWithIdentity(const ReadinessRequest& request,
                                                  const ReadinessContext& ctx,
                                                  const valgrind_tool::ValgrindIdentity& identity) {
  return decideCallgrind(request, ctx, &identity);
}

/* ----------------------------- Helpers ----------------------------- */

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
  request.backend = "callgrind";
  return checkCallgrindRequest(request, CTX);
}

#ifdef __linux__
/// Switch callgrind's instrumentation for this process on or off.
/// callgrind_control takes the process id as a trailing argument, not as an
/// option. It exits 0 whether or not the command reached the process, so its
/// status is not read.
void switchInstrumentation(const char* state) {
  const std::string CMD = std::string{"callgrind_control -i "} + state + " " +
                          std::to_string(::getpid()) + " >/dev/null 2>&1";
  [[maybe_unused]] const int RC = std::system(CMD.c_str());
}
#endif

} // namespace

/* ----------------------------- CallgrindProfiler Methods ----------------------------- */

CallgrindProfiler::CallgrindProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Decided first and reported as the registry reports its own decisions; a
  // request that cannot collect creates nothing.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (ProfilerRegistry::instance().reportDecision("callgrind", DECISION)) {
    plan_ = readyPlan(DECISION);
    artifactDir_ = profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_,
                                                    "callgrind");
  }
  applyPlan();
}

CallgrindProfiler::CallgrindProfiler(const PerfConfig& cfg, std::string testName,
                                     std::shared_ptr<const valgrind_tool::ValgrindPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "callgrind");
  applyPlan();
}

void CallgrindProfiler::applyPlan() {
  runningUnderValgrind_ = plan_ != nullptr;
  wrappedByRunner_ = plan_ && plan_->launch == LaunchContext::RUNNER_WRAPPED;
  canToggle_ = plan_ && plan_->canToggle;
}

void CallgrindProfiler::beforeMeasure() {
#ifdef __linux__
  if (canToggle_) {
    switchInstrumentation("on");
  }
#endif
}

void CallgrindProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (!runningUnderValgrind_) {
    return; // Not wrapped at all -- nothing to do.
  }
  if (canToggle_) {
    switchInstrumentation("off");
  }

  // Valgrind writes the profile when the process exits, to the file its
  // --callgrind-out-file names: callgrind.out in artifactDir_ under bench
  // run's wrap. Any other wrap names its own. The profile is complete only
  // then, so nothing here reads it: bench run checks it, and annotates it for
  // --profile-analyze, after the exit.
  std::printf("\n=== Callgrind Profile ===\n");
  if (wrappedByRunner_) {
    std::printf("Output: %s\n", artifactDir_.c_str());
    std::printf("   bench run checks the profile after valgrind has written it%s\n",
                cfg_.profileAnalyze ? ", then annotates it" : "");
  } else {
    std::printf("Output: where the wrap's --callgrind-out-file points (by default "
                "callgrind.out.<pid> in the working directory)\n");
    std::printf("   valgrind writes the profile when this process exits; read it then with\n");
    std::printf("   callgrind_annotate <profile> (or kcachegrind)\n");
  }
  std::printf("\n");
#endif
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeCallgrindProfiler(const PerfConfig& cfg,
                                                const std::string& testName) {
  return std::make_unique<CallgrindProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedCallgrindProfiler(const PerfConfig& cfg,
                                                       const std::string& testName,
                                                       const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<CallgrindProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("callgrind", ::vernier::bench::checkCallgrindRequest,
                                   ::vernier::bench::makePlannedCallgrindProfiler,
                                   "Install valgrind: apt install valgrind.")
