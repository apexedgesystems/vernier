/**
 * @file ProfilerNsight.cu
 * @brief NVIDIA Nsight profiler integration for GPU benchmarks.
 *
 * nsys and ncu record a process only when they start it; neither can attach
 * to one that is already running. The backend therefore never launches either
 * tool: it marks each measured window with an NVTX range, stays passive under
 * a session, and outside one prints the command that captures the run.
 */

#include "src/bench/inc/ProfilerNsight.hpp"

#include <cstdio>
#include <memory>
#include <string>

#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerNsightChecks.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

using detail::shellQuote;

NsightProfiler::NsightProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {

  // The folder carries the name of the tool that was asked for: `.ncu` for
  // --profile ncu, `.nsight` for --profile nsight in any of its modes.
  const char* suffix = (cfg_.profileTool == "ncu") ? "ncu" : "nsight";
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, suffix);

  // The mode, by the readiness check's own parser: --profile ncu is Nsight
  // Compute, and replay stays the same --profile-args opt-in for both names.
  // The registry refuses a request holding a word the parser does not take
  // before it builds a profiler; built directly, the other words decide.
  (void)parseNsightMode(cfg_.profileTool == "ncu" ? "ncu" : "nsight", cfg_.profileArgs, mode_);

  // Once per test: the session that owns the capture, or the command that
  // would capture this run.
  const std::string SESSION = profiler_env::nsightSessionTool();
  if (!SESSION.empty()) {
    std::fprintf(stderr,
                 "[nsight] this process runs under %s, which writes the report when the "
                 "process exits.\n",
                 SESSION.c_str());
  } else {
    printWrapCommand();
  }
}

NsightProfiler::~NsightProfiler() { popRange(); }

void NsightProfiler::beforeMeasure() {
  // Name the measured window after the test: a labeled range in the nsys
  // timeline, and a range ncu's --nvtx filters can select. Popped in
  // afterMeasure(); never pushed twice.
#if VERNIER_NVTX_USABLE
  if (!nvtxRangePush_) {
    ::nvtxRangePushA(testName_.c_str());
    nvtxRangePush_ = true;
  }
#endif
}

void NsightProfiler::afterMeasure(const Stats& /*s*/) { popRange(); }

void NsightProfiler::popRange() noexcept {
#if VERNIER_NVTX_USABLE
  if (nvtxRangePush_) {
    ::nvtxRangePop();
    nvtxRangePush_ = false;
  }
#endif
}

void NsightProfiler::printWrapCommand() const {
  // The flags this run was given, so the printed command selects the same mode.
  // Every value that comes from the run is quoted as one shell word, so a
  // folder or an argument with a space or a quote in it stays one argument.
  std::string flags = "--profile " + shellQuote(cfg_.profileTool);
  if (!cfg_.profileArgs.empty()) {
    flags += " --profile-args " + shellQuote(cfg_.profileArgs);
  }

  if (mode_ == NsightMode::Systems) {
    const std::string OUTPUT = shellQuote(artifactDir_ + "/profile");
    std::fprintf(stderr,
                 "\n[nsight] No nsys session: nsys cannot attach to a running process, so this\n"
                 "[nsight] run is not captured. Start the binary under nsys:\n"
                 "[nsight]   nsys profile -o %s -t cuda,nvtx --force-overwrite true \\\n"
                 "[nsight]       <this-binary> %s [...]\n"
                 "[nsight] or let bench run start it, which also writes the summary reports:\n"
                 "[nsight]   bench run <this-binary> --profile nsight -- [...]\n\n",
                 OUTPUT.c_str(), flags.c_str());
    return;
  }

  // ncu replays every kernel launch many times, so the command keeps the
  // launch count small; a full-length run takes hours.
  const bool REPLAY = (mode_ == NsightMode::ComputeReplay);
  const std::string METRICS =
      REPLAY ? " --metrics " + shellQuote(replayMetrics_.toNcuMetricString()) : "";
  const std::string OUTPUT =
      shellQuote(artifactDir_ + (REPLAY ? "/kernel_replay" : "/kernel_profile"));
  std::fprintf(stderr,
               "\n[nsight] No ncu session: ncu cannot attach to a running process, so this\n"
               "[nsight] run is not captured. Start the binary under ncu, with few launches:\n"
               "[nsight]   ncu%s -o %s -f --target-processes all \\\n"
               "[nsight]       <this-binary> %s --cycles 3 --repeats 1 [...]\n",
               METRICS.c_str(), OUTPUT.c_str(), flags.c_str());
  if (!REPLAY) {
    std::fprintf(stderr,
                 "[nsight] or: bench run <this-binary> --profile ncu --cycles 3 --repeats 1 "
                 "-- [...]\n");
  }
  std::fprintf(stderr, "[nsight] Reading GPU performance counters needs root on some systems, "
                       "Jetson included.\n\n");
}

std::unique_ptr<Profiler> makeNsightProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<NsightProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

namespace vernier {
namespace bench {

namespace {

/** @brief The Nsight backend for a request the readiness check let collect. */
std::unique_ptr<Profiler> makePlannedNsightProfiler(const PerfConfig& cfg,
                                                    const std::string& testName,
                                                    const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::make_unique<NsightProfiler>(cfg, testName);
}

} // namespace

} // namespace bench
} // namespace vernier

// The checks are libbench's (ProfilerNsightChecks.cpp), which registers them
// with a passive profiler; these registrations replace that one with the
// Nsight backend in every build that has it.
VERNIER_REGISTER_READINESS_BACKEND("nsight", ::vernier::bench::checkNsightRequest,
                                   ::vernier::bench::makePlannedNsightProfiler,
                                   "Install NVIDIA Nsight Systems (nsys) or Nsight Compute "
                                   "(ncu), the tool the mode needs.",
                                   "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE")

// Nsight Compute as its own first-class name: the same backend, whose mode
// the tool name selects. Kernel replay's metrics need many launches, so it
// stays a pass of its own (--profile-args replay).
VERNIER_REGISTER_READINESS_BACKEND("ncu", ::vernier::bench::checkNsightRequest,
                                   ::vernier::bench::makePlannedNsightProfiler,
                                   "Install NVIDIA Nsight Compute (ncu).",
                                   "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE")
