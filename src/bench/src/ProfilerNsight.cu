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
#include <cstdlib>
#include <string>

#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

namespace vernier {
namespace bench {

namespace {

/**
 * @brief @p word as one POSIX shell word: unchanged when every character is
 * safe, otherwise in single quotes with each quote written as '\''.
 */
std::string shellQuote(const std::string& word) {
  static constexpr const char* SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "0123456789_@%+=:,./-";
  if (!word.empty() && word.find_first_not_of(SAFE) == std::string::npos) {
    return word;
  }
  std::string quoted = "'";
  for (const char C : word) {
    if (C == '\'') {
      quoted += "'\\''";
    } else {
      quoted += C;
    }
  }
  quoted += "'";
  return quoted;
}

} // namespace

NsightProfiler::NsightProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {

  // The folder carries the name of the tool that was asked for: `.ncu` for
  // --profile ncu, `.nsight` for --profile nsight in any of its modes.
  const char* suffix = (cfg_.profileTool == "ncu") ? "ncu" : "nsight";
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, suffix);

  if (cfg_.profileTool == "ncu") {
    // First-class Nsight Compute (--profile ncu). Replay stays the same
    // --profile-args opt-in as the nsight spelling.
    const bool REPLAY = cfg_.profileArgs.find("replay") != std::string::npos;
    mode_ = REPLAY ? NsightMode::ComputeReplay : NsightMode::Compute;
  } else if (cfg_.profileArgs.find("replay") != std::string::npos) {
    mode_ = NsightMode::ComputeReplay;
  } else if (cfg_.profileArgs.find("ncu") != std::string::npos ||
             cfg_.profileArgs.find("compute") != std::string::npos) {
    mode_ = NsightMode::Compute;
  } else {
    mode_ = NsightMode::Systems;
  }

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

EnvReport checkNsightEnvironment() {
  const bool nsys = std::system("command -v nsys >/dev/null 2>&1") == 0;
  const bool ncu = std::system("command -v ncu  >/dev/null 2>&1") == 0;
  if (!nsys && !ncu) {
    return EnvReport{EnvReport::Status::Error, "neither nsys nor ncu found on PATH",
                     "Install CUDA toolkit + Nsight (devtools repo on Ubuntu)."};
  }
  if (!nsys) {
    return EnvReport{EnvReport::Status::Warning, "ncu present but nsys missing",
                     "Install nsight-systems-cli for timeline profiling."};
  }
  if (!ncu) {
    return EnvReport{EnvReport::Status::Warning, "nsys present but ncu missing",
                     "Install nsight-compute for kernel analysis."};
  }
  // Both present; under a Docker PID namespace attach-by-pid is unreliable, so
  // wrap nsys/ncu externally around the binary.
  if (std::system("grep -q docker /proc/1/cgroup 2>/dev/null") == 0) {
    return EnvReport{
        EnvReport::Status::Warning, "nsys + ncu available; running in Docker (PID namespace)",
        "Wrap nsys/ncu externally around the binary (attach-by-pid is unreliable here)."};
  }
  return EnvReport{EnvReport::Status::Ok, "nsys + ncu available", ""};
}

EnvReport checkNcuEnvironment() {
  if (std::system("command -v ncu >/dev/null 2>&1") != 0) {
    return EnvReport{EnvReport::Status::Error, "ncu not found on PATH",
                     "Install nsight-compute (CUDA toolkit devtools repo on Ubuntu)."};
  }
  if (profiler_env::isInContainer()) {
    return EnvReport{EnvReport::Status::Warning,
                     "ncu available; running in a container (PID namespace)",
                     "Use bench run --profile ncu (wraps ncu around the binary automatically)."};
  }
  return EnvReport{EnvReport::Status::Ok, "ncu available", ""};
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND(
    "nsight", ::vernier::bench::makeNsightProfiler, ::vernier::bench::checkNsightEnvironment,
    "Install NVIDIA Nsight tools (nsys/ncu) and ensure a CUDA-capable GPU is visible.")

// Nsight Compute as its own first-class name: same backend implementation,
// constructor selects Compute mode from the tool name. Kernel replay's rich
// metrics inherently need many launches, so it remains a separate pass from
// single-launch timing (--profile-args replay) -- the flag is still the one
// entry point.
VERNIER_REGISTER_PROFILER_BACKEND(
    "ncu", ::vernier::bench::makeNsightProfiler, ::vernier::bench::checkNcuEnvironment,
    "Install NVIDIA Nsight Compute (ncu) and ensure a CUDA-capable GPU is visible.")