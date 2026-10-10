/**
 * @file ProfilerNsightChecks.cpp
 * @brief The Nsight backend's mode parser and readiness check, and libbench's
 * default registration of `nsight` and `ncu` with a passive profiler.
 */

#include "src/bench/inc/ProfilerNsightChecks.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

namespace vernier {
namespace bench {

namespace {

/** @brief How long the doctor's `<tool> --version` may run. */
constexpr int VERSION_PROBE_TIMEOUT_MS = 10000;

/** @brief The driver parameter file whose RmProfilingAdminOnly the ncu rows read. */
constexpr const char* DRIVER_PARAMS = "/proc/driver/nvidia/params";

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
  std::string out;
  for (const std::string& PART : parts) {
    out += (out.empty() ? "" : separator) + PART;
  }
  return out;
}

/** @brief "--profile <backend>[ --profile-args <words>]", the words as `bench run` reads them. */
std::string requestText(const std::string& backend, const std::string& profileArgs) {
  const std::string WORDS = join(valgrind_tool::modeWords(profileArgs), ",");
  return "--profile " + backend + (WORDS.empty() ? std::string{} : " --profile-args " + WORDS);
}

/** @brief The line of a `--version` output that names the version, else its last line. */
std::string versionLine(const std::string& output) {
  std::istringstream in(output);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("ersion") != std::string::npos) {
      return outputTail(line, 1);
    }
  }
  return outputTail(output, 1);
}

/** @brief True when @p driverParams says `RmProfilingAdminOnly: 1`. */
bool countersRootOnly(const std::string& driverParams) {
  std::ifstream in(driverParams);
  std::string line;
  constexpr std::string_view KEY = "RmProfilingAdminOnly:";
  while (std::getline(in, line)) {
    if (line.rfind(KEY, 0) == 0) {
      const std::size_t VALUE = line.find_first_not_of(" \t", KEY.size());
      return VALUE != std::string::npos && line[VALUE] == '1';
    }
  }
  return false;
}

std::string installHint(const std::string& tool) {
  return tool == "nsys"
             ? "Install the CUDA toolkit's Nsight Systems (nsys), or add its bin folder (for "
               "example /usr/local/cuda/bin) to PATH."
             : "Install the CUDA toolkit's Nsight Compute (ncu), or add its bin folder (for "
               "example /usr/local/cuda/bin) to PATH.";
}

/**
 * @brief The remedy of a request no Nsight tool started: the command that
 * captures it, and `bench run` where it wraps the mode.
 */
std::string wrapRemedy(const std::string& backend, NsightMode mode,
                       const std::string& profileArgs) {
  const std::string REQUEST = requestText(backend, profileArgs);
  switch (mode) {
  case NsightMode::Systems:
    return "Wrap it: nsys profile -o ./profile -t cuda,nvtx --force-overwrite true <this-binary> " +
           REQUEST + " [...]; or run it with bench run " + REQUEST +
           ", which wraps it and writes the summary reports.";
  case NsightMode::Compute:
    return "Wrap it, with few launches (ncu replays each one): ncu -o ./kernel_profile -f "
           "--target-processes all <this-binary> " +
           REQUEST + " --cycles 3 --repeats 1 [...]; or run it with bench run " + REQUEST +
           " --cycles 3 --repeats 1, which wraps it.";
  case NsightMode::ComputeReplay:
    break;
  }
  return "Wrap it, with few launches (ncu replays each one): ncu --metrics " +
         ReplayMetrics{}.toNcuMetricString() +
         " -o ./kernel_replay -f --target-processes all <this-binary> " + REQUEST +
         " --cycles 3 --repeats 1 [...]; bench run does not wrap a kernel replay.";
}

/** @brief "; the driver gives ... root only ..." for an ncu mode outside root, else "". */
std::string countersNote(NsightMode mode, const ReadinessContext& ctx,
                         const std::string& driverParams) {
  if (mode == NsightMode::Systems || ctx.euid() == 0 || !countersRootOnly(driverParams)) {
    return {};
  }
  return "; the driver gives GPU performance counters to root only (RmProfilingAdminOnly: 1 in " +
         driverParams + "), and this process is not root";
}

/** @brief The doctor's decision: whether the mode's tool runs here. */
ReadinessResult probeTool(const std::string& tool, NsightMode mode, const ReadinessContext& ctx,
                          const std::string& driverParams) {
  const auto RESOLVED = resolveExecutable(tool, ctx);
  if (!RESOLVED) {
    return readinessResult(ReadinessCause::MISSING, tool + " not found on PATH", installHint(tool));
  }
  if (!RESOLVED->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, RESOLVED->path + " is not an executable file",
                           "Reinstall it, or fix PATH so it finds a working " + tool + ".");
  }
  const ProbeResult VERSION =
      runBoundedProbe({RESOLVED->path, "--version"}, VERSION_PROBE_TIMEOUT_MS, ctx);
  if (!VERSION.succeeded()) {
    const std::string TAIL = outputTail(VERSION.output, 3);
    return readinessResult(ReadinessCause::UNUSABLE,
                           RESOLVED->path + " --version: " + VERSION.describe() +
                               (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Reinstall it, or fix PATH so it finds a working " + tool + ".");
  }
  return readinessResult(ReadinessCause::UNVERIFIED,
                         RESOLVED->path + " runs here (" + versionLine(VERSION.output) +
                             "); whether it captures the benchmark's GPU work is not checked "
                             "before the run" +
                             countersNote(mode, ctx, driverParams),
                         "");
}

/** @brief A run's decision: whether the mode's tool started this process. */
ReadinessResult decideRuntime(const std::string& backend, const std::string& tool, NsightMode mode,
                              const std::string& profileArgs, const ReadinessContext& ctx,
                              const std::string& driverParams) {
  const std::string SESSION = profiler_env::nsightSessionTool(ctx);
  if (SESSION.empty()) {
    if (!resolveExecutable(tool, ctx)) {
      return readinessResult(ReadinessCause::MISSING, tool + " not found on PATH",
                             installHint(tool));
    }
    return readinessResult(ReadinessCause::MISSING,
                           backend + " collects only when " + tool + " starts the process, and " +
                               tool + " did not start this one",
                           wrapRemedy(backend, mode, profileArgs));
  }
  if (SESSION != tool) {
    return readinessResult(ReadinessCause::UNSUPPORTED,
                           requestText(backend, profileArgs) + " needs " + tool +
                               ", and this process runs under " + SESSION,
                           wrapRemedy(backend, mode, profileArgs));
  }
  return readinessResult(ReadinessCause::UNVERIFIED,
                         SESSION +
                             " started this process and writes its report when the "
                             "process exits; whether it captured the benchmark's GPU work "
                             "is not checked from inside the process" +
                             countersNote(mode, ctx, driverParams),
                         "");
}

/** @brief The analysis a request with --profile-analyze promises, which Nsight has none of. */
ReadinessResult analysisDecision(const std::string& backend) {
  return readinessResult(
      ReadinessCause::UNSUPPORTED,
      "--profile-analyze: " + backend +
          " has no analysis of its own; the capture still runs and its report is kept",
      backend == "ncu" ? "Read the report with ncu --import after the process exits, and drop "
                         "--profile-analyze."
                       : "Read the report with nsys stats (an ncu report with ncu --import) after "
                         "the process exits, and drop --profile-analyze.",
      ReadinessStage::ANALYSIS);
}

} // namespace

std::optional<ReadinessResult> parseNsightMode(const std::string& backend,
                                               const std::string& profileArgs, NsightMode& mode) {
  const bool NCU = backend == "ncu";
  bool compute = NCU;
  bool replay = false;
  std::optional<ReadinessResult> refused;
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    if (WORD == "replay") {
      replay = true;
    } else if (!NCU && (WORD == "compute" || WORD == "ncu")) {
      compute = true;
    } else if (!refused) {
      refused =
          valgrind_tool::refusedWord(NCU ? "ncu" : "nsight", WORD,
                                     NCU ? std::vector<std::string>{"replay"}
                                         : std::vector<std::string>{"compute", "ncu", "replay"});
    }
  }
  mode = replay ? NsightMode::ComputeReplay : (compute ? NsightMode::Compute : NsightMode::Systems);
  return refused;
}

const char* nsightModeTool(NsightMode mode) { return mode == NsightMode::Systems ? "nsys" : "ncu"; }

ReadinessResult checkNsightRequestWith(const ReadinessRequest& request, const ReadinessContext& ctx,
                                       const std::string& driverParams) {
  const std::string BACKEND = request.backend == "ncu" ? "ncu" : "nsight";
  NsightMode mode = NsightMode::Systems;
  if (auto refused = parseNsightMode(BACKEND, request.profileArgs, mode)) {
    return *refused;
  }
  const std::string TOOL = nsightModeTool(mode);
  ReadinessResult result =
      request.scope == ReadinessScope::RUNTIME
          ? decideRuntime(BACKEND, TOOL, mode, request.profileArgs, ctx, driverParams)
          : probeTool(TOOL, mode, ctx, driverParams);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(std::move(result), analysisDecision(BACKEND));
}

ReadinessResult checkNsightRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  return checkNsightRequestWith(request, ctx, DRIVER_PARAMS);
}

namespace {

/**
 * @brief libbench's profiler for a request Nsight may capture: it names the
 * backend and its folder and does nothing else (no NVTX without CUDA). A CUDA
 * build's registration replaces it with the Nsight backend.
 */
std::unique_ptr<Profiler> makePassiveNsightProfiler(const PerfConfig& cfg,
                                                    const std::string& testName,
                                                    const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  const char* SUFFIX = ProfilerRegistry::canonicalName(cfg.profileTool) == "ncu" ? "ncu" : "nsight";
  return std::make_unique<detail::NoOpProfiler>(
      "nsight",
      profiler_env::resolveArtifactDir(cfg.profileTool, cfg.artifactRoot, testName, SUFFIX));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_FALLBACK("nsight", ::vernier::bench::checkNsightRequest,
                                    ::vernier::bench::makePassiveNsightProfiler,
                                    "Install NVIDIA Nsight Systems (nsys) or Nsight Compute "
                                    "(ncu), the tool the mode needs.",
                                    "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE")

VERNIER_REGISTER_READINESS_FALLBACK("ncu", ::vernier::bench::checkNsightRequest,
                                    ::vernier::bench::makePassiveNsightProfiler,
                                    "Install NVIDIA Nsight Compute (ncu).",
                                    "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE")
